"""Rater-panel composition audit on real trust ratings.

The model predicts how the pooled signals respond to who sits on the rater panel:
population-mean trust and bridging-MF follow how many raters each group contributes,
while bridging-min ignores group sizes and responds only to how beliefs are spread
within each group. This script applies the same panel changes to the real
Pennycook & Rand (2019) ratings, holding panel size fixed:

  share  70% Republican vs. 70% Democratic panels (uniform within each party)
  null   two panels drawn from the sample's own composition (sampling noise only)
  tilt   raters weighted by exp(+2(cons - 1/2)) vs. exp(-2(cons - 1/2)), where cons is
         self-reported conservatism in [0, 1]; this shifts both group sizes and the
         spread of views within each party, as in the model's tilted panel

For each signal we compute outlets' percentile rank within the study under the
right-leaning and the left-leaning panel, and regress the rank change (right minus
left) on outlet lean (Robertson et al., 0 = left, 1 = right) and quality (Lin et al.),
with a study dummy. A positive lean slope means right-leaning outlets gain rank when
the panel leans right. 95% CIs resample raters (new panels) and outlets together.
This tests the scoring mechanism on real ratings, not producer adaptation.
Writes data/derived/panel_audit.csv.
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from attention_model_v2 import mf_intercepts  # noqa: E402

D = os.path.join(HERE, "derived")
B = int(sys.argv[1]) if len(sys.argv) > 1 else 200
SIGNALS = ["pooled", "bridging_min", "bridging_mf", "bridging_mf_uncentered"]


def fl(x):
    return np.nan if x in ("", None) else float(x)


def load_study(ratings, study):
    rs = [r for r in ratings if r["study"] == study]
    doms = sorted({r["domain"] for r in rs})
    rids = sorted({r["rid"] for r in rs})
    di = {d: j for j, d in enumerate(doms)}
    ri = {r: i for i, r in enumerate(rids)}
    R = np.full((len(rids), len(doms)), np.nan)
    rep = np.zeros(len(rids))
    cons = np.full(len(rids), np.nan)
    for r in rs:
        i = ri[r["rid"]]
        R[i, di[r["domain"]]] = float(r["trust"])
        rep[i] = int(r["rep"])
        cons[i] = fl(r["cons"])
    keep = ~np.isnan(R).any(1) & ~np.isnan(cons)
    return doms, R[keep], rep[keep], cons[keep]


def scores(R, rep):
    dem, rp = R[rep == 0].mean(0), R[rep == 1].mean(0)
    return dict(pooled=R.mean(0), bridging_min=np.minimum(dem, rp),
                bridging_mf=mf_intercepts(R, iters=100)[0],
                bridging_mf_uncentered=mf_intercepts(R, iters=100, center=False)[0])


def pct_rank(x):
    return np.argsort(np.argsort(x)) / (len(x) - 1)


def panel_weights(kind, side, rep, cons):
    """Sampling weights for a right-leaning (side=+1) or left-leaning (side=-1) panel."""
    if kind == "null":
        return np.full(len(rep), 1 / len(rep))
    if kind == "share":
        target = 0.7 if side > 0 else 0.3
        w = np.where(rep == 1, target / (rep == 1).sum(), (1 - target) / (rep == 0).sum())
    else:
        w = np.exp(side * 2 * (cons - 0.5))
    return w / w.sum()


def rank_shift(R, rep, cons, kind, rng):
    n = len(R)
    out = {}
    for side in (+1, -1):
        idx = rng.choice(n, n, replace=True, p=panel_weights(kind, side, rep, cons))
        s = scores(R[idx], rep[idx])
        out[side] = {k: pct_rank(v) for k, v in s.items()}
    return {k: out[+1][k] - out[-1][k] for k in SIGNALS}


def ols(y, X):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def main():
    with open(os.path.join(D, "outlets.csv")) as f:
        outlets = {r["domain"]: r for r in csv.DictReader(f)}
    with open(os.path.join(D, "ratings.csv")) as f:
        ratings = list(csv.DictReader(f))
    studies = {st: load_study(ratings, st) for st in ("s1", "s2")}
    for st, (doms, R, rep, cons) in studies.items():
        for kind in ("share", "tilt"):
            w = panel_weights(kind, +1, rep, cons)
            print(f"{st} {kind}: right panel {w[rep == 1].sum():.2f} Rep, "
                  f"left panel {panel_weights(kind, -1, rep, cons)[rep == 1].sum():.2f} Rep")

    # Outlet covariates (rows ordered study by study, outlets with lean and quality only).
    meta = []
    for st, (doms, *_ ) in studies.items():
        for j, d in enumerate(doms):
            q, ln = fl(outlets[d]["quality"]), fl(outlets[d]["lean"])
            if not (np.isnan(q) or np.isnan(ln)):
                meta.append((st, j, d, q, ln))
    lean = np.array([m[4] for m in meta])
    qual = np.array([m[3] for m in meta])
    s2 = np.array([m[0] == "s2" for m in meta], float)
    dom_id = np.unique([m[2] for m in meta], return_inverse=True)[1]
    X = np.column_stack([np.ones_like(lean), lean, qual, s2])
    # Sensitivity (Study 2 only): MBFC slant for position, fact-checker ratings for quality.
    s2meta = [(j, fl(outlets[d]["mbfc_slant"]), fl(outlets[d]["factcheck"]))
              for j, d in enumerate(studies["s2"][0])]
    s2meta = [m for m in s2meta if not (np.isnan(m[1]) or np.isnan(m[2]))]
    X2 = np.column_stack([np.ones(len(s2meta)), [m[1] for m in s2meta], [m[2] for m in s2meta]])

    rng = np.random.default_rng(7)
    rows = []
    for kind in ("null", "share", "tilt"):
        draws2 = {k: [] for k in SIGNALS}
        draws = {k: [] for k in SIGNALS}
        absd = {k: [] for k in SIGNALS}
        for b in range(B):
            shift = {st: rank_shift(R, rep, cons, kind, rng)
                     for st, (doms, R, rep, cons) in studies.items()}
            # Resample outlets (by domain, across both studies) for the regression.
            pick = rng.integers(0, dom_id.max() + 1, dom_id.max() + 1)
            wd = np.bincount(pick, minlength=dom_id.max() + 1)[dom_id].astype(float)
            sw = np.sqrt(wd)
            for k in SIGNALS:
                y = np.array([shift[m[0]][k][m[1]] for m in meta])
                draws[k].append(ols(y * sw, X * sw[:, None])[1])
                y2 = np.array([shift["s2"][k][m[0]] for m in s2meta])
                p2 = rng.integers(0, len(y2), len(y2))
                draws2[k].append(ols(y2[p2], X2[p2])[1])
                absd[k].append(np.mean(np.concatenate([np.abs(shift[st][k]) for st in shift])))
            if (b + 1) % 50 == 0:
                print(f"  {kind}: {b + 1}/{B}")
        for k in SIGNALS:
            sl = np.array(draws[k])
            ad = np.array(absd[k])
            rows.append(dict(panel=kind, signal=k, lean_slope=sl.mean(),
                             lo=np.percentile(sl, 2.5), hi=np.percentile(sl, 97.5),
                             mean_abs_shift=ad.mean(), n_outlets=len(meta),
                             s2_mbfc_fc_slope=np.mean(draws2[k]),
                             s2_lo=np.percentile(draws2[k], 2.5),
                             s2_hi=np.percentile(draws2[k], 97.5), n_s2=len(s2meta), B=B))
            print(f"{kind:5s} {k:24s} slope on lean {sl.mean():+.3f} "
                  f"[{np.percentile(sl, 2.5):+.3f},{np.percentile(sl, 97.5):+.3f}]  "
                  f"mean |rank shift| {ad.mean():.3f}  S2 MBFC/fact-checker slope "
                  f"{np.mean(draws2[k]):+.3f} [{np.percentile(draws2[k], 2.5):+.3f},"
                  f"{np.percentile(draws2[k], 97.5):+.3f}]")

    with open(os.path.join(D, "panel_audit.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
