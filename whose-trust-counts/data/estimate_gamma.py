"""Step 1: estimate credulity gamma from real trust ratings.

The model's perception rule (paper Eq. 1) is
    chat_us = (1 - gamma) c_s + gamma (1 - |beta_s - beta_u|).
We regress lay trust ratings (Pennycook & Rand 2019) on outlet quality c_s (Lin et al.
2023 aggregate expert score) and rater-outlet agreement A_us = 1 - |beta_s - beta_u|,
with outlet position beta_s from Robertson et al. (2018) audience partisanship and rater
position beta_u from self-reported conservatism. Both regressors live on [0,1] like in the
model, so gamma_hat = b_A / (b_c + b_A).

Uncertainty: pigeonhole bootstrap (resample respondents and outlets independently).
Writes data/derived/gamma_estimates.csv.
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "derived")
B = int(sys.argv[1]) if len(sys.argv) > 1 else 300


def load():
    with open(os.path.join(D, "outlets.csv")) as f:
        outlets = {r["domain"]: r for r in csv.DictReader(f)}
    with open(os.path.join(D, "ratings.csv")) as f:
        ratings = list(csv.DictReader(f))
    return outlets, ratings


def fl(x):
    return np.nan if x in ("", None) else float(x)


def build(outlets, ratings, quality="quality", lean="lean", rater="cons",
          studies=("s1", "s2"), familiar_only=False):
    rows = []
    for r in ratings:
        if r["study"] not in studies:
            continue
        if familiar_only and r["familiar"] != "1":
            continue
        o = outlets[r["domain"]]
        c, b, bu = fl(o[quality]), fl(o[lean]), fl(r[rater])
        if np.isnan(c) or np.isnan(b) or np.isnan(bu):
            continue
        rows.append((r["rid"], r["domain"], float(r["trust"]), c, 1 - abs(b - bu)))
    rid = np.unique([x[0] for x in rows], return_inverse=True)[1]
    dom = np.unique([x[1] for x in rows], return_inverse=True)[1]
    y, c, a = (np.array([x[k] for x in rows]) for k in (2, 3, 4))
    return rid, dom, y, c, a


def demean(v, groups, w):
    out = v.copy()
    for g in groups:
        s = np.bincount(g, w * out, minlength=g.max() + 1)
        n = np.bincount(g, w, minlength=g.max() + 1)
        out = out - (s / np.maximum(n, 1e-12))[g]
    return out


def fit(rid, dom, y, c, a, w, fe):
    """Weighted OLS; fe in {'none','resp'}. Returns (b_c, b_A)."""
    if fe == "none":
        X = np.column_stack([np.ones_like(c), c, a])
        sw = np.sqrt(w)
        beta = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
        return beta[1], beta[2]
    yd, cd, ad = (demean(v, [rid], w) for v in (y, c, a))
    X = np.column_stack([cd, ad])
    sw = np.sqrt(w)
    beta = np.linalg.lstsq(X * sw[:, None], yd * sw, rcond=None)[0]
    return beta[0], beta[1]


def fit_within_outlet(rid, dom, y, a, w, iters=30):
    """b_A with respondent AND outlet fixed effects (alternating projections)."""
    yd, ad = y.copy(), a.copy()
    for _ in range(iters):
        yd, ad = demean(yd, [rid, dom], w), demean(ad, [rid, dom], w)
    return float((w * ad * yd).sum() / (w * ad * ad).sum())


def estimate(data, fe, rng, B):
    rid, dom, y, c, a = data
    w = np.ones_like(y)
    bc, ba = fit(rid, dom, y, c, a, w, fe)
    g = ba / (bc + ba)
    nr, nd = rid.max() + 1, dom.max() + 1
    boots = []
    for _ in range(B):
        wr = np.bincount(rng.integers(0, nr, nr), minlength=nr)
        wd = np.bincount(rng.integers(0, nd, nd), minlength=nd)
        wb = (wr[rid] * wd[dom]).astype(float)
        keep = wb > 0
        args = (rid[keep], dom[keep], y[keep], c[keep], a[keep], wb[keep])
        # re-index groups after dropping zero-weight rows
        r2 = np.unique(args[0], return_inverse=True)[1]
        bcb, bab = fit(r2, args[1], args[2], args[3], args[4], args[5], fe)
        boots.append(bab / (bcb + bab))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(b_c=bc, b_A=ba, gamma=g, lo=lo, hi=hi, n=len(y),
                n_resp=int(nr), n_outlets=int(nd))


SPECS = [
    # name, build kwargs, fixed effects
    ("main: conservatism, Lin quality, Robertson lean", dict(), "resp"),
    ("no fixed effects", dict(), "none"),
    ("familiar outlets only", dict(familiar_only=True), "resp"),
    ("study 1 only", dict(studies=("s1",)), "resp"),
    ("study 2 only", dict(studies=("s2",)), "resp"),
    ("party (0/1) as rater position", dict(rater="rep"), "resp"),
    ("study 2, fact-checker quality", dict(studies=("s2",), quality="factcheck"), "resp"),
    ("study 2, MBFC slant (incl. fake sites)", dict(studies=("s2",), lean="mbfc_slant"), "resp"),
    ("study 2, fact-checker + MBFC", dict(studies=("s2",), quality="factcheck",
                                         lean="mbfc_slant"), "resp"),
]


def main():
    outlets, ratings = load()
    rng = np.random.default_rng(0)
    res = []
    for name, kw, fe in SPECS:
        data = build(outlets, ratings, **kw)
        r = estimate(data, fe, rng, B)
        rid, dom, y, c, a = data
        r["b_A_within_outlet"] = fit_within_outlet(rid, dom, y, a, np.ones_like(y))
        r["spec"] = name
        res.append(r)
        print(f"{name:42s} gamma={r['gamma']:.2f} [{r['lo']:.2f},{r['hi']:.2f}]  "
              f"b_c={r['b_c']:.3f} b_A={r['b_A']:.3f} (within-outlet b_A={r['b_A_within_outlet']:.3f})"
              f"  n={r['n']} resp={r['n_resp']} outlets={r['n_outlets']}", flush=True)
    with open(os.path.join(D, "gamma_estimates.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["spec", "gamma", "lo", "hi", "b_c", "b_A",
                                          "b_A_within_outlet", "n", "n_resp", "n_outlets"])
        w.writeheader()
        for r in res:
            w.writerow({k: r[k] for k in w.fieldnames})


if __name__ == "__main__":
    main()
