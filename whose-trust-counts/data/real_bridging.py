"""Step 2: compute the paper's ranking signals on real outlets and ask what they reward.

For each Pennycook & Rand (2019) study, every respondent rated every outlet, the same
dense setting as the model. We compute, per outlet:
  pooled       mean trust over all respondents (population-pooled)
  bridging_min min over Democrats / Republicans of group-mean trust (paper's bridging-min)
  bridging_mf  source intercept of the Community Notes-style MF (paper's bridging-MF,
               zero-mean user factor) and an uncentered variant closer to deployed CN
  audience     mean trust among respondents who recognise the outlet (readers proxy)
  factchecker  mean of 8 professional fact-checkers (Study 2 outlets only)

Then, on outlets with quality and lean data, regress each score on quality (Lin et al.)
and centrality, cent = 1 - |lean - 1/2| (lean from Robertson et al.), with a study
dummy. The implied centrality weight w = b_cent / (b_q + b_cent) is the empirical
counterpart of gamma in the paper's Eqs. 2-3. Outlet-bootstrap 95% CIs.
Writes data/derived/real_scores.csv and data/derived/real_bridging.csv.
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from attention_model_v2 import mf_intercepts  # noqa: E402

D = os.path.join(HERE, "derived")
B = int(sys.argv[1]) if len(sys.argv) > 1 else 1000


def mf_uncentered(R, lam_i=0.15, lam_f=0.03, iters=200):
    """Same MF without centering the user factor (as Community Notes is deployed)."""
    nU, nS = R.shape
    rng = np.random.default_rng(0)
    mu, iu, fu = R.mean(), np.zeros(nU), rng.normal(0, 0.1, nU)
    is_, fs = np.zeros(nS), rng.normal(0, 0.1, nS)
    L = np.diag([lam_i, lam_f])
    for _ in range(iters):
        X = np.column_stack([np.ones(nS), fs])
        sol = np.linalg.solve(X.T @ X + L, X.T @ (R - mu - is_[None, :]).T)
        iu, fu = sol[0], sol[1]
        X = np.column_stack([np.ones(nU), fu])
        sol = np.linalg.solve(X.T @ X + L, X.T @ (R - mu - iu[:, None]))
        is_, fs = sol[0], sol[1]
        mu = (R - iu[:, None] - is_[None, :] - np.outer(fu, fs)).mean()
    return is_


def fl(x):
    return np.nan if x in ("", None) else float(x)


def study_scores(ratings, study):
    rs = [r for r in ratings if r["study"] == study]
    doms = sorted({r["domain"] for r in rs})
    rids = sorted({r["rid"] for r in rs})
    di = {d: j for j, d in enumerate(doms)}
    ri = {r: i for i, r in enumerate(rids)}
    R = np.full((len(rids), len(doms)), np.nan)
    F = np.zeros_like(R)
    rep = np.zeros(len(rids))
    for r in rs:
        i, j = ri[r["rid"]], di[r["domain"]]
        R[i, j] = float(r["trust"])
        F[i, j] = r["familiar"] == "1"
        rep[i] = int(r["rep"])
    full = ~np.isnan(R).any(1)  # MF needs a dense matrix: keep complete raters
    R, F, rep = R[full], F[full], rep[full]
    is_c, _ = mf_intercepts(R, iters=200)
    out = {}
    for j, d in enumerate(doms):
        out[d] = dict(
            study=study, domain=d,
            pooled=R[:, j].mean(),
            dem=R[rep == 0, j].mean(), rep=R[rep == 1, j].mean(),
            bridging_min=min(R[rep == 0, j].mean(), R[rep == 1, j].mean()),
            bridging_mf=is_c[j],
            audience=R[F[:, j] == 1, j].mean() if F[:, j].sum() >= 20 else np.nan,
            n_audience=int(F[:, j].sum()))
    unc = mf_uncentered(R)
    for j, d in enumerate(doms):
        out[d]["bridging_mf_uncentered"] = unc[j]
    print(f"{study}: {full.sum()} complete raters of {len(full)}, {len(doms)} outlets; "
          f"corr(MF intercept, pooled mean) = {np.corrcoef(is_c, R.mean(0))[0, 1]:.3f}, "
          f"uncentered = {np.corrcoef(unc, R.mean(0))[0, 1]:.3f}")
    return out


def wfit(y, X, w):
    sw = np.sqrt(w)
    return np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]


def main():
    with open(os.path.join(D, "outlets.csv")) as f:
        outlets = {r["domain"]: r for r in csv.DictReader(f)}
    with open(os.path.join(D, "ratings.csv")) as f:
        ratings = list(csv.DictReader(f))
    rows = []
    for st in ("s1", "s2"):
        for d, r in study_scores(ratings, st).items():
            o = outlets[d]
            r.update(quality=fl(o["quality"]), lean=fl(o["lean"]),
                     factchecker=fl(o["factcheck"]) if st == "s2" else np.nan)
            rows.append(r)
    keys = list(rows[0])
    with open(os.path.join(D, "real_scores.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    signals = ["pooled", "bridging_min", "bridging_mf", "bridging_mf_uncentered",
               "audience", "factchecker"]
    rng = np.random.default_rng(1)
    res = []
    for sig in signals:
        sub = [r for r in rows if not np.isnan(r["quality"]) and not np.isnan(r["lean"])
               and not np.isnan(r[sig])]
        y = np.array([r[sig] for r in sub])
        q = np.array([r["quality"] for r in sub])
        cent = 1 - np.abs(np.array([r["lean"] for r in sub]) - 0.5)
        s2 = np.array([r["study"] == "s2" for r in sub], float)
        X = np.column_stack([np.ones_like(y), q, cent] + ([s2] if s2.std() > 0 else []))
        dom = np.unique([r["domain"] for r in sub], return_inverse=True)[1]
        b = wfit(y, X, np.ones_like(y))
        wt = b[2] / (b[1] + b[2])
        boots = []
        for _ in range(B):
            wd = np.bincount(rng.integers(0, dom.max() + 1, dom.max() + 1),
                             minlength=dom.max() + 1)[dom].astype(float)
            bb = wfit(y, X, wd)
            boots.append(bb[2] / (bb[1] + bb[2]))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rq = np.corrcoef(np.argsort(np.argsort(y)), np.argsort(np.argsort(q)))[0, 1]
        rc = np.corrcoef(np.argsort(np.argsort(y)), np.argsort(np.argsort(cent)))[0, 1]
        res.append(dict(signal=sig, w_cent=wt, lo=lo, hi=hi, b_q=b[1], b_cent=b[2],
                        spearman_quality=rq, spearman_cent=rc, n=len(y)))
        print(f"{sig:24s} w_cent={wt:.2f} [{lo:.2f},{hi:.2f}]  b_q={b[1]:.3f} "
              f"b_cent={b[2]:.3f}  rho(q)={rq:.2f} rho(cent)={rc:.2f}  n={len(y)}")
    # Cleaner test (Study 2): does a lay signal reward centrality *beyond* what
    # professional fact-checkers' ratings of the same outlets reflect?
    print("\nStudy 2, controlling for fact-checker ratings instead of Lin quality:")
    for sig in signals[:-1]:
        sub = [r for r in rows if r["study"] == "s2" and not np.isnan(r["lean"])
               and not np.isnan(r[sig]) and not np.isnan(r["factchecker"])]
        y = np.array([r[sig] for r in sub])
        q = np.array([r["factchecker"] for r in sub])
        cent = 1 - np.abs(np.array([r["lean"] for r in sub]) - 0.5)
        X = np.column_stack([np.ones_like(y), q, cent])
        b = wfit(y, X, np.ones_like(y))
        boots = []
        for _ in range(B):
            wd = np.bincount(rng.integers(0, len(y), len(y)), minlength=len(y)).astype(float)
            bb = wfit(y, X, wd)
            boots.append(bb[2] / (bb[1] + bb[2]))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        wt = b[2] / (b[1] + b[2])
        res.append(dict(signal=sig + "|factchecker", w_cent=wt, lo=lo, hi=hi, b_q=b[1],
                        b_cent=b[2], spearman_quality=np.nan, spearman_cent=np.nan, n=len(y)))
        print(f"{sig:24s} w_cent={wt:.2f} [{lo:.2f},{hi:.2f}]  b_fc={b[1]:.3f} b_cent={b[2]:.3f}  n={len(y)}")
    qq = np.array([r["quality"] for r in rows if not np.isnan(r["quality"]) and not np.isnan(r["lean"])])
    cc = 1 - np.abs(np.array([r["lean"] for r in rows if not np.isnan(r["quality"]) and not np.isnan(r["lean"])]) - 0.5)
    print(f"corr(quality, centrality) across outlets = {np.corrcoef(qq, cc)[0, 1]:.2f}")
    with open(os.path.join(D, "real_bridging.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0]))
        w.writeheader()
        w.writerows(res)


if __name__ == "__main__":
    main()
