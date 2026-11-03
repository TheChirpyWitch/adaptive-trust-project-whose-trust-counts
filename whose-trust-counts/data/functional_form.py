"""Functional-form checks for the credulity regression (main specification).

The model assumes trust is additive and linear in quality c_s and agreement a_us.
We check (1) curvature and interaction terms, (2) binned residuals of the linear
fit, and (3) what the coefficient scale implies. Respondent fixed effects throughout.
Writes data/derived/functional_form.csv.
"""
import csv
import os

import numpy as np

from estimate_gamma import D, build, demean, load


def ols_fe(rid, y, X):
    w = np.ones_like(y)
    yd = demean(y, [rid], w)
    Xd = np.column_stack([demean(X[:, j], [rid], w) for j in range(X.shape[1])])
    beta, *_ = np.linalg.lstsq(Xd, yd, rcond=None)
    resid = yd - Xd @ beta
    r2 = 1 - (resid ** 2).sum() / (yd ** 2).sum()
    return beta, resid, r2


def main():
    outlets, ratings = load()
    rid, dom, y, c, a = build(outlets, ratings)
    rows = []

    b, resid, r2 = ols_fe(rid, y, np.column_stack([c, a]))
    rows.append(dict(model="linear (main)", b_q=b[0], b_a=b[1], extra="", within_r2=r2))
    print(f"linear: b_q={b[0]:.3f} b_a={b[1]:.3f} gamma={b[1]/b.sum():.2f} within R2={r2:.3f}")

    bq, _, r2q = ols_fe(rid, y, np.column_stack([c, a, c ** 2, a ** 2]))
    rows.append(dict(model="+ squares", b_q=bq[0], b_a=bq[1],
                     extra=f"c^2={bq[2]:.3f}; a^2={bq[3]:.3f}", within_r2=r2q))
    print(f"+squares: {np.round(bq, 3)} within R2={r2q:.3f}")

    bi, _, r2i = ols_fe(rid, y, np.column_stack([c, a, c * a]))
    rows.append(dict(model="+ interaction", b_q=bi[0], b_a=bi[1],
                     extra=f"c*a={bi[2]:.3f}", within_r2=r2i))
    print(f"+interaction: {np.round(bi, 3)} within R2={r2i:.3f}")

    # Average marginal effects in the quadratic model, and the implied gamma.
    me_q = bq[0] + 2 * bq[2] * c.mean()
    me_a = bq[1] + 2 * bq[3] * a.mean()
    print(f"quadratic AMEs: q={me_q:.3f} a={me_a:.3f} gamma={me_a/(me_q+me_a):.2f}")
    rows.append(dict(model="quadratic, average marginal effects", b_q=me_q, b_a=me_a,
                     extra=f"gamma={me_a/(me_q+me_a):.2f}", within_r2=r2q))

    # Binned residuals of the linear fit by deciles of agreement and of quality.
    for name, v in (("agreement", a), ("quality", c)):
        qs = np.quantile(v, np.linspace(0, 1, 11))
        k = np.clip(np.searchsorted(qs, v, side="right") - 1, 0, 9)
        means = [resid[k == j].mean() for j in range(10)]
        print(f"binned residuals by {name}:", np.round(means, 3))
        rows.append(dict(model=f"binned residuals by {name} decile", b_q="", b_a="",
                         extra="; ".join(f"{m:.3f}" for m in means), within_r2=""))

    print(f"sd of trust (rescaled) = {y.std():.3f}; b_q+b_a = {b.sum():.3f}")
    with open(os.path.join(D, "functional_form.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "b_q", "b_a", "extra", "within_r2"])
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
