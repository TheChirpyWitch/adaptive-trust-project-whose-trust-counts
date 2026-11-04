"""Is gamma_hat an artifact of how quality and positions are scaled?

gamma_hat = b_A / (b_c + b_A) compares two coefficients, so it depends on the units
of outlet quality c_s and of agreement 1 - |beta_s - beta_u|. The main estimate puts
both on [0,1] as published (Lin quality score; Robertson lean mapped from [-1,1]).
Here we re-estimate under other monotone rescalings of quality, outlet lean and
rater position. Writes data/derived/gamma_scaling.csv.
Usage: python data/gamma_scaling.py [bootstrap reps, default 200]
"""
import csv, os, sys
import numpy as np
from estimate_gamma import load, build, estimate, D

B = int(sys.argv[1]) if len(sys.argv) > 1 else 200


def pct(vals):
    """Percentile rank in [0,1] (ties averaged)."""
    v = np.asarray(vals, float)
    order = v.argsort().argsort().astype(float)
    for x in np.unique(v):                      # average ranks for ties
        order[v == x] = order[v == x].mean()
    return order / max(len(v) - 1, 1)


def minmax(vals):
    v = np.asarray(vals, float)
    return (v - v.min()) / (v.max() - v.min())


def transform_outlets(outlets, key, f):
    doms = [d for d, r in outlets.items() if r[key] not in ("", None)]
    new = f([float(outlets[d][key]) for d in doms])
    out = {d: dict(r) for d, r in outlets.items()}
    for d, v in zip(doms, new):
        out[d][key] = v
    return out


def transform_raters(ratings, f):
    by_rid = {}
    for r in ratings:
        if r["cons"] != "":
            by_rid[r["rid"]] = float(r["cons"])
    rids = list(by_rid)
    new = dict(zip(rids, f([by_rid[i] for i in rids])))
    return [dict(r, cons=new.get(r["rid"], "")) for r in ratings]


def main():
    outlets, ratings = load()
    variants = [
        ("main (as published)", outlets, ratings),
        ("quality: min-max to [0,1]", transform_outlets(outlets, "quality", minmax), ratings),
        ("quality: percentile rank", transform_outlets(outlets, "quality", pct), ratings),
        ("lean: min-max to [0,1]", transform_outlets(outlets, "lean", minmax), ratings),
        ("lean: percentile rank", transform_outlets(outlets, "lean", pct), ratings),
        ("rater position: percentile rank", outlets, transform_raters(ratings, pct)),
        ("all three: percentile rank",
         transform_outlets(transform_outlets(outlets, "quality", pct), "lean", pct),
         transform_raters(ratings, pct)),
    ]
    rng = np.random.default_rng(0)
    res = []
    for name, o, r in variants:
        e = estimate(build(o, r), "resp", rng, B)
        e["spec"] = name
        res.append(e)
        print(f"{name:36s} gamma={e['gamma']:.2f} [{e['lo']:.2f},{e['hi']:.2f}]", flush=True)
    with open(os.path.join(D, "gamma_scaling.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["spec", "gamma", "lo", "hi", "b_c", "b_A", "n"])
        w.writeheader()
        for e in res:
            w.writerow({k: e[k] for k in w.fieldnames})


if __name__ == "__main__":
    main()
