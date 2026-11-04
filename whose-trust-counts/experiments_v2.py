"""Experiment grid for model v2. Usage: python experiments_v2.py <name>

Writes results/<name>.csv (one row per run, tail means) and
results/ts/<name>.npz (per-step time series, rows in the same order as the CSV).
"""
import sys, os, csv, json, itertools, numpy as np
from multiprocessing import Pool
from attention_model_v2 import AttentionModelV2, POLICIES, ACCESS_METRICS

POPS = ["balanced_archetypes", "credibility_seeking", "demanding", "entertainment_credulous"]
METRICS = ["aud_c", "mean_c", "aud_e", "coverage_spread", "exposure",
           "perc_minus_true_c", "true_sat", "perc_sat", "gini"]
TS_METRICS = ["aud_c", "mean_c", "aud_e", "coverage_spread",
              "spread_axis1", "spread_axis2", "exposure", "rewires"]
T, TAIL = 400, 50
PROFIT_K = 20       # trial length for the profit-based adaptation rule (pilot: 10-40 and T up to 6000 give the same pattern)

def run(cfg):
    cfg = dict(cfg)
    steps = cfg.pop("T", T)          # a grid may set its own horizon
    tail = cfg.pop("tail", TAIL)     # steps averaged for the reported outcome
    varied = cfg.pop("varied", None) # label only (sensitivity grid)
    transfer = cfg.pop("transfer", None)
    if transfer is not None:
        # Evolve coverage in a full adapting run, then start a fresh run from the
        # same seed (same initial accuracy, entertainment, network) with coverage
        # frozen at the evolved positions: each source's own ("matched") or a
        # random permutation of them ("shuffled").
        src = AttentionModelV2(**dict(cfg, adapt_beta=True)).run(steps)
        beta = src.X[:, 2].copy()
        if transfer == "shuffled":
            beta = np.random.default_rng([cfg["seed"], 29]).permutation(beta)
        m = AttentionModelV2(**dict(cfg, adapt_beta=False, beta_init=tuple(beta))).run(steps)
    else:
        m = AttentionModelV2(**cfg).run(steps)
    ts = np.array([m.h[k] for k in TS_METRICS], dtype=np.float32).T   # steps × metrics
    out = dict(cfg)
    if transfer is not None:
        out["transfer"] = transfer
        out["adapt_beta"] = False
    out["T"] = steps
    out["tail"] = tail
    if varied is not None:
        out["varied"] = varied
    out.setdefault("adapt_beta", True)
    out.setdefault("score_norm", "rank")
    for k in METRICS:
        out[k] = float(np.mean(m.h[k][-tail:]))
    out["rewires"] = float(np.mean(m.h["rewires"][-tail:]))
    out.setdefault("belief_dims", 1)
    out["coverage_mean"] = float(np.mean(m.X[:, 2]))            # final step, axis 1
    out["center_dist"] = float(np.mean(np.abs(m.X[:, 2] - 0.5)))
    out.setdefault("prior_dist", "uniform")
    for k in [f"{m}_{g}" for m in ACCESS_METRICS for g in "LCR"]:
        out[k] = float(np.nanmean(m.h[k][-tail:]))      # group-level access (belief thirds)
    for k in ["spread_axis1", "spread_axis2"]:
        out[k] = float(np.mean(m.h[k][-tail:]))
    return out, ts

def run_indexed(item):
    i, cfg = item
    return i, run(cfg)

def grid(name):
    seeds = range(10)
    if name == "main":
        return [dict(user_dist=p, policy=pol, rewire_prob=rp, seed=s)
                for p, pol, rp, s in itertools.product(POPS, POLICIES, [0.5, 0.0], seeds)]
    if name == "kappa":
        return [dict(user_dist="balanced_archetypes", policy=pol, kappa_c=kc, seed=s)
                for pol, kc, s in itertools.product(POLICIES, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8], seeds)]
    if name == "gamma":
        return [dict(user_dist=f"gamma_{g}", policy=pol, seed=s)
                for pol, g, s in itertools.product(POLICIES, [0.0, 0.2, 0.4, 0.6, 0.8], seeds)]
    if name == "frozen_beta":
        return [dict(user_dist=f"gamma_{g}", policy=pol, adapt_beta=ab, seed=s)
                for pol, g, ab, s in itertools.product(["crowd_trust", "bridging", "expert"],
                                                       [0.0, 0.2, 0.4, 0.6, 0.8], [True, False], seeds)]
    if name == "bridging_mf":
        return [dict(user_dist=f"gamma_{g}", policy=pol, adapt_beta=ab, seed=s)
                for pol, g, ab, s in itertools.product(["bridging_mf"],
                                                       [0.0, 0.2, 0.4, 0.6, 0.8], [True, False], seeds)]
    if name == "expert_realism":
        return [dict(user_dist=d, policy="expert", expert_noise=en, expert_coverage=ec, seed=s)
                for d, en, ec, s in itertools.product(["balanced_archetypes", "gamma_0.6"],
                                                      [0.05, 0.1, 0.2, 0.3], [1.0, 0.5, 0.25], seeds)]
    if name == "norm_check":
        return [dict(user_dist=f"gamma_{g}", policy=pol, adapt_beta=ab, score_norm=sn, seed=s)
                for pol, g, ab, sn, s in itertools.product(
                    ["crowd_trust", "bridging", "bridging_mf", "expert"],
                    [0.0, 0.4, 0.6, 0.8], [True, False], ["rank", "minmax"], seeds)]
    # ── Final paper grids: rank-normalised scores (default), more seeds ──
    TRUST = ["crowd_trust", "bridging", "bridging_mf", "expert"]
    if name == "paper_gamma":
        s30 = range(30)
        on = [dict(user_dist=f"gamma_{g}", policy=pol, seed=s)
              for pol, g, s in itertools.product(POLICIES, [0.0, 0.2, 0.4, 0.6, 0.8], s30)]
        off = [dict(user_dist=f"gamma_{g}", policy=pol, adapt_beta=False, seed=s)
               for pol, g, s in itertools.product(TRUST, [0.0, 0.2, 0.4, 0.6, 0.8], s30)]
        return on + off
    if name == "robust_beliefs":
        # 2-D belief space (bridging groups split on axis 1 only) and a
        # polarized 1-D population with no users near the centre.
        gs, s20 = [0.0, 0.39, 0.6, 0.8], range(20)
        out = []
        for extra in [dict(belief_dims=2), dict(prior_dist="bimodal")]:
            out += [dict(user_dist=f"gamma_{g}", policy=pol, seed=s, **extra)
                    for pol, g, s in itertools.product(POLICIES, gs, s20)]
            out += [dict(user_dist=f"gamma_{g}", policy=pol, adapt_beta=False, seed=s, **extra)
                    for pol, g, s in itertools.product(TRUST, gs, s20)]
        return out
    if name == "empirical_gamma":
        # gamma estimated from Pennycook & Rand (2019) ratings (data/estimate_gamma.py):
        # party-position spec 0.22, main 0.39 with 95% CI [0.29, 0.48], familiar-only 0.58
        gs, s30 = [0.22, 0.29, 0.39, 0.48, 0.58], range(30)
        on = [dict(user_dist=f"gamma_{g}", policy=pol, seed=s)
              for pol, g, s in itertools.product(POLICIES, gs, s30)]
        off = [dict(user_dist=f"gamma_{g}", policy=pol, adapt_beta=False, seed=s)
               for pol, g, s in itertools.product(TRUST, gs, s30)]
        return on + off
    if name == "paper_kappa":
        return [dict(user_dist=f"gamma_{g}", policy=pol, kappa_c=kc, seed=s)
                for pol, g, kc, s in itertools.product(POLICIES, [0.2, 0.6],
                                                       [0.2, 0.4, 0.6, 0.8, 1.0, 1.2], range(20))]
    if name == "paper_pops":
        return [dict(user_dist=p, policy=pol, rewire_prob=rp, seed=s)
                for p, pol, rp, s in itertools.product(POPS, POLICIES, [0.5, 0.0], range(20))]
    if name == "paper_expert":
        return [dict(user_dist=d, policy="expert", expert_noise=en, expert_coverage=ec, seed=s)
                for d, en, ec, s in itertools.product(["gamma_0.2", "gamma_0.6"],
                                                      [0.05, 0.1, 0.2, 0.3], [1.0, 0.5, 0.25], range(20))]
    # ── Validation grids ──
    if name == "sensitivity":
        # One-at-a-time sweep of fixed parameters around the defaults, at the
        # estimated credulity (0.39) and high credulity (0.8).
        alts = {"lam": [0.1, 0.5, 0.7], "n_sources": [15, 60], "n_users": [300, 1200],
                "k": [2, 5], "sigma_prop": [0.025, 0.1], "mu_experiment": [0.1, 0.4],
                "rank_temp": [0.05, 0.2], "noise_c": [0.05, 0.2],
                "social_proof_weight": [0.0, 0.6]}
        variants = [dict(varied="default")] + [
            {"varied": p, p: v} for p, vals in alts.items() for v in vals]
        return [dict(user_dist=f"gamma_{g}", policy=pol, adapt_beta=ab, seed=s, **var)
                for var in variants
                for pol, g, ab, s in itertools.product(
                    ["crowd_trust", "bridging", "bridging_mf"], [0.39, 0.8], [True, False], range(20))]
    if name == "horizon":
        # Same cells as the main sweep, run three times longer, to check steady state.
        gs, s20 = [0.39, 0.6, 0.8], range(20)
        on = [dict(user_dist=f"gamma_{g}", policy=pol, seed=s, T=1200)
              for pol, g, s in itertools.product(POLICIES, gs, s20)]
        off = [dict(user_dist=f"gamma_{g}", policy=pol, adapt_beta=False, seed=s, T=1200)
               for pol, g, s in itertools.product(TRUST, gs, s20)]
        return on + off
    if name == "fine_gamma":
        # Locate where frozen-coverage protection fails (predicted 2/3 and 4/5).
        gs = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
        return [dict(user_dist=f"gamma_{g}", policy=pol, adapt_beta=ab, seed=s)
                for pol, g, ab, s in itertools.product(
                    ["bridging", "bridging_mf"], gs, [True, False], range(30))]
    if name == "move_cost":
        # Repositioning is no longer free: a coverage move of distance d must
        # beat the current strategy by kappa_move * d.
        return [dict(user_dist=f"gamma_{g}", policy=pol, kappa_move=km, seed=s, varied="kappa_move")
                for pol, g, km, s in itertools.product(
                    ["crowd_trust", "bridging", "bridging_mf"], [0.39, 0.6, 0.8],
                    [0.02, 0.1, 0.5], range(20))]
    if name == "sparse_mf":
        # Community Notes-like rating: users rate what they read plus a sparse,
        # optionally self-selected (homophilous) sample of other sources.
        return [dict(user_dist=f"gamma_{g}", policy="bridging_mf", adapt_beta=ab,
                     rating_density=rd, rating_homophily=rh, seed=s)
                for g, ab, rd, rh, s in itertools.product(
                    [0.39, 0.6, 0.8], [True, False], [0.05, 0.2], [0.0, 5.0], range(20))]
    if name == "pilot_long":
        # How long until runs level off? Key cells, 3000 steps.
        return [dict(user_dist=f"gamma_{g}", policy=pol, adapt_beta=ab, seed=s, T=3000)
                for pol, g, ab, s in itertools.product(
                    TRUST, [0.39, 0.6, 0.8], [True, False], range(10))] + \
               [dict(user_dist=f"gamma_{g}", policy=pol, seed=s, T=3000)
                for pol, g, s in itertools.product(["popularity", "engagement"], [0.39], range(10))]
    if name == "norm_check_extra":
        # Seeds 10-29 for the normalization check (seeds 0-9 are in norm_check.csv).
        return [dict(user_dist=f"gamma_{g}", policy=pol, adapt_beta=ab, score_norm=sn, seed=s)
                for pol, g, ab, sn, s in itertools.product(
                    TRUST, [0.0, 0.4, 0.6, 0.8], [True, False], ["rank", "minmax"], range(10, 30))]
    if name == "expert_lean":
        # Experts as a biased rater pool: they also reward agreement with their
        # own position (centered, or left of center).
        return [dict(user_dist=f"gamma_{g}", policy="expert", adapt_beta=ab,
                     expert_gamma=eg, expert_center=ec, seed=s)
                for g, ab, eg, ec, s in itertools.product(
                    [0.39, 0.6, 0.8], [True, False], [0.0, 0.1, 0.25, 0.5], [0.5, 0.3], range(20))
                if not (eg == 0.0 and ec != 0.5)]
    if name == "corrected_trust":
        # Pooled trust with the agreement term estimated and subtracted, as the
        # platform's position estimates get noisier.
        return [dict(user_dist=f"gamma_{g}", policy="corrected_trust", adapt_beta=ab,
                     position_noise=pn, seed=s)
                for g, ab, pn, s in itertools.product(
                    [0.39, 0.6, 0.8], [True, False], [0.0, 0.05, 0.1, 0.2], range(20))]
    if name == "mf_uncentered":
        # Community Notes-style MF with and without re-centering the user factor.
        return [dict(user_dist=f"gamma_{g}", policy="bridging_mf", adapt_beta=ab,
                     mf_center=mc, seed=s)
                for g, ab, mc, s in itertools.product(
                    [0.39, 0.6, 0.8], [True, False], [True, False], range(20))]
    if name == "frozen_central":
        # Frozen coverage at dispersed (uniform) vs. clustered central positions.
        conds = [dict(adapt_beta=True), dict(adapt_beta=False),
                 dict(adapt_beta=False, beta_init_sd=0.08)]
        return [dict(user_dist=f"gamma_{g}", policy=pol, seed=s, **cd)
                for pol, g, cd, s in itertools.product(
                    ["crowd_trust", "bridging", "bridging_mf"], [0.39, 0.6, 0.8], conds, range(20))]
    if name == "rater_pool":
        # Rater panel that differs from the user population: group over-sampling
        # (70% of panel weight on the left half) or a continuous tilt to the left.
        pools = [dict(), dict(rater_left_share=0.7), dict(rater_tilt=2.0)]
        return [dict(user_dist=f"gamma_{g}", policy=pol, seed=s, **pl)
                for pol, g, pl, s in itertools.product(
                    ["crowd_trust", "bridging", "bridging_mf"], [0.39, 0.8], pools, range(20))]
    if name == "profit_rule":
        # Sources keep a change only if realized profit over a trial beats the old strategy.
        return [dict(user_dist=f"gamma_{g}", policy=pol, adapt_beta=ab,
                     adapt_rule="profit", trial_len=PROFIT_K, seed=s)
                for pol, g, ab, s in itertools.product(
                    TRUST, [0.39, 0.8], [True, False], range(20))] + \
               [dict(user_dist=f"gamma_{g}", policy=pol, adapt_rule="profit",
                     trial_len=PROFIT_K, seed=s)
                for pol, g, s in itertools.product(["popularity", "engagement"], [0.39, 0.8], range(20))]
    if name == "learning_rule":
        # Matched comparison: the main objective J, learned through the same
        # 20-step trials as the profit rule (same proposal rate and horizon).
        return [dict(user_dist=f"gamma_{g}", policy=pol, adapt_beta=ab,
                     adapt_rule="trial_J", trial_len=PROFIT_K, seed=s)
                for pol, g, ab, s in itertools.product(
                    TRUST, [0.39, 0.8], [True, False], range(20))] + \
               [dict(user_dist=f"gamma_{g}", policy=pol, adapt_rule="trial_J",
                     trial_len=PROFIT_K, seed=s)
                for pol, g, s in itertools.product(["popularity", "engagement"], [0.39, 0.8], range(20))]
    if name == "pop_mean":
        # Population-mean trust baseline, dense and sparse/self-selected ratings.
        dense = [dict(user_dist=f"gamma_{g}", policy="pop_mean", adapt_beta=ab, seed=s)
                 for g, ab, s in itertools.product([0.0, 0.2, 0.39, 0.6, 0.8], [True, False], range(20))]
        sparse = [dict(user_dist=f"gamma_{g}", policy="pop_mean", adapt_beta=ab,
                       rating_density=rd, rating_homophily=rh, seed=s)
                  for g, ab, rd, rh, s in itertools.product(
                      [0.39, 0.6, 0.8], [True, False], [0.05, 0.2], [0.0, 5.0], range(20))]
        return dense + sparse
    if name == "init_acc":
        # Initial accuracy correlated with centrality, or a wider initial range.
        return [dict(user_dist=f"gamma_{g}", policy=pol, adapt_beta=ab, init_acc=ia, seed=s)
                for pol, g, ab, ia, s in itertools.product(
                    TRUST, [0.39, 0.8], [True, False], ["corr", "wide"], range(20))]
    if name == "freeze_evolved":
        # Adapt for 1,000 steps, then freeze coverage at the evolved configuration.
        return [dict(user_dist=f"gamma_{g}", policy=pol, freeze_at=1000, seed=s)
                for pol, g, s in itertools.product(TRUST, [0.39, 0.8], range(20))]
    if name == "profit_rank":
        # Realized profit plus a direct rank term, learned through 20-step trials.
        return [dict(user_dist=f"gamma_{g}", policy=pol, adapt_beta=ab,
                     adapt_rule="trial_profit_rank", trial_len=PROFIT_K, seed=s)
                for pol, g, ab, s in itertools.product(
                    TRUST, [0.39, 0.8], [True, False], range(20))] + \
               [dict(user_dist=f"gamma_{g}", policy=pol, adapt_rule="trial_profit_rank",
                     trial_len=PROFIT_K, seed=s)
                for pol, g, s in itertools.product(["popularity", "engagement"], [0.39, 0.8], range(20))]
    if name == "cred_scale":
        # Perceived credibility on the survey's compressed scale (coefficients sum to 0.58).
        return [dict(user_dist=f"gamma_{g}", policy=pol, adapt_beta=ab, cred_scale=0.58, seed=s)
                for pol, g, ab, s in itertools.product(
                    TRUST, [0.39, 0.58], [True, False], range(20))]
    if name == "group_access":
        # Group-level access across the empirically informed credulity range.
        return [dict(user_dist=f"gamma_{g}", policy=pol, adapt_beta=ab, seed=s)
                for pol, g, ab, s in itertools.product(
                    TRUST, [0.22, 0.39, 0.58], [True, False], range(20))]
    if name == "nonlinear_perc":
        # Perception from the fitted quality x agreement model instead of Eq. 1.
        return [dict(user_dist="gamma_0.39", policy=pol, adapt_beta=ab,
                     perception="interaction", seed=s)
                for pol, ab, s in itertools.product(TRUST, [True, False], range(20))]
    if name == "transfer":
        # Evolved coverage transplanted into a fresh run and frozen.
        return [dict(user_dist=f"gamma_{g}", policy=pol, transfer=tr, seed=s)
                for pol, g, tr, s in itertools.product(
                    ["bridging", "bridging_mf"], [0.39, 0.8], ["matched", "shuffled"], range(20))]
    if name == "access_radius":
        # Nearby accuracy and availability at several radii, coverage adapting.
        return [dict(user_dist=f"gamma_{g}", policy=pol, seed=s)
                for pol, g, s in itertools.product(TRUST, [0.22, 0.39, 0.58, 0.8], range(20))]
    raise ValueError(name)

if __name__ == "__main__":
    # python experiments_v2.py <name> [--T steps] [--out dir]
    #   --T sets the horizon for every run that doesn't set its own
    #   --out writes to another folder (default results)
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("name"); ap.add_argument("--T", type=int); ap.add_argument("--out", default="results")
    ap.add_argument("--tail", type=int, help="steps averaged for reported outcomes (default 50)")
    ap.add_argument("--max-seed", type=int, help="keep only runs with seed < this")
    a = ap.parse_args()
    name, OUT = a.name, a.out
    cfgs = grid(name)
    if a.T:
        cfgs = [dict(c, T=c.get("T", a.T)) for c in cfgs]
    if a.tail:
        cfgs = [dict(c, tail=a.tail) for c in cfgs]
    if a.max_seed is not None:
        cfgs = [c for c in cfgs if c["seed"] < a.max_seed]
    os.makedirs(OUT, exist_ok=True)
    # Each finished run is saved to {OUT}/partial/{name}/ right away, so an
    # interrupted grid resumes where it stopped when rerun with the same arguments.
    part = f"{OUT}/partial/{name}"
    os.makedirs(part, exist_ok=True)
    todo = [(i, c) for i, c in enumerate(cfgs) if not os.path.exists(f"{part}/{i}.npz")]
    done = len(cfgs) - len(todo)
    if done:
        print(f"{name}: resuming, {done}/{len(cfgs)} runs already saved", flush=True)
    with Pool() as pool:
        for i, (row, ts) in pool.imap_unordered(run_indexed, todo, chunksize=1):
            np.savez_compressed(f"{part}/{i}.tmp.npz", row=json.dumps(row), ts=ts)
            os.replace(f"{part}/{i}.tmp.npz", f"{part}/{i}.npz")
            done += 1
            if done % 50 == 0 or done == len(cfgs):
                print(f"{name}: {done}/{len(cfgs)} runs", flush=True)
    rows, series = [], []
    for i in range(len(cfgs)):
        z = np.load(f"{part}/{i}.npz")
        rows.append(json.loads(str(z["row"]))); series.append(z["ts"])
    fields = list(dict.fromkeys(k for r in rows for k in r))   # union, first-seen order
    with open(f"{OUT}/{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader(); w.writerows(rows)
    os.makedirs(f"{OUT}/ts", exist_ok=True)
    np.savez_compressed(f"{OUT}/ts/{name}.npz", ts=np.stack(series),
                        metrics=np.array(TS_METRICS), configs=json.dumps(cfgs))
    print(f"saved {OUT}/{name}.csv and {OUT}/ts/{name}.npz  ({len(rows)} runs)")
    import shutil; shutil.rmtree(part)
