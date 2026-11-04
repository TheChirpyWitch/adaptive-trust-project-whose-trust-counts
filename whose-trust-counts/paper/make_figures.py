"""Build all paper figures and LaTeX tables from ../results/*.csv
(T=2000, mean of last 500 steps), falling back to ../results (T=400)
for grids not yet rerun.

Usage (from paper/):  python3 make_figures.py
Writes figs/*.pdf and tables/*.tex.
"""
import csv, collections, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(__file__), "..", "results")
RES_OLD = os.path.join(os.path.dirname(__file__), "..", "results")
FIG = os.path.join(os.path.dirname(__file__), "figs")
TAB = os.path.join(os.path.dirname(__file__), "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

# ── Style: validated categorical slots 1–4 for trust signals (identity),
#    recessive grays for the two baselines. Markers + direct labels give a
#    second encoding (slots 3–4 are below 3:1 contrast on white).
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
STYLE = {
    "popularity":  dict(color="#9a9893", ls="--", marker="", label="Popularity"),
    "engagement":  dict(color="#6f6d68", ls=":",  marker="", label="Engagement"),
    "crowd_trust": dict(color="#2a78d6", ls="-",  marker="o", label="Audience trust"),
    "bridging":    dict(color="#eb6834", ls="-",  marker="s", label="Bridging-min"),
    "bridging_mf": dict(color="#1baf7a", ls="-",  marker="^", label="Bridging-MF"),
    "expert":      dict(color="#eda100", ls="-",  marker="D", label="Expert"),
}
SHORT = {"popularity": "Popularity", "engagement": "Engagement", "crowd_trust": "Audience trust",
         "bridging": "Bridging-min", "bridging_mf": "Bridging-MF", "expert": "Expert"}
POLS = list(STYLE)
TRUST = ["crowd_trust", "bridging", "bridging_mf", "expert"]
GAMMAS = [0.0, 0.2, 0.4, 0.6, 0.8]

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8, "axes.titlesize": 8.5,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "pdf.fonttype": 42, "lines.linewidth": 1.5,
    "lines.markersize": 4.5,
})


def load(name):
    path = os.path.join(RES, f"{name}.csv")
    if not os.path.exists(path):
        print(f"[{name}: using results (T=400)]")
        path = os.path.join(RES_OLD, f"{name}.csv")
    return list(csv.DictReader(open(path)))


def agg(rows, keyfn, met):
    g = collections.defaultdict(list)
    for r in rows:
        g[keyfn(r)].append(float(r[met]))
    return {k: (np.mean(v), 1.96 * np.std(v, ddof=1) / np.sqrt(len(v)), len(v)) for k, v in g.items()}


def gamma_of(r):
    return float(r["user_dist"].split("_")[1])


def line_panel(ax, stats, xs, pols, label_ends=True):
    for p in pols:
        st = STYLE[p]
        m = np.array([stats[(p, x)][0] for x in xs])
        ci = np.array([stats[(p, x)][1] for x in xs])
        ax.fill_between(xs, m - ci, m + ci, color=st["color"], alpha=0.12, lw=0)
        ax.plot(xs, m, color=st["color"], ls=st["ls"], marker=st["marker"],
                markeredgecolor="white", markeredgewidth=0.6, label=st["label"])


# ── Figure 1: credibility and coverage spread vs credulity ───────────────
def fig_gamma():
    rows = [r for r in load("paper_gamma") if r["adapt_beta"] == "True"]
    c = agg(rows, lambda r: (r["policy"], gamma_of(r)), "aud_c")
    s = agg(rows, lambda r: (r["policy"], gamma_of(r)), "coverage_spread")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    line_panel(axes[0], c, GAMMAS, POLS)
    axes[0].set(xlabel="Credulity $\\gamma$", ylabel="Consumed accuracy",
                title="(a) Consumed accuracy", ylim=(-0.02, 1.02))
    line_panel(axes[1], s, GAMMAS, POLS)
    axes[1].axhline(0.289, color=INK2, lw=0.8, ls=(0, (1, 2)))
    axes[1].text(0.8, 0.305, "uniform initial coverage", color=INK2, fontsize=6.5,
                 va="bottom", ha="right")
    axes[1].set(xlabel="Credulity $\\gamma$", ylabel="Coverage spread (s.d. of $\\beta_s$)",
                title="(b) Coverage spread", ylim=(0, 0.36))
    # Population-mean trust baseline (dense ratings), which dense bridging-MF approximates.
    pm = load_new("pop_mean")
    if pm is not None:
        pm = [r for r in pm if r["adapt_beta"] == "True" and r.get("rating_density", "") in ("", "1.0")]
        for ax, met in ((axes[0], "aud_c"), (axes[1], "coverage_spread")):
            st = agg(pm, gamma_of, met)
            xs = sorted(st)
            ax.plot(xs, [st[x][0] for x in xs], color="#0d6b49", ls=(0, (4, 1.5)), lw=1.0,
                    marker="x", markersize=3.5, label="Population mean", zorder=3)
    for ax in axes:
        ax.set_xticks(GAMMAS)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=7, bbox_to_anchor=(0.5, -0.04), fontsize=7)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(FIG, "fig_gamma.pdf"), bbox_inches="tight")
    plt.close(fig)
    return c, s


# ── Figure 2: coverage adapts vs frozen (mechanism test) ─────────────────
def fig_frozen():
    rows = load("paper_gamma")
    c = agg(rows, lambda r: (r["policy"], gamma_of(r), r["adapt_beta"]), "aud_c")
    gs = [0.4, 0.6, 0.8]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1), sharey=True)
    ys = np.arange(len(TRUST))[::-1] * 1.0
    off = 0.17   # adapting just above the signal's row, frozen just below: never overlap
    for ax, g in zip(axes, gs):
        for y, p in zip(ys, TRUST):
            (a, ea, _), (f, ef, _) = c[(p, g, "True")], c[(p, g, "False")]
            col, mk = STYLE[p]["color"], STYLE[p]["marker"]
            ax.axhspan(y - 0.42, y + 0.42, color=GRID, alpha=0.35, lw=0, zorder=0)
            ax.plot([a, f], [y + off, y - off], color=col, lw=0.9, alpha=0.6, zorder=1)
            ax.errorbar(a, y + off, xerr=ea, fmt="none", ecolor=col, elinewidth=1, capsize=0, zorder=2)
            ax.errorbar(f, y - off, xerr=ef, fmt="none", ecolor=col, elinewidth=1, capsize=0, zorder=2)
            ax.scatter([a], [y + off], s=30, color=col, edgecolor="white", lw=0.6, marker=mk, zorder=3)
            ax.scatter([f], [y - off], s=30, facecolor="white", edgecolor=col, lw=1.2, marker=mk, zorder=3)
        ax.set_title(f"$\\gamma = {g}$")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(ys.min() - 0.55, ys.max() + 0.55)
        ax.set_xlabel("Consumed accuracy")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([SHORT[p] for p in TRUST])
    axes[-1].scatter([], [], s=30, color=INK2, label="coverage adapts (upper)")
    axes[-1].scatter([], [], s=30, facecolor="white", edgecolor=INK2, lw=1.2, label="coverage frozen (lower)")
    h, l = axes[-1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=7, handletextpad=0.3,
               columnspacing=1.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(os.path.join(FIG, "fig_frozen.pdf"), bbox_inches="tight")
    plt.close(fig)
    return c


# ── Figure 3: cost of credibility sweep ──────────────────────────────────
def fig_kappa():
    rows = load("paper_kappa")
    ks = sorted({float(r["kappa_c"]) for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=True)
    out = {}
    for ax, g in zip(axes, [0.2, 0.6]):
        sub = [r for r in rows if gamma_of(r) == g]
        st = agg(sub, lambda r: (r["policy"], float(r["kappa_c"])), "aud_c")
        out[g] = st
        line_panel(ax, st, ks, POLS)
        ax.set(title=f"$\\gamma = {g}$", xlabel="Cost of accuracy $\\kappa_c$", ylim=(-0.02, 1.02))
        ax.set_xticks(ks)
    axes[0].set_ylabel("Consumed accuracy")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.04), fontsize=7)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(FIG, "fig_kappa.pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


# ── Figure 4: expert error × coverage heatmaps ───────────────────────────
def fig_expert():
    rows = load("paper_expert")
    noises = [0.05, 0.1, 0.2, 0.3]
    covs = [1.0, 0.5, 0.25]
    from matplotlib.colors import LinearSegmentedColormap
    ramp = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    cmap = LinearSegmentedColormap.from_list("blue", ramp)
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.0), sharey=True)
    out = {}
    for ax, g in zip(axes, [0.2, 0.6]):
        sub = [r for r in rows if gamma_of(r) == g]
        st = agg(sub, lambda r: (float(r["expert_noise"]), float(r["expert_coverage"])), "aud_c")
        out[g] = st
        M = np.array([[st[(n, cv)][0] for cv in covs] for n in noises])
        ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(len(noises)):
            for j in range(len(covs)):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if M[i, j] > 0.55 else INK)
        ax.set_xticks(range(len(covs)))
        ax.set_xticklabels([f"{int(cv*100)}%" for cv in covs])
        ax.set_xlabel("Share of sources rated")
        ax.set_title(f"$\\gamma = {g}$")
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_yticks(range(len(noises)))
    axes[0].set_yticklabels([str(n) for n in noises])
    axes[0].set_ylabel("Expert error s.d. $\\sigma_x$")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_expert.pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


# ── Table: populations × policy, switching on / off ──────────────────────
POPNAME = {"credibility_seeking": "Accuracy-focused", "demanding": "Demanding",
           "balanced_archetypes": "Mixed (4 types)", "entertainment_credulous": "Entertainment-seeking, credulous"}


def tab_pops():
    rows = load("paper_pops")
    st = agg(rows, lambda r: (r["user_dist"], r["policy"], r["rewire_prob"]), "aud_c")
    lines = [r"\begin{tabular}{l" + "c" * len(POLS) + "}", r"\toprule",
             "Population & " + " & ".join(SHORT[p] for p in POLS) + r" \\", r"\midrule"]
    for pop in POPNAME:
        cells = []
        for p in POLS:
            on, off = st[(pop, p, "0.5")][0], st[(pop, p, "0.0")][0]
            cells.append(f"{on:.2f} / {off:.2f}")
        lines.append(POPNAME[pop] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "pops.tex"), "w").write("\n".join(lines))
    return st


# ── Table: normalization robustness (rank vs min-max), 10 seeds ──────────
def tab_norm():
    rows = load("norm_check") + load("norm_check_extra")   # seeds 0-9 and 10-29
    c = agg(rows, lambda r: (r["score_norm"], r["policy"], gamma_of(r), r["adapt_beta"]), "aud_c")
    s = agg(rows, lambda r: (r["score_norm"], r["policy"], gamma_of(r), r["adapt_beta"]), "coverage_spread")
    lines = [r"\begin{tabular}{llcccc}", r"\toprule",
             r"Scaling & Signal & Acc.\ (adapt) & Acc.\ (frozen) & Spread $\gamma{=}0$ & Spread $\gamma{=}0.8$ \\",
             r"\midrule"]
    for sn, name in [("rank", "Rank"), ("minmax", "Min--max")]:
        for p in TRUST:
            lines.append(f"{name} & {SHORT[p]} & {c[(sn, p, 0.8, 'True')][0]:.2f} & "
                         f"{c[(sn, p, 0.8, 'False')][0]:.2f} & {s[(sn, p, 0.0, 'True')][0]:.2f} & "
                         f"{s[(sn, p, 0.8, 'True')][0]:.2f} \\\\")
            name = ""
        if sn == "rank":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "norm.tex"), "w").write("\n".join(lines))


# ── Figure: empirical credulity and model outcomes at that credulity ────
DER = os.path.join(os.path.dirname(__file__), "..", "data", "derived")
SPEC_LABEL = {
    "main: conservatism, Lin quality, Robertson lean": "Main",
    "no fixed effects": "No respondent FE",
    "familiar outlets only": "Recognized outlets only",
    "study 1 only": "Study 1 (MTurk)",
    "study 2 only": "Study 2 (Lucid)",
    "party (0/1) as rater position": "Party as rater position",
    "study 2, fact-checker quality": "S2, fact-checker quality",
    "study 2, MBFC slant (incl. fake sites)": "S2, MBFC slant",
    "study 2, fact-checker + MBFC": "S2, fact-checker + MBFC",
}


def fig_empirical():
    est = list(csv.DictReader(open(os.path.join(DER, "gamma_estimates.csv"))))
    rows = load("empirical_gamma")
    c = agg(rows, lambda r: (r["policy"], gamma_of(r), r["adapt_beta"]), "aud_c")
    s = agg(rows, lambda r: (r["policy"], gamma_of(r), r["adapt_beta"]), "coverage_spread")
    gs = sorted({gamma_of(r) for r in rows})
    main = next(e for e in est if e["spec"].startswith("main"))
    lo, hi = float(main["lo"]), float(main["hi"])

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3), gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
    ax = axes[0]
    ys = np.arange(len(est))[::-1]
    for y, e in zip(ys, est):
        g, l, h = float(e["gamma"]), float(e["lo"]), float(e["hi"])
        bold = e is main
        ax.plot([l, h], [y, y], color=INK if bold else INK2, lw=1.6 if bold else 1)
        ax.scatter([g], [y], s=22 if bold else 14, color=INK if bold else INK2, zorder=3)
    for t, lab in [(0.5, "1/2"), (2 / 3, "2/3"), (0.8, "4/5")]:
        ax.axvline(t, color=GRID, lw=1, zorder=0)
        ax.text(t, ys[0] + 0.9, lab, ha="center", va="bottom", fontsize=6.5, color=INK2)
    ax.set_yticks(ys)
    ax.set_yticklabels([SPEC_LABEL[e["spec"]] for e in est], fontsize=6.5)
    ax.set_xlim(0, 0.85)
    ax.set_ylim(-0.6, ys[0] + 1.6)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Estimated credulity $\\hat\\gamma$ (95% CI)")
    ax.set_title("(a) Credulity estimated from data")

    for ax in axes[1:]:
        ax.axvspan(lo, hi, color=GRID, alpha=0.6, lw=0, zorder=0)
    line_panel(axes[1], {(p, g): s[(p, g, "True")] for p in TRUST for g in gs}, gs, TRUST)
    axes[1].axhline(s[("expert", gs[0], "False")][0], color=INK2, lw=0.8, ls=":")
    axes[1].text(gs[0], s[("expert", gs[0], "False")][0] + 0.004, "initial", ha="left",
                 va="bottom", fontsize=6.5, color=INK2)
    axes[1].set(xlabel="Credulity $\\gamma$", ylabel="Coverage spread (s.d. of $\\beta_s$)",
                ylim=(0.08, 0.31), title="(b) Coverage spread")
    for p in ["bridging", "bridging_mf"]:
        st = STYLE[p]
        for adapt, fill in [("True", st["color"]), ("False", "white")]:
            m = [c[(p, g, adapt)][0] for g in gs]
            axes[2].plot(gs, m, color=st["color"], ls="-" if adapt == "True" else "--",
                         marker=st["marker"], markerfacecolor=fill, markeredgecolor=st["color"])
    axes[2].plot([], [], color=INK2, ls="-", label="coverage adapts")
    axes[2].plot([], [], color=INK2, ls="--", label="coverage frozen")
    axes[2].legend(loc="lower left", fontsize=6.5)
    axes[2].set(xlabel="Credulity $\\gamma$", ylabel="Consumed accuracy", ylim=(-0.02, 1.02),
                title="(c) Bridging accuracy")
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.62, -0.08), fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_empirical.pdf"), bbox_inches="tight")
    plt.close(fig)
    return c, s, gs


def tab_real():
    rb = {r["signal"]: r for r in csv.DictReader(open(os.path.join(DER, "real_bridging.csv")))}
    names = [("pooled", "Population mean"), ("bridging_min", "Bridging-min"),
             ("bridging_mf", "Bridging-MF"), ("audience", "Audience (recognizers)"),
             ("factchecker", "Fact-checkers")]

    def cell(k):
        if k not in rb:
            return "--"
        r = rb[k]
        if abs(float(r["b_q"])) < 0.05:  # share undefined when quality coefficient ~ 0
            return f"$b_q={float(r['b_q']):.3f}$"
        return f"{float(r['w_cent']):.2f} [{float(r['lo']):.2f}, {float(r['hi']):.2f}]"

    lines = [r"\begin{tabular}{lcc}", r"\toprule",
             r"Signal & Quality = expert aggregate & Quality = fact-checkers (S2) \\", r"\midrule"]
    for k, n in names:
        lines.append(f"{n} & {cell(k)} & {cell(k + '|factchecker')} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "real.tex"), "w").write("\n".join(lines))


def tab_panel_audit():
    """Real-data rater-panel audit: rank change under right- vs. left-leaning panels."""
    f = os.path.join(DER, "panel_audit.csv")
    if not os.path.exists(f):
        return None
    pa = {(r["panel"], r["signal"]): r for r in csv.DictReader(open(f))}
    names = [("pooled", "Population mean"), ("bridging_min", "Bridging-min"),
             ("bridging_mf", "Bridging-MF"), ("bridging_mf_uncentered", "Bridging-MF, uncentered")]
    panels = [("null", "Same composition"), ("share", "70\\% party share"), ("tilt", "Ideology tilt")]

    def cell(r, a="lean_slope", lo="lo", hi="hi"):
        f = lambda x: "0.00" if abs(float(x)) < 0.005 else f"{float(x):+.2f}"
        return f"${f(r[a])}$ [${f(r[lo])}$, ${f(r[hi])}$]"

    lines = [r"\begin{tabular}{llcc}", r"\toprule",
             r"Panels & Signal & Lean slope & Slope, S2 alt.\ measures \\", r"\midrule"]
    for i, (pk, pn) in enumerate(panels):
        if i:
            lines.append(r"\midrule")
        for k, n in names:
            r = pa[(pk, k)]
            lines.append(f"{pn if k == 'pooled' else ''} & {n} & {cell(r)} & "
                         f"{cell(r, 's2_mbfc_fc_slope', 's2_lo', 's2_hi')} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "panel_audit.tex"), "w").write("\n".join(lines))
    return pa


def tab_beliefs():
    """2-D belief space and polarized priors: credibility (adapt / frozen) and spread."""
    rows = load("robust_beliefs")
    key = lambda r: ("2d" if r["belief_dims"] == "2" else "bimodal", r["policy"], gamma_of(r), r["adapt_beta"])
    c = agg(rows, key, "aud_c")
    s1 = agg(rows, key, "spread_axis1")
    s2 = agg(rows, key, "spread_axis2")
    gs = [0.39, 0.8]
    lines = [r"\begin{tabular}{ll" + "ccc" * len(gs) + "}", r"\toprule",
             r" & & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{$\gamma={g}$}}" for g in gs) + r" \\",
             " ".join(rf"\cmidrule(lr){{{3 + 3 * i}-{5 + 3 * i}}}" for i in range(len(gs))),
             r"Beliefs & Signal & " + " & ".join(["Adapt & Frozen & Spread"] * len(gs)) + r" \\", r"\midrule"]
    for setting, name in [("2d", "2-D uniform"), ("bimodal", "Polarized")]:
        for p in POLS:
            cells = []
            for g in gs:
                a = c[(setting, p, g, "True")][0]
                f = f"{c[(setting, p, g, 'False')][0]:.2f}" if (setting, p, g, "False") in c else "--"
                if setting == "2d":
                    sp = f"{s1[(setting, p, g, 'True')][0]:.2f} / {s2[(setting, p, g, 'True')][0]:.2f}"
                else:
                    sp = f"{s1[(setting, p, g, 'True')][0]:.2f}"
                cells += [f"{a:.2f}", f, sp]
            lines.append(f"{name} & {SHORT[p]} & " + " & ".join(cells) + r" \\")
            name = ""
        if setting == "2d":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "beliefs.tex"), "w").write("\n".join(lines))
    print("beliefs (adapt|frozen cred; spread axis1/axis2):")
    for setting in ["2d", "bimodal"]:
        for p in POLS:
            print(f"  {setting:7s} {p:12s}", " ".join(
                f"γ{g}: {c[(setting, p, g, 'True')][0]:.2f}|"
                f"{c[(setting, p, g, 'False')][0] if (setting, p, g, 'False') in c else float('nan'):.2f} "
                f"sp {s1[(setting, p, g, 'True')][0]:.3f}/{s2[(setting, p, g, 'True')][0]:.3f}"
                for g in [0.0, 0.39, 0.6, 0.8]))


# ── Table: sensitivity to fixed parameters (one at a time) ───────────────
SENS = [("default", None, "Default")] + [
    ("lam", v, f"$\\lambda={v}$") for v in ["0.1", "0.5", "0.7"]] + [
    ("n_sources", v, f"$n_S={v}$") for v in ["15", "60"]] + [
    ("n_users", v, f"$n_U={v}$") for v in ["300", "1200"]] + [
    ("k", v, f"$k={v}$") for v in ["2", "5"]] + [
    ("sigma_prop", v, f"$\\sigma={v}$") for v in ["0.025", "0.1"]] + [
    ("mu_experiment", v, f"$\\mu={v}$") for v in ["0.1", "0.4"]] + [
    ("rank_temp", v, f"Rank temperature $\\tau={v}$") for v in ["0.05", "0.2"]] + [
    ("noise_c", v, f"Perception noise on $\\hat c$: ${v}$") for v in ["0.05", "0.2"]] + [
    ("social_proof_weight", v, f"Reliance on rank $\\omega={v}$") for v in ["0.0", "0.6"]]


def tab_sens():
    rows = load("sensitivity")
    def var(r):
        return ("default", None) if r["varied"] == "default" else (r["varied"], r[r["varied"]])
    key = lambda r: var(r) + (r["policy"], gamma_of(r), r["adapt_beta"])
    c = agg(rows, key, "aud_c")
    s = agg(rows, key, "coverage_spread")
    # Paired (same-seed) adapting-minus-frozen drops and their 95% CIs.
    pair = collections.defaultdict(dict)
    for r in rows:
        pair[var(r) + (r["policy"], gamma_of(r), r["seed"])][r["adapt_beta"]] = float(r["aud_c"])
    drops = collections.defaultdict(list)
    for kk, v in pair.items():
        if len(v) == 2:
            drops[kk[:-1]].append(v["True"] - v["False"])
    def straddles(kk):
        d = np.array(drops[kk]); h = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
        return d.mean() - h < 0.15 < d.mean() + h
    def cell(k, pol, g):
        a, f = c[k + (pol, g, "True")][0], c[k + (pol, g, "False")][0]
        return f"{a:.2f} / {f:.2f}" + (r"$^\dagger$" if straddles(k + (pol, g)) else "")
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"Variant & Spread, $\gamma{=}0.8$ & Bridging-min, $\gamma{=}0.8$ & Bridging-MF, $\gamma{=}0.39$ & Bridging-min, $\gamma{=}0.39$ \\",
             r" & aud. / min / MF & adapt / frozen & adapt / frozen & adapt / frozen \\", r"\midrule"]
    checks = {}
    for p, v, label in SENS:
        k = (p, v)
        sp = [s[k + (pol, 0.8, "True")][0] for pol in ["crowd_trust", "bridging", "bridging_mf"]]
        bm8 = (c[k + ("bridging", 0.8, "True")][0], c[k + ("bridging", 0.8, "False")][0])
        mf39 = (c[k + ("bridging_mf", 0.39, "True")][0], c[k + ("bridging_mf", 0.39, "False")][0])
        bm39 = (c[k + ("bridging", 0.39, "True")][0], c[k + ("bridging", 0.39, "False")][0])
        lines.append(f"{label} & {sp[0]:.2f} / {sp[1]:.2f} / {sp[2]:.2f} & {cell(k, 'bridging', 0.8)} & "
                     f"{cell(k, 'bridging_mf', 0.39)} & {cell(k, 'bridging', 0.39)} \\\\")
        if p == "default":
            lines.append(r"\midrule")
        checks[label] = dict(
            compress=max(sp[1], sp[2]) <= sp[0] - 0.05,          # pooled well below crowd
            collapse=bm8[1] <= 0.2 and bm8[0] - bm8[1] >= 0.3,   # freezing collapses bridging-min
            mf_free=mf39[0] - mf39[1] <= 0.1,                    # MF needs little convergence at 0.39
            min_needs=bm39[0] - bm39[1] >= 0.15,                 # bridging-min needs it at 0.39
            vals=(sp, bm8, mf39, bm39))
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "sens.tex"), "w").write("\n".join(lines))
    return checks


# ── Tables: biased experts and agreement-corrected trust ─────────────────
def tab_expert_lean():
    rows = load("expert_lean")
    key = lambda r: (float(r["expert_gamma"]), float(r["expert_center"]), gamma_of(r), r["adapt_beta"])
    c, s, m = (agg(rows, key, met) for met in ["aud_c", "coverage_spread", "coverage_mean"])
    gs = [0.39, 0.8]
    lines = [r"\begin{tabular}{cc" + "ccc" * len(gs) + "}", r"\toprule",
             r" & & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{$\gamma={g}$}}" for g in gs) + r" \\",
             " ".join(rf"\cmidrule(lr){{{3 + 3 * i}-{5 + 3 * i}}}" for i in range(len(gs))),
             r"$\gamma_E$ & $\beta_E$ & " + " & ".join(["Adapt / frozen & Spread & Position"] * len(gs)) + r" \\",
             r"\midrule"]
    for eg, ec in [(0.0, 0.5), (0.1, 0.5), (0.1, 0.3), (0.25, 0.5), (0.25, 0.3), (0.5, 0.5), (0.5, 0.3)]:
        cells = []
        for g in gs:
            k = (eg, ec, g)
            cells += [f"{c[k + ('True',)][0]:.2f} / {c[k + ('False',)][0]:.2f}",
                      f"{s[k + ('True',)][0]:.2f}", f"{m[k + ('True',)][0]:.2f}"]
        lines.append(f"{eg:g} & {'--' if eg == 0 else f'{ec:g}'} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "expert_lean.tex"), "w").write("\n".join(lines))


def tab_corrected():
    rows = load("corrected_trust")
    key = lambda r: (float(r["position_noise"]), gamma_of(r), r["adapt_beta"])
    c, s = agg(rows, key, "aud_c"), agg(rows, key, "coverage_spread")
    gs = [0.39, 0.6, 0.8]
    lines = [r"\begin{tabular}{c" + "cc" * len(gs) + "}", r"\toprule",
             r" & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{$\gamma={g}$}}" for g in gs) + r" \\",
             " ".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(gs))),
             r"Position error & " + " & ".join(["Adapt / frozen & Spread"] * len(gs)) + r" \\", r"\midrule"]
    for pn in [0.0, 0.05, 0.1, 0.2]:
        cells = []
        for g in gs:
            k = (pn, g)
            cells += [f"{c[k + ('True',)][0]:.2f} / {c[k + ('False',)][0]:.2f}", f"{s[k + ('True',)][0]:.2f}"]
        lines.append(f"{pn:g} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "corrected.tex"), "w").write("\n".join(lines))


# ── Tables: reviewer-response experiments (skipped until their grids finish) ──
def load_new(name):
    path = os.path.join(RES, f"{name}.csv")
    return list(csv.DictReader(open(path))) if os.path.exists(path) else None


PNAME = {"crowd_trust": "Audience trust", "bridging": "Bridging-min", "bridging_mf": "Bridging-MF",
         "expert": "Expert", "popularity": "Popularity", "engagement": "Engagement"}


def tab_frozen_central():
    rows = load_new("frozen_central")
    if rows is None:
        return None
    cond = lambda r: "adapt" if r["adapt_beta"] == "True" else ("central" if r.get("beta_init_sd") else "dispersed")
    key = lambda r: (r["policy"], gamma_of(r), cond(r))
    c, s = agg(rows, key, "aud_c"), agg(rows, key, "coverage_spread")
    gs = [0.39, 0.6, 0.8]
    lines = [r"\begin{tabular}{l" + "c" * len(gs) + "}", r"\toprule",
             r"Signal & " + " & ".join(rf"$\gamma={g}$" for g in gs) + r" \\", r"\midrule"]
    for pol in ["crowd_trust", "bridging", "bridging_mf"]:
        cells = [" / ".join(f"{c[(pol, g, k)][0]:.2f}" for k in ["adapt", "dispersed", "central"]) for g in gs]
        lines.append(f"{PNAME[pol]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "frozen_central.tex"), "w").write("\n".join(lines))
    return c, s


def tab_rater_pool():
    rows = load_new("rater_pool")
    if rows is None:
        return None
    pool = lambda r: "left70" if r.get("rater_left_share") else ("tilt" if r.get("rater_tilt") else "users")
    key = lambda r: (r["policy"], gamma_of(r), pool(r))
    mets = ["coverage_mean", "coverage_spread", "aud_c"] + [f"{m}_{g}" for m in ("near", "dist") for g in "LCR"]
    st = {m: agg(rows, key, m) for m in mets}
    names = {"users": "All users", "left70": "Left group 70\\%", "tilt": "Tilted left"}
    lines = [r"\begin{tabular}{llcccccc}", r"\toprule",
             r" & & \multicolumn{2}{c}{Mean position} & \multicolumn{4}{c}{$\gamma=0.8$} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-8}",
             r"Signal & Rater panel & $\gamma=0.39$ & $\gamma=0.8$ & Spread & Accuracy & Nearby accuracy L / C / R & Distance L / C / R \\",
             r"\midrule"]
    for pol in ["crowd_trust", "bridging", "bridging_mf"]:
        for i, pl in enumerate(["users", "left70", "tilt"]):
            k39, k8 = (pol, 0.39, pl), (pol, 0.8, pl)
            near = " / ".join(f"{st[f'near_{g}'][k8][0]:.2f}" for g in "LCR")
            dist = " / ".join(f"{st[f'dist_{g}'][k8][0]:.2f}" for g in "LCR")
            lines.append(f"{PNAME[pol] if i == 0 else ''} & {names[pl]} & {st['coverage_mean'][k39][0]:.2f} & "
                         f"{st['coverage_mean'][k8][0]:.2f} & {st['coverage_spread'][k8][0]:.2f} & "
                         f"{st['aud_c'][k8][0]:.2f} & {near} & {dist} \\\\")
        if pol != "bridging_mf":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "rater_pool.tex"), "w").write("\n".join(lines))
    return st


def tab_mf_uncentered():
    rows = load_new("mf_uncentered")
    if rows is None:
        return None
    key = lambda r: (r["mf_center"], gamma_of(r), r["adapt_beta"])
    c, s = agg(rows, key, "aud_c"), agg(rows, key, "coverage_spread")
    gs = [0.39, 0.6, 0.8]
    lines = [r"\begin{tabular}{l" + "cc" * len(gs) + "}", r"\toprule",
             r" & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{$\gamma={g}$}}" for g in gs) + r" \\",
             " ".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(gs))),
             r"User factor & " + " & ".join(["Adapt / frozen & Spread"] * len(gs)) + r" \\", r"\midrule"]
    for mc, lab in [("True", "Re-centered (main)"), ("False", "Not re-centered")]:
        cells = []
        for g in gs:
            cells += [f"{c[(mc, g, 'True')][0]:.2f} / {c[(mc, g, 'False')][0]:.2f}", f"{s[(mc, g, 'True')][0]:.2f}"]
        lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "mf_uncentered.tex"), "w").write("\n".join(lines))
    return c, s


def tab_group_access():
    """Nearby accuracy by belief third across the empirical credulity range (adapting)."""
    rows = load_new("group_access")
    if rows is None:
        return
    st = collections.defaultdict(list)
    for r in rows:
        if r["adapt_beta"] != "True":
            continue
        near = [float(r[f"near_{t}"]) for t in "LCR"]
        st[(r["policy"], r["user_dist"])].append((near[1], min(near[0], near[2]), float(r["aud_c"])))
    gs = ["gamma_0.22", "gamma_0.39", "gamma_0.58"]
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r"Signal & $\gamma=0.22$ & $\gamma=0.39$ & $\gamma=0.58$ \\", r"\midrule"]
    for pol in ["crowd_trust", "bridging", "bridging_mf", "expert"]:
        cells = []
        for g in gs:
            a = np.array(st[(pol, g)])
            cells.append(f"{a[:, 0].mean():.2f} / {a[:, 1].mean():.2f}")
        lines.append(f"{PNAME[pol]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "group_access.tex"), "w").write("\n".join(lines))
    for pol in ["crowd_trust", "bridging", "bridging_mf", "expert"]:
        for g in gs:
            a = np.array(st[(pol, g)])
            hw = 1.96 * a[:, 1].std(ddof=1) / np.sqrt(len(a))
            print(f"group access {pol} {g}: middle {a[:,0].mean():.2f} worse-outer {a[:,1].mean():.2f} (±{hw:.2f}) acc {a[:,2].mean():.2f}")



def tab_transfer():
    """Evolved coverage transplanted into fresh runs (matched / shuffled), vs. other freezes."""
    tr, fe = load_new("transfer"), load_new("freeze_evolved")
    pg = load("paper_gamma") + load("empirical_gamma")   # gamma 0.8 / 0.39 baselines
    if tr is None or fe is None:
        return None
    dense = lambda r: r.get("rating_density", "") in ("", "1.0")
    mean = lambda rows: (np.mean(rows), 1.96 * np.std(rows, ddof=1) / np.sqrt(len(rows)))
    out = {}
    for pol in ["bridging", "bridging_mf"]:
        for g in [0.39, 0.8]:
            pick = lambda rows, f: [float(r["aud_c"]) for r in rows
                                    if r["policy"] == pol and abs(gamma_of(r) - g) < 1e-9 and f(r)]
            out[(pol, g)] = [
                mean(pick(pg, lambda r: r["adapt_beta"] == "True" and dense(r))),
                mean(pick(pg, lambda r: r["adapt_beta"] == "False" and dense(r))),
                mean(pick(fe, lambda r: True)),
                mean(pick(tr, lambda r: r["transfer"] == "matched")),
                mean(pick(tr, lambda r: r["transfer"] == "shuffled"))]
    lines = [r"\begin{tabular}{llccccc}", r"\toprule",
             r"Signal & $\gamma$ & Adapting & Frozen at start & Frozen at step 1{,}000 & Transplanted, own & Transplanted, shuffled \\",
             r"\midrule"]
    for (pol, g), v in out.items():
        lines.append(f"{PNAME[pol]} & {g} & " + " & ".join(f"{m:.2f}" for m, _ in v) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "transfer.tex"), "w").write("\n".join(lines))
    for k, v in out.items():
        print("transfer", k, " ".join(f"{m:.2f}±{h:.2f}" for m, h in v))
    return out


def tab_nonlinear():
    """Fitted quality x agreement perception: accuracy adapt / frozen and spread."""
    rows = load_new("nonlinear_perc")
    if rows is None:
        return None
    acc = agg(rows, lambda r: (r["policy"], r["adapt_beta"]), "aud_c")
    sp = agg(rows, lambda r: (r["policy"], r["adapt_beta"]), "coverage_spread")
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r"Signal & Accuracy, adapting & Accuracy, frozen & Spread \\", r"\midrule"]
    for pol in TRUST:
        lines.append(f"{PNAME[pol]} & {acc[(pol, 'True')][0]:.2f} & {acc[(pol, 'False')][0]:.2f} & "
                     f"{sp[(pol, 'True')][0]:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "nonlinear.tex"), "w").write("\n".join(lines))
    for pol in TRUST:
        print(f"nonlinear {pol}: acc {acc[(pol,'True')][0]:.2f}±{acc[(pol,'True')][1]:.2f} / "
              f"{acc[(pol,'False')][0]:.2f}±{acc[(pol,'False')][1]:.2f} spread {sp[(pol,'True')][0]:.3f}")
    return acc, sp


def tab_access_radius():
    """Availability of an accurate (>= 0.8) source within radius r, middle / worse outer third."""
    rows = load_new("access_radius")
    if rows is None:
        return None
    st = collections.defaultdict(list)
    for r in rows:
        vals = []
        for tag in ("10", "15", "25"):
            a = [float(r[f"avail{tag}_{t}"]) for t in "LCR"]
            vals += [a[1], min(a[0], a[2])]
        n = [float(r[f"near_{t}"]) for t in "LCR"]
        vals += [n[1], min(n[0], n[2])]
        st[(r["policy"], gamma_of(r))].append(vals)
    gs = [0.22, 0.39, 0.58, 0.8]
    lines = [r"\begin{tabular}{ll" + "c" * len(gs) + "}", r"\toprule",
             r"Radius & Signal & " + " & ".join(f"$\\gamma={g}$" for g in gs) + r" \\", r"\midrule"]
    for ti, tag in enumerate(("0.10", "0.15", "0.25")):
        if ti:
            lines.append(r"\midrule")
        for pol in TRUST:
            cells = []
            for g in gs:
                a = np.array(st[(pol, g)])
                cells.append(f"{a[:, 2 * ti].mean():.2f} / {a[:, 2 * ti + 1].mean():.2f}")
            lines.append(f"{tag if pol == TRUST[0] else ''} & {PNAME[pol]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "access_radius.tex"), "w").write("\n".join(lines))
    for pol in TRUST:
        for g in gs:
            a = np.array(st[(pol, g)])
            hw = 1.96 * a.std(0, ddof=1) / np.sqrt(len(a))
            print(f"access {pol} {g}: " + " ".join(f"{m:.2f}±{h:.2f}" for m, h in zip(a.mean(0), hw)))
    return st


def tab_profit_rule():
    rows = load_new("profit_rule")
    if rows is None:
        return None
    key = lambda r: (r["policy"], gamma_of(r), r["adapt_beta"])
    c, s = agg(rows, key, "aud_c"), agg(rows, key, "coverage_spread")
    gs = [0.39, 0.8]
    lines = [r"\begin{tabular}{l" + "cc" * len(gs) + "}", r"\toprule",
             r" & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{$\gamma={g}$}}" for g in gs) + r" \\",
             " ".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(gs))),
             r"Signal & " + " & ".join(["Adapt / frozen & Spread"] * len(gs)) + r" \\", r"\midrule"]
    for pol in ["popularity", "engagement"] + TRUST:
        cells = []
        for g in gs:
            fr = c.get((pol, g, "False"))
            cells += [f"{c[(pol, g, 'True')][0]:.2f} / " + (f"{fr[0]:.2f}" if fr else "--"),
                      f"{s[(pol, g, 'True')][0]:.2f}"]
        lines.append(f"{PNAME[pol]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "profit_rule.tex"), "w").write("\n".join(lines))
    return c, s


def tab_learning():
    """Main rule vs. trial-based J (matched learning) vs. profit-based rule."""
    lr, pr = load_new("learning_rule"), load_new("profit_rule")
    if lr is None or pr is None:
        return None
    main = [r for r in load("empirical_gamma") if gamma_of(r) == 0.39] + \
           [r for r in load("paper_gamma") if gamma_of(r) == 0.8]
    key = lambda r: (r["policy"], gamma_of(r), r["adapt_beta"])
    gs = [0.39, 0.8]
    lines = [r"\begin{tabular}{ll" + "cc" * len(gs) + "}", r"\toprule",
             r" & & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{$\gamma={g}$}}" for g in gs) + r" \\",
             " ".join(rf"\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(gs))),
             r"Rule & Signal & " + " & ".join(["Adapt / frozen & Spread"] * len(gs)) + r" \\"]
    variants = [("Main (predicted $J$)", main), ("Trial-based $J$", lr), ("Profit-based", pr)]
    prk = load_new("profit_rank")
    if prk is not None:
        variants.append(("Profit plus rank", prk))
    for name, rows in variants:
        c, s = agg(rows, key, "aud_c"), agg(rows, key, "coverage_spread")
        lines.append(r"\midrule")
        for i, pol in enumerate(["popularity", "engagement"] + TRUST):
            cells = []
            for g in gs:
                fr = c.get((pol, g, "False"))
                cells += [f"{c[(pol, g, 'True')][0]:.2f} / " + (f"{fr[0]:.2f}" if fr else "--"),
                          f"{s[(pol, g, 'True')][0]:.2f}"]
            lines.append(f"{name if i == 0 else ''} & {PNAME[pol]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "learning.tex"), "w").write("\n".join(lines))


def tab_learning_main():
    """Compact main-text learning-rule comparison: spread at γ=0.8 and bridging-min adapt/frozen."""
    lr, pr, prk = load_new("learning_rule"), load_new("profit_rule"), load_new("profit_rank")
    if lr is None or pr is None or prk is None:
        return None
    main = [r for r in load("empirical_gamma") if gamma_of(r) == 0.39] + \
           [r for r in load("paper_gamma") if gamma_of(r) == 0.8]
    key = lambda r: (r["policy"], gamma_of(r), r["adapt_beta"])
    lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
             r" & \multicolumn{3}{c}{Coverage spread, $\gamma=0.8$} & \multicolumn{3}{c}{Consumed accuracy, adapt / frozen} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Learning rule (value judged) & Min & MF & Audience & Min, $\gamma=0.39$ & Min, $\gamma=0.8$ & MF, $\gamma=0.8$ \\",
             r"\midrule"]
    for name, rows in [("Predicted $J_s$ (main)", main), ("Realized $J_s$, 20-step trials", lr),
                       ("Realized profit $d_sm_s$", pr), ("Realized profit plus rank", prk)]:
        c, s = agg(rows, key, "aud_c"), agg(rows, key, "coverage_spread")
        sp = [s[(p, 0.8, "True")][0] for p in ["bridging", "bridging_mf", "crowd_trust"]]
        af = lambda p, g: f"{c[(p, g, 'True')][0]:.2f} / {c[(p, g, 'False')][0]:.2f}"
        lines.append(f"{name} & {sp[0]:.2f} & {sp[1]:.2f} & {sp[2]:.2f} & {af('bridging', 0.39)} & "
                     f"{af('bridging', 0.8)} & {af('bridging_mf', 0.8)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "learning_main.tex"), "w").write("\n".join(lines))


# ── Table: credulity regressions, every specification ────────────────────
GAMMA_SPECS = {
    "main: conservatism, Lin quality, Robertson lean": "Main (conservatism, Lin quality, Robertson lean)",
    "no fixed effects": "No respondent fixed effects",
    "familiar outlets only": "Recognized outlets only",
    "study 1 only": "Study 1 only (MTurk)",
    "study 2 only": "Study 2 only (Lucid)",
    "party (0/1) as rater position": "Party (0/1) as rater position",
    "study 2, fact-checker quality": "Study 2, fact-checker quality",
    "study 2, MBFC slant (incl. fake sites)": "Study 2, MBFC slant (incl.\\ fake-news sites)",
    "study 2, fact-checker + MBFC": "Study 2, fact-checker quality + MBFC slant",
}


def tab_gamma_reg():
    path = os.path.join(os.path.dirname(__file__), "..", "data", "derived", "gamma_estimates.csv")
    rows = list(csv.DictReader(open(path)))
    lines = [r"\begin{tabular}{lccccrrr}", r"\toprule",
             r"Specification & $\hat b_q$ & $\hat b_a$ & $\hat\gamma$ & 95\% CI & $\hat b_a$ (outlet FE) & Ratings & Resp.\ / outlets \\",
             r"\midrule"]
    for r in rows:
        f = {k: float(r[k]) for k in ["gamma", "lo", "hi", "b_c", "b_A", "b_A_within_outlet"]}
        lines.append(f"{GAMMA_SPECS[r['spec']]} & {f['b_c']:.3f} & {f['b_A']:.3f} & {f['gamma']:.2f} & "
                     f"{f['lo']:.2f}--{f['hi']:.2f} & {f['b_A_within_outlet']:.3f} & "
                     f"{int(r['n']):,} & {int(r['n_resp']):,} / {r['n_outlets']} \\\\".replace(",", "{,}"))
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "gamma_reg.tex"), "w").write("\n".join(lines))


if __name__ == "__main__":
    tab_beliefs()
    em = fig_empirical()
    tab_real()
    tab_panel_audit()
    print("empirical gamma grid (adapt|frozen cred; spread):")
    for p in TRUST:
        print(f"  {p:12s}", " ".join(
            f"γ{g}: {em[0][(p, g, 'True')][0]:.2f}|{em[0][(p, g, 'False')][0]:.2f} sp {em[1][(p, g, 'True')][0]:.3f}"
            for g in em[2]))
    for p in ["popularity", "engagement"]:
        print(f"  {p:12s}", " ".join(f"γ{g}: {em[0][(p, g, 'True')][0]:.2f} sp {em[1][(p, g, 'True')][0]:.3f}" for g in em[2]))
    c, s = fig_gamma()
    fz = fig_frozen()
    kp = fig_kappa()
    ex = fig_expert()
    tab_pops()
    tab_norm()
    tab_gamma_reg()
    tab_group_access()
    tab_learning()
    tab_learning_main()
    for f in (tab_transfer, tab_nonlinear, tab_access_radius):
        print(f.__name__, 'written' if f() is not None else 'skipped (grid not finished)')
    for f in (tab_frozen_central, tab_rater_pool, tab_mf_uncentered, tab_profit_rule):
        print(f.__name__, 'written' if f() is not None else 'skipped (grid not finished)')
    tab_expert_lean()
    tab_corrected()
    print("sensitivity checks (compress, collapse, mf_free, min_needs):")
    for lab, ck in tab_sens().items():
        print(f"  {lab:32s}", ck["compress"], ck["collapse"], ck["mf_free"], ck["min_needs"],
              " ".join(f"{x:.2f}" for t in ck["vals"] for x in t))
    # Print the numbers the text cites.
    print("consumed credibility / spread by gamma (adapting):")
    for p in POLS:
        print(f"  {p:12s}", " ".join(f"{c[(p, g)][0]:.2f}/{s[(p, g)][0]:.2f}" for g in GAMMAS))
    print("adapt vs frozen:")
    for p in TRUST:
        print(f"  {p:12s}", " ".join(f"γ{g}: {fz[(p, g, 'True')][0]:.2f}|{fz[(p, g, 'False')][0]:.2f}"
                                    for g in GAMMAS))
    print("kappa (γ=0.6):")
    for p in POLS:
        print(f"  {p:12s}", " ".join(f"{kp[0.6][(p, k)][0]:.2f}" for k in [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]))
    print("expert (γ=0.6) noise×coverage:", {k: round(v[0], 2) for k, v in ex[0.6].items()})
