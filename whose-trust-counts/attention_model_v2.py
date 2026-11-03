"""
Attention Allocation Model v2 — selection, not drift
====================================================

Changes from v1 (attention_model.py), each fixing a way results were
built in by construction:

1. No forced directional change. Clickbait and decay are gone as
   mechanisms. Sources experiment with SYMMETRIC proposals on (e, c, β)
   and keep a proposal only if it raises their expected payoff
   ("A/B testing"). Credibility falls only if the environment rewards it.

2. Credibility and entertainment are costly. Per-user margin
       m(e, c) = 1 − κ_c·c − κ_e·e
   so accuracy is a real investment that must pay for itself in audience.

3. Pandering is emergent, not imposed. There is no directed β mutation.
   A random β proposal is kept when it raises audience satisfaction or
   platform score — i.e. only when the environment rewards it.

4. Perceived credibility is a convex mix, not an additive bonus:
       ĉ = (1 − γ)·c + γ·alignment,   alignment = 1 − |β_s − β_u|
   γ = how much a user judges credibility by agreement instead of accuracy.
   (v1 used c + γ·alignment clipped at 1, which made the perception gap
   positive by construction and hid it whenever c was high.)

5. A platform ranks sources. When a user drops a source, the replacement
   is drawn from the platform's ranking (softmax over scores), and the
   user's expectation of an untried source uses the platform score in
   place of raw popularity. Ranking policies:
       popularity   degree                        (= v1 behaviour)
       engagement   mean perceived e of audience
       crowd_trust  mean perceived c of audience   (survey-based trust)
       bridging     min over belief groups of mean perceived c
                    from a panel of ALL users
       bridging_mf  matrix-factorisation intercept on all users'
                    perceived c (the Community Notes algorithm)
       expert       true c + noise                  (fact-checker ratings);
                    with expert_gamma > 0 the experts also reward agreement
                    with their own position expert_center (a biased rater pool)
       corrected_trust  population mean perceived c minus an estimated
                    agreement term: the platform regresses ratings on
                    agreement computed from estimated positions (error s.d.
                    position_noise) and subtracts it. Not in POLICIES, so the
                    paper grids that loop over POLICIES are unchanged.

Source payoff used both for imitation and for accepting proposals:
       J_s = m(e_s, c_s) · [ (1 − λ)·V_s + λ·R_s ]
   V_s = mean expected preference utility of the source's current audience
   R_s = platform score (normalised to [0, 1])
Imitation copies the visible (e, c) of a neighbour with higher realised
profit (degree × margin). β is hidden and never imitated.

Users are fixed types [w_e, w_c, α, β_u, γ] as in v1.

Belief space (robustness extension). With belief_dims = D > 1, coverage β_s and
priors β_u are points in [0,1]^D and alignment = 1 − ‖β_s − β_u‖₂ / √D, which
reduces to the 1-D rule when D = 1. Bridging groups still split on axis 1 only
(a partisan split), so axes ≥ 2 are dimensions the platform does not measure.
prior_dist = "bimodal" draws axis-1 priors from two clusters (Beta(6,18) and its
mirror image) with no users near the centre.
"""

import numpy as np

POLICIES = ["popularity", "engagement", "crowd_trust", "bridging", "bridging_mf", "expert"]
EXTRA_POLICIES = ["corrected_trust", "pop_mean"]   # pop_mean: population-mean trust baseline


def mf_intercepts(R, lam_i=0.15, lam_f=0.03, iters=30, init=None, center=True):
    """Community Notes-style matrix factorisation on a dense rating matrix.

        R[u, s] ≈ μ + i_u + i_s + f_u · f_s     (1-D factor)

    Intercepts are regularised more than factors (as in Community Notes),
    so agreement explained by viewpoint goes into f_u·f_s and only
    cross-viewpoint appeal is left in i_s. Returns (i_s, state) where state
    can warm-start the next call.
    """
    nU, nS = R.shape
    if init is None:
        rng = np.random.default_rng(0)
        mu, iu, fu = R.mean(), np.zeros(nU), rng.normal(0, 0.1, nU)
        is_, fs = np.zeros(nS), rng.normal(0, 0.1, nS)
    else:
        mu, iu, fu, is_, fs = (np.array(x, copy=True) if np.ndim(x) else x for x in init)
    # Penalties scaled to per-rating loss (sum over all ratings, as in CN)
    L = np.diag([lam_i, lam_f])
    for _ in range(iters):
        # users: regress (R − μ − i_s) on [1, f_s]
        X = np.column_stack([np.ones(nS), fs])
        Y = (R - mu - is_[None, :]).T                     # nS × nU
        sol = np.linalg.solve(X.T @ X + L, X.T @ Y)      # 2 × nU
        iu, fu = sol[0], sol[1]
        if center:                                 # zero-mean factor: identifies intercept
            fu = fu - fu.mean()
        # sources: regress (R − μ − i_u) on [1, f_u]
        X = np.column_stack([np.ones(nU), fu])
        Y = R - mu - iu[:, None]                          # nU × nS
        sol = np.linalg.solve(X.T @ X + L, X.T @ Y)      # 2 × nS
        is_, fs = sol[0], sol[1]
        mu = (R - iu[:, None] - is_[None, :] - np.outer(fu, fs)).mean()
    return is_, (mu, iu, fu, is_, fs)


def mf_intercepts_masked(R, W, lam_i=0.15, lam_f=0.03, iters=30, init=None, center=True):
    """Same model as mf_intercepts, fit only on observed ratings (W[u, s] = 1).

    Penalties are added to per-user / per-source sums of squared error, as in the
    dense version, so sparsely rated sources are shrunk harder toward zero, as in
    Community Notes.
    """
    nU, nS = R.shape
    W = W.astype(float)
    if init is None:
        rng = np.random.default_rng(0)
        mu = (W * R).sum() / max(W.sum(), 1.0)
        iu, fu = np.zeros(nU), rng.normal(0, 0.1, nU)
        is_, fs = np.zeros(nS), rng.normal(0, 0.1, nS)
    else:
        mu, iu, fu, is_, fs = (np.array(x, copy=True) if np.ndim(x) else x for x in init)
    L = np.diag([lam_i, lam_f])

    def solve(Wm, Xf, Y):
        # One ridge regression per row of Wm: rows weight the columns of Y.
        f = Xf[:, 1]
        a00, a01, a11 = Wm.sum(1) + L[0, 0], Wm @ f, Wm @ (f * f) + L[1, 1]
        WY = Wm * Y
        b0, b1 = WY.sum(1), WY @ f
        det = a00 * a11 - a01 * a01                           # closed-form 2×2 solve
        return np.column_stack([(a11 * b0 - a01 * b1) / det, (a00 * b1 - a01 * b0) / det])

    for _ in range(iters):
        Xs = np.column_stack([np.ones(nS), fs])
        sol = solve(W, Xs, R - mu - is_[None, :])            # nU × 2
        iu, fu = sol[:, 0], sol[:, 1]
        if center:
            fu = fu - fu.mean()
        Xu = np.column_stack([np.ones(nU), fu])
        sol = solve(W.T, Xu, (R - mu - iu[:, None]).T)       # nS × 2
        is_, fs = sol[:, 0], sol[:, 1]
        resid = R - iu[:, None] - is_[None, :] - np.outer(fu, fs)
        mu = (W * resid).sum() / max(W.sum(), 1.0)
    return is_, (mu, iu, fu, is_, fs)


def init_user_types(dist: str, n: int, rng) -> np.ndarray:
    """(n, 5) array [w_e, w_c, α, β_u, γ]; same archetypes as v1."""
    def block(m, we, wc, a, g):
        return np.column_stack([we(m), wc(m), a(m), rng.uniform(0, 1, m), np.full(m, g)])

    hi = lambda m: rng.beta(5, 2, m)
    lo = lambda m: rng.beta(2, 5, m)
    a_pref = lambda m: rng.uniform(0.6, 1.0, m)

    if dist == "demanding":
        return block(n, hi, hi, a_pref, 0.10)
    if dist == "credibility_seeking":
        return block(n, lo, hi, a_pref, 0.05)
    if dist == "entertainment_credulous":
        return block(n, hi, lo, a_pref, 0.75)
    if dist == "apathetic":
        return block(n, lo, lo, lo, 0.30)
    if dist == "balanced_archetypes":
        q, extra = n // 4, n - 4 * (n // 4)
        arr = np.vstack([
            block(q, hi, hi, a_pref, 0.10),
            block(q, lo, hi, a_pref, 0.05),
            block(q, hi, lo, a_pref, 0.75),
            block(q + extra, lo, lo, lo, 0.30),
        ])
        rng.shuffle(arr)
        return arr
    if dist.startswith("gamma_"):
        # Uniform population with a set credulity, e.g. "gamma_0.5".
        g = float(dist.split("_")[1])
        return block(n, lambda m: rng.beta(3, 3, m), lambda m: rng.beta(3, 3, m), a_pref, g)
    raise ValueError(f"Unknown user_dist: {dist}")


class AttentionModelV2:
    def __init__(
        self,
        n_sources: int = 30,
        n_users: int = 600,
        k: int = 3,                    # sources per user (fixed)
        user_dist: str = "balanced_archetypes",
        policy: str = "popularity",
        # Source economics
        kappa_c: float = 0.5,          # cost of credibility
        kappa_e: float = 0.2,          # cost of entertainment
        lam: float = 0.3,              # weight sources put on platform score
        # Source adaptation
        mu_experiment: float = 0.2,    # prob. a source runs an A/B test per step
        sigma_prop: float = 0.05,      # proposal step size (symmetric)
        sigma_ab: float = 0.005,       # noise in A/B test readout
        imitation: bool = True,
        adapt_beta: bool = True,       # False = coverage frozen (no pandering possible)
        # User observation
        noise_e: float = 0.05,
        noise_c: float = 0.10,
        social_proof_weight: float = 0.30,
        # Rewiring
        rewire_prob: float = 0.5,
        improvement_margin: float = 0.02,
        rank_temp: float = 0.10,       # softmax temperature for recommendations
        expert_noise: float = 0.05,
        expert_coverage: float = 1.0,  # share of sources fact-checkers rate
        # Biased experts: score = (1 - expert_gamma)(c + error)
        #   + expert_gamma * (1 - |beta_s - expert_center|)   (axis 1)
        expert_gamma: float = 0.0,
        expert_center: float = 0.5,
        # corrected_trust: s.d. of the platform's error in estimated positions
        position_noise: float = 0.0,
        # How raw platform scores become a [0, 1] ranking signal. Must be the
        # same for every policy so policies are compared on equal footing.
        #   "rank"   percentile rank (what a feed position actually conveys)
        #   "minmax" (r − min) / (max − min)
        #   "max"    r / max  (legacy; MF intercepts were shifted by min first)
        score_norm: str = "rank",
        # Cost of repositioning: a proposal that moves coverage by distance d
        # must beat the current strategy by kappa_move * d to be adopted.
        kappa_move: float = 0.0,
        # Sparse, self-selected ratings for bridging_mf. Each step a user rates
        # the sources it holds plus each other source with probability
        # ~ rating_density * exp(-rating_homophily * distance), scaled so the
        # mean probability is rating_density. 1.0 = everyone rates everything.
        rating_density: float = 1.0,
        rating_homophily: float = 0.0,
        belief_dims: int = 1,          # dimensions of the belief/coverage space
        prior_dist: str = "uniform",   # "uniform" or "bimodal" (axis 1)
        # bridging_mf: re-center the user factor after each update (False = plain
        # regularized fit, closer to the Community Notes objective).
        mf_center: bool = True,
        # Rater panel for the population-pooled signals (bridging, bridging_mf),
        # drawn once from the users with weights; None = every user rates.
        #   rater_left_share: share of panel weight on the left group (beta_u < 1/2)
        #   rater_tilt: weights ∝ exp(-rater_tilt * (beta_u - 1/2)) (continuous tilt)
        rater_left_share: float = None,
        rater_tilt: float = 0.0,
        # Initial coverage: None = U(0, 1); a number = clip(N(0.5, sd)) on axis 1
        # (drawn from a separate stream, so everything else is unchanged).
        beta_init_sd: float = None,
        # How sources evaluate a proposal:
        #   "surrogate": one-step predicted payoff J = m[(1-lam) p + lam R] (default)
        #   "profit": adopt the proposal for trial_len steps and keep it only if
        #             realized profit d*m over the trial's second half beats the
        #             mean over the trial_len steps before it.
        #   "trial_J": like "profit" (same trials and horizon) but judged by the
        #             realized surrogate J = m[(1-lam) p + lam R] instead of d*m.
        #   "trial_profit_rank": like "profit" but judged by realized profit plus
        #             a direct rank term, m[(1-lam) d/d_mean + lam R].
        adapt_rule: str = "surrogate",
        trial_len: int = 10,
        # Initial accuracy: None = U(0.7, 1); "corr" = U(0.7, 1) marginals with
        # rank correlation ~0.75 with centrality 1-|beta-1/2|; "wide" = U(0.3, 1).
        init_acc: str = None,
        # Freeze coverage from this step on (None = never): sources first adapt,
        # then keep the configuration they reached.
        freeze_at: int = None,
        # Users' perceived credibility enters their utility on a compressed
        # scale, 0.5 + cred_scale * (c_hat - 0.5) (1 = model scale). Platform
        # scores use the unscaled ratings (rank-normalized, so unaffected).
        cred_scale: float = 1.0,
        seed: int = 0,
    ):
        assert policy in POLICIES + EXTRA_POLICIES, policy
        self.rng = rng = np.random.default_rng(seed)
        self.nS, self.nU, self.k = n_sources, n_users, k
        self.policy = policy
        self.kappa_c, self.kappa_e, self.lam = kappa_c, kappa_e, lam
        self.mu_experiment, self.sigma_prop, self.sigma_ab = mu_experiment, sigma_prop, sigma_ab
        self.imitation = imitation
        self.adapt_beta = adapt_beta
        self.noise_e, self.noise_c = noise_e, noise_c
        self.spw = social_proof_weight
        self.rewire_prob, self.margin = rewire_prob, improvement_margin
        self.rank_temp, self.expert_noise = rank_temp, expert_noise
        self.score_norm = score_norm
        self.D = belief_dims
        # Fixed set of sources the fact-checkers cover; unrated sources get
        # the mean rated score (no information either way).
        n_rated = max(1, int(round(expert_coverage * n_sources)))
        self.rated = np.zeros(n_sources, dtype=bool)
        self.rated[rng.choice(n_sources, n_rated, replace=False)] = True
        self.expert_bias = rng.normal(0, expert_noise, n_sources)
        self._mf_state = None
        self.kappa_move = kappa_move
        self.rating_density, self.rating_homophily = rating_density, rating_homophily
        self._mask = None
        # Separate stream, so sparse ratings never shift the main random stream.
        self.mask_rng = np.random.default_rng([seed, 7])
        self.expert_gamma, self.expert_center = expert_gamma, expert_center
        # Persistent errors in the platform's estimates of source and user
        # positions (corrected_trust only), from their own random stream.
        pos_rng = np.random.default_rng([seed, 11])
        self.pos_err_s = pos_rng.normal(0, position_noise, (n_sources, belief_dims))
        self.pos_err_u = pos_rng.normal(0, position_noise, (n_users, belief_dims))
        self.mf_center = mf_center
        assert adapt_rule in ("surrogate", "profit", "trial_J", "trial_profit_rank"), adapt_rule
        self.freeze_at, self.cred_scale = freeze_at, cred_scale
        self.adapt_rule, self.trial_len = adapt_rule, trial_len
        self.trial_rng = np.random.default_rng([seed, 17])

        # Sources [e, c, β_1..β_D]; all start credible, spread across beliefs.
        self.X = np.column_stack([
            rng.uniform(0.05, 0.60, n_sources),
            rng.uniform(0.70, 1.00, n_sources),
            rng.uniform(0.00, 1.00, (n_sources, belief_dims)),
        ])
        self.U = init_user_types(user_dist, n_users, rng)
        self.we, self.wc, self.alpha, bu1, self.gamma = self.U.T
        # Extra draws happen only for non-default settings, so the default
        # 1-D uniform model consumes the random stream exactly as before.
        if prior_dist == "bimodal":
            side = rng.random(n_users) < 0.5
            x = rng.beta(6, 18, n_users)
            bu1 = np.where(side, x, 1 - x)
        elif prior_dist != "uniform":
            raise ValueError(prior_dist)
        extra = [rng.uniform(0, 1, n_users) for _ in range(belief_dims - 1)]
        self.BU = np.column_stack([bu1] + extra)       # users × D
        self.bu = self.BU[:, 0]
        self.wsum = np.maximum(self.we + self.wc, 1e-6)
        self.group = (self.bu >= 0.5).astype(int)   # belief halves, for bridging
        if beta_init_sd is not None:
            init_rng = np.random.default_rng([seed, 13])
            self.X[:, 2] = np.clip(init_rng.normal(0.5, beta_init_sd, n_sources), 0, 1)
        if init_acc is not None:
            acc_rng = np.random.default_rng([seed, 23])
            if init_acc == "wide":
                self.X[:, 1] = acc_rng.uniform(0.3, 1.0, n_sources)
            elif init_acc == "corr":
                cent = 1.0 - np.abs(self.X[:, 2] - 0.5)
                z = 0.75 * (cent - cent.mean()) / cent.std() + np.sqrt(1 - 0.75 ** 2) * acc_rng.normal(size=n_sources)
                r = z.argsort().argsort()
                self.X[:, 1] = 0.7 + 0.3 * (r + 0.5) / n_sources
            else:
                raise ValueError(init_acc)
        # Rater panel (indices into users) for the pooled signals.
        self.panel = None
        if rater_left_share is not None or rater_tilt:
            w = np.exp(-rater_tilt * (self.bu - 0.5))
            if rater_left_share is not None:
                left = self.group == 0
                w = np.where(left, w * rater_left_share / w[left].sum(),
                             w * (1 - rater_left_share) / w[~left].sum())
            panel_rng = np.random.default_rng([seed, 19])
            self.panel = panel_rng.choice(n_users, n_users, replace=True, p=w / w.sum())
        # Profit-rule state: realized-profit history and running trials.
        self.ph = np.zeros((trial_len, n_sources))
        self.trial_start = np.full(n_sources, -1)
        self.trial_old = np.zeros((n_sources, 2 + belief_dims))
        self.trial_base = np.zeros(n_sources)
        self.trial_sum = np.zeros(n_sources)
        self.trial_n = np.zeros(n_sources)

        # Adjacency: each user holds exactly k sources.
        self.A = np.zeros((n_users, n_sources), dtype=bool)
        for u in range(n_users):
            self.A[u, rng.choice(n_sources, k, replace=False)] = True

        self.t = 0
        self.h = {key: [] for key in [
            "mean_c", "aud_c", "mean_e", "aud_e", "coverage_spread",
            "exposure", "perc_minus_true_c", "true_sat", "perc_sat",
            "gini", "rewires", "c_gap_by_group", "spread_axis1", "spread_axis2",
        ] + [f"{m}_{g}" for m in ("acc", "dist", "near") for g in "LCR"]}

    # ── Perception ────────────────────────────────────────────────────
    def _pref(self, X, users=None, noisy=False):
        """Preference utility matrix (users × sources) for strategies X."""
        u = slice(None) if users is None else users
        we, wc, bu, g, ws = self.we[u], self.wc[u], self.bu[u], self.gamma[u], self.wsum[u]
        e, c = X[:, 0][None, :], X[:, 1][None, :]
        if self.D == 1:
            align = 1.0 - np.abs(X[:, 2][None, :] - bu[:, None])
        else:
            d = X[None, :, 2:] - self.BU[u][:, None, :]
            align = 1.0 - np.sqrt((d ** 2).sum(-1)) / np.sqrt(self.D)
        pe = np.broadcast_to(e, align.shape)
        pc = (1 - g[:, None]) * c + g[:, None] * align
        pu = pc if self.cred_scale == 1.0 else 0.5 + self.cred_scale * (pc - 0.5)
        if noisy:
            pe = np.clip(pe + self.rng.normal(0, self.noise_e, pe.shape), 0, 1)
            pu = np.clip(pu + self.rng.normal(0, self.noise_c, pu.shape), 0, 1)
            if self.cred_scale == 1.0:
                pc = pu
        return (we[:, None] * pe + wc[:, None] * pu) / ws[:, None], pe, pc

    # ── Platform scores ───────────────────────────────────────────────
    def _scores(self, X, A, commit=True):
        """Raw platform score per source under the current policy."""
        deg = A.sum(0).astype(float)
        if self.policy == "popularity":
            return deg
        if self.policy == "expert":
            # Persistent per-source rating error (a fact-checker's misjudgement
            # sticks), not fresh noise each step, which would average out.
            r = X[:, 1] + self.expert_bias
            if self.expert_gamma > 0:
                agree = 1.0 - np.abs(X[:, 2] - self.expert_center)
                r = (1 - self.expert_gamma) * r + self.expert_gamma * agree
            r[~self.rated] = r[self.rated].mean()
            return r
        _, pe, pc = self._pref(X)
        if self.policy == "corrected_trust":
            # Agreement from estimated positions; its weight is estimated by a
            # within-source regression of ratings on agreement (source fixed
            # effects), then removed from every rating before averaging.
            Bs = X[:, 2:] + self.pos_err_s
            Bu = self.BU + self.pos_err_u
            d = np.sqrt(((Bs[None, :, :] - Bu[:, None, :]) ** 2).sum(-1)) / np.sqrt(self.D)
            a = 1.0 - d
            at, pt = a - a.mean(0), pc - pc.mean(0)
            g_hat = float(np.clip((at * pt).sum() / max((at ** 2).sum(), 1e-12), 0.0, 0.95))
            return (pc - g_hat * a).mean(0)
        if self.panel is not None and self.policy in ("bridging", "bridging_mf", "pop_mean"):
            pc = pc[self.panel]
        if self.policy == "bridging_mf":
            if self.rating_density < 1.0:
                if self._mask is None:
                    self._draw_mask()
                i_s, state = mf_intercepts_masked(pc, self._mask,
                                                  iters=8 if self._mf_state else 30,
                                                  init=self._mf_state, center=self.mf_center)
            else:
                i_s, state = mf_intercepts(pc, iters=8 if self._mf_state else 30,
                                           init=self._mf_state, center=self.mf_center)
            if commit:
                self._mf_state = state
            return i_s - i_s.min() if self.score_norm == "max" else i_s
        if self.policy == "pop_mean":
            if self.rating_density < 1.0:
                if self._mask is None:
                    self._draw_mask()
                W = self._mask
                return (pc * W).sum(0) / np.maximum(W.sum(0), 1)
            return pc.mean(0)
        if self.policy in ("engagement", "crowd_trust"):
            M = pe if self.policy == "engagement" else pc
            num = (M * A).sum(0)
            return np.where(deg > 0, num / np.maximum(deg, 1), 0.0)
        # bridging: every user is a potential rater; take the worse group
        grp = self.group if self.panel is None else self.group[self.panel]
        g0 = pc[grp == 0].mean(0)
        g1 = pc[grp == 1].mean(0)
        return np.minimum(g0, g1)

    def _draw_mask(self):
        """Who rates whom this step (sparse bridging_mf only)."""
        B = self.X[:, 2:]
        d = np.sqrt(((B[None, :, :] - self.BU[:, None, :]) ** 2).sum(-1)) / np.sqrt(self.D)
        w = np.exp(-self.rating_homophily * d)
        p = np.clip(self.rating_density * w / w.mean(), 0, 1)
        self._mask = self.A | (self.mask_rng.random(p.shape) < p)

    def _norm(self, r):
        if self.score_norm == "rank":
            gt = (r[:, None] > r[None, :]).sum(1)
            eq = (r[:, None] == r[None, :]).sum(1) - 1
            return (gt + 0.5 * eq) / max(len(r) - 1, 1)
        if self.score_norm == "minmax":
            span = r.max() - r.min()
            return (r - r.min()) / span if span > 0 else np.full_like(r, 0.5)
        m = r.max()
        return r / m if m > 0 else np.zeros_like(r)

    def _margin(self, X):
        return 1.0 - self.kappa_c * X[:, 1] - self.kappa_e * X[:, 0]

    def _payoff_one(self, s, x, R_others):
        """Expected payoff J for source s if it used strategy x."""
        X = self.X.copy()
        X[s] = x
        aud = np.flatnonzero(self.A[:, s])
        if aud.size:
            V = self._pref(x[None, :], users=aud)[0].mean()
        else:
            V = 0.5
        if self.policy == "popularity":
            R = self._norm(R_others)[s]          # unaffected by own strategy
        else:
            raw = self._scores(X, self.A, commit=False)
            R = self._norm(raw)[s]
        m = 1.0 - self.kappa_c * x[1] - self.kappa_e * x[0]
        return m * ((1 - self.lam) * V + self.lam * R)

    # ── Step ──────────────────────────────────────────────────────────
    def step(self):
        rng = self.rng
        if self.freeze_at is not None and self.t == self.freeze_at:
            self.adapt_beta = False
        deg = self.A.sum(0).astype(float)

        # 1. Imitation of visible (e, c) from a more profitable competitor.
        if self.imitation:
            profit = deg * self._margin(self.X)
            shared = (self.A.T.astype(int) @ self.A.astype(int)) > 0
            np.fill_diagonal(shared, False)
            newX = self.X.copy()
            pmax = max(profit.max(), 1e-9)
            for s in range(self.nS):
                if self.trial_start[s] >= 0:
                    continue
                nb = np.flatnonzero(shared[s])
                if nb.size == 0:
                    continue
                j = rng.choice(nb)
                if profit[j] > profit[s] and rng.random() < (profit[j] - profit[s]) / pmax:
                    newX[s, :2] = self.X[j, :2]
            self.X = newX

        # 2. Experimentation: symmetric proposal, keep if payoff improves.
        if self.policy in ("bridging_mf", "pop_mean") and self.rating_density < 1.0:
            self._draw_mask()
        if self.adapt_rule in ("profit", "trial_J", "trial_profit_rank"):
            self._start_trials()
            todo = []
        else:
            R_now = self._scores(self.X, self.A)
            todo = np.flatnonzero(rng.random(self.nS) < self.mu_experiment)
        for s in todo:
            x_old = self.X[s].copy()
            step = rng.normal(0, self.sigma_prop, 2 + self.D)
            if not self.adapt_beta:
                step[2:] = 0.0
            x_new = np.clip(x_old + step, 0, 1)
            j_old = self._payoff_one(s, x_old, R_now) + rng.normal(0, self.sigma_ab)
            j_new = self._payoff_one(s, x_new, R_now) + rng.normal(0, self.sigma_ab)
            if self.kappa_move:
                j_new -= self.kappa_move * np.linalg.norm(x_new[2:] - x_old[2:])
            if j_new > j_old:
                self.X[s] = x_new

        # 3. Users: consider dropping their worst source for a recommendation.
        deg = self.A.sum(0).astype(float)
        pop = deg / max(deg.max(), 1.0)
        R = self._norm(self._scores(self.X, self.A))
        pref_noisy = self._pref(self.X, noisy=True)[0]
        util = self.alpha[:, None] * pref_noisy + (1 - self.alpha[:, None]) * pop[None, :]
        prior = np.clip(0.5 + self.spw * R, 0, 1)
        exp_util = self.alpha[:, None] * prior[None, :] + (1 - self.alpha[:, None]) * pop[None, :]

        util_conn = np.where(self.A, util, np.inf)
        worst = util_conn.argmin(1)
        worst_val = util_conn[np.arange(self.nU), worst]
        gumbel = rng.gumbel(size=(self.nU, self.nS))
        cand_score = np.where(self.A, -np.inf, R[None, :] / self.rank_temp + gumbel)
        cand = cand_score.argmax(1)
        cand_val = exp_util[np.arange(self.nU), cand]
        switch = (cand_val > worst_val + self.margin) & (rng.random(self.nU) < self.rewire_prob)
        idx = np.flatnonzero(switch)
        self.A[idx, worst[idx]] = False
        self.A[idx, cand[idx]] = True

        if self.adapt_rule in ("profit", "trial_J", "trial_profit_rank"):
            self._end_trials(R)
        self._record(int(switch.sum()))
        self.t += 1

    def _start_trials(self):
        """Idle sources start a trial with prob. mu once they have a profit history."""
        if self.t < self.trial_len:
            return
        r = self.trial_rng
        idle = self.trial_start < 0
        for s in np.flatnonzero(idle & (r.random(self.nS) < self.mu_experiment)):
            step = r.normal(0, self.sigma_prop, 2 + self.D)
            if not self.adapt_beta:
                step[2:] = 0.0
            self.trial_old[s] = self.X[s]
            self.trial_base[s] = self.ph[:, s].mean()
            self.trial_sum[s] = self.trial_n[s] = 0.0
            self.trial_start[s] = self.t
            self.X[s] = np.clip(self.X[s] + step, 0, 1)

    def _end_trials(self, R=None):
        """Record realized profit (or realized J); close trials that have run trial_len steps."""
        if self.adapt_rule == "trial_J":
            pref = self._pref(self.X)[0]
            deg = self.A.sum(0)
            V = np.where(deg > 0, (pref * self.A).sum(0) / np.maximum(deg, 1), 0.5)
            profit = self._margin(self.X) * ((1 - self.lam) * V + self.lam * R)
        elif self.adapt_rule == "trial_profit_rank":
            d_rel = self.A.sum(0) / (self.nU * self.k / self.nS)
            profit = self._margin(self.X) * ((1 - self.lam) * d_rel + self.lam * R)
        else:
            profit = self.A.sum(0) * self._margin(self.X)
        self.ph[self.t % self.trial_len] = profit
        on = self.trial_start >= 0
        age = self.t - self.trial_start
        late = on & (age >= self.trial_len // 2)
        self.trial_sum[late] += profit[late]
        self.trial_n[late] += 1
        for s in np.flatnonzero(on & (age >= self.trial_len - 1)):
            gain = self.trial_sum[s] / max(self.trial_n[s], 1) - self.trial_base[s]
            if self.kappa_move:
                gain -= self.kappa_move * np.linalg.norm(self.X[s, 2:] - self.trial_old[s, 2:])
            if gain <= 0:
                self.X[s] = self.trial_old[s]
            self.trial_start[s] = -1

    # ── Metrics ───────────────────────────────────────────────────────
    def _record(self, rewires):
        A, X = self.A, self.X
        deg = A.sum(0).astype(float)
        w = deg / deg.sum()
        pref, pe, pc = self._pref(X)
        true = (self.we[:, None] * X[:, 0] + self.wc[:, None] * X[:, 1]) / self.wsum[:, None]
        k = A.sum(1)
        h = self.h
        h["mean_c"].append(X[:, 1].mean())
        h["aud_c"].append((w * X[:, 1]).sum())       # credibility users actually consume
        h["mean_e"].append(X[:, 0].mean())
        h["aud_e"].append((w * X[:, 0]).sum())
        B = X[:, 2:]
        # Spread = RMS distance of coverage from its centroid / √D (= s.d. when D = 1)
        h["coverage_spread"].append(np.sqrt(((B - B.mean(0)) ** 2).sum(1).mean() / self.D))
        h["spread_axis1"].append(B[:, 0].std())
        h["spread_axis2"].append(B[:, 1].std() if self.D > 1 else np.nan)
        # Exposure: mean belief distance between users and the sources they hold
        dist = np.sqrt(((B[None, :, :] - self.BU[:, None, :]) ** 2).sum(-1)) / np.sqrt(self.D)
        h["exposure"].append(dist[A].mean())
        h["perc_minus_true_c"].append(((pc - X[:, 1][None, :]) * A).sum() / A.sum())
        h["true_sat"].append(((true * A).sum(1) / k).mean())
        h["perc_sat"].append(((pref * A).sum(1) / k).mean())
        v = np.sort(deg)
        n = len(v)
        h["gini"].append((2 * (np.arange(1, n + 1) * v).sum()) / (n * v.sum()) - (n + 1) / n)
        h["rewires"].append(rewires)
        # Per belief third (L < 1/3 <= C < 2/3 <= R): accuracy of held sources,
        # distance to held sources, and best accuracy among sources within 0.15.
        held_c = (A * X[:, 1][None, :]).sum(1) / k
        held_d = (dist * A).sum(1) / k
        near = np.where(dist <= 0.15, X[:, 1][None, :], 0.0).max(1)
        third = np.minimum((self.bu * 3).astype(int), 2)
        for j, g in enumerate("LCR"):
            sel = third == j
            h[f"acc_{g}"].append(held_c[sel].mean() if sel.any() else np.nan)
            h[f"dist_{g}"].append(held_d[sel].mean() if sel.any() else np.nan)
            h[f"near_{g}"].append(near[sel].mean() if sel.any() else np.nan)

    def run(self, T=500):
        for _ in range(T):
            self.step()
        return self

    def per_user_consumed_c(self):
        """Mean true credibility of each user's sources (for harm-by-group)."""
        return (self.A * self.X[:, 1][None, :]).sum(1) / self.A.sum(1)
