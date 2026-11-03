# Whose Trust Counts? Code and data

Code for the agent-based model, the simulation experiments, the empirical analysis of
real trust ratings, and every figure and table in the paper.

## Requirements

Python 3.10+ with `numpy` and `matplotlib`. `openpyxl` is needed only to rebuild the
derived data from the raw files (`data/prepare_data.py`).

## Layout

| Path | What it is |
|---|---|
| `attention_model_v2.py` | The model (Section 3; Appendix C gives the full ODD description). Every option defaults to the configuration used in the main text. |
| `experiments_v2.py` | All simulation grids. `python experiments_v2.py <grid> --T 2000 --tail 500 --out results` runs one grid in parallel and writes `results/<grid>.csv` (one row per run). Interrupted grids resume. |
| `run_all.sh` | Runs every grid the paper uses. |
| `results/` | Outputs of every grid used in the paper, so figures and tables can be rebuilt without rerunning. |
| `paper/make_figures.py` | Reads `results/` and `data/derived/` and writes every figure (`paper/figs/`) and table (`paper/tables/`). |
| `data/prepare_data.py` | Builds de-identified analysis files from the three public datasets (see `data/raw/README.md`). |
| `data/estimate_gamma.py` | Credulity regressions (Section 6; Appendix B). |
| `data/gamma_scaling.py` | Re-estimates credulity under other rescalings of quality and position. |
| `data/functional_form.py` | Curvature, interaction, and binned-residual checks of the credulity regression. |
| `data/real_bridging.py` | Ranking signals computed on real outlets and what they reward (Section 6). |
| `data/derived/` | De-identified derived data: one row per (respondent, outlet) rating, one row per outlet, and the estimates above. |

## Which grid produces which result

| Grid | Paper |
|---|---|
| `paper_gamma` | Figs. 2–3; Section 5.1 and Appendix B (normalization, rank rows) |
| `freeze_evolved` | Section 5.1, P2 (coverage frozen after sources converge) |
| `fine_gamma` | Section 5.1, P3 |
| `paper_pops` | Table 2 |
| `paper_expert` | Expert-rating heatmap |
| `rater_pool`, `group_access` | Skewed rater panels and group-level access |
| `empirical_gamma` | Fig. 4b–c (model across the estimated range) |
| `profit_rule`, `learning_rule`, `profit_rank` | What sources must respond to (profit-based, matched trial-based, and profit-plus-rank learning) |
| `robust_beliefs`, `move_cost`, `sparse_mf`, `mf_uncentered`, `pop_mean`, `init_acc`, `sensitivity` | Robustness (Section 5 and Appendix B; `pop_mean` also in Fig. 2) |
| `cred_scale` | Section 6 (perception on the survey's scale) |
| `paper_kappa`, `norm_check`, `norm_check_extra`, `frozen_central`, `expert_lean`, `corrected_trust` | Appendix B |
| `pilot_long` | Horizon check (outcomes level off by about step 1,500) |

## Reproducing the paper

```sh
sh run_all.sh                       # all simulation grids (several hours on 8 cores)
python data/estimate_gamma.py       # credulity estimates
python data/gamma_scaling.py
python data/functional_form.py
python data/real_bridging.py
python paper/make_figures.py        # figures and tables
```

`run_all.sh` sets `VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1`: each grid already uses
one process per core. To rebuild only the figures and tables from the included results,
run the last command.

## Data sources

The derived files are built from public data:

- Pennycook & Rand (2019), *PNAS*, lay and fact-checker trust ratings of news outlets, OSF https://osf.io/6bptd/
- Lin et al. (2023), *PNAS Nexus*, aggregate domain-quality ratings, https://github.com/hauselin/domain-quality-ratings
- Robertson et al. (2018), *PACM HCI*, partisan audience bias scores

Only rating, recognition and political-identity variables are kept; IP addresses,
locations, zip codes, free-text responses and original respondent IDs are dropped.
Please cite the original datasets if you use the derived files.
