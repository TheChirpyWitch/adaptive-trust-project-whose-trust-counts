#!/bin/sh
# Run every simulation grid used in the paper (T=2000 steps, outcomes averaged
# over the last 500). Each grid resumes if interrupted; finished grids are skipped.
cd "$(dirname "$0")"
export VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for g in paper_gamma paper_pops paper_expert paper_kappa norm_check norm_check_extra robust_beliefs empirical_gamma fine_gamma sensitivity move_cost sparse_mf expert_lean corrected_trust mf_uncentered frozen_central rater_pool profit_rule learning_rule pop_mean init_acc group_access freeze_evolved profit_rank cred_scale nonlinear_perc transfer access_radius pilot_long; do
  [ -f "results/$g.csv" ] && { echo "skip $g"; continue; }
  python3 experiments_v2.py "$g" --T 2000 --tail 500 --out results
done
