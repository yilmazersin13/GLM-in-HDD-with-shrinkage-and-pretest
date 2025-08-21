High-Dimensional GLM Shrinkage Estimation — Simulation Study (R)

End-to-end R code to compare penalized, shrinkage, and pretest estimators for high-dimensional logistic GLMs. The script runs simulations, summarizes metrics, and saves plots/tables to a dated folder.

Methods

FM: Elastic-Net full model
SM: Screening + LASSO submodel
S: Stein-type shrinkage toward SM
PS: Positive-part Stein
PT: Bootstrap-calibrated pretest (chooses FM vs SM)

Requirements
R ≥ 4.2 and:
install.packages(c(
  "glmnet","MASS","Matrix","ggplot2","dplyr","tidyr","gridExtra",
  "parallel","ncvreg","reshape2","doParallel","ggridges"
))

Quick Start

Clone the repo and open the script.

For a fast smoke test: set QUICK_TEST <- TRUE.
For full runs: keep FALSE and (optionally) enable USE_PARALLEL <- TRUE.

Run: Rscript sim_study.R (or source in RStudio).

Parallel notes: Windows uses PSOCK via doParallel; Linux/macOS use mclapply. Cores are detected automatically.

Key Settings

Simulation grid: SIM_GRID (n, p, s, cor_type, rho)
Replications / bootstrap: N_REPS, B_BOOT

Pretest level: ALPHA
Signal range: SIGNAL_RANGE
Submodel size: k rule inside fit_submodel


Outputs (in simulation_results_fixed_YYYY-MM-DD/)

complete_results.(RData|csv)
summary_statistics.csv
Figures: fig1_… to fig8_… (MSE, trends, ratios, shrinkage, ranks, accuracy/log-loss, ridges, selection)
Tables: table1_main_results.csv, table2_ranking_summary.csv
simulation_summary_report.txt (settings + overall ranking)
