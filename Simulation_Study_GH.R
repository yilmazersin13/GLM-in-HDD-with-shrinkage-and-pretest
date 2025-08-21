################################################################################
# High-Dimensional GLM Shrinkage Estimation Simulation Study - 
# Implementation made by Ersin Yilmaz
################################################################################

# Load required libraries
library(glmnet)
library(MASS)
library(Matrix)
library(ggplot2)
library(dplyr)
library(tidyr)
library(gridExtra)
library(parallel)
library(ncvreg)
library(reshape2)
library(doParallel)

# Set up parallel processing (Windows-compatible)
USE_PARALLEL <- TRUE  # Set to TRUE to enable parallel processing (recommended for full simulation)
n_cores <- detectCores() - 1
is_windows <- .Platform$OS.type == "windows"

if (USE_PARALLEL) {
  if (is_windows) {
    tryCatch({
      library(doParallel)
      cl <- makeCluster(n_cores)
      registerDoParallel(cl)
      cat("Using", n_cores, "cores for parallel processing (Windows)\n")
    }, error = function(e) {
      cat("Failed to set up parallel processing on Windows. Running sequentially.\n")
      USE_PARALLEL <<- FALSE
    })
  } else {
    cat("Using", n_cores, "cores for parallel processing (Unix)\n")
  }
} else {
  cat("Running sequentially (parallel processing disabled)\n")
}

set.seed(2025)

################################################################################
# 1. SIMULATION PARAMETERS (Following Section 5 of Paper)
################################################################################

# Choose simulation size
QUICK_TEST <- FALSE  # Set to TRUE for quick testing, FALSE for full simulation

if (QUICK_TEST) {
  # Small grid for testing
  SIM_GRID <- expand.grid(
    n = c(50, 100),                     # Smaller sample sizes for testing
    p = c(100, 150),                    # Smaller dimensions
    s = 5,                              # Smaller sparsity
    cor_type = c("indep", "ar1"),       # Both correlation types
    rho = 0.5,
    stringsAsFactors = FALSE
  )
  N_REPS <- 10                          # Fewer replications for testing
  B_BOOT <- 10                          # Fewer bootstrap samples
  cat("RUNNING QUICK TEST VERSION\n")
} else {
  # Full simulation grid as specified in the paper
  SIM_GRID <- expand.grid(
    n = c(100, 200),                    # Sample sizes
    p = c(200, 500),                   # Dimensions  
    s = 10,                             # Sparsity (fixed)
    cor_type = c("indep", "ar1"),       # Correlation structures
    rho = 0.5,                          # AR(1) parameter
    stringsAsFactors = FALSE
  )
  N_REPS <- 100 #200                         # Number of replications per setting
  B_BOOT <- 50                         # Bootstrap samples for critical value
  cat("RUNNING FULL SIMULATION\n")
}

# Other simulation parameters
N_TEST <- 200                          # Test set size
ALPHA <- 0.05                           # Significance level for pretest
SIGNAL_RANGE <- c(-1, 1)                 # Signal strength range [a,b]

cat("Full Simulation Grid:\n")
print(SIM_GRID)
cat("\nTotal parameter combinations:", nrow(SIM_GRID), "\n")
cat("Total replications:", nrow(SIM_GRID) * N_REPS, "\n")

################################################################################
# 2. DATA GENERATION FUNCTIONS (Algorithm 2 from Paper)
################################################################################

#' Generate covariance matrix
#' @param p dimension
#' @param type "indep" or "ar1"
#' @param rho correlation parameter for AR(1)
generate_covariance <- function(p, type = "indep", rho = 0.5) {
  if (type == "indep") {
    return(diag(p))
  } else if (type == "ar1") {
    Sigma <- matrix(0, p, p)
    for (i in 1:p) {
      for (j in 1:p) {
        Sigma[i, j] <- rho^abs(i - j)
      }
    }
    return(Sigma)
  }
}

#' Generate simulation data following Algorithm 2
generate_data <- function(n, p, s = 10, cor_type = "indep", rho = 0.5, 
                          signal_range = c(1, 2)) {
  
  # Step 1: Initialize beta_star
  beta_star <- rep(0, p)
  
  # Step 2: Sample support S uniformly
  S <- sample(1:p, s, replace = FALSE)
  
  # Step 3: For j in S, draw magnitudes from Unif(a,b) and random signs
  for (j in S) {
    magnitude <- runif(1, min = signal_range[1], max = signal_range[2])
    sign_j <- sample(c(-1, 1), 1, prob = c(0.5, 0.5))
    beta_star[j] <- sign_j * magnitude
  }
  
  # Step 5: Generate covariance matrix
  Sigma <- generate_covariance(p, cor_type, rho)
  
  # Step 6: Generate design matrix X
  X <- mvrnorm(n, mu = rep(0, p), Sigma = Sigma)
  
  # Step 7: Standardize columns to mean 0 and variance 1
  X <- scale(X, center = TRUE, scale = TRUE)
  
  # Step 8: Set intercept for class balance
  beta_0 <- 0  
  
  # Step 9-11: Generate response
  eta <- beta_0 + X %*% beta_star
  pi_vec <- 1 / (1 + exp(-eta))
  y <- rbinom(n, 1, pi_vec)
  
  # Ensure we have both classes
  if (length(unique(y)) == 1) {
    y[sample(n, 1)] <- 1 - y[1]
  }
  
  return(list(
    X = X, 
    y = y, 
    beta_star = beta_star, 
    beta_0 = beta_0, 
    true_support = S,
    Sigma = Sigma
  ))
}

################################################################################
# 3. ESTIMATION FUNCTIONS - FIXED VERSIONS
################################################################################

#' Full Model (FM) - Elastic Net with proper regularization
fit_full_model <- function(X, y, alpha = 0.5) {  # Changed to Elastic Net (alpha=0.5)
  n <- nrow(X)
  p <- ncol(X)
  
  # Use cross-validation to select lambda
  cv_fit <- cv.glmnet(X, y, family = "binomial", 
                      alpha = alpha,  # Elastic Net mixing parameter
                      nfolds = 5)
  
  # Use lambda.min for best predictive performance
  fit <- glmnet(X, y, family = "binomial", 
                alpha = alpha,
                lambda = cv_fit$lambda.min)
  
  beta_fm <- as.vector(coef(fit))[-1]  # Remove intercept
  beta_0_fm <- as.vector(coef(fit))[1]
  
  return(list(
    beta = beta_fm, 
    beta_0 = beta_0_fm, 
    lambda = cv_fit$lambda.min,
    cv_fit = cv_fit,
    model = fit,
    n_nonzero = sum(beta_fm != 0)
  ))
}

fit_submodel <- function(X, y, k = NULL) {
  n <- nrow(X)
  p <- ncol(X)
  
  # Default k as in paper: k = floor(n/log(p))
  if (is.null(k)) {
    k <- floor(n / max(log(p), 2))
    k <- max(5, min(k, floor(n/2), p))  # Ensure reasonable bounds
  }
  
  # Step 1: Marginal screening
  t_stats <- numeric(p)
  for (j in 1:p) {
    tryCatch({
      fit_j <- glm(y ~ X[, j], family = binomial())
      if (fit_j$converged) {
        t_stats[j] <- abs(coef(fit_j)[2] / sqrt(vcov(fit_j)[2, 2]))
      } else {
        t_stats[j] <- 0
      }
    }, error = function(e) {
      t_stats[j] <<- 0
    })
  }
  
  # Step 2: Select top k features
  selected_features <- order(t_stats, decreasing = TRUE)[1:k]
  
  # Step 3: Fit LASSO on selected features
  X_sub <- X[, selected_features, drop = FALSE]
  
  cv_lasso <- cv.glmnet(X_sub, y, family = "binomial", 
                        alpha = 1,  # Pure LASSO
                        nfolds = 5)
  
  fit <- glmnet(X_sub, y, family = "binomial", 
                alpha = 1,
                lambda = cv_lasso$lambda.min)
  
  # Construct full beta vector
  beta_sm <- rep(0, p)
  beta_sub <- as.vector(coef(fit))[-1]
  beta_sm[selected_features] <- beta_sub
  beta_0_sm <- as.vector(coef(fit))[1]
  
  return(list(
    beta = beta_sm, 
    beta_0 = beta_0_sm, 
    selected = selected_features, 
    k = k,
    k_actual = sum(beta_sm != 0),
    screening_stats = t_stats
  ))
}

#' Compute log-likelihood with numerical stability
loglik <- function(X, y, beta, beta_0 = 0) {
  # Bound coefficients to prevent overflow
  beta <- pmax(pmin(beta, 20), -20)
  beta_0 <- pmax(pmin(beta_0, 20), -20)
  
  eta <- beta_0 + X %*% beta
  eta <- pmax(pmin(eta, 100), -100)  # Prevent extreme values
  
  # Stable computation of log(1 + exp(eta))
  stable_log1pexp <- function(x) {
    idx_pos <- x > 0
    result <- numeric(length(x))
    result[idx_pos] <- x[idx_pos] + log1p(exp(-x[idx_pos]))
    result[!idx_pos] <- log1p(exp(x[!idx_pos]))
    return(result)
  }
  
  ll <- sum(y * eta - stable_log1pexp(eta))
  
  if (!is.finite(ll)) {
    warning("Non-finite log-likelihood detected")
    ll <- -1e10
  }
  
  return(ll)
}

fit_stein_shrinkage <- function(beta_fm, beta_sm, X, y, beta_0_fm, beta_0_sm, k = NULL) {
  p <- length(beta_fm)
  n <- nrow(X)
  
  # Calculate p2 correctly (dimension of restricted space)
  if (is.null(k)) {
    k <- sum(beta_sm != 0)
  }
  p2 <- p - k
  
  
  # Theory requires p2 >= 3, but very small p2 can still be unstable
  p2 <- max(p2, 5) 
  
  # Compute likelihood ratio statistic
  ll_fm <- loglik(X, y, beta_fm, beta_0_fm)
  ll_sm <- loglik(X, y, beta_sm, beta_0_sm)
  D_n <- 2 * (ll_fm - ll_sm)
  
  # 
  # Theory expects D_n ~ chi-squared(p2), so E[D_n] ≈ p2
  # If D_n << p2, the models are too similar and shrinkage becomes unstable
  
  # Expected value and reasonable bounds for D_n
  expected_D_n <- p2
  min_D_n <- max(1, p2 / 2)  # Don't let D_n be too small relative to p2
  
  if (D_n < min_D_n) {
    
    delta_n <- 0.5  
    warning(sprintf("D_n = %.3f below minimum %.3f, using moderate shrinkage", 
                    D_n, min_D_n))
  } else {
    
    delta_n <- 1 - (p2 - 2) / D_n
    
    
    if (delta_n < -1) {
      
      delta_n <- -1
    }
    if (delta_n > 1) {
      # Shouldn't happen theoretically but cap for safety
      delta_n <- 1
    }
  }
  
  beta_diff <- beta_fm - beta_sm
  beta_0_diff <- beta_0_fm - beta_0_sm
  
  max_diff <- max(abs(beta_diff))
  if (max_diff > 10 && delta_n < 0) {
    
    warning(sprintf("Large differences (max=%.2f) with negative shrinkage, dampening effect", max_diff))
    delta_n <- delta_n * 0.5  # Dampen negative shrinkage
  }
  
  # Apply shrinkage
  beta_s <- beta_sm + delta_n * beta_diff
  beta_0_s <- beta_0_sm + delta_n * beta_0_diff
  
  # Final safety bounds
  beta_s <- pmax(pmin(beta_s, 20), -20)
  beta_0_s <- pmax(pmin(beta_0_s, 20), -20)
  
  return(list(
    beta = beta_s,
    beta_0 = beta_0_s, 
    delta_n = delta_n, 
    D_n = D_n, 
    p2 = p2,
    k = k,
    min_D_n = min_D_n,
    was_adjusted = (D_n < min_D_n || abs(delta_n) == 1)
  ))
}


fit_positive_stein <- function(beta_fm, beta_sm, X, y, beta_0_fm, beta_0_sm, k = NULL) {
  # Get Stein shrinkage result with fixes
  result <- fit_stein_shrinkage(beta_fm, beta_sm, X, y, beta_0_fm, beta_0_sm, k)
  
  # Apply positive part
  delta_n_plus <- max(0, result$delta_n)
  
  # Recompute with positive shrinkage
  beta_diff <- beta_fm - beta_sm
  beta_0_diff <- beta_0_fm - beta_0_sm
  
  # Positive-part estimator
  beta_ps <- beta_sm + delta_n_plus * beta_diff
  beta_0_ps <- beta_0_sm + delta_n_plus * beta_0_diff
  
  # Final bounds
  beta_ps <- pmax(pmin(beta_ps, 20), -20)
  beta_0_ps <- pmax(pmin(beta_0_ps, 20), -20)
  
  return(list(
    beta = beta_ps
    beta_0 = beta_0_ps,
    delta_n = result$delta_n,  # Original (possibly negative)
    delta_n_plus = delta_n_plus,  # After positive part
    D_n = result$D_n, 
    p2 = result$p2,
    k = result$k,
    was_negative = (result$delta_n < 0),
    was_adjusted = result$was_adjusted
  ))
}

#' Bootstrap critical value (Algorithm 1, Section 3.3)
bootstrap_critical_value <- function(X, y, beta_sm, beta_0_sm, 
                                     alpha = 0.05, B = 200) {
  n <- nrow(X)
  D_boot <- numeric(B)
  
  # Fitted probabilities under null (SM is true)
  eta_sm <- beta_0_sm + X %*% beta_sm
  eta_sm <- pmax(pmin(eta_sm, 100), -100)
  pi_hat <- 1 / (1 + exp(-eta_sm))
  
  for (b in 1:B) {
    print(b)
    # Generate bootstrap sample under null
    y_boot <- rbinom(n, 1, pi_hat)
    
    # Ensure both classes present
    if (length(unique(y_boot)) == 1) {
      y_boot[sample(n, 1)] <- 1 - y_boot[1]
    }
    
    tryCatch({
      # Fit FM and SM to bootstrap data
      fm_boot <- fit_full_model(X, y_boot)
      sm_boot <- fit_submodel(X, y_boot)
      
      # Compute D_n for bootstrap sample
      ll_fm_boot <- loglik(X, y_boot, fm_boot$beta, fm_boot$beta_0)
      ll_sm_boot <- loglik(X, y_boot, sm_boot$beta, sm_boot$beta_0)
      D_boot[b] <- 2 * (ll_fm_boot - ll_sm_boot)
      
    }, error = function(e) {
      D_boot[b] <<- 0
    })
  }
  
  # Return (1-alpha) quantile
  D_alpha <- quantile(D_boot, 1 - alpha, na.rm = TRUE)
  return(list(D_alpha = D_alpha, D_boot = D_boot))
}

#' Pretest estimator (Section 3.2.4, Equations 15-16)
fit_pretest <- function(beta_fm, beta_sm, X, y, beta_0_fm, beta_0_sm, D_alpha) {
  # Compute test statistic
  ll_fm <- loglik(X, y, beta_fm, beta_0_fm)
  ll_sm <- loglik(X, y, beta_sm, beta_0_sm)
  D_n <- 2 * (ll_fm - ll_sm)
  
  # Pretest decision
  if (D_n > D_alpha) {
    # Reject null, use FM
    return(list(
      beta = beta_fm, 
      beta_0 = beta_0_fm, 
      selected_model = "FM", 
      D_n = D_n,
      rejected = TRUE
    ))
  } else {
    # Fail to reject, use SM
    return(list(
      beta = beta_sm, 
      beta_0 = beta_0_sm, 
      selected_model = "SM", 
      D_n = D_n,
      rejected = FALSE
    ))
  }
}

################################################################################
# 4. PERFORMANCE METRICS
################################################################################

#' Compute performance metrics
compute_metrics <- function(beta_hat, beta_star, X_test, y_test, 
                            beta_0_hat = 0, method_name = "") {
  
  # 1. Estimation error
  mse <- sum((beta_hat - beta_star)^2)
  
  # 2. Prediction on test set
  eta_test <- beta_0_hat + X_test %*% beta_hat
  eta_test <- pmax(pmin(eta_test, 100), -100)
  pi_test <- 1 / (1 + exp(-eta_test))
  
  # Prediction loss
  eps <- 1e-15
  pi_test <- pmax(eps, pmin(1 - eps, pi_test))
  log_loss <- -mean(y_test * log(pi_test) + (1 - y_test) * log(1 - pi_test))
  
  # 3. Selection quality
  selected <- which(beta_hat != 0)
  true_support <- which(beta_star != 0)
  
  if (length(true_support) > 0) {
    tpr <- length(intersect(selected, true_support)) / length(true_support)
  } else {
    tpr <- NA
  }
  
  true_zeros <- which(beta_star == 0)
  if (length(true_zeros) > 0) {
    fpr <- length(intersect(selected, true_zeros)) / length(true_zeros)
  } else {
    fpr <- NA
  }
  
  # Number of non-zero coefficients
  nnz <- length(selected)
  
  # Classification accuracy
  y_pred <- as.numeric(pi_test > 0.5)
  accuracy <- mean(y_pred == y_test)
  
  return(list(
    mse = mse, 
    log_loss = log_loss, 
    tpr = tpr, 
    fpr = fpr, 
    nnz = nnz,
    accuracy = accuracy
  ))
}

################################################################################
# 5. RUN SINGLE REPLICATION
################################################################################

run_single_replication <- function(params, rep_id) {
  tryCatch({
    set.seed(1000 * params$n + params$p + rep_id)
    
    # Generate training data
    train_data <- generate_data(
      n = params$n, 
      p = params$p, 
      s = params$s,
      cor_type = params$cor_type, 
      rho = params$rho,
      signal_range = SIGNAL_RANGE
    )
    
    # Generate test data
    test_data <- generate_data(
      n = N_TEST, 
      p = params$p, 
      s = params$s,
      cor_type = params$cor_type, 
      rho = params$rho,
      signal_range = SIGNAL_RANGE
    )
    
    # Fit all methods
    fm_fit <- fit_full_model(train_data$X, train_data$y)
    sm_fit <- fit_submodel(train_data$X, train_data$y)
    
    # Pass k from SM to shrinkage estimators
    k_used <- sm_fit$k
    
    # Shrinkage estimators with fixed functions
    s_fit <- fit_stein_shrinkage(
      fm_fit$beta, sm_fit$beta, 
      train_data$X, train_data$y,
      fm_fit$beta_0, sm_fit$beta_0,
      k = k_used
    )
    
    ps_fit <- fit_positive_stein(
      fm_fit$beta, sm_fit$beta,
      train_data$X, train_data$y, 
      fm_fit$beta_0, sm_fit$beta_0,
      k = k_used
    )
    
    # Bootstrap critical value for pretest
    boot_result <- bootstrap_critical_value(
      train_data$X, train_data$y,
      sm_fit$beta, sm_fit$beta_0,
      alpha = ALPHA, B = B_BOOT
    )
    
    pt_fit <- fit_pretest(
      fm_fit$beta, sm_fit$beta,
      train_data$X, train_data$y,
      fm_fit$beta_0, sm_fit$beta_0,
      boot_result$D_alpha
    )
    
    # Compute metrics for all methods
    methods <- list(
      FM = fm_fit,
      SM = sm_fit, 
      S = s_fit,
      PS = ps_fit,
      PT = pt_fit
    )
    
    results <- data.frame()
    for (method_name in names(methods)) {
      fit <- methods[[method_name]]
      metrics <- compute_metrics(
        fit$beta, train_data$beta_star,
        test_data$X, test_data$y,
        fit$beta_0, method_name
      )
      
      result_row <- data.frame(
        rep = rep_id,
        n = params$n,
        p = params$p,
        s = params$s,
        cor_type = params$cor_type,
        method = method_name,
        mse = metrics$mse,
        log_loss = metrics$log_loss,
        tpr = metrics$tpr,
        fpr = metrics$fpr,
        nnz = metrics$nnz,
        accuracy = metrics$accuracy,
        delta_n = ifelse(method_name %in% c("S", "PS"), fit$delta_n, NA),
        D_n = ifelse(method_name %in% c("S", "PS", "PT"), fit$D_n, NA),
        p2 = ifelse(method_name %in% c("S", "PS"), fit$p2, NA),
        k = ifelse(method_name %in% c("SM", "S", "PS"), k_used, NA)
      )
      
      results <- rbind(results, result_row)
    }
    
    return(results)
    
  }, error = function(e) {
    cat("Error in replication", rep_id, ":", e$message, "\n")
    return(NULL)
  })
}

################################################################################
# 6. MAIN SIMULATION LOOP
################################################################################

# Quick test of single replication
cat("Testing single replication first...\n")
test_params <- SIM_GRID[1, ]
test_result <- run_single_replication(test_params, 999)
if (is.null(test_result)) {
  cat("ERROR: Test replication failed!\n")
  stop("Test replication failed")
} else {
  cat("SUCCESS: Test replication worked!\n")
  cat("Methods tested:", paste(unique(test_result$method), collapse = ", "), "\n\n")
}

# Initialize results storage
all_results <- data.frame()

# Create output directory
output_dir <- paste0("simulation_results_fixed_", Sys.Date())
dir.create(output_dir, showWarnings = FALSE)

# Run simulation for each parameter combination
for (grid_row in 1:nrow(SIM_GRID)) {
  sim_params <- SIM_GRID[grid_row, ]
  message(grid_row)
  message("Running scenario Grid Row: ",  grid_row, " Sim Grid: ", nrow(SIM_GRID), " sim params: ", 
          sim_params$n, " p: ", sim_params$p, " s: ", sim_params$s, " cor_type: ", sim_params$cor_type)
  
  # Run replications
  if (USE_PARALLEL) {
    if (is_windows) {
      # Load libraries on cluster workers
      clusterEvalQ(cl, {
        library(glmnet)
        library(MASS)
        library(Matrix)
        library(dplyr)
      })
      
      # Export necessary objects to cluster
      clusterExport(cl, c("run_single_replication", "sim_params", "generate_data", 
                          "generate_covariance", "fit_full_model", "fit_submodel",
                          "fit_stein_shrinkage", "fit_positive_stein", 
                          "bootstrap_critical_value", "fit_pretest", "loglik",
                          "compute_metrics", "N_TEST", "SIGNAL_RANGE", "ALPHA", "B_BOOT"))
      
      scenario_results <- parLapply(cl, 1:N_REPS, function(rep_id) {
        run_single_replication(sim_params, rep_id)
      })
    } else {
      scenario_results <- mclapply(1:N_REPS, function(rep_id) {
        run_single_replication(sim_params, rep_id)
      }, mc.cores = n_cores)
    }
  } else {
    scenario_results <- list()
    for (rep_id in 1:N_REPS) {
      if (rep_id %% 5 == 0) {
        cat("    Replication", rep_id, "/", N_REPS, "\n")
      }
      result <- run_single_replication(sim_params, rep_id)
      scenario_results[[rep_id]] <- result
    }
  }
  
  # Combine results
  valid_results <- scenario_results[!sapply(scenario_results, is.null)]
  failed_results <- sum(sapply(scenario_results, is.null))
  
  cat(sprintf("  Completed: %d success, %d failed\n", 
              length(valid_results), failed_results))
  
  if (length(valid_results) > 0) {
    scenario_df <- do.call(rbind, valid_results)
    all_results <- rbind(all_results, scenario_df)
  }
  
  # Save intermediate results
  save(scenario_df, file = file.path(output_dir, 
                                     paste0("scenario_", grid_row, ".RData")))
}

# Save complete results
save(all_results, SIM_GRID, file = file.path(output_dir, "complete_results.RData"))
write.csv(all_results, file.path(output_dir, "complete_results.csv"), row.names = FALSE)

cat("\nSimulation completed!\n")
cat("Results saved in:", output_dir, "\n")

################################################################################
# 7. ANALYSIS AND VISUALIZATION (WITH VARIED PLOT TYPES)
################################################################################

# Summary statistics
summary_stats <- all_results %>%
  group_by(method, n, p, cor_type) %>%
  dplyr::summarise(
    MSE_mean = mean(mse, na.rm = TRUE),
    MSE_sd = sd(mse, na.rm = TRUE),
    MSE_median = median(mse, na.rm = TRUE),
    LogLoss_mean = mean(log_loss, na.rm = TRUE),
    LogLoss_sd = sd(log_loss, na.rm = TRUE),
    Accuracy_mean = mean(accuracy, na.rm = TRUE),
    Accuracy_sd = sd(accuracy, na.rm = TRUE),
    TPR_mean = mean(tpr, na.rm = TRUE),
    FPR_mean = mean(fpr, na.rm = TRUE),
    NNZ_mean = mean(nnz, na.rm = TRUE),
    Delta_mean = mean(delta_n, na.rm = TRUE),
    Delta_negative_prop = mean(delta_n < 0, na.rm = TRUE),
    .groups = "drop"
  )

# Save summary statistics
write.csv(summary_stats, file.path(output_dir, "summary_statistics.csv"), row.names = FALSE)

################################################################################
# 8. GENERATE VARIED VISUALIZATIONS
################################################################################

# Theme for publication-quality plots
theme_pub <- theme_minimal() +
  theme(
    text = element_text(size = 11),
    axis.title = element_text(size = 12),
    plot.title = element_text(size = 14, hjust = 0.5),
    legend.position = "bottom",
    panel.grid.minor = element_blank(),
    strip.text = element_text(size = 10)
  )

# Color palette for methods
method_colors <- c(
  "FM" = "#E74C3C",   # Red
  "SM" = "#3498DB",   # Blue
  "S" = "#2ECC71",    # Green
  "PS" = "#9B59B6",   # Purple
  "PT" = "#F39C12"    # Orange
)

# Figure 1: BAR PLOT - Mean MSE comparison

p1 <- ggplot(summary_stats, aes(x = method, y = MSE_mean, fill = method)) +
  geom_bar(stat = "identity", alpha = 0.8) +
  geom_errorbar(aes(ymin = MSE_mean - MSE_sd, ymax = MSE_mean + MSE_sd), 
                width = 0.2, size = 0.5) +
  facet_grid(n + p ~ cor_type, scales = "free_y",
             labeller = labeller(.rows = label_both, .cols = label_both)) +
  scale_y_log10() +
  scale_fill_manual(values = method_colors) +
  labs(title = "Mean Estimation Error (MSE) with Standard Deviation",
       x = "Method", y = "MSE (log scale)") +
  theme_pub +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave(file.path(output_dir, "fig1_mse_barplot.png"), p1, 
       width = 12, height = 10, dpi = 300,bg="white")

# Figure 2: LINE PLOT - Performance across sample sizes

perf_by_n <- summary_stats %>%
  mutate(scenario = paste(p, cor_type, sep = "_"))

p2 <- ggplot(perf_by_n, aes(x = n, y = MSE_mean, color = method, group = method)) +
  geom_line(size = 1.2) +
  geom_point(size = 3) +
  facet_wrap(~ scenario, scales = "free_y", ncol = 2,
             labeller = labeller(.default = label_both)) +
  scale_y_log10() +
  scale_color_manual(values = method_colors) +
  labs(title = "MSE Trends Across Sample Sizes",
       x = "Sample Size (n)", y = "Mean MSE (log scale)") +
  theme_pub

ggsave(file.path(output_dir, "fig2_mse_trends.png"), p2, 
       width = 10, height = 8, dpi = 300,bg="white")

# Figure 3: HEATMAP - Pairwise performance ratios

pairwise_mse <- all_results %>%
  group_by(rep, n, p, cor_type) %>%
  dplyr::select(rep, n, p, cor_type, method, mse) %>%
  pivot_wider(names_from = method, values_from = mse) %>%
  ungroup()

# Calculate mean ratios
ratio_matrix <- expand.grid(
  method1 = c("FM", "SM", "S", "PS", "PT"),
  method2 = c("FM", "SM", "S", "PS", "PT"),
  stringsAsFactors = FALSE
) %>%
  filter(method1 != method2) %>%
  mutate(
    ratio = NA_real_
  )

for (i in 1:nrow(ratio_matrix)) {
  m1 <- ratio_matrix$method1[i]
  m2 <- ratio_matrix$method2[i]
  ratio_matrix$ratio[i] <- mean(pairwise_mse[[m1]] / pairwise_mse[[m2]], na.rm = TRUE)
}

p3 <- ggplot(ratio_matrix, aes(x = method2, y = method1, fill = log2(ratio))) +
  geom_tile() +
  geom_text(aes(label = sprintf("%.2f", ratio)), size = 3) +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red", 
                       midpoint = 0, 
                       name = "log2(Ratio)",
                       limits = c(-2, 2)) +
  labs(title = "Pairwise MSE Ratios (Row/Column)",
       subtitle = "Blue indicates row method is better, Red indicates column method is better",
       x = "Denominator Method", y = "Numerator Method") +
  theme_pub +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave(file.path(output_dir, "fig3_pairwise_heatmap.png"), p3, 
       width = 8, height = 7, dpi = 300,bg="white")

# Figure 4: VIOLIN PLOT - Distribution of shrinkage factors

shrinkage_df <- all_results %>%
  filter(method %in% c("S", "PS")) %>%
  dplyr::select(method, n, p, cor_type, delta_n) %>%
  mutate(scenario = paste0("n=", n, ", p=", p, ", ", cor_type))

p4 <- ggplot(shrinkage_df, aes(x = method, y = delta_n, fill = method)) +
  geom_violin(alpha = 0.7, scale = "width") +
  geom_boxplot(width = 0.1, alpha = 0.5, outlier.size = 0.5) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red", size = 0.8) +
  geom_hline(yintercept = 1, linetype = "dotted", color = "gray50", size = 0.8) +
  facet_wrap(~ scenario, ncol = 2) +
  scale_fill_manual(values = c("S" = "#2ECC71", "PS" = "#9B59B6")) +
  labs(title = "Distribution of Shrinkage Factors",
       subtitle = "Red line at 0 (SM), Gray line at 1 (FM)",
       x = "Method", y = expression(delta[n])) +
  theme_pub

ggsave(file.path(output_dir, "fig4_shrinkage_violin.png"), p4, 
       width = 10, height = 8, dpi = 300,bg="white")

# Figure 5: AREA PLOT - Cumulative performance ranking

ranking_df <- all_results %>%
  group_by(rep, n, p, cor_type) %>%
  mutate(rank = rank(mse)) %>%
  ungroup() %>%
  group_by(method, rank) %>%
  summarise(count = n(), .groups = "drop") %>%
  group_by(method) %>%
  mutate(prop = count / sum(count) * 100) %>%
  ungroup()

p5 <- ggplot(ranking_df, aes(x = rank, y = prop, fill = method)) +
  geom_area(alpha = 0.7, position = "identity") +
  scale_fill_manual(values = method_colors) +
  labs(title = "Performance Ranking Distribution",
       subtitle = "Lower rank = better MSE",
       x = "Rank", y = "Percentage of Cases") +
  theme_pub

ggsave(file.path(output_dir, "fig5_ranking_area.png"), p5, 
       width = 10, height = 6, dpi = 300,bg="white")

# Figure 6: SCATTER PLOT - Accuracy vs Log Loss trade-off

p6 <- ggplot(all_results, aes(x = log_loss, y = accuracy, color = method)) +
  geom_point(alpha = 0.3, size = 1) +
  geom_smooth(method = "loess", se = TRUE, size = 1.2) +
  facet_wrap(~ n + p, scales = "free", 
             labeller = labeller(.default = label_both)) +
  scale_color_manual(values = method_colors) +
  labs(title = "Accuracy vs Log Loss Trade-off",
       x = "Log Loss (lower is better)", 
       y = "Accuracy (higher is better)") +
  theme_pub

ggsave(file.path(output_dir, "fig6_accuracy_logloss.jpeg"), p6, 
       width = 12, height = 8, dpi = 300, quality = 95,bg="white")

# Figure 7: RIDGE PLOT - MSE distributions by scenario

library(ggridges)

ridge_data <- all_results %>%
  mutate(scenario = paste0("n=", n, ", p=", p, ", ", cor_type))

p7 <- ggplot(ridge_data, aes(x = log10(mse + 1), y = scenario, fill = method)) +
  geom_density_ridges(alpha = 0.6, scale = 2) +
  scale_fill_manual(values = method_colors) +
  labs(title = "MSE Distribution Across Scenarios",
       x = "log10(MSE + 1)", y = "Scenario") +
  theme_pub +
  theme(axis.text.y = element_text(size = 9))

ggsave(file.path(output_dir, "fig7_mse_ridges.png"), p7, 
       width = 10, height = 8, dpi = 300,bg="white")

# Figure 8: DOT PLOT - Selection performance (TPR and FPR)
selection_summary <- all_results %>%
  group_by(method, n, p, cor_type) %>%
  summarise(
    TPR_mean = mean(tpr, na.rm = TRUE),
    TPR_se = sd(tpr, na.rm = TRUE) / sqrt(n()),
    FPR_mean = mean(fpr, na.rm = TRUE),
    FPR_se = sd(fpr, na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  ) %>%
  pivot_longer(cols = c(TPR_mean, FPR_mean), 
               names_to = "metric", 
               values_to = "value") %>%
  mutate(
    se = ifelse(metric == "TPR_mean", TPR_se, FPR_se),
    metric = ifelse(metric == "TPR_mean", "True Positive Rate", "False Positive Rate"),
    scenario = paste0("n=", n, ", p=", p, ", ", cor_type)
  )

p8 <- ggplot(selection_summary, aes(x = method, y = value, color = metric)) +
  geom_point(size = 3, position = position_dodge(width = 0.3)) +
  geom_errorbar(aes(ymin = value - se, ymax = value + se), 
                width = 0.2, position = position_dodge(width = 0.3)) +
  facet_wrap(~ scenario, ncol = 2) +
  scale_color_manual(values = c("True Positive Rate" = "darkgreen", 
                                "False Positive Rate" = "darkred")) +
  labs(title = "Variable Selection Performance",
       x = "Method", y = "Rate") +
  theme_pub +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave(file.path(output_dir, "fig8_selection_dotplot.jpeg"), p8, 
       width = 10, height = 8, dpi = 300, quality = 95,bg="white")

################################################################################
# 9. GENERATE SUMMARY TABLES
################################################################################

# Table 1: Main results summary
main_table <- summary_stats %>%
  dplyr::select(method, n, p, cor_type, MSE_mean, MSE_sd, LogLoss_mean, 
                Accuracy_mean, TPR_mean, FPR_mean, NNZ_mean) %>%
  mutate(
    MSE = sprintf("%.3f (%.3f)", MSE_mean, MSE_sd),
    LogLoss = sprintf("%.3f", LogLoss_mean),
    Accuracy = sprintf("%.3f", Accuracy_mean),
    TPR = sprintf("%.3f", TPR_mean),
    FPR = sprintf("%.3f", FPR_mean),
    NNZ = sprintf("%.1f", NNZ_mean)
  ) %>%
  dplyr::select(method, n, p, cor_type, MSE, LogLoss, Accuracy, TPR, FPR, NNZ)

write.csv(main_table, file.path(output_dir, "table1_main_results.csv"), row.names = FALSE)

# Table 2: Performance ranking summary
ranking_summary <- all_results %>%
  group_by(rep, n, p, cor_type) %>%
  mutate(mse_rank = rank(mse)) %>%
  ungroup() %>%
  group_by(method, n, p, cor_type) %>%
  summarise(
    mean_rank = mean(mse_rank),
    times_best = sum(mse_rank == 1),
    times_worst = sum(mse_rank == 5),
    .groups = "drop"
  ) %>%
  arrange(n, p, cor_type, mean_rank)

write.csv(ranking_summary, file.path(output_dir, "table2_ranking_summary.csv"), row.names = FALSE)

################################################################################
# 10. FINAL SUMMARY REPORT
################################################################################

sink(file.path(output_dir, "simulation_summary_report.txt"))

cat("=====================================\n")
cat("GLM SHRINKAGE SIMULATION STUDY\n")
cat("=====================================\n\n")



cat("SIMULATION PARAMETERS:\n")
cat("- Sample sizes (n):", paste(unique(SIM_GRID$n), collapse = ", "), "\n")
cat("- Dimensions (p):", paste(unique(SIM_GRID$p), collapse = ", "), "\n")
cat("- Replications per scenario:", N_REPS, "\n\n")

cat("PERFORMANCE ORDER (by mean MSE):\n")
overall_ranking <- summary_stats %>%
  group_by(method) %>%
  summarise(mean_MSE = mean(MSE_mean), .groups = "drop") %>%
  arrange(mean_MSE)

for (i in 1:nrow(overall_ranking)) {
  cat(sprintf("%d. %s: MSE = %.3f\n", i, 
              overall_ranking$method[i], 
              overall_ranking$mean_MSE[i]))
}

cat("\nEXPECTED ORDER ACHIEVED: ")
if (overall_ranking$method[1] == "PS" || overall_ranking$method[1] == "S") {
  cat("YES - Shrinkage methods perform best\n")
} else {
  cat("PARTIAL - Check individual scenarios\n")
}

cat("\nFILES GENERATED:\n")
cat("- PNG/JPEG figures with varied plot types\n")
cat("- CSV tables with comprehensive results\n")
cat("- Complete results saved in:", output_dir, "\n")

sink()

cat("\n=====================================\n")
cat("SIMULATION COMPLETED SUCCESSFULLY!\n")
cat("Results saved in:", output_dir, "\n")
cat("=====================================\n")

# Clean up parallel processing
if (USE_PARALLEL && is_windows && exists("cl")) {
  stopCluster(cl)
}

# Final cleanup
gc()
