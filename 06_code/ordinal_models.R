load_ordinal_packages <- function() {
  suppressPackageStartupMessages({
    library(readr)
    library(dplyr)
    library(ordinal)
    library(emmeans)
    library(tibble)
  })
}

read_ratings <- function(path) {
  ratings <- read_csv(path, show_col_types = FALSE)
  dimensions <- paste0("D", 1:8)
  metadata <- c("rater_id", "scoring_id", "output_id", "case_id",
                "participant_id", "arm", "model_key")
  required <- c(metadata, dimensions)
  if (!all(required %in% names(ratings))) {
    stop("Ratings CSV is missing: ", paste(setdiff(required, names(ratings)), collapse = ", "))
  }
  if (nrow(ratings) != 4800L || anyNA(ratings[required]) ||
      anyDuplicated(ratings[c("scoring_id", "rater_id")])) {
    stop("Expected 4,800 complete, unique output-by-rater records.")
  }
  if (!setequal(unique(ratings$rater_id), paste0("rater_0", 1:3)) ||
      length(unique(ratings$scoring_id)) != 1600L ||
      any(table(ratings$scoring_id) != 3L)) {
    stop("Expected 1,600 outputs scored by rater_01, rater_02 and rater_03.")
  }
  stable <- unique(ratings[setdiff(metadata, "rater_id")])
  if (nrow(stable) != 1600L || anyDuplicated(stable$output_id) ||
      length(unique(stable$case_id)) != 100L ||
      length(unique(stable$participant_id)) != 16L) {
    stop("Output metadata must agree across raters: 1,600 outputs, 100 cases, 16 participants.")
  }
  model_levels <- c("siliconflow_primary", "openrouter_gpt_oss_20b_free",
                    "siliconflow_deepseek_v4_flash_paid", "siliconflow_minimax_m2_5_paid")
  if (!setequal(unique(ratings$arm), c("A", "B", "C", "D")) ||
      !setequal(unique(ratings$model_key), model_levels)) {
    stop("Ratings must contain the four study endpoints and prompt arms.")
  }
  if (!all(vapply(ratings[dimensions], is.numeric, logical(1)))) {
    stop("D1-D8 must be numeric scores.")
  }
  score_matrix <- as.matrix(ratings[dimensions])
  if (any(!is.finite(score_matrix)) || any(score_matrix < 1 | score_matrix > 5) ||
      any(score_matrix != floor(score_matrix))) stop("D1-D8 must be integers from 1 to 5.")
  dat <- ratings[rep(seq_len(nrow(ratings)), each = 8L), metadata]
  dat$dimension <- rep(dimensions, times = nrow(ratings))
  dat$score <- as.vector(t(score_matrix))
  dat
}

# Portable fitting helpers for the primary interaction CLMMs.
# Definitions only: sourcing this file does not fit a model or load archives.
# Reconstructs the numerical path used in the saved Stage 2/3 analysis from
# the supplied ratings. Numeric estimates are never copied from the paper.
# The temporary ordinal namespace patch is process-local; run dimensions
# serially in a process, or use separate R processes for parallel fitting.

lrrif_primary_formula <- function() {
  score_ord ~ model_key * arm + rater_id +
    (1 | case_id) + (1 | output_id)
}

lrrif_additive_formula <- function() {
  score_ord ~ model_key + arm + rater_id +
    (1 | case_id) + (1 | output_id)
}

lrrif_capture_conditions <- function(expr) {
  warnings <- messages <- character()
  value <- tryCatch(withCallingHandlers(expr,
    warning = function(w) {
      warnings <<- c(warnings, conditionMessage(w))
      invokeRestart("muffleWarning")
    }, message = function(m) {
      messages <<- c(messages, conditionMessage(m))
      invokeRestart("muffleMessage")
    }), error = function(e) e)
  list(value = if (inherits(value, "error")) NULL else value,
       warnings = warnings, messages = messages,
       error = if (inherits(value, "error")) conditionMessage(value) else "")
}

lrrif_with_stable_logit_curvature <- function(expr) {
  ns <- asNamespace("ordinal")
  original <- get("hess.u", ns, inherits = FALSE)
  original_body <- body(original)
  expected <- paste0("phi2 <- rho$wts * (((rho$p1 - rho$p2)/rho$fitted)^2 - ",
                     "((g1 - g2)/rho$fitted))")
  if (length(original_body) < 4L ||
      !identical(deparse(original_body[[4L]], width.cutoff = 500L), expected)) {
    stop("Unsupported ordinal::hess.u source; the audited D2-only curvature patch was not applied.")
  }
  patched <- original
  patched_body <- original_body
  patched_body[[4L]] <- quote({
    # Exact logit conditional-curvature identity; no clipping, probability,
    # likelihood, gradient, or subsequent Matrix-factorization change.
    phi2 <- rho$wts * (stats::dlogis(rho$eta1) + stats::dlogis(rho$eta2))
    stopifnot(all(is.finite(phi2)), all(phi2 >= 0))
  })
  body(patched) <- patched_body
  stopifnot(identical(as.list(original_body)[-4L],
                      as.list(body(patched))[-4L]))
  was_locked <- bindingIsLocked("hess.u", ns)
  # Register restoration before changing the binding, including error exits.
  on.exit({
    if (bindingIsLocked("hess.u", ns)) unlockBinding("hess.u", ns)
    assign("hess.u", original, envir = ns)
    if (was_locked) lockBinding("hess.u", ns)
  }, add = TRUE)
  if (was_locked) unlockBinding("hess.u", ns)
  assign("hess.u", patched, envir = ns)
  if (was_locked) lockBinding("hess.u", ns)
  force(expr)
}

lrrif_primary_control <- function(dimension, enlarged_outer_budget = FALSE) {
  if (identical(dimension, "D2")) {
    ordinal::clmm.control(maxIter = 500, gradTol = 1e-6,
      iter.max = 1000, eval.max = 2000, step.min = 0.001, step.max = 0.1)
  } else if (enlarged_outer_budget) {
    ordinal::clmm.control(maxIter = 200, gradTol = 1e-4,
      iter.max = 1000L, eval.max = 2000L)
  } else {
    ordinal::clmm.control(maxIter = 200, gradTol = 1e-4)
  }
}

lrrif_clmm_eligible <- function(fit) {
  if (is.null(fit) || !inherits(fit, "clmm") || is.null(fit$Hessian) ||
      !identical(as.integer(fit$optRes$convergence), 0L) ||
      any(!is.finite(fit$coefficients)) || any(!is.finite(fit$Hessian))) {
    return(FALSE)
  }
  eig <- eigen(fit$Hessian, symmetric = TRUE, only.values = TRUE)$values
  all(is.finite(eig)) && all(eig > 0)
}

lrrif_clmm_start_from_fit <- function(fit, d, interaction = TRUE) {
  design <- if (interaction) ~ model_key * arm + rater_id else
    ~ model_key + arm + rater_id
  beta_names <- colnames(stats::model.matrix(design, data = d))[-1L]
  if (!all(names(fit$beta) %in% beta_names)) {
    stop("Initialization coefficients do not align with the target fixed-effect design.")
  }
  beta_start <- stats::setNames(rep(0, length(beta_names)), beta_names)
  beta_start[names(fit$beta)] <- fit$beta
  # ST2par encodes these random-intercept diagonal entries as SDs, not log SDs.
  # Preserve the actual internal group order and verify it before optimization.
  list(c(fit$alpha, beta_start), unname(ordinal:::ST2par(fit$ST)))
}

lrrif_check_nested_start <- function(formula, d, control, start, reference_fit,
                                     tolerance = 1e-6) {
  rho <- ordinal::clmm(formula, data = d, link = "logit", threshold = "flexible",
    Hess = TRUE, nAGQ = 1, control = control, start = start, doFit = FALSE)
  if (!identical(names(rho$dims$nlev.re), names(reference_fit$ST))) {
    stop("Random-group ordering changed between the initial and target models.")
  }
  par <- ordinal:::getPar.clmm(rho)
  if (!isTRUE(all.equal(unname(par), unname(c(start[[1L]], start[[2L]]))))) {
    stop("Initial parameter encoding differs from the supplied fixed/SD start.")
  }
  value <- ordinal:::getNLA(rho, par)
  reference <- reference_fit$optRes$objective
  if (!is.finite(value) || !is.finite(reference) || abs(value - reference) >= tolerance) {
    stop("Initial target objective differs from its nested/reference objective.")
  }
  invisible(value)
}

lrrif_fit_primary_dimension <- function(d, dimension) {
  if (!requireNamespace("ordinal", quietly = TRUE)) stop("Package ordinal is required.")
  if (!dimension %in% paste0("D", 1:8)) stop("Unknown score dimension.")
  needed <- c("score_ord", "model_key", "arm", "rater_id", "case_id", "output_id")
  if (!all(needed %in% names(d)) || anyNA(d[needed])) stop("Incomplete model data.")
  # Match the original treatment coding independently of caller options.
  old_options <- options(contrasts = c("contr.treatment", "contr.poly"))
  on.exit(options(old_options), add = TRUE)
  d <- as.data.frame(d)
  d$score_ord <- ordered(as.character(d$score_ord), levels = as.character(1:5))
  d$model_key <- factor(d$model_key, levels = c("siliconflow_primary",
    "openrouter_gpt_oss_20b_free", "siliconflow_deepseek_v4_flash_paid",
    "siliconflow_minimax_m2_5_paid"))
  d$arm <- factor(d$arm, levels = c("A", "B", "C", "D"))
  d$rater_id <- factor(d$rater_id, levels = paste0("rater_0", 1:3))
  d$case_id <- factor(d$case_id)
  d$output_id <- factor(d$output_id)
  if (anyNA(d[needed]) || any(table(d$model_key) == 0L) ||
      any(table(d$arm) == 0L) || any(table(d$rater_id) == 0L)) {
    stop("Unexpected scores, endpoint/arm/rater levels, or missing factor levels.")
  }
  attempts <- list()
  run_fit <- function(stage, formula, control, start = NULL, stable = FALSE) {
    started <- proc.time()[["elapsed"]]
    call_fit <- function() {
      if (is.null(start)) {
        ordinal::clmm(formula, data = d, link = "logit", threshold = "flexible",
          Hess = TRUE, nAGQ = 1, control = control)
      } else {
        ordinal::clmm(formula, data = d, link = "logit", threshold = "flexible",
          Hess = TRUE, nAGQ = 1, control = control, start = start)
      }
    }
    result <- lrrif_capture_conditions(if (stable)
      lrrif_with_stable_logit_curvature(call_fit()) else call_fit())
    attempts[[length(attempts) + 1L]] <<- list(stage = stage, control = control,
      start = start, stable_logit_curvature = stable,
      elapsed_seconds = proc.time()[["elapsed"]] - started,
      warnings = result$warnings, messages = result$messages, error = result$error,
      optimizer_code = if (is.null(result$value)) NA_integer_ else
        result$value$optRes$convergence)
    result$value
  }
  additive_formula <- lrrif_additive_formula()
  main_formula <- lrrif_primary_formula()
  control <- lrrif_primary_control(dimension)
  if (identical(dimension, "D2")) {
    # Original Stage 1 fit supplies D2 fixed effects and case SD. This is an
    # initialization fit only; it is not substituted for the primary model.
    case_fit <- run_fit("case_only_D2_initialization",
      score_ord ~ model_key + arm + rater_id + (1 | case_id),
      ordinal::clmm.control(maxIter = 200, gradTol = 1e-4))
    if (!lrrif_clmm_eligible(case_fit)) {
      return(list(fit = NULL, additive_fit = NULL, attempts = attempts,
        model_name = "fit_failed", message = "D2 case-only initialization fit failed diagnostics."))
    }
    # The original study has 1,600 output groups versus 100 case groups;
    # ordinal orders these random-intercept SD parameters output, then case.
    # Build the order by group cardinality and verify it against doFit=FALSE.
    group_order <- names(sort(c(output_id = nlevels(d$output_id),
      case_id = nlevels(d$case_id)), decreasing = TRUE))
    random_start <- c(output_id = 0.5,
      case_id = as.numeric(case_fit$ST$case_id[1L, 1L]))
    additive_start <- list(c(case_fit$alpha, case_fit$beta),
      unname(random_start[group_order]))
    initial_rho <- ordinal::clmm(additive_formula, data = d, link = "logit",
      threshold = "flexible", Hess = TRUE, nAGQ = 1, control = control,
      start = additive_start, doFit = FALSE)
    if (!identical(names(initial_rho$dims$nlev.re), group_order)) {
      stop("D2 initialization random SD order differs from ordinal internal ordering.")
    }
    additive_fit <- run_fit("additive_D2_stable_curvature", additive_formula,
      control, start = additive_start, stable = TRUE)
  } else {
    additive_fit <- run_fit("additive_initialization", additive_formula, control)
  }
  if (!lrrif_clmm_eligible(additive_fit)) {
    return(list(fit = NULL, additive_fit = additive_fit, attempts = attempts,
      model_name = "fit_failed", message = "Additive initialization fit failed diagnostics."))
  }
  main_start <- lrrif_clmm_start_from_fit(additive_fit, d, interaction = TRUE)
  check <- lrrif_capture_conditions(if (identical(dimension, "D2"))
    lrrif_with_stable_logit_curvature(lrrif_check_nested_start(main_formula,
      d, control, main_start, additive_fit)) else
    lrrif_check_nested_start(main_formula, d, control, main_start, additive_fit))
  if (nzchar(check$error)) {
    return(list(fit = NULL, additive_fit = additive_fit, attempts = attempts,
      model_name = "fit_failed", message = check$error))
  }
  fit <- run_fit("interaction_from_additive_zero_interactions", main_formula,
    control, start = main_start, stable = identical(dimension, "D2"))
  if (dimension %in% c("D5", "D6") && !is.null(fit) &&
      identical(as.integer(fit$optRes$convergence), 1L) &&
      identical(fit$optRes$message, "iteration limit reached without convergence (10)")) {
    retry_control <- lrrif_primary_control(dimension, enlarged_outer_budget = TRUE)
    retry_start <- lrrif_clmm_start_from_fit(fit, d, interaction = TRUE)
    # Retry from the returned full-model numerical point, not from zero
    # interactions, and do not treat that unconverged point as an estimate.
    retry_check <- lrrif_capture_conditions(lrrif_check_nested_start(main_formula,
      d, retry_control, retry_start, fit))
    if (nzchar(retry_check$error)) {
      return(list(fit = NULL, additive_fit = additive_fit, attempts = attempts,
        model_name = "fit_failed", message = retry_check$error))
    }
    fit <- run_fit("interaction_D5_D6_outer_budget_retry", main_formula,
      retry_control, start = retry_start)
  }
  eligible <- lrrif_clmm_eligible(fit)
  if (eligible) {
    stopifnot(length(fit$beta) == 17L,
      sum(grepl(":", names(fit$beta), fixed = TRUE)) == 9L,
      setequal(names(fit$ST), c("case_id", "output_id")))
  }
  list(fit = if (eligible) fit else NULL, additive_fit = additive_fit,
    returned_fit = fit, attempts = attempts,
    model_name = if (eligible) "full_interaction_fixed_rater_case_output" else "fit_failed",
    message = if (eligible) "" else "Primary interaction fit failed diagnostics; inspect numerical attempts.")
}

# Primary CLMM exports, ported from the archived 2026-10-13 postprocessing.
# These functions only summarize an already fitted model; they do not fit/update it.

primary_fixed_effects <- function(fit, dimension, fitted_model) {
  coef_table <- as.data.frame(coef(summary(fit)))
  coef_table$term <- rownames(coef_table)
  rownames(coef_table) <- NULL
  names(coef_table) <- make.names(names(coef_table))
  # fit$beta contains every location coefficient, including rater and interaction.
  fixed <- coef_table[coef_table$term %in% names(fit$beta), , drop = FALSE]
  stopifnot(nrow(fixed) == length(fit$beta))
  tibble::tibble(
    dimension = dimension,
    fitted_model = fitted_model,
    term = fixed$term,
    estimate_log_odds = fixed$Estimate,
    std_error = fixed$Std..Error,
    z_value = fixed$z.value,
    p_value = fixed$Pr...z..,
    odds_ratio = exp(fixed$Estimate),
    or_95_low = exp(fixed$Estimate - qnorm(0.975) * fixed$Std..Error),
    or_95_high = exp(fixed$Estimate + qnorm(0.975) * fixed$Std..Error),
    interpretation_direction = ifelse(fixed$Estimate >= 0,
                                     "higher_score_odds", "lower_score_odds")
  )
}

primary_marginal_emmeans <- function(reference_grid, dimension, fitted_model,
                                   by_var, model_labels) {
  stopifnot(by_var %in% c("model_key", "arm"))
  emm <- emmeans::emmeans(reference_grid, specs = by_var, weights = "equal")
  tab <- as.data.frame(summary(emm, infer = c(TRUE, FALSE),
                               level = 0.95, adjust = "none"))
  stopifnot(nrow(tab) == 4L)
  tab$dimension <- dimension
  tab$fitted_model <- fitted_model
  tab$factor_type <- by_var
  tab$ci_level <- 0.95
  tab$ci_method <- "pointwise_asymptotic_Wald"
  tab$ci_adjustment <- "none"
  tab$scale <- "latent_logit"
  tab$weights <- "equal"
  tab$rater_handling <- "equal_average_over_three_observed_rater_fixed_effects"
  tab$case_handling <- "case_random_effect_zero_on_latent_scale"
  tab$output_handling <- "output_random_effect_zero_on_latent_scale"
  if (by_var == "model_key") {
    tab$model_label <- unname(model_labels[as.character(tab$model_key)])
  }
  tab <- tab[c("dimension", "fitted_model", "factor_type",
               setdiff(names(tab), c("dimension", "fitted_model", "factor_type")))]
  list(emm = emm, table = tibble::as_tibble(tab))
}

primary_marginal_pairwise <- function(marginal_emm, dimension, fitted_model,
                                     by_var) {
  tab <- as.data.frame(summary(pairs(marginal_emm, adjust = "none"),
                               infer = c(TRUE, TRUE), level = 0.95,
                               adjust = "none"))
  stopifnot(nrow(tab) == 6L)
  tab$p_raw <- tab$p.value
  tab$p_holm <- p.adjust(tab$p_raw, method = "holm", n = 6L)
  # Preserve the prior file contract: downstream code treats p.value as Holm p.
  tab$p.value <- tab$p_holm
  tab$dimension <- dimension
  tab$fitted_model <- fitted_model
  tab$factor_type <- by_var
  tab$family_id <- paste(dimension, by_var, sep = ":")
  tab$family_size <- 6L
  tab$ci_level <- 0.95
  tab$ci_method <- "pointwise_asymptotic_Wald"
  tab$ci_adjustment <- "none"
  tab$p_adjustment <- "Holm_within_dimension_and_factor_six_comparisons"
  tab$scale <- "latent_logit"
  tab$weights <- "equal"
  tab$rater_handling <- "equal_average_over_three_observed_rater_fixed_effects"
  tab$case_handling <- "case_random_effect_zero_on_latent_scale"
  tab$output_handling <- "output_random_effect_zero_on_latent_scale"
  tab <- tab[c("dimension", "fitted_model", "factor_type",
               setdiff(names(tab), c("dimension", "fitted_model", "factor_type")))]
  tibble::as_tibble(tab)
}

primary_cell_emmeans <- function(reference_grid, dimension, fitted_model,
                                model_labels) {
  emm <- emmeans::emmeans(reference_grid, specs = ~ model_key * arm,
                          weights = "equal")
  tab <- as.data.frame(summary(emm, infer = c(TRUE, FALSE),
                               level = 0.95, adjust = "none"))
  stopifnot(nrow(tab) == 16L)
  tab$dimension <- dimension
  tab$fitted_model <- fitted_model
  tab$factor_type <- "model_key:arm"
  tab$model_label <- unname(model_labels[as.character(tab$model_key)])
  tab$ci_lower <- tab$asymp.LCL
  tab$ci_upper <- tab$asymp.UCL
  tab$ci_level <- 0.95
  tab$ci_method <- "pointwise_asymptotic_Wald"
  tab$ci_adjustment <- "none"
  tab$scale <- "latent_logit"
  tab$weights <- "equal"
  tab$rater_handling <- "equal_average_over_three_observed_rater_fixed_effects"
  tab$case_handling <- "case_random_effect_zero_on_latent_scale"
  tab$output_handling <- "output_random_effect_zero_on_latent_scale"
  tab <- tab[c("dimension", "fitted_model", "factor_type",
               setdiff(names(tab), c("dimension", "fitted_model", "factor_type")))]
  tibble::as_tibble(tab)
}

primary_arm_within_model_pairwise <- function(reference_grid, dimension,
                                             fitted_model, model_labels) {
  emm <- emmeans::emmeans(reference_grid, specs = ~ arm | model_key,
                          weights = "equal")
  tab <- as.data.frame(summary(pairs(emm, adjust = "none"),
                               infer = c(TRUE, TRUE), level = 0.95,
                               adjust = "none"))
  stopifnot(nrow(tab) == 24L)
  tab$p_raw <- tab$p.value
  tab$p_holm <- NA_real_
  for (model in unique(as.character(tab$model_key))) {
    idx <- which(as.character(tab$model_key) == model)
    stopifnot(length(idx) == 6L)
    tab$p_holm[idx] <- p.adjust(tab$p_raw[idx], method = "holm", n = 6L)
  }
  tab$p.value <- tab$p_holm
  tab$dimension <- dimension
  tab$fitted_model <- fitted_model
  tab$factor_type <- "arm_within_model_key"
  tab$conditioning_model <- as.character(tab$model_key)
  tab$model_label <- unname(model_labels[tab$conditioning_model])
  tab$ci_lower <- tab$asymp.LCL
  tab$ci_upper <- tab$asymp.UCL
  tab$family_id <- paste(dimension, tab$conditioning_model, "arm", sep = ":")
  tab$family_size <- 6L
  tab$ci_level <- 0.95
  tab$ci_method <- "pointwise_asymptotic_Wald"
  tab$ci_adjustment <- "none"
  tab$p_adjustment <- "Holm_within_dimension_and_endpoint_six_arm_comparisons"
  tab$scale <- "latent_logit"
  tab$weights <- "equal"
  tab$rater_handling <- "equal_average_over_three_observed_rater_fixed_effects"
  tab$case_handling <- "case_random_effect_zero_on_latent_scale"
  tab$output_handling <- "output_random_effect_zero_on_latent_scale"
  tab <- tab[c("dimension", "fitted_model", "factor_type",
               setdiff(names(tab), c("dimension", "fitted_model", "factor_type")))]
  tibble::as_tibble(tab)
}

extract_primary_results <- function(fit, dimension, fitted_model, model_labels) {
  reference_grid <- emmeans::ref_grid(fit, mode = "latent")
  stopifnot(nrow(reference_grid@grid) == 48L,
            all(is.finite(reference_grid@bhat)),
            all(is.finite(reference_grid@V)))
  model_emm <- primary_marginal_emmeans(reference_grid, dimension, fitted_model,
                                       "model_key", model_labels)
  arm_emm <- primary_marginal_emmeans(reference_grid, dimension, fitted_model,
                                     "arm", model_labels)
  list(
    fixed_effects = primary_fixed_effects(fit, dimension, fitted_model),
    model_emmeans = model_emm$table,
    arm_emmeans = arm_emm$table,
    model_pairwise = primary_marginal_pairwise(model_emm$emm, dimension,
                                              fitted_model, "model_key"),
    arm_pairwise = primary_marginal_pairwise(arm_emm$emm, dimension,
                                            fitted_model, "arm"),
    model_arm_emmeans = primary_cell_emmeans(reference_grid, dimension,
                                            fitted_model, model_labels),
    arm_within_model_pairwise = primary_arm_within_model_pairwise(
      reference_grid, dimension, fitted_model, model_labels)
  )
}

validate_primary_fit <- function(fit, d) {
  formula_text <- function(x) gsub("[[:space:]]", "", paste(deparse(x), collapse = ""))
  if (!lrrif_clmm_eligible(fit) ||
      !identical(formula_text(fit$formula), formula_text(lrrif_primary_formula())) ||
      !identical(fit$link, "logit") || !identical(fit$threshold, "flexible") ||
      fit$nAGQ != 1L || nrow(fit$model) != 4800L ||
      !setequal(names(fit$ST), c("case_id", "output_id")) ||
      length(fit$beta) != 17L ||
      sum(grepl(":", names(fit$beta), fixed = TRUE)) != 9L ||
      sum(grepl("^rater_id", names(fit$beta))) != 2L) {
    stop("Primary model formula, numerical diagnostics or fixed/random structure does not match the protocol.")
  }
  for (column in c("score_ord", "model_key", "arm", "rater_id", "case_id", "output_id")) {
    if (!identical(as.character(fit$model[[column]]), as.character(d[[column]]))) {
      stop("Primary model/input rows differ in ", column, ".")
    }
  }
  for (column in c("model_key", "arm", "rater_id")) {
    if (!identical(levels(fit$model[[column]]), levels(d[[column]]))) {
      stop("Primary model/input factor references differ in ", column, ".")
    }
  }
  invisible(fit)
}

primary_fit_status <- function(fit, dimension, input_mode, message = "") {
  hessian_eigenvalues <- eigen(fit$Hessian, symmetric = TRUE, only.values = TRUE)$values
  tibble::tibble(
    dimension = dimension, status = "fit_ok",
    fitted_model = "full_interaction_fixed_rater_case_output",
    n_observations = nrow(fit$model), logLik = as.numeric(logLik(fit)),
    AIC = AIC(fit), BIC = BIC(fit), message = message, input_mode = input_mode,
    model_formula = paste(deparse(fit$formula, width.cutoff = 500L), collapse = " "),
    n_cases = length(unique(fit$model$case_id)),
    n_outputs = length(unique(fit$model$output_id)),
    n_raters = length(unique(fit$model$rater_id)),
    optimizer_code = fit$optRes$convergence,
    optimizer_message = paste(fit$optRes$message, collapse = " | "),
    max_abs_gradient = max(abs(fit$gradient)),
    hessian_min_eigenvalue = min(hessian_eigenvalues),
    hessian_positive_definite = all(hessian_eigenvalues > 0),
    case_random_sd = as.numeric(fit$ST$case_id[1L, 1L]),
    output_random_sd = as.numeric(fit$ST$output_id[1L, 1L]),
    link = fit$link, threshold = fit$threshold, nAGQ = fit$nAGQ,
    inner_maxIter = fit$control$ctrl$maxIter,
    inner_gradTol = fit$control$ctrl$gradTol,
    outer_control = paste(capture.output(dput(fit$control$optCtrl)), collapse = " ")
  )
}

run_primary_models <- function(dat, output_dir, primary_model_dir = NULL) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  model_dir <- file.path(output_dir, "model_objects")
  dir.create(model_dir, showWarnings = FALSE)
  model_levels <- c("siliconflow_primary", "openrouter_gpt_oss_20b_free",
                    "siliconflow_deepseek_v4_flash_paid", "siliconflow_minimax_m2_5_paid")
  model_labels <- c(siliconflow_primary = "Qwen3-8B baseline",
                   openrouter_gpt_oss_20b_free = "GPT-OSS-20B",
                   siliconflow_deepseek_v4_flash_paid = "DeepSeek-V4-Flash",
                   siliconflow_minimax_m2_5_paid = "MiniMax-M2.5")
  old_options <- options(contrasts = c("contr.treatment", "contr.poly"))
  on.exit(options(old_options), add = TRUE)
  dat <- as.data.frame(dat)
  dat$score_ord <- ordered(dat$score, levels = 1:5)
  dat$model_key <- factor(dat$model_key, levels = model_levels)
  dat$arm <- factor(dat$arm, levels = c("A", "B", "C", "D"))
  dat$rater_id <- factor(dat$rater_id, levels = paste0("rater_0", 1:3))
  dat$case_id <- factor(dat$case_id)
  dat$output_id <- factor(dat$output_id)
  if (anyNA(dat[c("score_ord", "model_key", "arm", "rater_id", "case_id", "output_id")])) {
    stop("Incomplete primary model data after factor coding.")
  }
  filenames <- c(
    fixed_effects = "PRIMARY_CLMM_FIXED_EFFECTS.csv",
    model_emmeans = "PRIMARY_CLMM_MODEL_EMMEANS_LATENT.csv",
    arm_emmeans = "PRIMARY_CLMM_ARM_EMMEANS_LATENT.csv",
    model_pairwise = "PRIMARY_CLMM_MODEL_PAIRWISE_LATENT.csv",
    arm_pairwise = "PRIMARY_CLMM_ARM_PAIRWISE_LATENT.csv",
    model_arm_emmeans = "PRIMARY_CLMM_MODEL_ARM_EMMEANS_LATENT.csv",
    arm_within_model_pairwise = "PRIMARY_CLMM_ARM_WITHIN_MODEL_PAIRWISE_LATENT.csv"
  )
  tables <- stats::setNames(lapply(filenames, function(x) list()), names(filenames))
  fit_rows <- list()
  for (dimension_name in paste0("D", 1:8)) {
    d <- dat[dat$dimension == dimension_name, , drop = FALSE]
    cat("Primary CLMM: ", dimension_name, "\n", sep = "")
    flush.console()
    result <- tryCatch({
      input_mode <- if (is.null(primary_model_dir)) "fit_from_ratings" else "verified_saved_model"
      if (is.null(primary_model_dir)) {
        fitted <- lrrif_fit_primary_dimension(d, dimension_name)
        saveRDS(fitted$attempts, file.path(model_dir, paste0(dimension_name, "_numerical_attempts.rds")))
        if (!is.null(fitted$additive_fit)) {
          saveRDS(fitted$additive_fit, file.path(model_dir, paste0(dimension_name, "_additive_clmm_fit.rds")))
        }
        if (!is.null(fitted$returned_fit)) {
          saveRDS(fitted$returned_fit, file.path(model_dir, paste0(dimension_name, "_returned_interaction_fit.rds")))
        }
        if (is.null(fitted$fit)) stop(fitted$message)
        fit <- fitted$fit
        fit_message <- paste(unique(unlist(lapply(fitted$attempts, `[[`, "warnings"))), collapse = " | ")
      } else {
        # Reuse is explicit. A saved object is accepted only for identical
        # rating rows, factor references, model formula and valid diagnostics.
        source_path <- file.path(primary_model_dir, paste0(dimension_name, "_interaction_clmm_fit.rds"))
        fit <- readRDS(source_path)
        fit_message <- "Saved model verified against every supplied rating row; no refit."
      }
      validate_primary_fit(fit, d)
      saveRDS(fit, file.path(model_dir, paste0(dimension_name, "_interaction_clmm_fit.rds")))
      derived <- extract_primary_results(fit, dimension_name,
        "full_interaction_fixed_rater_case_output", model_labels)
      list(status = primary_fit_status(fit, dimension_name, input_mode, fit_message), tables = derived)
    }, error = function(e) e)
    if (inherits(result, "error")) {
      fit_rows[[dimension_name]] <- tibble::tibble(
        dimension = dimension_name, status = "fit_or_postprocessing_failed",
        fitted_model = "full_interaction_fixed_rater_case_output",
        n_observations = nrow(d), logLik = NA_real_, AIC = NA_real_, BIC = NA_real_,
        message = conditionMessage(result)
      )
    } else {
      fit_rows[[dimension_name]] <- result$status
      for (name in names(filenames)) tables[[name]][[dimension_name]] <- result$tables[[name]]
    }
    # Persist progress and explicit failures without substituting another model.
    readr::write_csv(dplyr::bind_rows(fit_rows), file.path(output_dir, "PRIMARY_CLMM_FIT_STATUS.csv"))
  }
  for (name in names(filenames)) {
    readr::write_csv(dplyr::bind_rows(tables[[name]]), file.path(output_dir, filenames[[name]]))
  }
  capture.output(sessionInfo(), file = file.path(output_dir, "PRIMARY_CLMM_R_SESSION_INFO.txt"))
  invisible(dplyr::bind_rows(fit_rows))
}

# Fit only the additive member of the primary CLMM comparison. This helper
# follows the additive phase of lrrif_fit_primary_dimension, including D2's
# numerical initialization and temporary logit-curvature identity.
lrrif_fit_additive_dimension <- function(d, dimension) {
  if (!requireNamespace("ordinal", quietly = TRUE)) stop("Package ordinal is required.")
  if (!dimension %in% paste0("D", 1:8)) stop("Unknown score dimension.")
  needed <- c("score_ord", "model_key", "arm", "rater_id", "case_id", "output_id")
  if (!all(needed %in% names(d)) || anyNA(d[needed])) stop("Incomplete model data.")
  old_options <- options(contrasts = c("contr.treatment", "contr.poly"))
  on.exit(options(old_options), add = TRUE)
  d <- as.data.frame(d)
  d$score_ord <- ordered(as.character(d$score_ord), levels = as.character(1:5))
  d$model_key <- factor(d$model_key, levels = c("siliconflow_primary",
    "openrouter_gpt_oss_20b_free", "siliconflow_deepseek_v4_flash_paid",
    "siliconflow_minimax_m2_5_paid"))
  d$arm <- factor(d$arm, levels = c("A", "B", "C", "D"))
  d$rater_id <- factor(d$rater_id, levels = paste0("rater_0", 1:3))
  d$case_id <- factor(d$case_id)
  d$output_id <- factor(d$output_id)
  if (anyNA(d[needed]) || any(table(d$model_key) == 0L) ||
      any(table(d$arm) == 0L) || any(table(d$rater_id) == 0L)) {
    stop("Unexpected scores, endpoint/arm/rater levels, or missing factor levels.")
  }
  attempts <- list()
  run_fit <- function(stage, formula, control, start = NULL, stable = FALSE) {
    started <- proc.time()[["elapsed"]]
    call_fit <- function() {
      if (is.null(start)) {
        ordinal::clmm(formula, data = d, link = "logit", threshold = "flexible",
          Hess = TRUE, nAGQ = 1, control = control)
      } else {
        ordinal::clmm(formula, data = d, link = "logit", threshold = "flexible",
          Hess = TRUE, nAGQ = 1, control = control, start = start)
      }
    }
    result <- lrrif_capture_conditions(if (stable)
      lrrif_with_stable_logit_curvature(call_fit()) else call_fit())
    attempts[[length(attempts) + 1L]] <<- list(stage = stage, control = control,
      start = start, stable_logit_curvature = stable,
      elapsed_seconds = proc.time()[["elapsed"]] - started,
      warnings = result$warnings, messages = result$messages, error = result$error,
      optimizer_code = if (is.null(result$value)) NA_integer_ else
        result$value$optRes$convergence)
    result$value
  }
  additive_formula <- lrrif_additive_formula()
  control <- lrrif_primary_control(dimension)
  if (identical(dimension, "D2")) {
    case_fit <- run_fit("case_only_D2_initialization",
      score_ord ~ model_key + arm + rater_id + (1 | case_id),
      ordinal::clmm.control(maxIter = 200, gradTol = 1e-4))
    if (!lrrif_clmm_eligible(case_fit)) {
      return(list(fit = NULL, attempts = attempts,
        message = "D2 case-only initialization fit failed diagnostics."))
    }
    group_order <- names(sort(c(output_id = nlevels(d$output_id),
      case_id = nlevels(d$case_id)), decreasing = TRUE))
    random_start <- c(output_id = 0.5,
      case_id = as.numeric(case_fit$ST$case_id[1L, 1L]))
    additive_start <- list(c(case_fit$alpha, case_fit$beta),
      unname(random_start[group_order]))
    started <- proc.time()[["elapsed"]]
    initial <- lrrif_capture_conditions({
      initial_rho <- ordinal::clmm(additive_formula, data = d, link = "logit",
        threshold = "flexible", Hess = TRUE, nAGQ = 1, control = control,
        start = additive_start, doFit = FALSE)
      if (!identical(names(initial_rho$dims$nlev.re), group_order)) {
        stop("D2 initialization random SD order differs from ordinal internal ordering.")
      }
      TRUE
    })
    attempts[[length(attempts) + 1L]] <- list(stage = "D2_additive_start_order_check",
      control = control, start = additive_start, stable_logit_curvature = FALSE,
      elapsed_seconds = proc.time()[["elapsed"]] - started,
      warnings = initial$warnings, messages = initial$messages, error = initial$error,
      optimizer_code = NA_integer_)
    if (nzchar(initial$error)) {
      return(list(fit = NULL, attempts = attempts, message = initial$error))
    }
    fit <- run_fit("additive_D2_stable_curvature", additive_formula,
      control, start = additive_start, stable = TRUE)
  } else {
    fit <- run_fit("additive_initialization", additive_formula, control)
  }
  eligible <- lrrif_clmm_eligible(fit)
  if (eligible) {
    stopifnot(length(fit$beta) == 8L,
      !any(grepl(":", names(fit$beta), fixed = TRUE)),
      setequal(names(fit$ST), c("case_id", "output_id")))
  }
  list(fit = if (eligible) fit else NULL, attempts = attempts,
    message = if (eligible) "" else
      "Additive CLMM fit failed diagnostics; inspect numerical attempts.")
}

prepare_interaction_data <- function(dat) {
  dat <- as.data.frame(dat)
  dat$score_ord <- ordered(dat$score, levels = 1:5)
  dat$model_key <- factor(dat$model_key, levels = c("siliconflow_primary",
    "openrouter_gpt_oss_20b_free", "siliconflow_deepseek_v4_flash_paid",
    "siliconflow_minimax_m2_5_paid"))
  dat$arm <- factor(dat$arm, levels = c("A", "B", "C", "D"))
  dat$rater_id <- factor(dat$rater_id, levels = paste0("rater_0", 1:3))
  dat$case_id <- factor(dat$case_id)
  dat$output_id <- factor(dat$output_id)
  if (anyNA(dat[c("score_ord", "model_key", "arm", "rater_id", "case_id", "output_id")])) {
    stop("Incomplete interaction comparison data after factor coding.")
  }
  dat
}

validate_interaction_pair <- function(additive, interaction, d) {
  validate_primary_fit(interaction, d)
  formula_text <- function(x) gsub("[[:space:]]", "", paste(deparse(x), collapse = ""))
  expected_contrasts <- stats::setNames(rep(list("contr.treatment"), 3L),
                                       c("model_key", "arm", "rater_id"))
  specifications <- list(
    additive = list(fit = additive, formula = lrrif_additive_formula(),
                    design = ~ model_key + arm + rater_id, beta_count = 8L),
    interaction = list(fit = interaction, formula = lrrif_primary_formula(),
                       design = ~ model_key * arm + rater_id, beta_count = 17L)
  )
  for (name in names(specifications)) {
    specification <- specifications[[name]]
    fit <- specification$fit
    if (!lrrif_clmm_eligible(fit) ||
        !identical(formula_text(fit$formula), formula_text(specification$formula)) ||
        !identical(fit$link, "logit") || !identical(fit$threshold, "flexible") ||
        fit$nAGQ != 1L || nrow(fit$model) != 4800L ||
        !setequal(names(fit$ST), c("case_id", "output_id")) ||
        length(fit$beta) != specification$beta_count ||
        !identical(fit$contrasts, expected_contrasts)) {
      stop("Invalid ", name, " CLMM structure, coding or numerical diagnostics.")
    }
    expected_beta <- colnames(stats::model.matrix(specification$design, data = d))[-1L]
    if (!identical(names(fit$beta), expected_beta)) {
      stop("The ", name, " CLMM fixed-effect coding differs from the rating design.")
    }
    for (column in c("score_ord", "model_key", "arm", "rater_id", "case_id", "output_id")) {
      if (!identical(as.character(fit$model[[column]]), as.character(d[[column]]))) {
        stop("The ", name, " CLMM/input rows differ in ", column, ".")
      }
      # ordinal drops unused response categories; compare the fitted response
      # to observed input levels rather than requiring all five categories.
      expected_levels <- if (column == "score_ord") levels(droplevels(d[[column]])) else levels(d[[column]])
      if (!identical(levels(fit$model[[column]]), expected_levels)) {
        stop("The ", name, " CLMM/input levels differ in ", column, ".")
      }
    }
    for (group in c("case_id", "output_id")) {
      if (!identical(dim(fit$ST[[group]]), c(1L, 1L))) {
        stop("Comparison requires case and output random intercepts only.")
      }
    }
  }
  if (!identical(names(additive$alpha), names(interaction$alpha)) ||
      !identical(additive$y.levels, interaction$y.levels) ||
      !all(names(additive$beta) %in% names(interaction$beta)) ||
      sum(grepl(":", names(interaction$beta), fixed = TRUE)) != 9L) {
    stop("CLMM pair is not the required nested comparison with nine added interactions.")
  }
  invisible(TRUE)
}

clmm_interaction_lrt <- function(additive, interaction, d, dimension, input_mode, message = "") {
  validate_interaction_pair(additive, interaction, d)
  ll_add <- logLik(additive)
  ll_full <- logLik(interaction)
  df_diff <- attr(ll_full, "df") - attr(ll_add, "df")
  LR_stat <- 2 * (as.numeric(ll_full) - as.numeric(ll_add))
  if (length(df_diff) != 1L || !is.finite(df_diff) || df_diff != 9L ||
      !is.finite(as.numeric(ll_add)) || !is.finite(as.numeric(ll_full)) ||
      !is.finite(LR_stat) || LR_stat < 0) {
    stop("Invalid CLMM likelihood ratio or parameter difference; no value was clamped.")
  }
  p_raw <- pchisq(LR_stat, df = df_diff, lower.tail = FALSE)
  tibble::tibble(
    dimension = dimension, status = "fit_ok",
    fitted_model = "fixed_rater_case_output_clmm_comparison",
    n_observations = nrow(d), additive_AIC = AIC(additive),
    interaction_AIC = AIC(interaction), LR_stat = LR_stat,
    df_diff = as.integer(df_diff), p_value = p_raw, p_raw = p_raw,
    p_holm = NA_real_, delta_AIC = AIC(interaction) - AIC(additive),
    additive_logLik = as.numeric(ll_add), interaction_logLik = as.numeric(ll_full),
    additive_df = attr(ll_add, "df"), interaction_df = attr(ll_full, "df"),
    n_cases = length(unique(d$case_id)), n_outputs = length(unique(d$output_id)),
    n_raters = length(unique(d$rater_id)), input_mode = input_mode,
    additive_optimizer_code = additive$optRes$convergence,
    interaction_optimizer_code = interaction$optRes$convergence,
    additive_formula = paste(deparse(additive$formula, width.cutoff = 500L), collapse = " "),
    interaction_formula = paste(deparse(interaction$formula, width.cutoff = 500L), collapse = " "),
    reference_distribution = "asymptotic_chisquare_9df",
    family_size = 8L, p_adjustment = "Holm_across_eight_dimension_interaction_tests",
    message = message
  )
}

run_interaction_screen <- function(dat, output_dir, primary_model_dir = NULL,
                                   additive_model_dir = NULL) {
  if (is.null(primary_model_dir) && !is.null(additive_model_dir)) {
    stop("An additive saved-model directory requires a paired primary saved-model directory.")
  }
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  model_dir <- file.path(output_dir, "model_objects")
  dir.create(model_dir, showWarnings = FALSE)
  old_options <- options(contrasts = c("contr.treatment", "contr.poly"))
  on.exit(options(old_options), add = TRUE)
  dat <- prepare_interaction_data(dat)
  model_labels <- c(siliconflow_primary = "Qwen3-8B baseline",
    openrouter_gpt_oss_20b_free = "GPT-OSS-20B",
    siliconflow_deepseek_v4_flash_paid = "DeepSeek-V4-Flash",
    siliconflow_minimax_m2_5_paid = "MiniMax-M2.5")
  status_rows <- fixed_rows <- emm_rows <- list()
  for (dimension in paste0("D", 1:8)) {
    d <- dat[dat$dimension == dimension, , drop = FALSE]
    cat("CLMM interaction comparison: ", dimension, "\n", sep = "")
    flush.console()
    result <- tryCatch({
      if (is.null(primary_model_dir)) {
        fitted <- lrrif_fit_primary_dimension(d, dimension)
        saveRDS(fitted$attempts, file.path(model_dir, paste0(dimension, "_numerical_attempts.rds")))
        if (is.null(fitted$fit) || is.null(fitted$additive_fit)) stop(fitted$message)
        interaction <- fitted$fit
        additive <- fitted$additive_fit
        input_mode <- "paired_fits_from_ratings"
        message <- paste(unique(unlist(lapply(fitted$attempts, `[[`, "warnings"))), collapse = " | ")
      } else {
        full_path <- file.path(primary_model_dir, paste0(dimension, "_interaction_clmm_fit.rds"))
        interaction <- readRDS(full_path)
        validate_primary_fit(interaction, d)
        additive_dir <- if (is.null(additive_model_dir)) primary_model_dir else additive_model_dir
        additive_path <- file.path(additive_dir, paste0(dimension, "_additive_clmm_fit.rds"))
        if (file.exists(additive_path)) {
          additive <- readRDS(additive_path)
          input_mode <- "verified_saved_clmm_pair"
          message <- "Both saved CLMMs verified against every supplied rating row; no refit."
        } else if (!is.null(additive_model_dir)) {
          stop("Explicit additive model is missing: ", additive_path)
        } else {
          # A primary-only cache does not contain the null model. Compute only
          # the missing additive member using the same initialization/controls.
          fitted <- lrrif_fit_additive_dimension(d, dimension)
          saveRDS(fitted$attempts, file.path(model_dir, paste0(dimension, "_additive_numerical_attempts.rds")))
          if (is.null(fitted$fit)) stop(fitted$message)
          additive <- fitted$fit
          input_mode <- "verified_primary_and_additive_fit_from_ratings"
          message <- paste(unique(unlist(lapply(fitted$attempts, `[[`, "warnings"))), collapse = " | ")
        }
      }
      status <- clmm_interaction_lrt(additive, interaction, d, dimension, input_mode, message)
      saveRDS(additive, file.path(model_dir, paste0(dimension, "_additive_clmm_fit.rds")))
      saveRDS(interaction, file.path(model_dir, paste0(dimension, "_interaction_clmm_fit.rds")))
      fixed <- primary_fixed_effects(interaction, dimension,
                                     "full_interaction_fixed_rater_case_output")
      reference_grid <- emmeans::ref_grid(interaction, mode = "latent")
      emm <- primary_cell_emmeans(reference_grid, dimension,
                                  "full_interaction_fixed_rater_case_output", model_labels)
      list(status = status, fixed = fixed, emmeans = emm)
    }, error = function(e) e)
    if (inherits(result, "error")) {
      status_rows[[dimension]] <- tibble::tibble(
        dimension = dimension, status = "comparison_or_postprocessing_failed",
        fitted_model = "fixed_rater_case_output_clmm_comparison",
        n_observations = nrow(d), additive_AIC = NA_real_, interaction_AIC = NA_real_,
        LR_stat = NA_real_, df_diff = NA_integer_, p_value = NA_real_,
        p_raw = NA_real_, p_holm = NA_real_, delta_AIC = NA_real_,
        family_size = 8L, message = conditionMessage(result)
      )
    } else {
      status_rows[[dimension]] <- result$status
      fixed_rows[[dimension]] <- result$fixed
      emm_rows[[dimension]] <- result$emmeans
    }
    readr::write_csv(dplyr::bind_rows(status_rows),
                    file.path(output_dir, "PRIMARY_ORDINAL_MODEL_ARM_INTERACTION_SCREEN.csv"))
  }
  status <- dplyr::bind_rows(status_rows)
  complete <- nrow(status) == 8L && all(status$status == "fit_ok") && all(is.finite(status$p_raw))
  status$family_complete <- complete
  if (complete) status$p_holm <- p.adjust(status$p_raw, method = "holm", n = 8L)
  # A failed dimension leaves the entire eight-test adjusted family unavailable.
  readr::write_csv(status, file.path(output_dir, "PRIMARY_ORDINAL_MODEL_ARM_INTERACTION_SCREEN.csv"))
  readr::write_csv(dplyr::bind_rows(fixed_rows),
                  file.path(output_dir, "PRIMARY_ORDINAL_MODEL_ARM_INTERACTION_FIXED_EFFECTS.csv"))
  readr::write_csv(dplyr::bind_rows(emm_rows),
                  file.path(output_dir, "PRIMARY_ORDINAL_MODEL_ARM_INTERACTION_EMMEANS.csv"))
  capture.output(sessionInfo(), file = file.path(output_dir,
                    "PRIMARY_ORDINAL_MODEL_ARM_INTERACTION_SCREEN_R_SESSION_INFO.txt"))
  invisible(status)
}

print_usage <- function() {
  cat("Usage: Rscript ordinal_models.R --ratings CSV --output-dir DIR [--mode primary|interaction|all] [--primary-model-dir DIR] [--additive-model-dir DIR]\n")
}

main <- function(args = commandArgs(trailingOnly = TRUE)) {
  if (any(args %in% c("--help", "-h"))) {
    print_usage()
    return(invisible(NULL))
  }
  options <- list(mode = "all")
  seen <- character()
  allowed <- c("--ratings", "--output-dir", "--mode", "--primary-model-dir", "--additive-model-dir")
  k <- 1L
  while (k <= length(args)) {
    flag <- args[[k]]
    if (!(flag %in% allowed) || flag %in% seen) stop("Unknown or repeated option: ", flag)
    if (k == length(args) || startsWith(args[[k + 1L]], "--")) stop("Missing value for ", flag)
    options[[substring(flag, 3L)]] <- args[[k + 1L]]
    seen <- c(seen, flag)
    k <- k + 2L
  }
  if (is.null(options$ratings) || is.null(options[["output-dir"]])) {
    stop("--ratings and --output-dir are required.")
  }
  if (!(options$mode %in% c("primary", "interaction", "all"))) stop("Invalid --mode.")
  ratings_path <- normalizePath(path.expand(options$ratings), mustWork = TRUE)
  output_dir <- path.expand(options[["output-dir"]])
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  output_dir <- normalizePath(output_dir, mustWork = TRUE)
  primary_model_dir <- if (is.null(options[["primary-model-dir"]])) NULL else
    normalizePath(path.expand(options[["primary-model-dir"]]), mustWork = TRUE)
  additive_model_dir <- if (is.null(options[["additive-model-dir"]])) NULL else
    normalizePath(path.expand(options[["additive-model-dir"]]), mustWork = TRUE)
  load_ordinal_packages()
  dat <- read_ratings(ratings_path)
  primary_status <- NULL
  interaction_status <- NULL
  if (options$mode %in% c("primary", "all")) {
    primary_status <- run_primary_models(dat, file.path(output_dir, "primary"), primary_model_dir)
  }
  if (options$mode %in% c("interaction", "all")) {
    if (!is.null(primary_status) && any(primary_status$status != "fit_ok")) {
      stop("Primary analysis failed; interaction comparisons were not run from incomplete model outputs.")
    }
    # Retain an explicitly supplied source cache, including any additive
    # members it contains. Fresh all-mode fits are cached by run_primary_models.
    paired_primary_dir <- if (options$mode == "all" && is.null(primary_model_dir))
      file.path(output_dir, "primary", "model_objects") else primary_model_dir
    interaction_status <- run_interaction_screen(dat, file.path(output_dir, "interaction"),
      primary_model_dir = paired_primary_dir, additive_model_dir = additive_model_dir)
  }
  failed <- (!is.null(primary_status) && any(primary_status$status != "fit_ok")) ||
            (!is.null(interaction_status) && any(interaction_status$status != "fit_ok"))
  if (failed) stop("One or more fits failed; inspect the fit-status CSVs. No alternative model was substituted.")
  invisible(NULL)
}

if (sys.nframe() == 0L) main()
