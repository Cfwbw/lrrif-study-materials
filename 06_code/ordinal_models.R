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

run_primary_models <- function(dat, output_dir) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  fit_status_out <- file.path(output_dir, "PRIMARY_CLMM_FIT_STATUS.csv")
  fixed_effects_out <- file.path(output_dir, "PRIMARY_CLMM_FIXED_EFFECTS.csv")
  model_emmeans_out <- file.path(output_dir, "PRIMARY_CLMM_MODEL_EMMEANS_LATENT.csv")
  arm_emmeans_out <- file.path(output_dir, "PRIMARY_CLMM_ARM_EMMEANS_LATENT.csv")
  model_pairwise_out <- file.path(output_dir, "PRIMARY_CLMM_MODEL_PAIRWISE_LATENT.csv")
  arm_pairwise_out <- file.path(output_dir, "PRIMARY_CLMM_ARM_PAIRWISE_LATENT.csv")
  session_info_out <- file.path(output_dir, "PRIMARY_CLMM_R_SESSION_INFO.txt")

  model_levels <- c(
    "siliconflow_primary",
    "openrouter_gpt_oss_20b_free",
    "siliconflow_deepseek_v4_flash_paid",
    "siliconflow_minimax_m2_5_paid"
  )
  model_labels <- c(
    siliconflow_primary = "Qwen3-8B baseline",
    openrouter_gpt_oss_20b_free = "GPT-OSS-20B",
    siliconflow_deepseek_v4_flash_paid = "DeepSeek-V4-Flash",
    siliconflow_minimax_m2_5_paid = "MiniMax-M2.5"
  )
  arm_levels <- c("A", "B", "C", "D")
  dimensions <- paste0("D", 1:8)

  safe_fit <- function(d) {
    fit <- tryCatch(
      clmm(
        score_ord ~ model_key + arm + (1 | case_id) + (1 | rater_id),
        data = d,
        link = "logit",
        Hess = TRUE,
        nAGQ = 1,
        control = clmm.control(maxIter = 200, gradTol = 1e-4)
      ),
      error = function(e) e
    )
    if (inherits(fit, "error")) {
      return(list(fit = NULL, model_name = "fit_failed", message = conditionMessage(fit)))
    }
    list(fit = fit, model_name = "full", message = "")
  }

  extract_fixed_effects <- function(fit, dimension, fitted_model) {
    coef_table <- as.data.frame(coef(summary(fit)))
    coef_table$term <- rownames(coef_table)
    rownames(coef_table) <- NULL
    names(coef_table) <- make.names(names(coef_table))

    # Keep location/fixed-effect terms only; exclude ordinal threshold terms.
    fixed <- coef_table %>%
      filter(grepl("^model_key|^arm", term)) %>%
      transmute(
        dimension = dimension,
        fitted_model = fitted_model,
        term = term,
        estimate_log_odds = Estimate,
        std_error = Std..Error,
        z_value = z.value,
        p_value = Pr...z..,
        odds_ratio = exp(Estimate),
        or_95_low = exp(Estimate - 1.96 * Std..Error),
        or_95_high = exp(Estimate + 1.96 * Std..Error),
        interpretation_direction = ifelse(Estimate >= 0, "higher_score_odds", "lower_score_odds")
      )

    fixed
  }

  extract_emmeans <- function(fit, dimension, fitted_model, by_var) {
    emm <- tryCatch(
      as.data.frame(emmeans(fit, specs = as.formula(paste0("~ ", by_var)), mode = "latent")),
      error = function(e) NULL
    )
    if (is.null(emm)) {
      return(tibble())
    }

    out <- as_tibble(emm) %>%
      mutate(
        dimension = dimension,
        fitted_model = fitted_model,
        factor_type = by_var
      ) %>%
      relocate(dimension, fitted_model, factor_type)

    if (by_var == "model_key") {
      out <- out %>% mutate(model_label = unname(model_labels[as.character(model_key)]))
    }
    out
  }

  extract_pairwise <- function(fit, dimension, fitted_model, by_var) {
    pw <- tryCatch(
      as.data.frame(pairs(emmeans(fit, specs = as.formula(paste0("~ ", by_var)), mode = "latent"), adjust = "holm")),
      error = function(e) NULL
    )
    if (is.null(pw)) {
      return(tibble())
    }

    as_tibble(pw) %>%
      mutate(
        dimension = dimension,
        fitted_model = fitted_model,
        factor_type = by_var
      ) %>%
      relocate(dimension, fitted_model, factor_type)
  }

  dat <- dat %>%
    mutate(
      score_ord = ordered(score, levels = 1:5),
      model_key = factor(model_key, levels = model_levels),
      arm = factor(arm, levels = arm_levels),
      case_id = factor(case_id),
      rater_id = factor(rater_id)
    )

  fit_rows <- list()
  fixed_rows <- list()
  model_emm_rows <- list()
  arm_emm_rows <- list()
  model_pairwise_rows <- list()
  arm_pairwise_rows <- list()

  for (dimension_name in dimensions) {
    d <- dat %>% filter(dimension == dimension_name)
    fit_result <- safe_fit(d)

    if (is.null(fit_result$fit)) {
      fit_rows[[dimension_name]] <- tibble(
        dimension = dimension_name,
        status = "fit_failed",
        fitted_model = fit_result$model_name,
        n_observations = nrow(d),
        logLik = NA_real_,
        AIC = NA_real_,
        BIC = NA_real_,
        message = fit_result$message
      )
      next
    }

    fit <- fit_result$fit
    fitted_model <- fit_result$model_name
    fit_rows[[dimension_name]] <- tibble(
      dimension = dimension_name,
      status = "fit_ok",
      fitted_model = fitted_model,
      n_observations = nrow(d),
      logLik = as.numeric(logLik(fit)),
      AIC = AIC(fit),
      BIC = BIC(fit),
      message = fit_result$message
    )

    fixed_rows[[dimension_name]] <- extract_fixed_effects(fit, dimension_name, fitted_model)
    model_emm_rows[[dimension_name]] <- extract_emmeans(fit, dimension_name, fitted_model, "model_key")
    arm_emm_rows[[dimension_name]] <- extract_emmeans(fit, dimension_name, fitted_model, "arm")
    model_pairwise_rows[[dimension_name]] <- extract_pairwise(fit, dimension_name, fitted_model, "model_key")
    arm_pairwise_rows[[dimension_name]] <- extract_pairwise(fit, dimension_name, fitted_model, "arm")
  }

  fit_status <- bind_rows(fit_rows)
  fixed_effects <- bind_rows(fixed_rows)
  model_emmeans <- bind_rows(model_emm_rows)
  arm_emmeans <- bind_rows(arm_emm_rows)
  model_pairwise <- bind_rows(model_pairwise_rows)
  arm_pairwise <- bind_rows(arm_pairwise_rows)

  write_csv(fit_status, fit_status_out)
  write_csv(fixed_effects, fixed_effects_out)
  write_csv(model_emmeans, model_emmeans_out)
  write_csv(arm_emmeans, arm_emmeans_out)
  write_csv(model_pairwise, model_pairwise_out)
  write_csv(arm_pairwise, arm_pairwise_out)

  capture.output(sessionInfo(), file = session_info_out)
  invisible(fit_status)
}

run_interaction_screen <- function(dat, output_dir) {
  out_dir <- output_dir
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  model_levels <- c(
    "siliconflow_primary",
    "openrouter_gpt_oss_20b_free",
    "siliconflow_deepseek_v4_flash_paid",
    "siliconflow_minimax_m2_5_paid"
  )
  arm_levels <- c("A", "B", "C", "D")
  dimensions <- paste0("D", 1:8)

  extract_fixed <- function(fit, dimension, fitted_model) {
    coef_table <- as.data.frame(coef(summary(fit)))
    coef_table$term <- rownames(coef_table)
    rownames(coef_table) <- NULL
    names(coef_table) <- make.names(names(coef_table))
    as_tibble(coef_table) %>%
      filter(grepl("^model_key|^arm", term)) %>%
      transmute(
        dimension = dimension,
        fitted_model = fitted_model,
        term = term,
        estimate_log_odds = Estimate,
        std_error = Std..Error,
        z_value = z.value,
        p_value = Pr...z..,
        odds_ratio = exp(Estimate)
      )
  }

  dat <- dat %>%
    mutate(
      score_ord = ordered(score, levels = 1:5),
      model_key = factor(model_key, levels = model_levels),
      arm = factor(arm, levels = arm_levels)
    )

  status_rows <- list()
  fixed_rows <- list()
  emm_rows <- list()

  for (dimension_name in dimensions) {
    d <- dat %>% filter(dimension == dimension_name)
    additive <- tryCatch(clm(score_ord ~ model_key + arm, data = d, link = "logit", Hess = TRUE), error = function(e) e)
    interaction <- tryCatch(clm(score_ord ~ model_key * arm, data = d, link = "logit", Hess = TRUE), error = function(e) e)
    if (!inherits(additive, "error") && !inherits(interaction, "error")) {
      lrt <- as.data.frame(anova(additive, interaction))
      p_lrt <- if ("Pr(>Chisq)" %in% names(lrt)) lrt[2, "Pr(>Chisq)"] else NA_real_
      lr_stat <- if ("LR.stat" %in% names(lrt)) lrt[2, "LR.stat"] else NA_real_
      df_diff <- if ("df" %in% names(lrt)) lrt[2, "df"] else NA_real_
      status_rows[[dimension_name]] <- tibble(
        dimension = dimension_name,
        status = "fit_ok",
        fitted_model = "fixed_effect_clm",
        n_observations = nrow(d),
        additive_AIC = AIC(additive),
        interaction_AIC = AIC(interaction),
        LR_stat = lr_stat,
        df_diff = df_diff,
        p_value = p_lrt,
        message = ""
      )
      fixed_rows[[dimension_name]] <- extract_fixed(interaction, dimension_name, "fixed_effect_clm_interaction")
      emm <- tryCatch(
        as.data.frame(emmeans(interaction, specs = ~ model_key | arm, mode = "latent")),
        error = function(e) NULL
      )
      if (!is.null(emm)) {
        emm_rows[[dimension_name]] <- as_tibble(emm) %>%
          mutate(dimension = dimension_name, fitted_model = "fixed_effect_clm_interaction") %>%
          relocate(dimension, fitted_model)
      }
    } else {
      status_rows[[dimension_name]] <- tibble(
        dimension = dimension_name,
        status = "fit_failed",
        fitted_model = "fixed_effect_clm",
        n_observations = nrow(d),
        additive_AIC = NA_real_,
        interaction_AIC = NA_real_,
        LR_stat = NA_real_,
        df_diff = NA_real_,
        p_value = NA_real_,
        message = paste(
          if (inherits(additive, "error")) additive$message else "",
          if (inherits(interaction, "error")) interaction$message else "",
          sep = " | "
        )
      )
    }
  }

  status <- bind_rows(status_rows) %>%
    mutate(p_holm = p.adjust(p_value, method = "holm"))
  fixed <- bind_rows(fixed_rows)
  emm <- bind_rows(emm_rows)

  write_csv(status, file.path(out_dir, "PRIMARY_ORDINAL_MODEL_ARM_INTERACTION_SCREEN.csv"))
  write_csv(fixed, file.path(out_dir, "PRIMARY_ORDINAL_MODEL_ARM_INTERACTION_FIXED_EFFECTS.csv"))
  write_csv(emm, file.path(out_dir, "PRIMARY_ORDINAL_MODEL_ARM_INTERACTION_EMMEANS.csv"))
  capture.output(sessionInfo(), file = file.path(out_dir, "PRIMARY_ORDINAL_MODEL_ARM_INTERACTION_SCREEN_R_SESSION_INFO.txt"))
  invisible(status)
}

print_usage <- function() {
  cat("Usage: Rscript ordinal_models.R --ratings CSV --output-dir DIR [--mode primary|interaction|all]\n")
}

main <- function(args = commandArgs(trailingOnly = TRUE)) {
  if (any(args %in% c("--help", "-h"))) {
    print_usage()
    return(invisible(NULL))
  }
  options <- list(mode = "all")
  seen <- character()
  allowed <- c("--ratings", "--output-dir", "--mode")
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
  load_ordinal_packages()
  dat <- read_ratings(ratings_path)
  primary_status <- NULL
  interaction_status <- NULL
  if (options$mode %in% c("primary", "all")) {
    primary_status <- run_primary_models(dat, file.path(output_dir, "primary"))
  }
  if (options$mode %in% c("interaction", "all")) {
    interaction_status <- run_interaction_screen(dat, file.path(output_dir, "interaction"))
  }
  failed <- (!is.null(primary_status) && any(primary_status$status != "fit_ok")) ||
            (!is.null(interaction_status) && any(interaction_status$status != "fit_ok"))
  if (failed) stop("One or more fits failed; inspect the fit-status CSVs. No alternative model was substituted.")
  invisible(NULL)
}

if (sys.nframe() == 0L) main()
