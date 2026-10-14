from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

DIMS = [f'D{i}' for i in range(1, 9)]
MODEL_LABELS = {
    'siliconflow_primary': 'Qwen3-8B baseline',
    'openrouter_gpt_oss_20b_free': 'GPT-OSS-20B',
    'siliconflow_deepseek_v4_flash_paid': 'DeepSeek-V4-Flash',
    'siliconflow_minimax_m2_5_paid': 'MiniMax-M2.5',
}


def result_directory(output_dir, section):
    target = Path(output_dir) / section
    target.mkdir(parents=True, exist_ok=True)
    return target


def read_scores(ratings_path, audit_path=None):
    ratings = pd.read_csv(ratings_path)
    keys = ['scoring_id', 'output_id', 'case_id', 'participant_id', 'arm', 'model_key']
    if ratings[keys + ['rater_id'] + DIMS].isna().any().any():
        raise ValueError('Missing rating scores or identifiers.')
    if len(ratings) != 4800 or ratings.duplicated(['scoring_id', 'rater_id']).any():
        raise ValueError('Expected 4800 distinct output-by-rater rows.')
    raters = sorted(ratings.rater_id.unique())
    if len(raters) != 3 or not ratings[DIMS].isin([1, 2, 3, 4, 5]).all().all():
        raise ValueError('Expected three raters and integer scores from 1 to 5.')
    for key in keys[1:]:
        if not ratings.groupby('scoring_id')[key].nunique().eq(1).all():
            raise ValueError(f'Inconsistent {key} within an output.')
    mapping = ratings[keys].drop_duplicates().sort_values('scoring_id').reset_index(drop=True)
    if (len(mapping), mapping.output_id.nunique(), mapping.case_id.nunique(), mapping.participant_id.nunique()) != (1600, 1600, 100, 16):
        raise ValueError('Expected 1600 outputs, 100 cases and 16 participants.')
    design = pd.MultiIndex.from_product([mapping.case_id.unique(), list(MODEL_LABELS), list('ABCD')])
    observed = pd.MultiIndex.from_frame(mapping[['case_id', 'model_key', 'arm']])
    if len(observed.unique()) != 1600 or set(observed) != set(design):
        raise ValueError('Incomplete case-by-model-by-arm design.')
    index = pd.MultiIndex.from_product([mapping.scoring_id, raters], names=['scoring_id', 'rater_id'])
    raw = ratings.set_index(['scoring_id', 'rater_id']).reindex(index)[DIMS].to_numpy()
    if not np.isfinite(raw).all():
        raise ValueError('Each output must have all three ratings.')
    values = raw.astype(np.int8).reshape(1600, 3, 8).transpose(0, 2, 1)
    ranges = np.ptp(values, axis=2)
    high = (ranges.max(axis=1) >= 3) | (ranges[:, [3, 4, 6]].max(axis=1) >= 2)
    scores = mapping[['scoring_id', 'case_id', 'arm', 'model_key']].copy()
    scores['high_disagreement'] = np.where(high, 'yes', 'no')
    for j, dimension in enumerate(DIMS):
        scores[f'{dimension}_median_score'] = np.median(values[:, j, :], axis=1)
        # The six-setting and participant screens use dimension means rounded to three decimals.
        scores[f'{dimension}_mean_score'] = [round(float(x), 3) for x in values[:, j, :].mean(axis=1)]
    if audit_path is not None:
        flags = pd.read_csv(audit_path, keep_default_na=False)
        if len(flags) != 1600 or flags.output_id.duplicated().any() or set(flags.output_id) != set(mapping.output_id):
            raise ValueError('Audit flags must cover all 1600 rated output IDs exactly once.')
        if not flags.audit_flag.isin(['yes', 'no']).all():
            raise ValueError('Audit flag must be yes or no.')
        mapping = mapping.merge(flags[['output_id', 'audit_flag', 'audit_flag_type']], on='output_id', validate='one_to_one', sort=False)
        scores['audit_flag_type'] = mapping.audit_flag_type.replace('', 'none')
    return scores, mapping, values


def run_clmm_summary(clmm_dir: Path, output_dir: Path) -> None:
    ANALYSIS_DIR = Path(clmm_dir)
    OUT_DIR = result_directory(output_dir, 'clmm_summary')
    MODEL_ORDER = [
        "siliconflow_primary",
        "openrouter_gpt_oss_20b_free",
        "siliconflow_deepseek_v4_flash_paid",
        "siliconflow_minimax_m2_5_paid",
    ]

    MODEL_LABELS = {
        "siliconflow_primary": "Qwen3-8B baseline",
        "openrouter_gpt_oss_20b_free": "GPT-OSS-20B",
        "siliconflow_deepseek_v4_flash_paid": "DeepSeek-V4-Flash",
        "siliconflow_minimax_m2_5_paid": "MiniMax-M2.5",
    }

    ARM_ORDER = ["A", "B", "C", "D"]

    def read_inputs() -> dict[str, pd.DataFrame]:
        return {
            "fit": pd.read_csv(ANALYSIS_DIR / "PRIMARY_CLMM_FIT_STATUS.csv"),
            "model_emm": pd.read_csv(ANALYSIS_DIR / "PRIMARY_CLMM_MODEL_EMMEANS_LATENT.csv"),
            "arm_emm": pd.read_csv(ANALYSIS_DIR / "PRIMARY_CLMM_ARM_EMMEANS_LATENT.csv"),
            "model_pairwise": pd.read_csv(ANALYSIS_DIR / "PRIMARY_CLMM_MODEL_PAIRWISE_LATENT.csv"),
            "arm_pairwise": pd.read_csv(ANALYSIS_DIR / "PRIMARY_CLMM_ARM_PAIRWISE_LATENT.csv"),
        }

    def prepare_tables(inputs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        fit = inputs["fit"].copy()
        fit_table = fit[
            ["dimension", "status", "fitted_model", "n_observations", "logLik", "AIC", "BIC"]
        ].copy()
        for col in ["logLik", "AIC", "BIC"]:
            fit_table[col] = fit_table[col].round(3)

        model_emm = inputs["model_emm"].copy()
        model_emm["model_label"] = model_emm["model_key"].map(MODEL_LABELS).fillna(model_emm["model_key"])
        model_by_dim = model_emm[
            ["dimension", "model_key", "model_label", "emmean", "SE", "asymp.LCL", "asymp.UCL"]
        ].copy()
        model_by_dim = model_by_dim.sort_values(
            ["dimension", "model_key"], key=lambda s: s.map({m: i for i, m in enumerate(MODEL_ORDER)}).fillna(s)
            if s.name == "model_key"
            else s
        )

        model_overall = (
            model_emm.groupby(["model_key", "model_label"], as_index=False)
            .agg(
                mean_latent_emm=("emmean", "mean"),
                min_dimension_emm=("emmean", "min"),
                max_dimension_emm=("emmean", "max"),
            )
            .sort_values("mean_latent_emm", ascending=False)
        )
        model_overall.insert(0, "rank", range(1, len(model_overall) + 1))

        arm_emm = inputs["arm_emm"].copy()
        arm_by_dim = arm_emm[["dimension", "arm", "emmean", "SE", "asymp.LCL", "asymp.UCL"]].copy()
        arm_by_dim["arm"] = pd.Categorical(arm_by_dim["arm"], categories=ARM_ORDER, ordered=True)
        arm_by_dim = arm_by_dim.sort_values(["dimension", "arm"])

        arm_overall = (
            arm_emm.groupby("arm", as_index=False)
            .agg(
                mean_latent_emm=("emmean", "mean"),
                min_dimension_emm=("emmean", "min"),
                max_dimension_emm=("emmean", "max"),
            )
            .sort_values("mean_latent_emm", ascending=False)
        )
        arm_overall.insert(0, "rank", range(1, len(arm_overall) + 1))

        model_pw = inputs["model_pairwise"].copy()
        model_vs_qwen_rows = []
        for _, row in model_pw.iterrows():
            contrast = row["contrast"]
            if not contrast.startswith("siliconflow_primary - "):
                continue
            comparator = contrast.replace("siliconflow_primary - ", "")
            model_vs_qwen_rows.append(
                {
                    "dimension": row["dimension"],
                    "comparator_model_key": comparator,
                    "comparator_model_label": MODEL_LABELS.get(comparator, comparator),
                    "latent_difference_vs_qwen": -row["estimate"],
                    "SE": row["SE"],
                    "holm_adjusted_p": row["p.value"],
                    "direction": "higher_than_qwen" if -row["estimate"] > 0 else "lower_than_qwen",
                    "significant_p_lt_0_05": bool(row["p.value"] < 0.05),
                }
            )
        model_vs_qwen = pd.DataFrame(model_vs_qwen_rows)

        arm_pw = inputs["arm_pairwise"].copy()
        arm_vs_a_rows = []
        for _, row in arm_pw.iterrows():
            contrast = row["contrast"]
            if not contrast.startswith("A - "):
                continue
            comparator = contrast.replace("A - ", "")
            arm_vs_a_rows.append(
                {
                    "dimension": row["dimension"],
                    "comparator_arm": comparator,
                    "latent_difference_vs_arm_a": -row["estimate"],
                    "SE": row["SE"],
                    "holm_adjusted_p": row["p.value"],
                    "direction": "higher_than_A" if -row["estimate"] > 0 else "lower_than_A",
                    "significant_p_lt_0_05": bool(row["p.value"] < 0.05),
                }
            )
        arm_vs_a = pd.DataFrame(arm_vs_a_rows)

        pairwise_summary = pd.DataFrame(
            [
                {
                    "factor": "model",
                    "n_significant_holm_p_lt_0_05": int((model_pw["p.value"] < 0.05).sum()),
                    "n_total_pairwise_tests": len(model_pw),
                    "n_non_significant": int((model_pw["p.value"] >= 0.05).sum()),
                },
                {
                    "factor": "arm",
                    "n_significant_holm_p_lt_0_05": int((arm_pw["p.value"] < 0.05).sum()),
                    "n_total_pairwise_tests": len(arm_pw),
                    "n_non_significant": int((arm_pw["p.value"] >= 0.05).sum()),
                },
            ]
        )

        for df in [model_by_dim, model_overall, arm_by_dim, arm_overall, model_vs_qwen, arm_vs_a]:
            for col in ["emmean", "SE", "asymp.LCL", "asymp.UCL", "mean_latent_emm", "min_dimension_emm", "max_dimension_emm", "latent_difference_vs_qwen", "latent_difference_vs_arm_a", "holm_adjusted_p"]:
                if col in df.columns:
                    df[col] = df[col].round(3)

        return {
            "fit_table": fit_table,
            "model_overall": model_overall,
            "arm_overall": arm_overall,
            "model_by_dim": model_by_dim,
            "arm_by_dim": arm_by_dim,
            "model_vs_qwen": model_vs_qwen,
            "arm_vs_a": arm_vs_a,
            "pairwise_summary": pairwise_summary,
        }

    def write_tables(tables: dict[str, pd.DataFrame]) -> None:
        outputs = {
            "fit_table": "PRIMARY_CLMM_MANUSCRIPT_TABLE_FIT_STATUS.csv",
            "model_overall": "PRIMARY_CLMM_MANUSCRIPT_TABLE_MODEL_OVERALL_LATENT.csv",
            "arm_overall": "PRIMARY_CLMM_MANUSCRIPT_TABLE_ARM_OVERALL_LATENT.csv",
            "model_by_dim": "PRIMARY_CLMM_MANUSCRIPT_TABLE_MODEL_BY_DIMENSION_LATENT.csv",
            "arm_by_dim": "PRIMARY_CLMM_MANUSCRIPT_TABLE_ARM_BY_DIMENSION_LATENT.csv",
            "model_vs_qwen": "PRIMARY_CLMM_MANUSCRIPT_TABLE_MODEL_VS_QWEN_CONTRASTS.csv",
            "arm_vs_a": "PRIMARY_CLMM_MANUSCRIPT_TABLE_ARM_VS_A_CONTRASTS.csv",
            "pairwise_summary": "PRIMARY_CLMM_MANUSCRIPT_TABLE_PAIRWISE_SUMMARY.csv",
        }
        for key, name in outputs.items():
            tables[key].to_csv(OUT_DIR / name, index=False)

    write_tables(prepare_tables(read_inputs()))


def run_output_level_sensitivity(scores, mapping, output_dir):
    folder = result_directory(output_dir, 'output_level')
    OUT_SUMMARY = folder / 'PRIMARY_OUTPUT_LEVEL_SENSITIVITY_SUMMARY.csv'
    OUT_DISAGREEMENT = folder / 'PRIMARY_OUTPUT_LEVEL_DISAGREEMENT_SUMMARY.csv'
    mapping = mapping[['scoring_id', 'participant_id', 'output_id']]
    MODEL_LABELS = {
        "siliconflow_primary": "Qwen3-8B baseline",
        "openrouter_gpt_oss_20b_free": "GPT-OSS-20B",
        "siliconflow_deepseek_v4_flash_paid": "DeepSeek-V4-Flash",
        "siliconflow_minimax_m2_5_paid": "MiniMax-M2.5",
    }

    CLMM_ENDPOINT_ORDER = [
        "MiniMax-M2.5",
        "GPT-OSS-20B",
        "DeepSeek-V4-Flash",
        "Qwen3-8B baseline",
    ]
    # Reference order of the descriptive cross-dimension marginal latent EMM
    # index from the primary interaction CLMMs (Supplementary Table S8).
    CLMM_ARM_ORDER = ["A", "C", "D", "B"]

    DIMENSION_SETS = {
        "All 8 dimensions": ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8"],
        "Excluding D6": ["D1", "D2", "D3", "D4", "D5", "D7", "D8"],
        "Core D1/D3/D5": ["D1", "D3", "D5"],
    }


    def ordered_summary(series: pd.Series, label_map: dict[str, str] | None = None) -> tuple[list[str], str]:
        values = series.sort_values(ascending=False)
        if label_map:
            labels = [label_map.get(str(idx), str(idx)) for idx in values.index]
        else:
            labels = [str(idx) for idx in values.index]
        value_text = "; ".join(
            f"{label}={value:.3f}" for label, value in zip(labels, values.to_numpy())
        )
        return labels, value_text


    df = scores.merge(mapping, on="scoring_id", how="left")

    rows: list[dict[str, object]] = []
    for dimension_set, dimensions in DIMENSION_SETS.items():
        for aggregation in ["median", "mean"]:
            cols = [f"{dimension}_{aggregation}_score" for dimension in dimensions]
            tmp = df.copy()
            tmp["output_level_score"] = tmp[cols].mean(axis=1)
            model_series = tmp.groupby("model_key")["output_level_score"].mean()
            arm_series = tmp.groupby("arm")["output_level_score"].mean()
            endpoint_order, endpoint_values = ordered_summary(model_series, MODEL_LABELS)
            arm_order, arm_values = ordered_summary(arm_series)
            rows.append(
                {
                    "dimension_set": dimension_set,
                    "aggregation": aggregation,
                    "endpoint_ordering": " > ".join(endpoint_order),
                    "endpoint_values": endpoint_values,
                    "arm_ordering": " > ".join(arm_order),
                    "arm_values": arm_values,
                    "same_endpoint_as_clmm": endpoint_order == CLMM_ENDPOINT_ORDER,
                    "same_arm_as_clmm": arm_order == CLMM_ARM_ORDER,
                }
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_SUMMARY, index=False)

    df["model_label"] = df["model_key"].map(MODEL_LABELS)
    df["audit_flag_type"] = df["audit_flag_type"].fillna("none")
    disagreement_rows: list[dict[str, object]] = []
    for factor in ["model_label", "arm", "audit_flag_type"]:
        table = (
            df.groupby([factor, "high_disagreement"], dropna=False)
            .size()
            .reset_index(name="n")
        )
        table["factor"] = factor
        table = table.rename(columns={factor: "level"})
        disagreement_rows.extend(table[["factor", "level", "high_disagreement", "n"]].to_dict("records"))
    disagreement = pd.DataFrame(disagreement_rows)
    disagreement.to_csv(OUT_DISAGREEMENT, index=False)



def run_participant_adjusted_sensitivity(scores, mapping, output_dir):
    import statsmodels.formula.api as smf
    OUT_DIR = result_directory(output_dir, 'participant_adjusted')
    mapping = mapping[['scoring_id', 'participant_id']]
    MODEL_LEVELS = [
        "siliconflow_primary",
        "openrouter_gpt_oss_20b_free",
        "siliconflow_deepseek_v4_flash_paid",
        "siliconflow_minimax_m2_5_paid",
    ]
    MODEL_LABELS = {
        "siliconflow_primary": "Qwen3-8B baseline",
        "openrouter_gpt_oss_20b_free": "GPT-OSS-20B",
        "siliconflow_deepseek_v4_flash_paid": "DeepSeek-V4-Flash",
        "siliconflow_minimax_m2_5_paid": "MiniMax-M2.5",
    }
    ARM_LEVELS = ["A", "B", "C", "D"]


    df = scores.merge(mapping, on="scoring_id", how="left")
    df["model_key"] = pd.Categorical(df["model_key"], categories=MODEL_LEVELS)
    df["arm"] = pd.Categorical(df["arm"], categories=ARM_LEVELS)
    df["participant_id"] = pd.Categorical(df["participant_id"])

    fit_rows: list[dict[str, object]] = []
    coef_rows: list[dict[str, object]] = []
    emm_rows: list[dict[str, object]] = []

    participants = list(df["participant_id"].cat.categories)

    for dimension in [f"D{i}" for i in range(1, 9)]:
        score_col = f"{dimension}_mean_score"
        d = df[["participant_id", "model_key", "arm", score_col]].dropna().copy()
        d = d.rename(columns={score_col: "mean_score"})
        try:
            fit = smf.ols("mean_score ~ C(model_key) + C(arm) + C(participant_id)", data=d).fit(
                cov_type="cluster", cov_kwds={"groups": d["participant_id"]}
            )
            fit_rows.append(
                {
                    "dimension": dimension,
                    "status": "fit_ok",
                    "fitted_model": "participant_fixed_effect_cluster_se_ols",
                    "n_outputs": int(d.shape[0]),
                    "n_participants": int(d["participant_id"].nunique()),
                    "r_squared": round(float(fit.rsquared), 3),
                    "AIC": round(float(fit.aic), 3),
                    "BIC": round(float(fit.bic), 3),
                    "message": "participant fixed effects with participant-clustered standard errors",
                }
            )
            for term, estimate in fit.params.items():
                if term == "Intercept" or term.startswith("C(participant_id)"):
                    continue
                coef_rows.append(
                    {
                        "dimension": dimension,
                        "term": term,
                        "estimate_mean_score": round(float(estimate), 4),
                        "std_error_clustered": round(float(fit.bse.get(term, np.nan)), 4),
                        "t_value": round(float(fit.tvalues.get(term, np.nan)), 4),
                        "p_value_clustered": float(fit.pvalues.get(term, np.nan)),
                    }
                )

            grid_rows = []
            for model in MODEL_LEVELS:
                for arm in ARM_LEVELS:
                    for participant in participants:
                        grid_rows.append(
                            {"model_key": model, "arm": arm, "participant_id": participant}
                        )
            grid = pd.DataFrame(grid_rows)
            grid["model_key"] = pd.Categorical(grid["model_key"], categories=MODEL_LEVELS)
            grid["arm"] = pd.Categorical(grid["arm"], categories=ARM_LEVELS)
            grid["participant_id"] = pd.Categorical(grid["participant_id"], categories=participants)
            grid["predicted_mean_score"] = fit.predict(grid)

            for model, md in grid.groupby("model_key", observed=False):
                emm_rows.append(
                    {
                        "dimension": dimension,
                        "factor_type": "model_key",
                        "level": model,
                        "label": MODEL_LABELS[model],
                        "participant_adjusted_mean_score": round(float(md["predicted_mean_score"].mean()), 3),
                    }
                )
            for arm, ad in grid.groupby("arm", observed=False):
                emm_rows.append(
                    {
                        "dimension": dimension,
                        "factor_type": "arm",
                        "level": arm,
                        "label": arm,
                        "participant_adjusted_mean_score": round(float(ad["predicted_mean_score"].mean()), 3),
                    }
                )
        except Exception as exc:
            fit_rows.append(
                {
                    "dimension": dimension,
                    "status": "fit_failed",
                    "fitted_model": "participant_fixed_effect_cluster_se_ols",
                    "n_outputs": int(d.shape[0]),
                    "n_participants": int(d["participant_id"].nunique()),
                    "r_squared": np.nan,
                    "AIC": np.nan,
                    "BIC": np.nan,
                    "message": str(exc),
                }
            )

    fit_status = pd.DataFrame(fit_rows)
    coefs = pd.DataFrame(coef_rows)
    emmeans = pd.DataFrame(emm_rows)

    fit_status.to_csv(OUT_DIR / "PRIMARY_PARTICIPANT_ADJUSTED_FIXED_EFFECT_FIT_STATUS.csv", index=False)
    coefs.to_csv(OUT_DIR / "PRIMARY_PARTICIPANT_ADJUSTED_FIXED_EFFECT_COEFFICIENTS.csv", index=False)
    emmeans.to_csv(OUT_DIR / "PRIMARY_PARTICIPANT_ADJUSTED_FIXED_EFFECT_MARGINAL_MEANS.csv", index=False)

    model_summary = (
        emmeans[emmeans["factor_type"] == "model_key"]
        .groupby(["level", "label"], as_index=False)["participant_adjusted_mean_score"]
        .mean()
        .sort_values("participant_adjusted_mean_score", ascending=False)
    )
    arm_summary = (
        emmeans[emmeans["factor_type"] == "arm"]
        .groupby(["level", "label"], as_index=False)["participant_adjusted_mean_score"]
        .mean()
        .sort_values("participant_adjusted_mean_score", ascending=False)
    )

    model_summary.to_csv(OUT_DIR / 'PRIMARY_PARTICIPANT_ADJUSTED_MODEL_SUMMARY.csv', index=False)
    arm_summary.to_csv(OUT_DIR / 'PRIMARY_PARTICIPANT_ADJUSTED_ARM_SUMMARY.csv', index=False)

def run_complete_sensitivity(mapping, values, output_dir):
    """Use unrounded integer ratings for the four output-inclusion scenarios."""
    out = result_directory(output_dir, 'sensitivity')
    dimsets = {'all8': list(range(8)), 'excludeD6': [0,1,2,3,4,6,7], 'coreD1D3D5': [0,2,4]}
    ranges = np.ptp(values, axis=2)
    high = (ranges.max(axis=1) >= 3) | (ranges[:, [3,4,6]].max(axis=1) >= 2)
    audit = mapping.audit_flag.eq('yes').to_numpy()
    masks = {'all_outputs': np.ones(len(mapping), dtype=bool), 'exclude_audit_flagged': ~audit,
             'exclude_high_disagreement': ~high, 'exclude_audit_and_high_disagreement': ~(audit | high)}
    if (int(high.sum()), int(audit.sum()), int((high & audit).sum())) != (547, 146, 78):
        raise ValueError('Disagreement/audit counts do not match the study data.')
    aggregated = {'median': np.median(values, axis=2), 'mean': np.mean(values, axis=2)}
    output_values = {(d,a): v[:,cols].mean(axis=1) for d,cols in dimsets.items() for a,v in aggregated.items()}
    score_table = mapping[['scoring_id','output_id','case_id','participant_id','model_key','arm']].copy()
    score_table['audit_flag'] = np.where(audit, 'yes', 'no')
    score_table['high_disagreement'] = np.where(high, 'yes', 'no')
    for (d,a), scores in output_values.items():
        score_table[f'{d}_{a}'] = scores
    for j in range(8):
        score_table[f'D{j+1}_rater_range'] = ranges[:,j]
    score_table.to_csv(out/'output_level_scores_from_integer_ratings.csv', index=False, float_format='%.17g')
    counts, estimates = [], []
    for scenario, mask in masks.items():
        selected = mapping.loc[mask]
        counts.append({'scenario': scenario, 'n_outputs': len(selected), 'n_excluded_outputs': 1600-len(selected),
                       'n_cases': selected.case_id.nunique(), 'n_participants': selected.participant_id.nunique(),
                       'high_disagreement_outputs_retained': int(high[mask].sum()), 'audit_flagged_outputs_retained': int(audit[mask].sum())})
        for dimset in dimsets:
            for aggregation in ['median', 'mean']:
                frame = selected.copy()
                frame['value'] = output_values[(dimset, aggregation)][mask]
                for factor, column in [('endpoint','model_key'), ('arm','arm')]:
                    for level, group in frame.groupby(column, sort=True):
                        estimates.append({'scenario': scenario, 'dimension_set': dimset, 'aggregation': aggregation,
                                          'factor': factor, 'level': level, 'label': MODEL_LABELS.get(level,level),
                                          'n_outputs_scenario': len(frame), 'n_outputs': len(group),
                                          'n_cases': group.case_id.nunique(), 'n_participants': group.participant_id.nunique(),
                                          'point_estimate': float(group.value.to_numpy(dtype=np.float64).mean())})
    pd.DataFrame(counts).to_csv(out/'scenario_counts.csv', index=False)
    groups = pd.DataFrame(estimates)
    keys = ['scenario','dimension_set','aggregation','factor']
    groups['point_rank'] = groups.groupby(keys).point_estimate.rank(method='min', ascending=False).astype(int)
    groups.to_csv(out/'group_means.csv', index=False, float_format='%.17g')
    rows = []
    for key, group in groups.groupby(keys, sort=False):
        scenario, dimset, aggregation, factor = key
        group = group.sort_values(['point_rank','label'])
        rows.append({'scenario': scenario, 'dimension_set': dimset, 'aggregation': aggregation, 'factor': factor,
                     'n_outputs_scenario': int(group.n_outputs_scenario.iloc[0]),
                     'point_ordering': ' > '.join(group.label),
                     'point_values': '; '.join(f'{r.label}={r.point_estimate:.9f}' for r in group.itertuples())})
    ranks = pd.DataFrame(rows)
    baseline = ranks[ranks.scenario.eq('all_outputs')][['dimension_set','aggregation','factor','point_ordering']].rename(columns={'point_ordering':'all_outputs_point_ordering'})
    ranks = ranks.merge(baseline, on=['dimension_set','aggregation','factor'], validate='many_to_one')
    ranks['same_order_as_all_outputs'] = ranks.point_ordering.eq(ranks.all_outputs_point_ordering)
    ranks.to_csv(out/'point_rankings_and_changes.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ratings', type=Path, default=Path(__file__).resolve().parent.parent/'04_ratings/ratings_4800.csv')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--clmm-dir', type=Path, help='Defaults to RESULTS/primary.')
    parser.add_argument('--audit-flags', type=Path, help='Defaults to RESULTS/grounding_robustness/PRIMARY_OUTPUT_AUDIT_FLAGS.csv.')
    parser.add_argument('--mode', choices=['summary','output-level','participant','complete','all'], default='all')
    args = parser.parse_args()
    out = args.output_dir.expanduser().resolve()
    needs_audit = args.mode in ['output-level','complete','all']
    flags = args.audit_flags.expanduser().resolve() if args.audit_flags else out/'grounding_robustness/PRIMARY_OUTPUT_AUDIT_FLAGS.csv'
    if needs_audit and not flags.is_file():
        parser.error('Run grounding_robustness.py first or supply --audit-flags for the 1600 primary outputs.')
    clmm = args.clmm_dir.expanduser().resolve() if args.clmm_dir else out/'primary'
    if args.mode in ['summary','all']:
        run_clmm_summary(clmm, out)
    if args.mode != 'summary':
        scores, mapping, values = read_scores(args.ratings.expanduser().resolve(), flags if needs_audit else None)
        if args.mode in ['output-level','complete','all']:
            run_output_level_sensitivity(scores, mapping, out)
        if args.mode in ['participant','all']:
            run_participant_adjusted_sensitivity(scores, mapping, out)
        if args.mode in ['complete','all']:
            run_complete_sensitivity(mapping, values, out)


if __name__ == '__main__':
    main()
