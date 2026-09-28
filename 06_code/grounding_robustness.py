from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DIMS = [f"D{i}" for i in range(1, 9)]

HRV_FORBIDDEN_PATTERNS = [
    r"\bHRV\b",
    r"\bH\.R\.V\.\b",
    r"heart rate variability",
    r"heart-rate variability",
    r"\bHR variability\b",
    r"heart rate variation",
    r"heart-rate variation",
    r"HRV-related",
    r"autonomic balance",
    r"parasympathetic status",
    r"vagal tone",
    r"autonomic recovery",
    r"parasympathetic recovery",
]
RECOVERY_PROXY_RE = re.compile(r"nervous[- ]system recovery|overtraining|overreaching|diagnos", re.I)
TRUE_TREND_RE = re.compile(r"(?:7d|28d|delta|last3_direction|acwr|monotony|strain|rolling|trend)", re.I)


def parse_payload(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def payload_text(payload: dict[str, Any] | None, raw: str = "") -> str:
    return json.dumps(payload, ensure_ascii=False) if payload else str(raw or "")


def extract_visible_values(user_prompt: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for block in re.findall(r"===[^=]+===\n((?:- [^\n]+\n?)+)", str(user_prompt)):
        for name, value in re.findall(r"-\s*([a-z][a-z0-9_]+)\s*:\s*([^\n]+)", block):
            values[name] = value.strip()
    flat_match = re.search(r"Here are today's [^\n]*monitoring data\..*?\n\n(.+?)(?:\n\n|$)", str(user_prompt), re.DOTALL)
    if flat_match:
        for name, value in re.findall(r"([a-z][a-z0-9_]+)\s*=\s*([^,\n]+)", flat_match.group(1)):
            values[name] = value.strip()
    return values


def decision_caution_label(payload: dict[str, Any] | None, raw: str = "") -> str:
    text = " ".join(
        [
            str(payload.get("risk_identification", "")) if payload else "",
            str(payload.get("training_adjustment_suggestion", "")) if payload else "",
            raw or "",
        ]
    ).lower()
    if re.search(r"\b(stop|no training|medical|clinician|do not train)\b", text):
        return "stop_or_medical_caution"
    if re.search(r"\b(rest|active recovery|reduce|lower|easy|light|deload)\b", text):
        return "reduce_or_recovery"
    if re.search(r"\b(increase|progress|higher|harder)\b", text):
        return "increase_or_progress"
    if re.search(r"\b(maintain|same|normal|usual)\b", text):
        return "maintain"
    return "unclear"


def visible_grounding_issues(payload: dict[str, Any] | None, visible: dict[str, str]) -> list[str]:
    issues: list[str] = []
    evidence = payload.get("supporting_evidence", []) if payload else []
    if not isinstance(evidence, list):
        return ["supporting_evidence_not_list"]
    for item in evidence:
        if not isinstance(item, dict):
            issues.append("non_object_evidence")
            continue
        variable = str(item.get("variable", ""))
        direction = item.get("direction")
        if variable not in visible:
            issues.append(f"out_of_scope_variable:{variable}")
        elif str(visible.get(variable, "")).strip().lower() in {"missing", "nan", ""} and direction not in {None, "null", "", "→"}:
            issues.append(f"missing_value_non_null_direction:{variable}")
    return issues


def has_hrv_text(text: str) -> bool:
    return any(re.search(pattern, text, flags=re.I) for pattern in HRV_FORBIDDEN_PATTERNS)


def add_payload_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    payloads = [parse_payload(x) for x in out.get("parsed_json", pd.Series([""] * len(out))).fillna("")]
    out["_payload"] = payloads
    out["safety_flag"] = [p.get("safety_flag") if p else "unparsed" for p in payloads]
    # decision_bucket stores the derived decision-caution label.
    out["decision_bucket"] = [decision_caution_label(p, raw) for p, raw in zip(payloads, out.get("raw_text", pd.Series([""] * len(out))).fillna(""))]
    out["payload_text"] = [payload_text(p, raw) for p, raw in zip(payloads, out.get("raw_text", pd.Series([""] * len(out))).fillna(""))]
    out["hrv_specific_hit"] = out["payload_text"].map(has_hrv_text)
    out["recovery_proxy_or_diagnostic_hit"] = out["payload_text"].map(lambda x: bool(RECOVERY_PROXY_RE.search(str(x))))
    return out


def original_lookup(primary: pd.DataFrame) -> dict[tuple[str, str], dict[str, str]]:
    mini = primary[primary["model_key"].eq("siliconflow_minimax_m2_5_paid")].copy()
    lookup: dict[tuple[str, str], dict[str, str]] = {}
    for row in mini.itertuples(index=False):
        lookup[(row.case_id, row.arm)] = {
            "original_safety_flag": row.safety_flag,
            "original_decision_bucket": row.decision_bucket,
        }
    return lookup


def analyze_robustness(robustness: pd.DataFrame, primary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    original = original_lookup(primary)

    test = robustness[robustness["robustness_task"].eq("test_retest")]
    for (case_id, arm), group in test.groupby(["case_id", "arm"]):
        base = original.get((case_id, arm), {})
        rows.append(
            {
                "analysis": "test_retest",
                "unit": f"{case_id}_{arm}",
                "n_outputs": len(group),
                "original_safety_flag": base.get("original_safety_flag", ""),
                "original_decision_bucket": base.get("original_decision_bucket", ""),
                "safety_unique": group["safety_flag"].nunique(),
                "decision_unique": group["decision_bucket"].nunique(),
                "safety_stable": group["safety_flag"].nunique() == 1,
                "decision_stable": group["decision_bucket"].nunique() == 1,
                "safety_changed_vs_original_rate": float((group["safety_flag"] != base.get("original_safety_flag", "")).mean()) if base else pd.NA,
                "decision_changed_vs_original_rate": float((group["decision_bucket"] != base.get("original_decision_bucket", "")).mean()) if base else pd.NA,
                "word_count_range": int(group["word_count"].max() - group["word_count"].min()),
                "evidence_count_range": int(group["n_evidence"].max() - group["n_evidence"].min()),
            }
        )

    para = robustness[robustness["robustness_task"].eq("prompt_paraphrase")]
    for (case_id, arm), group in para.groupby(["case_id", "arm"]):
        base = original.get((case_id, arm), {})
        rows.append(
            {
                "analysis": "prompt_paraphrase",
                "unit": f"{case_id}_{arm}",
                "n_outputs": len(group),
                "original_safety_flag": base.get("original_safety_flag", ""),
                "original_decision_bucket": base.get("original_decision_bucket", ""),
                "safety_unique": group["safety_flag"].nunique(),
                "decision_unique": group["decision_bucket"].nunique(),
                "safety_stable": group["safety_flag"].nunique() == 1,
                "decision_stable": group["decision_bucket"].nunique() == 1,
                "safety_changed_vs_original_rate": float((group["safety_flag"] != base.get("original_safety_flag", "")).mean()) if base else pd.NA,
                "decision_changed_vs_original_rate": float((group["decision_bucket"] != base.get("original_decision_bucket", "")).mean()) if base else pd.NA,
                "word_count_range": int(group["word_count"].max() - group["word_count"].min()),
                "evidence_count_range": int(group["n_evidence"].max() - group["n_evidence"].min()),
            }
        )

    miss = robustness[robustness["robustness_task"].eq("missingness")]
    for (condition, case_id, arm), group in miss.groupby(["robustness_condition", "case_id", "arm"]):
        base = original.get((case_id, arm), {})
        rows.append(
            {
                "analysis": f"missingness_{condition}",
                "unit": f"{case_id}_{arm}",
                "n_outputs": len(group),
                "original_safety_flag": base.get("original_safety_flag", ""),
                "original_decision_bucket": base.get("original_decision_bucket", ""),
                "safety_unique": group["safety_flag"].nunique(),
                "decision_unique": group["decision_bucket"].nunique(),
                "safety_stable": group["safety_flag"].nunique() == 1,
                "decision_stable": group["decision_bucket"].nunique() == 1,
                "safety_changed_vs_original_rate": float((group["safety_flag"] != base.get("original_safety_flag", "")).mean()) if base else pd.NA,
                "decision_changed_vs_original_rate": float((group["decision_bucket"] != base.get("original_decision_bucket", "")).mean()) if base else pd.NA,
                "word_count_range": int(group["word_count"].max() - group["word_count"].min()),
                "evidence_count_range": int(group["n_evidence"].max() - group["n_evidence"].min()),
            }
        )
    return pd.DataFrame(rows)


def build_failure_table(primary: pd.DataFrame, robustness: pd.DataFrame, registry: pd.DataFrame, primary_scores: pd.DataFrame) -> pd.DataFrame:
    prompt_lookup = registry.set_index(["case_id", "arm", "robustness_task", "robustness_condition", "prompt_hash"])["user_prompt"].to_dict()
    main_prompt_lookup = read_input_csv(MAIN_REGISTRY).set_index(["case_id", "arm"])["user_prompt"].to_dict()

    primary_aug = primary.copy()
    if not primary_aug.empty:
        primary_aug["source"] = "primary"
        primary_aug["robustness_task"] = ""
        primary_aug["robustness_condition"] = ""
        primary_aug = primary_aug.merge(
            primary_scores[["case_id", "arm", "model_key", "audit_flag", "audit_flag_type", "D2_mean_score", "D4_mean_score", "D5_mean_score", "D7_mean_score", "overall_mean_aggregate"]],
            on=["case_id", "arm", "model_key"],
            how="left",
        )
    robustness_aug = robustness.copy()
    robustness_aug["source"] = "robustness"
    robustness_aug["audit_flag"] = "not_applicable"
    robustness_aug["audit_flag_type"] = ""
    robustness_aug["D2_mean_score"] = pd.NA
    robustness_aug["D4_mean_score"] = pd.NA
    robustness_aug["D5_mean_score"] = pd.NA
    robustness_aug["D7_mean_score"] = pd.NA
    robustness_aug["overall_mean_aggregate"] = pd.NA

    combined = pd.concat([primary_aug, robustness_aug], ignore_index=True, sort=False)
    combined = add_payload_columns(combined)
    rows: list[dict[str, Any]] = []
    for _, row in combined.iterrows():
        payload = row.get("_payload")
        if row.get("source") == "robustness":
            user_prompt = prompt_lookup.get((row.get("case_id"), row.get("arm"), row.get("robustness_task"), row.get("robustness_condition"), row.get("prompt_hash")), "")
        else:
            user_prompt = main_prompt_lookup.get((row.get("case_id"), row.get("arm")), "")
        visible = extract_visible_values(user_prompt)
        grounding_issues = visible_grounding_issues(payload, visible)
        text = row.get("payload_text", "")
        decision_label = row.get("decision_bucket", "")
        failure_modes: list[str] = []
        if grounding_issues or row.get("hrv_specific_hit") or row.get("recovery_proxy_or_diagnostic_hit"):
            failure_modes.append("unsupported_reasoning")
        if decision_label == "unclear" or (pd.notna(row.get("D2_mean_score", pd.NA)) and float(row.get("D2_mean_score")) <= 2.5):
            failure_modes.append("vague_recommendation")
        if re.search(r"\b(stop|avoid hard training|no training)\b", text, flags=re.I) and not re.search(r"\b(injury|severe|very low|multiple|high risk)\b", text, flags=re.I):
            failure_modes.append("overreaction")
        if re.search(r"\b(increase|progress|harder|high intensity)\b", text, flags=re.I) and re.search(r"\b(low readiness|poor sleep|injury|high fatigue|elevated soreness|high stress)\b", text, flags=re.I):
            failure_modes.append("over-persistence")
        if row.get("source") == "robustness" and row.get("robustness_task") == "missingness":
            if grounding_issues or not re.search(r"\b(missing|unavailable|masked|limited)\b", text, flags=re.I):
                failure_modes.append("missing-data breakdown")
        rows.append(
            {
                "source": row.get("source"),
                "case_id": row.get("case_id"),
                "arm": row.get("arm"),
                "model_key": row.get("model_key"),
                "robustness_task": row.get("robustness_task", ""),
                "robustness_condition": row.get("robustness_condition", ""),
                "output_id": row.get("output_id", ""),
                "status_code": row.get("status_code", ""),
                "safety_flag": row.get("safety_flag", ""),
                "decision_bucket": decision_label,
                "audit_flag": row.get("audit_flag", ""),
                "audit_flag_type": row.get("audit_flag_type", ""),
                "grounding_issue_count": len(grounding_issues),
                "grounding_issues": "; ".join(grounding_issues[:6]),
                "hrv_specific_hit": bool(row.get("hrv_specific_hit", False)),
                "recovery_proxy_or_diagnostic_hit": bool(row.get("recovery_proxy_or_diagnostic_hit", False)),
                "failure_modes": "; ".join(sorted(set(failure_modes))),
                "n_failure_modes": len(set(failure_modes)),
                "word_count": row.get("word_count", pd.NA),
                "n_evidence": row.get("n_evidence", pd.NA),
                "overall_mean_aggregate": row.get("overall_mean_aggregate", pd.NA),
            }
        )
    return pd.DataFrame(rows)


def select_case_studies(failure_table: pd.DataFrame, primary_scores: pd.DataFrame, primary: pd.DataFrame) -> pd.DataFrame:
    cases = read_input_csv(MAIN_CASES)
    mini_scores = primary_scores[primary_scores["model_key"].eq("siliconflow_minimax_m2_5_paid")].copy()
    case_means = mini_scores.groupby("case_id")["overall_mean_aggregate"].mean().reset_index(name="minimax_mean_score")
    case_info = cases.merge(case_means, on="case_id", how="left")

    high_load = case_info[
        (pd.to_numeric(case_info["srpe_total"], errors="coerce").fillna(0) >= case_info["srpe_total"].quantile(0.75))
        & (pd.to_numeric(case_info["readiness"], errors="coerce").fillna(10) <= 5)
    ].sort_values("minimax_mean_score", ascending=False)
    ambiguous = case_info[
        (pd.to_numeric(case_info["fatigue"], errors="coerce").between(2.5, 3.5))
        & (pd.to_numeric(case_info["readiness"], errors="coerce").between(5, 7))
    ].sort_values("minimax_mean_score", ascending=False)
    missing_cols = [c for c in case_info.columns if c.startswith("fb_")]
    case_info["fb_missing_count"] = case_info[missing_cols].isna().sum(axis=1)
    missing_case = case_info.sort_values(["fb_missing_count", "minimax_mean_score"], ascending=[False, False])
    failure_case = failure_table[failure_table["n_failure_modes"] > 0].sort_values(["n_failure_modes", "grounding_issue_count"], ascending=[False, False])

    choices = [
        ("high_load_low_recovery", high_load.iloc[0]["case_id"] if len(high_load) else case_info.iloc[0]["case_id"]),
        ("ambiguous_middle", ambiguous.iloc[0]["case_id"] if len(ambiguous) else case_info.iloc[1]["case_id"]),
        ("missing_data_case", missing_case.iloc[0]["case_id"]),
        ("illustrative_failure_case", failure_case.iloc[0]["case_id"] if len(failure_case) else case_info.iloc[2]["case_id"]),
    ]
    rows = []
    for study_type, case_id in choices:
        case_row = case_info[case_info["case_id"].eq(case_id)].iloc[0]
        ft = failure_table[failure_table["case_id"].eq(case_id)].sort_values(["n_failure_modes", "grounding_issue_count"], ascending=[False, False]).head(4)
        rows.append(
            {
                "case_study_type": study_type,
                "case_id": case_id,
                "participant_id": case_row.get("participant_id", ""),
                "date": case_row.get("date", ""),
                "selection_reason": describe_case_reason(study_type, case_row, ft),
                "selected_outputs": "; ".join(
                    f"{r.model_key}/{r.arm}/{r.source}:{r.failure_modes or 'no_failure_flag'}" for r in ft.itertuples(index=False)
                ),
            }
        )
    return pd.DataFrame(rows)


def describe_case_reason(study_type: str, case_row: pd.Series, ft: pd.DataFrame) -> str:
    if study_type == "high_load_low_recovery":
        return f"High load or low-recovery profile: readiness={case_row.get('readiness')}, srpe_total={case_row.get('srpe_total')}, fatigue={case_row.get('fatigue')}."
    if study_type == "ambiguous_middle":
        return f"Ambiguous mid-range state: fatigue={case_row.get('fatigue')}, readiness={case_row.get('readiness')}, mixed visible signals."
    if study_type == "missing_data_case":
        return f"High wearable/external missingness: fb_missing_count={case_row.get('fb_missing_count')}."
    return f"Illustrative failure case: highest detected failure-mode density among available primary/robustness outputs ({len(ft)} representative outputs)."


def read_input_csv(path: Path) -> pd.DataFrame:
    # Map legacy headers at input; preserve batch values, output IDs, and payloads.
    return pd.read_csv(path).rename(columns={
        "wave3_task": "robustness_task", "wave3_condition": "robustness_condition",
        "source_wave": "generation_batch",
    })


def require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {', '.join(missing)}")


def build_primary_audit(outputs: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    """Compute audit flags for the 1,600 primary outputs."""
    require_columns(outputs, ["study_component", "output_id", "case_id", "arm", "parsed_json", "raw_text"], "outputs")
    require_columns(registry, ["case_id", "arm", "user_prompt"], "main prompt registry")
    primary = outputs.loc[outputs["study_component"].eq("primary")]
    if primary["output_id"].duplicated().any() or registry.duplicated(["case_id", "arm"]).any():
        raise ValueError("Duplicate output IDs or case-arm prompt keys.")
    prompts = registry.set_index(["case_id", "arm"])["user_prompt"].to_dict()
    rows = []
    for row in primary.itertuples(index=False):
        key = (row.case_id, row.arm)
        if key not in prompts or not isinstance(prompts[key], str) or not prompts[key]:
            raise ValueError(f"Missing main prompt for {key}.")
        payload = parse_payload(row.parsed_json)
        if payload is None:
            raise ValueError(f"Invalid parsed payload for {row.output_id}.")
        visible = extract_visible_values(prompts[key])
        text = payload_text(payload, row.raw_text)
        flags = []
        if has_hrv_text(text):
            flags.append("hrv_specific_string")
        if RECOVERY_PROXY_RE.search(text):
            flags.append("recovery_proxy_or_diagnostic_wording")
        evidence = payload.get("supporting_evidence", [])
        if not isinstance(evidence, list):
            raise ValueError(f"Evidence is not a list for {row.output_id}.")
        for item in evidence:
            if isinstance(item, dict) and item.get("variable") not in visible:
                flags.append("exact_variable_grounding_issue:" + str(item.get("variable", "")))
        if not 2 <= len(evidence) <= 5:
            flags.append("evidence_count_outside_2_5:" + str(len(evidence)))
        if row.arm == "D" and not any(TRUE_TREND_RE.search(str(item.get("variable", ""))) for item in evidence if isinstance(item, dict)):
            flags.append("arm_d_no_true_trend_citation")
        flags = sorted(set(flags))
        rows.append({"output_id": row.output_id, "audit_flag": "yes" if flags else "no", "audit_flag_type": "; ".join(flags)})
    return pd.DataFrame(rows)


def build_primary_scores(ratings: pd.DataFrame, outputs: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    """Aggregate three ratings per output, rounding dimension means to three decimals."""
    metadata = ["scoring_id", "output_id", "case_id", "participant_id", "date", "arm", "model_key", "model_name", "generation_batch", "source_run_dir"]
    require_columns(ratings, metadata + ["rater_id"] + DIMS, "ratings")
    require_columns(outputs, ["output_id", "case_id", "participant_id", "date", "arm", "model_key", "provider"], "outputs")
    if ratings.duplicated(["scoring_id", "rater_id"]).any() or not ratings[DIMS].isin([1, 2, 3, 4, 5]).all().all():
        raise ValueError("Ratings must be unique output-rater rows with integer scores 1–5.")
    if outputs["output_id"].duplicated().any():
        raise ValueError("Output IDs must be unique.")
    output_lookup = outputs.set_index("output_id")
    flags = build_primary_audit(outputs, registry).set_index("output_id")
    rows = []
    for scoring_id, group in ratings.groupby("scoring_id", sort=True):
        group = group.sort_values("rater_id")
        if len(group) != 3 or group["rater_id"].nunique() != 3:
            raise ValueError(f"Expected three raters for {scoring_id}.")
        row = group.iloc[0]
        for field in metadata:
            if group[field].nunique(dropna=False) != 1:
                raise ValueError(f"Inconsistent {field} for {scoring_id}.")
        if row.output_id not in output_lookup.index or row.output_id not in flags.index:
            raise ValueError(f"Rated output {row.output_id} is absent from primary outputs.")
        output = output_lookup.loc[row.output_id]
        for field in ["case_id", "participant_id", "date", "arm", "model_key"]:
            if str(row[field]) != str(output[field]):
                raise ValueError(f"Rating/output {field} mismatch for {scoring_id}.")
        values = group[DIMS].to_numpy(dtype=int)
        ranges = np.ptp(values, axis=0)
        audit = flags.loc[row.output_id]
        result = {key: row[key] for key in ["scoring_id", "case_id", "arm", "model_key", "model_name"]}
        result.update(provider=output.provider, generation_batch=row.generation_batch, audit_flag=audit.audit_flag, audit_flag_type=audit.audit_flag_type)
        result.update(high_disagreement="yes" if ranges.max() >= 3 or ranges[[3, 4, 6]].max() >= 2 else "no",
                      max_dimension_range=int(ranges.max()), n_dimensions_range_ge2=int((ranges >= 2).sum()), n_dimensions_range_ge3=int((ranges >= 3).sum()))
        for j, dimension in enumerate(DIMS):
            result[dimension + "_median_score"] = float(np.median(values[:, j]))
            result[dimension + "_mean_score"] = round(float(values[:, j].mean()), 3)
            result[dimension + "_range"] = int(ranges[j])
        result["overall_median_aggregate"] = round(float(np.median(values, axis=0).mean()), 3)
        result["overall_mean_aggregate"] = 0.0
        result["n_low_reliability_dimension_flags"] = int((ranges[[3, 4, 5, 6, 7]] >= 2).sum())
        rows.append(result)
    aggregate = pd.DataFrame(rows)
    aggregate["overall_mean_aggregate"] = aggregate[[d + "_mean_score" for d in DIMS]].mean(axis=1).round(3)
    return aggregate


def audit_robustness(outputs: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    """Retain output-pattern counts separately from output-level flags and trend checks."""
    keys = ["case_id", "arm", "robustness_task", "robustness_condition", "prompt_hash"]
    merged = outputs.merge(registry[keys + ["user_prompt"]], on=keys, how="left", validate="one_to_one")
    if merged["user_prompt"].isna().any():
        raise ValueError("Some robustness outputs have no matching full prompt.")
    hrv_patterns = hrv_outputs = grounding_issues = trend_total = trend_cited = masked_total = leakage = 0
    for row in merged.itertuples(index=False):
        payload = parse_payload(row.parsed_json)
        text = payload_text(payload, row.raw_text)
        matches = [pattern for pattern in HRV_FORBIDDEN_PATTERNS if re.search(pattern, text, flags=re.I)]
        hrv_patterns += len(matches)
        hrv_outputs += bool(matches)
        visible = extract_visible_values(row.user_prompt)
        evidence = payload.get("supporting_evidence", []) if payload else []
        if isinstance(evidence, list):
            cited = {str(item.get("variable", "")) for item in evidence if isinstance(item, dict)}
            for item in evidence:
                if not isinstance(item, dict):
                    grounding_issues += 1
                    continue
                variable, direction = str(item.get("variable", "")), item.get("direction")
                if variable not in visible:
                    grounding_issues += 1
                elif str(visible.get(variable, "")).lower() == "missing" and direction not in {None, "null", "", "→"}:
                    grounding_issues += 1
            if row.arm == "D":
                trends = {k: v for k, v in visible.items() if TRUE_TREND_RE.search(k) and str(v).strip().lower() not in {"missing", "nan", ""}}
                if trends and row.robustness_condition != "mask_block_e":
                    trend_total += 1
                    trend_cited += bool(cited.intersection(trends))
                if row.robustness_condition == "mask_block_e":
                    masked_total += 1
                    leakage += any(TRUE_TREND_RE.search(variable) for variable in cited)
    rows = [
        ("hrv_pattern_matches", hrv_patterns, len(outputs), "output-pattern matches"),
        ("hrv_flagged_outputs", hrv_outputs, len(outputs), "outputs"),
        ("exact_variable_grounding_issues", grounding_issues, len(outputs), "evidence-item issues"),
        ("visible_trend_citation", trend_cited, trend_total, "eligible Arm D outputs"),
        ("masked_block_e_trend_leakage", leakage, masked_total, "masked Block E outputs"),
    ]
    return pd.DataFrame(rows, columns=["automated_check", "count", "denominator", "count_unit"])


def validate_inputs(outputs: pd.DataFrame, ratings: pd.DataFrame, main_registry: pd.DataFrame, registry: pd.DataFrame, cases: pd.DataFrame) -> None:
    require_columns(outputs, ["study_component", "output_id", "model_key", "case_id", "participant_id", "date", "arm", "prompt_hash", "robustness_task", "robustness_condition", "parsed_json", "raw_text", "status_code", "word_count", "n_evidence", "provider"], "outputs")
    require_columns(main_registry, ["case_id", "arm", "prompt_hash", "user_prompt"], "main prompt registry")
    keys = ["case_id", "arm", "robustness_task", "robustness_condition", "prompt_hash"]
    require_columns(registry, keys + ["user_prompt"], "robustness prompt registry")
    require_columns(cases, ["case_id", "participant_id", "date", "srpe_total", "readiness", "fatigue"], "main cases")
    primary = outputs.loc[outputs["study_component"].eq("primary")]
    robustness = outputs.loc[outputs["study_component"].ne("primary")]
    if len(outputs) != 2230 or len(primary) != 1600 or len(robustness) != 630 or outputs["output_id"].nunique() != 2230:
        raise ValueError("Expected 2,230 unique outputs: 1,600 primary and 630 robustness.")
    if len(ratings) != 4800 or ratings["output_id"].nunique() != 1600 or set(ratings["output_id"]) != set(primary["output_id"]):
        raise ValueError("Expected 4,800 ratings covering exactly the 1,600 primary outputs.")
    if len(main_registry) != 400 or main_registry.duplicated(["case_id", "arm"]).any() or len(registry) != 630 or registry.duplicated(keys).any():
        raise ValueError("Expected complete unique main (400) and robustness (630) prompt registries.")
    if len(cases) != 100 or cases["case_id"].duplicated().any() or set(cases["case_id"]) != set(primary["case_id"]):
        raise ValueError("The main case table must cover exactly the 100 primary cases.")
    if not any(column.startswith("fb_") for column in cases):
        raise ValueError("Main cases lack the fb_ wearable columns required for case selection.")
    for name, frame in [("main", main_registry), ("robustness", registry)]:
        if frame["user_prompt"].isna().any() or not frame["user_prompt"].astype(str).str.len().gt(0).all():
            raise ValueError(f"{name} registry contains empty prompt text.")
    matched = primary.merge(main_registry[["case_id", "arm", "prompt_hash"]], on=["case_id", "arm"], suffixes=("_output", "_registry"), how="left", validate="many_to_one")
    if not matched["prompt_hash_output"].eq(matched["prompt_hash_registry"]).all():
        raise ValueError("Primary output/prompt hashes do not match.")
    matched = robustness.merge(registry[keys], on=keys, how="left", indicator=True, validate="one_to_one")
    if not matched["_merge"].eq("both").all():
        raise ValueError("Robustness output/prompt hashes or conditions do not match.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", required=True, type=Path, help="Released outputs_2230.csv.")
    parser.add_argument("--ratings", required=True, type=Path, help="Released ratings_4800.csv.")
    parser.add_argument("--project-root", required=True, type=Path, help="Directory containing the original experiments/lrrif_v3_1 inputs.")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    global MAIN_REGISTRY, MAIN_CASES
    experiment = args.project_root.expanduser().resolve() / "experiments/lrrif_v3_1"
    MAIN_REGISTRY = experiment / "frozen/prompt_v1_0/prompts/main_prompt_registry.csv"
    MAIN_CASES = experiment / "data/main_100_cases.csv"
    registry_path = experiment / "main_runs/wave3_robustness_minimax/wave3_robustness_minimax_prompt_registry.csv"
    paths = [args.outputs, args.ratings, MAIN_REGISTRY, MAIN_CASES, registry_path]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        parser.error("Required inputs not found: " + "; ".join(missing))
    outputs, ratings, main_registry, registry, cases = [read_input_csv(path) for path in [args.outputs, args.ratings, MAIN_REGISTRY, registry_path, MAIN_CASES]]
    try:
        validate_inputs(outputs, ratings, main_registry, registry, cases)
        audit = build_primary_audit(outputs, main_registry)
        scores = build_primary_scores(ratings, outputs, main_registry)
        primary = add_payload_columns(outputs.loc[outputs["study_component"].eq("primary")].copy())
        robustness = add_payload_columns(outputs.loc[outputs["study_component"].ne("primary")].copy())
        stability = analyze_robustness(robustness, primary)
        failure_table = build_failure_table(primary, robustness, registry, scores)
        selection = select_case_studies(failure_table, scores, primary)
        grounding_summary = audit_robustness(robustness, registry)
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))
    destination = args.output_dir.expanduser().resolve() / "grounding_robustness"
    destination.mkdir(parents=True, exist_ok=True)
    audit.to_csv(destination / "PRIMARY_OUTPUT_AUDIT_FLAGS.csv", index=False)
    scores.to_csv(destination / "PRIMARY_AGGREGATED_SCORES.csv", index=False)
    stability.to_csv(destination / "ROBUSTNESS_OUTPUT_SUMMARY.csv", index=False)
    failure_table.to_csv(destination / "AUTOMATED_FAILURE_FLAGS.csv", index=False)
    selection.to_csv(destination / "CASE_STUDY_SELECTION.csv", index=False)
    grounding_summary.to_csv(destination / "ROBUSTNESS_GROUNDING_CHECKS.csv", index=False)
    stability.groupby("analysis").agg(
        n_units=("unit", "count"), safety_stability_rate=("safety_stable", "mean"), decision_stability_rate=("decision_stable", "mean"),
        mean_safety_changed_vs_original=("safety_changed_vs_original_rate", "mean"), mean_decision_changed_vs_original=("decision_changed_vs_original_rate", "mean"),
        mean_word_count_range=("word_count_range", "mean"), mean_evidence_count_range=("evidence_count_range", "mean"),
    ).reset_index().to_csv(destination / "LABEL_STABILITY_SUMMARY.csv", index=False)
    totals = failure_table.groupby("source").agg(
        n_outputs=("source", "size"), hrv_flagged_outputs=("hrv_specific_hit", "sum"), recovery_diagnostic_flagged_outputs=("recovery_proxy_or_diagnostic_hit", "sum"),
        n_failure_labels=("n_failure_modes", "sum"), mean_failure_labels=("n_failure_modes", "mean"),
    ).reset_index()
    totals.loc[len(totals)] = ["combined", len(failure_table), int(failure_table["hrv_specific_hit"].sum()),
                               int(failure_table["recovery_proxy_or_diagnostic_hit"].sum()),
                               int(failure_table["n_failure_modes"].sum()), float(failure_table["n_failure_modes"].mean())]
    totals.to_csv(destination / "AUTOMATED_SCREENING_TOTALS.csv", index=False)
    category_counts = failure_table.assign(failure_modes=failure_table["failure_modes"].replace("", "none")).groupby(["source", "failure_modes"]).size()
    all_groups = pd.MultiIndex.from_product(category_counts.index.levels, names=category_counts.index.names)
    categories = category_counts.reindex(all_groups, fill_value=0).rename("n_outputs").reset_index()
    categories["percent"] = categories["n_outputs"] / categories["source"].map(failure_table.groupby("source").size()) * 100
    categories.to_csv(destination / "AUTOMATED_FAILURE_CATEGORIES.csv", index=False)
    print(f"Wrote CSV results to {destination}")


if __name__ == "__main__":
    main()
