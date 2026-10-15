from __future__ import annotations
import argparse
from pathlib import Path
from io import StringIO
import re
from typing import Any
import numpy as np
import pandas as pd

TRUE_TREND_RE = re.compile(r"(?:7d|28d|delta|last3_direction|acwr|monotony|strain|rolling|trend)", re.I)

def _sample_with_weekend_balance(group: pd.DataFrame, n_take: int, rng: np.random.Generator) -> pd.DataFrame:
    if n_take <= 0 or group.empty:
        return group.head(0).copy()
    working = group.copy()
    working["is_weekend"] = pd.to_datetime(working["date"]).dt.dayofweek >= 5
    sampled_parts = []
    strata = [stratum for _, stratum in working.groupby("is_weekend", dropna=False)]
    if n_take >= len(strata):
        for stratum in strata:
            sampled_parts.append(stratum.sample(n=1, random_state=int(rng.integers(0, 2**31 - 1))))
    already = pd.concat(sampled_parts, ignore_index=False) if sampled_parts else working.head(0)
    remaining = working.drop(index=already.index, errors="ignore")
    remaining_slots = min(n_take - len(already), len(remaining))
    if remaining_slots > 0:
        sampled_parts.append(remaining.sample(n=remaining_slots, random_state=int(rng.integers(0, 2**31 - 1))))
    sampled = pd.concat(sampled_parts, ignore_index=False) if sampled_parts else working.sample(n=min(n_take, len(working)), random_state=int(rng.integers(0, 2**31 - 1)))
    return sampled.drop(columns=["is_weekend"], errors="ignore")

def stratified_sample_cases(frame: pd.DataFrame, n: int, seed: int, exclude_keys: set[tuple[str, str]] | None = None, prefix: str = "case") -> pd.DataFrame:
    exclude_keys = exclude_keys or set()
    valid = frame.loc[frame["valid_for_analysis"]].copy()
    valid = valid.loc[~valid.apply(lambda row: (str(row["participant_id"]), str(row["date"])) in exclude_keys, axis=1)].copy()
    valid = valid.sort_values(["participant_id", "date"]).reset_index(drop=True)
    rng = np.random.default_rng(seed)

    participants = sorted(valid["participant_id"].astype(str).unique())
    if len(participants) == 0:
        return valid.head(0).copy()
    if n < len(participants):
        raise ValueError(f"Requested n={n}, but at least {len(participants)} rows are needed to cover all participants.")

    per_participant_min = max(1, n // len(participants))
    sampled_parts = []
    for participant_id, group in valid.groupby("participant_id", sort=True):
        n_take = min(per_participant_min, len(group))
        sampled_parts.append(_sample_with_weekend_balance(group, n_take, rng))

    sampled = pd.concat(sampled_parts, ignore_index=False) if sampled_parts else valid.head(0)
    sampled_keys = set(zip(sampled["participant_id"].astype(str), sampled["date"].astype(str)))
    remaining = valid.loc[~valid.apply(lambda row: (str(row["participant_id"]), str(row["date"])) in sampled_keys, axis=1)].copy()
    remaining_slots = min(n - len(sampled), len(remaining))
    if remaining_slots > 0:
        remaining["is_weekend"] = pd.to_datetime(remaining["date"]).dt.dayofweek >= 5
        fill_parts = []
        for _, stratum in remaining.groupby(["participant_id", "is_weekend"], dropna=False):
            if len(fill_parts) >= remaining_slots:
                break
            proportional = max(1, round(remaining_slots * len(stratum) / max(len(remaining), 1)))
            take = min(proportional, len(stratum), remaining_slots - sum(len(part) for part in fill_parts))
            if take > 0:
                fill_parts.append(stratum.sample(n=take, random_state=int(rng.integers(0, 2**31 - 1))))
        fill = pd.concat(fill_parts, ignore_index=False) if fill_parts else remaining.head(0)
        if len(fill) < remaining_slots:
            fill_keys = set(zip(fill["participant_id"].astype(str), fill["date"].astype(str)))
            extra_pool = remaining.loc[~remaining.apply(lambda row: (str(row["participant_id"]), str(row["date"])) in fill_keys, axis=1)]
            extra_take = min(remaining_slots - len(fill), len(extra_pool))
            if extra_take > 0:
                fill = pd.concat([fill, extra_pool.sample(n=extra_take, random_state=int(rng.integers(0, 2**31 - 1)))], ignore_index=False)
        sampled = pd.concat([sampled, fill.drop(columns=["is_weekend"], errors="ignore")], ignore_index=False)

    sampled = sampled.sample(n=min(n, len(sampled)), random_state=seed).reset_index(drop=True)
    sampled.insert(0, "case_id", [f"{prefix}_{i:03d}" for i in range(1, len(sampled) + 1)])
    return sampled

def is_missing(value: Any) -> bool:
    if pd.isna(value):
        return True
    return str(value).strip().lower() in {"", "missing", "nan", "none", "null"}

def completeness_score(row: pd.Series) -> float:
    candidate_cols = [
        "fatigue",
        "mood",
        "readiness",
        "sleep_duration_h",
        "sleep_quality",
        "soreness",
        "stress",
        "training_count",
        "rpe_mean",
        "duration_total",
        "srpe_total",
        "fb_steps",
        "fb_calories",
        "fb_distance",
        "fb_exercise_count",
        "fb_exercise_duration_min",
        "fatigue_7d_mean",
        "readiness_7d_mean",
        "srpe_total_7d_mean",
        "acwr_srpe_7_28",
    ]
    present = [not is_missing(row.get(col)) for col in candidate_cols if col in row.index]
    return sum(present) / max(1, len(present))

def has_trend_signal(row: pd.Series) -> bool:
    for col in row.index:
        if TRUE_TREND_RE.search(str(col)) and not is_missing(row.get(col)):
            return True
    return False

def select_robustness_cases(cases: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    work = cases.copy()
    work["data_completeness"] = work.apply(completeness_score, axis=1)
    work["fatigue_bin"] = pd.cut(
        pd.to_numeric(work["fatigue"], errors="coerce"),
        bins=[-float("inf"), 2.5, 3.5, float("inf")],
        labels=["low", "mid", "high"],
    ).astype(str)
    work["readiness_bin"] = pd.cut(
        pd.to_numeric(work["readiness"], errors="coerce"),
        bins=[-float("inf"), 4.5, 6.5, float("inf")],
        labels=["low", "mid", "high"],
    ).astype(str)
    work["completeness_bin"] = pd.qcut(work["data_completeness"].rank(method="first"), q=3, labels=["low", "mid", "high"]).astype(str)
    work["has_trend_signal"] = work.apply(has_trend_signal, axis=1)
    work["safety_relevant"] = (
        (pd.to_numeric(work.get("fatigue"), errors="coerce").fillna(0) >= 4)
        | (pd.to_numeric(work.get("readiness"), errors="coerce").fillna(10) <= 4)
        | (pd.to_numeric(work.get("has_injury"), errors="coerce").fillna(0) == 1)
        | (pd.to_numeric(work.get("soreness"), errors="coerce").fillna(0) >= 4)
    )

    rng = work.sample(frac=1, random_state=42).reset_index(drop=True)
    selected: list[str] = []

    for _, group in rng.groupby("participant_id", sort=False):
        selected.append(group.iloc[0]["case_id"])
        if len(selected) >= 16:
            break

    strata = [
        ("fatigue_bin", "high"),
        ("fatigue_bin", "low"),
        ("readiness_bin", "high"),
        ("readiness_bin", "low"),
        ("completeness_bin", "high"),
        ("completeness_bin", "low"),
        ("has_trend_signal", True),
        ("has_trend_signal", False),
        ("safety_relevant", True),
    ]
    for col, val in strata:
        if len(selected) >= 30:
            break
        pool = rng[(rng[col] == val) & (~rng["case_id"].isin(selected))]
        if not pool.empty:
            selected.append(pool.iloc[0]["case_id"])

    for case_id in rng["case_id"]:
        if len(selected) >= 30:
            break
        if case_id not in selected:
            selected.append(case_id)

    selected = selected[:30]
    selected_df = work[work["case_id"].isin(selected)].copy()
    selected_df["robustness_selected"] = True
    selected_df["selection_order"] = selected_df["case_id"].map({cid: i + 1 for i, cid in enumerate(selected)})
    selected_df = selected_df.sort_values("selection_order").reset_index(drop=True)

    paraphrase_cases = (
        selected_df.sort_values(["safety_relevant", "has_trend_signal", "data_completeness"], ascending=[False, False, True])
        .head(10)["case_id"]
        .tolist()
    )
    return selected_df, selected, paraphrase_cases


def mark_eligible_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    required = {'participant_id', 'date', 'fb_minutes_asleep'}
    if not required.issubset(frame.columns):
        raise ValueError(f'Missing candidate columns: {sorted(required - set(frame.columns))}')
    if frame.duplicated(['participant_id', 'date']).any():
        raise ValueError('Duplicate participant/date candidate rows.')
    blocks = {
        'a': ['fatigue', 'readiness', 'mood', 'soreness', 'stress'],
        'b': ['sleep_duration_h', 'fb_sleep_score', 'fb_sleep_efficiency',
              'fb_minutes_asleep', 'fb_minutes_awake', 'fb_time_in_bed',
              'fb_deep_sleep_min', 'fb_sleep_deep_min', 'fb_sleep_light_min',
              'fb_sleep_rem_min', 'fb_sleep_wake_count', 'fb_sleep_resting_hr',
              'fb_resting_hr_precise'],
        'c': ['training_count', 'rpe_mean', 'rpe_max', 'duration_total', 'srpe_total'],
        'd': ['fb_steps', 'fb_calories', 'fb_distance', 'fb_heart_rate_bpm_mean',
              'fb_heart_rate_bpm_std', 'fb_hr_zone_below1', 'fb_hr_zone_1',
              'fb_hr_zone_2', 'fb_hr_zone_3', 'fb_exercise_count',
              'fb_exercise_calories', 'fb_exercise_duration_min',
              'fb_exercise_avg_hr', 'fb_exercise_steps', 'fb_exercise_distance'],
    }
    out = frame.copy()
    for block, columns in blocks.items():
        present = [c for c in columns if c in out.columns]
        out[f'has_block_{block}'] = out[present].notna().any(axis=1) if present else False
    out['valid_base_analysis'] = out[[f'has_block_{b}' for b in 'abcd']].all(axis=1)
    out['short_sleep_lt_120'] = pd.to_numeric(out['fb_minutes_asleep'], errors='coerce').lt(120).fillna(False)
    out['valid_for_analysis'] = out['valid_base_analysis'] & ~out['short_sleep_lt_120']
    return out


def select_study_cases(candidate_frame: pd.DataFrame):
    candidate_frame = mark_eligible_candidates(candidate_frame)
    pilot = stratified_sample_cases(candidate_frame, 20, 42, prefix="pilot")
    excluded = set(zip(pilot.participant_id.astype(str), pilot.date.astype(str)))
    primary = stratified_sample_cases(candidate_frame, 100, 43,
                                     exclude_keys=excluded, prefix="case")
    primary_for_robustness = pd.read_csv(StringIO(primary.to_csv(index=False)))
    robustness, _, paraphrase_case_ids = select_robustness_cases(primary_for_robustness)
    return pilot, primary, robustness, paraphrase_case_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', type=Path, required=True,
                        help='Cleaned PMData person-day CSV with current and prior-day trend variables.')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    candidates = pd.read_csv(args.candidates.expanduser())
    pilot, primary, robustness, paraphrase_ids = select_study_cases(candidates)
    expected = (20, 100, 30, 10)
    if (len(pilot), len(primary), len(robustness), len(paraphrase_ids)) != expected:
        raise ValueError('The candidate data cannot reproduce the specified 20/100/30/10 sample sizes.')
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in [('pilot_20_cases.csv', pilot), ('main_100_cases.csv', primary),
                        ('robustness_30_cases.csv', robustness),
                        ('paraphrase_10_cases.csv', robustness.set_index('case_id').loc[paraphrase_ids].reset_index())]:
        destination = out / name
        if destination == args.candidates.expanduser().resolve():
            raise ValueError('Output path would overwrite the candidate input.')
        frame.to_csv(destination, index=False)


if __name__ == '__main__':
    main()
