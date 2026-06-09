"""Exploratory analysis of signal-time traits vs 4h/24h outcomes for LONG_EXPLOSION signals.

Reads data/memecoin_signal_outcomes.csv and produces:
  - Feature summaries grouped by outcome_4h / outcome_24h
  - Diagnostic flag summaries
  - Rule experiment table
  - data/memecoin_signal_trait_summary_latest.csv
  - data/memecoin_rule_experiment_summary_latest.csv

Research only. No trading. No orders. No cron. No private credentials.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_OUTCOMES_PATH = Path("data/memecoin_signal_outcomes.csv")
TRAIT_SUMMARY_PATH = Path("data/memecoin_signal_trait_summary_latest.csv")
RULE_SUMMARY_PATH = Path("data/memecoin_rule_experiment_summary_latest.csv")

FEATURE_COLS = [
    "primary_ohlc_score",
    "spread_pct",
    "today_return_pct",
    "ret_15m_pct",
    "ret_1h_pct",
    "ret_4h_pct",
    "ret_24h_pct",
    "volume_ratio_1h",
    "volume_ratio_4h",
]

OUTCOME_4H_COL = "outcome_4h"
OUTCOME_24H_COL = "outcome_24h"

RULE_NAMES = [
    "baseline_all_long_explosion",
    "avoid_wide_spread",
    "avoid_terminal_spike",
    "require_positive_1h",
    "require_clean_continuation",
    "avoid_overextended_24h",
    "avoid_volume_climax_rollover",
    "tight_spread_only",
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_long_explosion(path: Path = DEFAULT_OUTCOMES_PATH) -> pd.DataFrame:
    """Load outcomes CSV and return only LONG_EXPLOSION rows with numeric coercion."""
    df = pd.read_csv(path, dtype=str)
    df = df[df["ohlc_signal_type"] == "LONG_EXPLOSION"].copy()

    numeric_cols = FEATURE_COLS + [
        "future_ret_4h_pct",
        "future_ret_24h_pct",
        "max_favorable_4h_pct",
        "max_adverse_4h_pct",
        "max_favorable_24h_pct",
        "max_adverse_24h_pct",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = float("nan")

    # Normalise outcome columns: blank strings -> NaN
    for col in [OUTCOME_4H_COL, OUTCOME_24H_COL, "outcome_15m", "outcome_1h"]:
        if col in df.columns:
            df[col] = df[col].replace("", pd.NA)
            df.loc[df[col] == "", col] = pd.NA

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Diagnostic flags
# ---------------------------------------------------------------------------


def add_diagnostic_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add boolean diagnostic flag columns and return a new df."""
    df = df.copy()

    df["is_rolling_over"] = (df["ret_1h_pct"] < 0) & (df["ret_4h_pct"] > 0)
    df["is_wide_spread"] = df["spread_pct"] > 1.0
    df["is_overextended_24h"] = df["ret_24h_pct"] > 25
    df["is_volume_climax"] = df["volume_ratio_4h"] > 10
    df["is_clean_continuation"] = (
        (df["ret_15m_pct"] > 0) & (df["ret_1h_pct"] > 0) & (df["ret_4h_pct"] > 0)
    )
    df["is_pullback_rebound_candidate"] = (
        (df["ret_1h_pct"] < 0) & (df["ret_4h_pct"] > 0) & (df["spread_pct"] <= 0.5)
    )
    df["danger_terminal_spike"] = (
        (df["spread_pct"] > 1.0) & (df["ret_1h_pct"] < 0) & (df["ret_4h_pct"] > 0)
    )

    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _outcome_counts(series: pd.Series) -> tuple[int, int, int]:
    """Return (success, flat, failure) counts from an outcome column slice."""
    success = int((series == "SUCCESS").sum())
    flat = int((series == "FLAT").sum())
    failure = int((series == "FAILURE").sum())
    return success, flat, failure


def _rate(num: int, denom: int) -> float:
    return round(num / denom, 4) if denom > 0 else float("nan")


def _fmt(val, pct: bool = True, decimals: int = 2) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "—"
    suffix = "%" if pct else ""
    return f"{val:.{decimals}f}{suffix}"


def _section(title: str) -> None:
    print(f"\n{'─' * 70}")
    print(f"  {title}")
    print(f"{'─' * 70}")


# ---------------------------------------------------------------------------
# Feature summaries grouped by outcome
# ---------------------------------------------------------------------------


def print_feature_summaries(df: pd.DataFrame) -> None:
    for outcome_col in [OUTCOME_4H_COL, OUTCOME_24H_COL]:
        _section(f"Feature summary grouped by {outcome_col}")
        rows_with_outcome = df[df[outcome_col].notna()]
        if rows_with_outcome.empty:
            print("  Not enough completed outcomes yet.")
            continue

        features = [c for c in FEATURE_COLS if c in df.columns]
        groups = rows_with_outcome.groupby(outcome_col)

        header = f"  {'outcome':<12}  {'n':>4}"
        for f in features:
            short = f.replace("_pct", "").replace("ratio_", "r_")[:10]
            header += f"  {short+'/avg':>13}{short+'/med':>13}"
        print(header)
        print("  " + "─" * min(200, 20 + len(features) * 28))

        for outcome_val, grp in groups:
            row = f"  {str(outcome_val):<12}  {len(grp):>4}"
            for feat in features:
                avg = grp[feat].mean()
                med = grp[feat].median()
                row += f"  {_fmt(avg):>13}{_fmt(med):>13}"
            print(row)


# ---------------------------------------------------------------------------
# Diagnostic flag summaries
# ---------------------------------------------------------------------------


def build_flag_summary(df: pd.DataFrame) -> pd.DataFrame:
    flag_cols = [
        "is_rolling_over",
        "is_wide_spread",
        "is_overextended_24h",
        "is_volume_climax",
        "is_clean_continuation",
        "is_pullback_rebound_candidate",
        "danger_terminal_spike",
    ]

    records = []
    for flag in flag_cols:
        if flag not in df.columns:
            continue
        flagged = df[df[flag] == True]  # noqa: E712
        n = len(flagged)

        sub4 = flagged.dropna(subset=["future_ret_4h_pct"])
        avg4 = sub4["future_ret_4h_pct"].mean() if not sub4.empty else float("nan")
        med4 = sub4["future_ret_4h_pct"].median() if not sub4.empty else float("nan")
        s4, fl4, f4 = _outcome_counts(sub4[OUTCOME_4H_COL].dropna()) if not sub4.empty else (0, 0, 0)

        sub24 = flagged.dropna(subset=["future_ret_24h_pct"])
        avg24 = sub24["future_ret_24h_pct"].mean() if not sub24.empty else float("nan")
        s24, fl24, f24 = _outcome_counts(sub24[OUTCOME_24H_COL].dropna()) if not sub24.empty else (0, 0, 0)

        records.append({
            "flag": flag,
            "n_flagged": n,
            "avg_future_ret_4h_pct": round(avg4, 4) if not pd.isna(avg4) else float("nan"),
            "median_future_ret_4h_pct": round(med4, 4) if not pd.isna(med4) else float("nan"),
            "success_4h": s4,
            "flat_4h": fl4,
            "failure_4h": f4,
            "avg_future_ret_24h_pct": round(avg24, 4) if not pd.isna(avg24) else float("nan"),
            "success_24h": s24,
            "flat_24h": fl24,
            "failure_24h": f24,
        })

    return pd.DataFrame(records)


def print_flag_summary(flag_df: pd.DataFrame) -> None:
    _section("Diagnostic Flag Summaries (LONG_EXPLOSION only)")
    if flag_df.empty:
        print("  No flags to report.")
        return

    col_w = 32
    print(
        f"\n  {'flag':<{col_w}}  {'n':>4}  {'avg4h':>7}  {'med4h':>7}  "
        f"{'s4':>4}  {'fl4':>4}  {'f4':>4}  "
        f"{'avg24h':>7}  {'s24':>4}  {'fl24':>4}  {'f24':>4}"
    )
    print("  " + "─" * (col_w + 70))

    for _, row in flag_df.iterrows():
        print(
            f"  {row['flag']:<{col_w}}  {int(row['n_flagged']):>4}  "
            f"{_fmt(row['avg_future_ret_4h_pct']):>7}  "
            f"{_fmt(row['median_future_ret_4h_pct']):>7}  "
            f"{int(row['success_4h']):>4}  {int(row['flat_4h']):>4}  {int(row['failure_4h']):>4}  "
            f"{_fmt(row['avg_future_ret_24h_pct']):>7}  "
            f"{int(row['success_24h']):>4}  {int(row['flat_24h']):>4}  {int(row['failure_24h']):>4}"
        )


# ---------------------------------------------------------------------------
# Rule experiments
# ---------------------------------------------------------------------------


def apply_rule(df: pd.DataFrame, rule_name: str) -> pd.Series:
    """Return a boolean mask for rows that PASS the rule (i.e., are kept)."""
    if rule_name == "baseline_all_long_explosion":
        return pd.Series([True] * len(df), index=df.index)
    if rule_name == "avoid_wide_spread":
        return df["spread_pct"] <= 1.0
    if rule_name == "avoid_terminal_spike":
        return ~df["danger_terminal_spike"]
    if rule_name == "require_positive_1h":
        return df["ret_1h_pct"] > 0
    if rule_name == "require_clean_continuation":
        return df["is_clean_continuation"]
    if rule_name == "avoid_overextended_24h":
        return df["ret_24h_pct"] <= 25
    if rule_name == "avoid_volume_climax_rollover":
        return ~((df["volume_ratio_4h"] > 10) & (df["ret_1h_pct"] < 0))
    if rule_name == "tight_spread_only":
        return df["spread_pct"] <= 0.5
    raise ValueError(f"Unknown rule: {rule_name}")


def build_rule_experiment_summary(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    total = len(df)

    for rule_name in RULE_NAMES:
        mask = apply_rule(df, rule_name)
        kept = df[mask]
        n_kept = len(kept)
        n_removed = total - n_kept

        sub4 = kept.dropna(subset=["future_ret_4h_pct"])
        avg4 = sub4["future_ret_4h_pct"].mean() if not sub4.empty else float("nan")
        med4 = sub4["future_ret_4h_pct"].median() if not sub4.empty else float("nan")
        avg_adv4 = sub4["max_adverse_4h_pct"].mean() if not sub4.empty else float("nan")
        s4, fl4, f4 = _outcome_counts(sub4[OUTCOME_4H_COL].dropna()) if not sub4.empty else (0, 0, 0)
        n4 = s4 + fl4 + f4
        sr4 = _rate(s4, n4)
        fr4 = _rate(f4, n4)

        sub24 = kept.dropna(subset=["future_ret_24h_pct"])
        avg24 = sub24["future_ret_24h_pct"].mean() if not sub24.empty else float("nan")
        s24, fl24, f24 = _outcome_counts(sub24[OUTCOME_24H_COL].dropna()) if not sub24.empty else (0, 0, 0)
        n24 = s24 + fl24 + f24
        sr24 = _rate(s24, n24)
        fr24 = _rate(f24, n24)

        records.append({
            "rule": rule_name,
            "rows_kept": n_kept,
            "rows_removed": n_removed,
            "avg_future_ret_4h_pct": round(avg4, 4) if not pd.isna(avg4) else float("nan"),
            "median_future_ret_4h_pct": round(med4, 4) if not pd.isna(med4) else float("nan"),
            "success_rate_4h": sr4,
            "failure_rate_4h": fr4,
            "avg_max_adverse_4h_pct": round(avg_adv4, 4) if not pd.isna(avg_adv4) else float("nan"),
            "avg_future_ret_24h_pct": round(avg24, 4) if not pd.isna(avg24) else float("nan"),
            "success_rate_24h": sr24,
            "failure_rate_24h": fr24,
        })

    return pd.DataFrame(records)


def print_rule_experiment_summary(rule_df: pd.DataFrame) -> None:
    _section("Rule Experiment Table")
    if rule_df.empty:
        print("  No rules to report.")
        return

    print(
        f"\n  {'rule':<40}  {'kept':>5}  {'rmvd':>5}  "
        f"{'avg4h':>7}  {'med4h':>7}  {'sr4h':>6}  {'fr4h':>6}  {'adv4h':>7}  "
        f"{'avg24h':>7}  {'sr24h':>6}  {'fr24h':>6}"
    )
    print("  " + "─" * 120)

    for _, row in rule_df.iterrows():
        print(
            f"  {row['rule']:<40}  {int(row['rows_kept']):>5}  {int(row['rows_removed']):>5}  "
            f"{_fmt(row['avg_future_ret_4h_pct']):>7}  "
            f"{_fmt(row['median_future_ret_4h_pct']):>7}  "
            f"{_fmt(row['success_rate_4h'], decimals=3):>6}  "
            f"{_fmt(row['failure_rate_4h'], decimals=3):>6}  "
            f"{_fmt(row['avg_max_adverse_4h_pct']):>7}  "
            f"{_fmt(row['avg_future_ret_24h_pct']):>7}  "
            f"{_fmt(row['success_rate_24h'], decimals=3):>6}  "
            f"{_fmt(row['failure_rate_24h'], decimals=3):>6}"
        )


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------


def write_trait_summary(flag_df: pd.DataFrame, path: Path = TRAIT_SUMMARY_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flag_df.to_csv(path, index=False)
    print(f"\n  -> Trait summary written to {path}")


def write_rule_summary(rule_df: pd.DataFrame, path: Path = RULE_SUMMARY_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rule_df.to_csv(path, index=False)
    print(f"  -> Rule experiment summary written to {path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(
    outcomes_path: Path = DEFAULT_OUTCOMES_PATH,
    trait_summary_path: Path = TRAIT_SUMMARY_PATH,
    rule_summary_path: Path = RULE_SUMMARY_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full analysis. Returns (flag_summary_df, rule_summary_df)."""

    print("\n" + "=" * 70)
    print("  Memecoin Signal Trait Analyser -- LONG_EXPLOSION")
    print("=" * 70)
    print()
    print("  *** This is exploratory analysis. Do not treat rules as validated")
    print("      until tested on future unseen snapshots. ***")
    print()

    df = load_long_explosion(outcomes_path)
    print(f"  Loaded {len(df)} LONG_EXPLOSION rows from {outcomes_path}")

    completed_4h = int(df["future_ret_4h_pct"].notna().sum())
    completed_24h = int(df["future_ret_24h_pct"].notna().sum())
    print(f"  Completed 4h outcomes : {completed_4h}")
    print(f"  Completed 24h outcomes: {completed_24h}")

    df = add_diagnostic_flags(df)

    print_feature_summaries(df)

    flag_df = build_flag_summary(df)
    print_flag_summary(flag_df)

    rule_df = build_rule_experiment_summary(df)
    print_rule_experiment_summary(rule_df)

    write_trait_summary(flag_df, trait_summary_path)
    write_rule_summary(rule_df, rule_summary_path)

    print()
    print("  Note: Optional future work: build backfill_recent_memecoin_signals.py")
    print("  to simulate missed snapshots using recent Kraken OHLC, limited by")
    print("  Kraken's 720-candle OHLC window.")
    print()
    print("=" * 70)

    return flag_df, rule_df


if __name__ == "__main__":
    main()
