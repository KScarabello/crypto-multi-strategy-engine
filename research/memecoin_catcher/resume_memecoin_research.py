"""Resume/catch-up command for the memecoin scanner research pipeline.

Evaluates outstanding outcomes for previously saved signals, then captures a
fresh scanner snapshot and evaluates that too.  Produces a before/after data-
state report so you can see exactly what changed.

This module is research-only.
- No trading.
- No orders.
- No private credentials.
- No cron changes.

Usage:
    python -m research.memecoin_catcher.resume_memecoin_research
    python -m research.memecoin_catcher.resume_memecoin_research --dry-run
    python -m research.memecoin_catcher.resume_memecoin_research --skip-fresh-snapshot
    python -m research.memecoin_catcher.resume_memecoin_research --skip-outcome-eval
    python -m research.memecoin_catcher.resume_memecoin_research --continue-on-error
"""

from __future__ import annotations

import argparse
import inspect
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

LOGGER = logging.getLogger(__name__)

DEFAULT_SIGNAL_HISTORY_PATH = Path("data/memecoin_signal_history.csv")
DEFAULT_OUTCOMES_PATH = Path("data/memecoin_signal_outcomes.csv")
LOG_PATH = Path("logs/memecoin_resume.log")

_ONE_DAY = timedelta(days=1)


# ---------------------------------------------------------------------------
# Data-state snapshot
# ---------------------------------------------------------------------------


def read_data_state(
    history_path: Path = DEFAULT_SIGNAL_HISTORY_PATH,
    outcomes_path: Path = DEFAULT_OUTCOMES_PATH,
) -> dict[str, Any]:
    """Return a dict describing the current state of the data files."""
    state: dict[str, Any] = {
        "history_rows": 0,
        "outcomes_rows": 0,
        "complete_15m": 0,
        "complete_1h": 0,
        "complete_4h": 0,
        "complete_24h": 0,
        "latest_snapshot_ts": None,
        "newest_signal_ts": None,
        "oldest_signal_ts": None,
    }

    if history_path.exists():
        try:
            hist = pd.read_csv(history_path, dtype=str)
            state["history_rows"] = len(hist)
            if "snapshot_ts_utc" in hist.columns:
                ts_series = pd.to_datetime(
                    hist["snapshot_ts_utc"], utc=True, errors="coerce"
                ).dropna()
                if not ts_series.empty:
                    state["newest_signal_ts"] = ts_series.max()
                    state["oldest_signal_ts"] = ts_series.min()
                    state["latest_snapshot_ts"] = ts_series.max()
        except Exception as exc:
            LOGGER.warning("Could not read history file %s: %s", history_path, exc)

    if outcomes_path.exists():
        try:
            out = pd.read_csv(outcomes_path, dtype=str)
            state["outcomes_rows"] = len(out)
            for col, key in (
                ("outcome_15m", "complete_15m"),
                ("outcome_1h", "complete_1h"),
                ("outcome_4h", "complete_4h"),
                ("outcome_24h", "complete_24h"),
            ):
                if col in out.columns:
                    state[key] = int((out[col].notna() & (out[col] != "")).sum())
        except Exception as exc:
            LOGGER.warning("Could not read outcomes file %s: %s", outcomes_path, exc)

    return state


def print_data_state(state: dict[str, Any], label: str = "Data state") -> None:
    """Print a formatted data-state block."""

    def _fmt(ts: Any) -> str:
        if ts is None:
            return "—"
        try:
            if hasattr(ts, "strftime"):
                return ts.strftime("%Y-%m-%dT%H:%M:%SZ")
            return str(ts)
        except Exception:
            return "—"

    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}")
    print(f"  signal history rows          : {state['history_rows']}")
    print(f"  outcome rows                 : {state['outcomes_rows']}")
    print(f"  completed 15m outcomes       : {state['complete_15m']}")
    print(f"  completed 1h  outcomes       : {state['complete_1h']}")
    print(f"  completed 4h  outcomes       : {state['complete_4h']}")
    print(f"  completed 24h outcomes       : {state['complete_24h']}")
    print(f"  latest snapshot ts           : {_fmt(state['latest_snapshot_ts'])}")
    print(f"  newest signal ts             : {_fmt(state['newest_signal_ts'])}")
    print(f"  oldest signal ts             : {_fmt(state['oldest_signal_ts'])}")
    print()


def check_downtime_warning(state: dict[str, Any], now: datetime | None = None) -> None:
    """Print a warning if the newest signal is more than 1 day old."""
    newest = state.get("newest_signal_ts")
    if newest is None:
        return
    _now = now if now is not None else datetime.now(tz=timezone.utc)
    try:
        if getattr(newest, "tzinfo", None) is None:
            newest = newest.replace(tzinfo=timezone.utc)
        delta = _now - newest
        if delta > _ONE_DAY:
            ts_str = newest.strftime("%Y-%m-%dT%H:%M:%SZ")
            print(
                f"\n⚠️  Scanner has not captured signals since {ts_str}. "
                "Missed snapshots are not recoverable unless a separate "
                "historical backfill is run."
            )
    except Exception as exc:
        LOGGER.debug("Could not compute downtime delta: %s", exc)


# ---------------------------------------------------------------------------
# Step runner
# ---------------------------------------------------------------------------


class StepResult:
    def __init__(
        self,
        name: str,
        success: bool,
        duration_s: float,
        skipped: bool = False,
        error: str = "",
    ) -> None:
        self.name = name
        self.success = success
        self.duration_s = duration_s
        self.skipped = skipped
        self.error = error

    def __str__(self) -> str:
        if self.skipped:
            status = "— skipped (dry-run)"
        elif self.success:
            status = "✓ OK"
        else:
            status = f"✗ FAILED: {self.error}"
        return f"  [{self.name}]  {self.duration_s:.1f}s  {status}"


def _run_step(
    name: str,
    fn: Callable[[], None],
    dry_run: bool,
    continue_on_error: bool,
) -> StepResult:
    print(f"\n→ Step: {name}")
    if dry_run:
        print("  (dry-run: skipping execution)")
        return StepResult(name=name, success=True, duration_s=0.0, skipped=True)

    t0 = time.monotonic()
    try:
        fn()
        elapsed = time.monotonic() - t0
        LOGGER.info("Step '%s' completed in %.1fs", name, elapsed)
        return StepResult(name=name, success=True, duration_s=elapsed)
    except Exception as exc:
        elapsed = time.monotonic() - t0
        LOGGER.exception("Step '%s' failed after %.1fs: %s", name, elapsed, exc)
        result = StepResult(
            name=name, success=False, duration_s=elapsed, error=str(exc)
        )
        if not continue_on_error:
            raise
        print(f"  ⚠  Step failed (--continue-on-error active): {exc}")
        return result


# ---------------------------------------------------------------------------
# Individual pipeline step functions (lazy imports to avoid import-time side
# effects and to allow the module to load even if optional deps are missing)
# ---------------------------------------------------------------------------


def _step_evaluate_outcomes(recompute: bool = True) -> None:
    from research.memecoin_catcher.evaluate_signal_outcomes import (
        run_outcome_evaluation,
    )

    sig = inspect.signature(run_outcome_evaluation)
    if "recompute" in sig.parameters:
        run_outcome_evaluation(recompute=recompute)
    else:
        LOGGER.info(
            "run_outcome_evaluation does not yet support recompute=; calling without it"
        )
        run_outcome_evaluation()


def _step_summarize_outcomes() -> None:
    try:
        from research.memecoin_catcher.summarize_signal_outcomes import (
            main as summarize_main,
        )
        summarize_main()
    except ImportError:
        LOGGER.warning(
            "summarize_signal_outcomes not found in this repo; skipping summary step"
        )
        print(
            "  (summarize_signal_outcomes not available in this repo — skipping)"
        )


def _step_fetch_universe() -> None:
    from research.memecoin_catcher.fetch_kraken_universe import (
        run_kraken_universe_scan,
    )
    run_kraken_universe_scan()


def _step_rank_candidates() -> None:
    from research.memecoin_catcher.rank_memecoin_candidates import (
        run_candidate_ranking,
    )
    run_candidate_ranking()


def _step_enrich_ohlc() -> None:
    from research.memecoin_catcher.enrich_candidates_with_ohlc import (
        main as enrich_main,
    )
    enrich_main()


def _step_save_snapshot() -> None:
    from research.memecoin_catcher.save_signal_snapshot import (
        run_signal_snapshot,
    )
    run_signal_snapshot()


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def run_resume(
    skip_fresh_snapshot: bool = False,
    skip_outcome_eval: bool = False,
    continue_on_error: bool = False,
    dry_run: bool = False,
    history_path: Path = DEFAULT_SIGNAL_HISTORY_PATH,
    outcomes_path: Path = DEFAULT_OUTCOMES_PATH,
    _now: datetime | None = None,
) -> list[StepResult]:
    """Run the full resume pipeline and return a list of StepResult objects."""
    start_ts = _now if _now is not None else datetime.now(tz=timezone.utc)
    t0_total = time.monotonic()

    print(f"\n{'═' * 60}")
    print("  Memecoin Research Resume")
    print(f"  Started : {start_ts.strftime('%Y-%m-%dT%H:%M:%SZ')} UTC")
    if dry_run:
        print("  Mode    : DRY-RUN (no files will be modified)")
    print(f"{'═' * 60}")

    state_before = read_data_state(history_path, outcomes_path)
    print_data_state(state_before, label="Data state BEFORE")
    check_downtime_warning(state_before, now=start_ts)

    print("\n  Planned steps:")
    if not skip_outcome_eval:
        print("    A  Evaluate existing signal outcomes (recompute=True)")
        print("    B  Summarize existing outcomes")
    if not skip_fresh_snapshot:
        print("    C1 Fetch Kraken universe")
        print("    C2 Rank memecoin candidates")
        print("    C3 Enrich candidates with OHLC")
        print("    C4 Save signal snapshot")
    if not skip_outcome_eval:
        print("    D  Evaluate outcomes (includes new snapshot)")
        print("    E  Summarize final state")
    if skip_outcome_eval and skip_fresh_snapshot:
        print("    (nothing to do — both --skip flags active)")

    print(
        "\n  Note: Optional future work: build backfill_recent_memecoin_signals.py"
        " to simulate missed snapshots using recent Kraken OHLC, limited by"
        " Kraken's 720-candle OHLC window."
    )

    results: list[StepResult] = []

    if not skip_outcome_eval:
        results.append(
            _run_step(
                "A: Evaluate existing signal outcomes",
                lambda: _step_evaluate_outcomes(recompute=True),
                dry_run=dry_run,
                continue_on_error=continue_on_error,
            )
        )
        results.append(
            _run_step(
                "B: Summarize existing outcomes",
                _step_summarize_outcomes,
                dry_run=dry_run,
                continue_on_error=continue_on_error,
            )
        )

    if not skip_fresh_snapshot:
        for step_name, step_fn in (
            ("C1: Fetch Kraken universe", _step_fetch_universe),
            ("C2: Rank memecoin candidates", _step_rank_candidates),
            ("C3: Enrich candidates with OHLC", _step_enrich_ohlc),
            ("C4: Save signal snapshot", _step_save_snapshot),
        ):
            results.append(
                _run_step(
                    step_name,
                    step_fn,
                    dry_run=dry_run,
                    continue_on_error=continue_on_error,
                )
            )

    if not skip_outcome_eval:
        results.append(
            _run_step(
                "D: Evaluate outcomes (post-snapshot)",
                lambda: _step_evaluate_outcomes(recompute=False),
                dry_run=dry_run,
                continue_on_error=continue_on_error,
            )
        )
        results.append(
            _run_step(
                "E: Summarize final state",
                _step_summarize_outcomes,
                dry_run=dry_run,
                continue_on_error=continue_on_error,
            )
        )

    if not dry_run:
        state_after = read_data_state(history_path, outcomes_path)
        print_data_state(state_after, label="Data state AFTER")

    total_elapsed = time.monotonic() - t0_total

    print(f"\n{'─' * 60}")
    print("  Step Results")
    print(f"{'─' * 60}")
    for r in results:
        print(r)

    n_failed = sum(1 for r in results if not r.success)
    print(
        f"\n  Total: {len(results)} steps, {n_failed} failed,"
        f" {total_elapsed:.1f}s elapsed"
    )
    print(f"{'═' * 60}\n")

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume/catch-up command for the memecoin scanner research pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--skip-fresh-snapshot",
        action="store_true",
        help="Only evaluate/summarize old saved signals; skip scanner steps.",
    )
    parser.add_argument(
        "--skip-outcome-eval",
        action="store_true",
        help="Only run a fresh scanner snapshot; skip evaluator/summarizer steps.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running steps after nonfatal failures.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned steps and data-state summary without modifying files.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
        ],
    )

    args = _parse_args(argv)
    run_resume(
        skip_fresh_snapshot=args.skip_fresh_snapshot,
        skip_outcome_eval=args.skip_outcome_eval,
        continue_on_error=args.continue_on_error,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
