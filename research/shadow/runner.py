"""Main entry point: process one or more bars for all shadow candidates."""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from research.shadow.specs import (
    FROZEN_CANDIDATES,
    SPEC_HASHES,
    PROSPECTIVE_START,
    SHADOW_INITIAL_CAPITAL,
    assert_all_spec_integrity,
)
from research.shadow.portfolio import (
    ShadowPortfolio,
    init_portfolio,
    load_portfolio,
    save_portfolio,
    compute_state_hash,
)
from research.shadow.signal_runner import BarDecision, run_bar
from research.shadow.ledger import (
    append_decision,
    get_last_decision_ts,
    is_bar_already_processed,
)
from research.shadow.monitor import generate_monitor_report

DEFAULT_STATE_DIR = Path("shadow_state")
DEFAULT_DATA_DIR = Path("data/local")


def _ensure_state_dirs(state_dir: Path) -> None:
    for sub in ["portfolios", "ledger", "decisions"]:
        (state_dir / sub).mkdir(parents=True, exist_ok=True)


def _load_or_init_portfolio(
    name: str,
    state_dir: Path,
    bar_ts: pd.Timestamp,
    dry_run: bool = False,
) -> ShadowPortfolio:
    portfolio = load_portfolio(name, state_dir)
    if portfolio is None:
        portfolio = init_portfolio(
            name=name,
            spec_hash=SPEC_HASHES[name],
            started_at=bar_ts.isoformat(),
            initial_capital=SHADOW_INITIAL_CAPITAL,
        )
    return portfolio


def process_bar(
    bar_ts: pd.Timestamp,
    close: pd.DataFrame,
    state_dir: Path,
    data_dir: Path,
    candidates: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, BarDecision]:
    """Process one bar for all specified candidates."""
    assert_all_spec_integrity()
    _ensure_state_dirs(state_dir)

    candidate_list = candidates if candidates is not None else list(FROZEN_CANDIDATES.keys())
    decisions: dict[str, BarDecision] = {}

    for name in candidate_list:
        if name not in FROZEN_CANDIDATES:
            print(f"[WARN] Unknown candidate {name!r}, skipping.", file=sys.stderr)
            continue

        if is_bar_already_processed(name, str(bar_ts), state_dir) and not dry_run:
            print(f"[SKIP] {name}: bar {bar_ts} already processed.")
            continue

        portfolio = _load_or_init_portfolio(name, state_dir, bar_ts, dry_run)

        # Apply previous bar's pending decision at current bar prices
        new_equity = portfolio.equity
        actual_return = 0.0

        if portfolio.pending_decision is not None:
            prev_holdings = portfolio.pending_decision.get("target_holdings", {})
            total_return = 0.0
            for sym, weight in prev_holdings.items():
                if sym in close.columns and bar_ts in close.index:
                    prev_ts_str = portfolio.pending_decision.get("decision_ts")
                    if prev_ts_str:
                        prev_ts = pd.Timestamp(prev_ts_str)
                        if prev_ts in close.index:
                            prev_price = float(close.loc[prev_ts, sym])
                            curr_price = float(close.loc[bar_ts, sym])
                            if prev_price > 0 and not pd.isna(prev_price) and not pd.isna(curr_price):
                                sym_return = (curr_price / prev_price) - 1.0
                                total_return += weight * sym_return

            new_equity = portfolio.equity * (1 + total_return)

            # Apply transaction costs if rebalanced at previous bar
            regime = FROZEN_CANDIDATES[name].get("regime")
            if regime and portfolio.pending_decision.get("should_rebalance"):
                fee_bps = regime.get("fee_bps", 10)
                slippage_bps = regime.get("slippage_bps", 5)
                for trade in portfolio.pending_decision.get("hypothetical_trades", []):
                    notional = trade.get("notional_estimate", 0)
                    cost = notional * (fee_bps + slippage_bps) / 10_000.0
                    new_equity -= cost
                    portfolio.total_fee_dollars += trade.get("fee_est", 0)
                    portfolio.total_slippage_dollars += trade.get("slip_est", 0)
                    portfolio.total_executed_notional += notional

            actual_return = total_return
            portfolio.holdings = dict(prev_holdings)
            portfolio.cash_weight = max(0.0, 1.0 - sum(prev_holdings.values()))

        portfolio.equity = new_equity
        portfolio.equity_history.append({"ts": str(bar_ts), "equity": new_equity})

        prev_portfolio_snapshot = ShadowPortfolio(**{**asdict(portfolio)})

        # Generate new signal for this bar
        decision = run_bar(
            candidate_name=name,
            bar_ts=bar_ts,
            close=close,
            portfolio=portfolio,
            spec_hash=SPEC_HASHES[name],
        )

        portfolio.bar_count += 1
        if decision.should_rebalance:
            portfolio.rebalance_count += 1
            portfolio.holdings = dict(decision.target_holdings)
            portfolio.cash_weight = max(0.0, 1.0 - sum(decision.target_holdings.values()))

        # Track regime transitions
        if portfolio.pending_decision is not None:
            prev_gate = portfolio.pending_decision.get("btc_regime_state", portfolio.last_gate)
            if prev_gate != decision.btc_regime_state:
                portfolio.regime_transitions += 1

        portfolio.last_updated_at = str(bar_ts)
        portfolio.pending_decision = {
            "decision_ts": decision.decision_ts,
            "execution_ts": decision.execution_ts,
            "target_holdings": decision.target_holdings,
            "should_rebalance": decision.should_rebalance,
            "hypothetical_trades": decision.hypothetical_trades,
            "btc_regime_state": decision.btc_regime_state,
        }

        if not dry_run:
            save_portfolio(portfolio, state_dir)
            close_last_row = None
            if bar_ts in close.index:
                close_last_row = close.loc[bar_ts].to_dict()
            append_decision(
                candidate_name=name,
                decision=decision,
                prev_portfolio=prev_portfolio_snapshot,
                new_equity=new_equity,
                actual_return=actual_return,
                state_dir=state_dir,
                close_last_row=close_last_row,
            )
        else:
            print(f"[DRY-RUN] {name}: bar={bar_ts}, target={decision.target_holdings}, "
                  f"regime={'ON' if decision.btc_regime_state else 'OFF'}, "
                  f"equity=${new_equity:.2f}")

        decisions[name] = decision

    return decisions


def _load_close_data(data_dir: Path) -> pd.DataFrame | None:
    """Try to load close price data from data_dir."""
    for pattern in ["close.parquet", "close.csv", "ohlcv.parquet", "prices.parquet"]:
        path = data_dir / pattern
        if path.exists():
            if path.suffix == ".parquet":
                return pd.read_parquet(path)
            else:
                return pd.read_csv(path, index_col=0, parse_dates=True)

    # Try per-symbol files
    close_frames = {}
    for sym in ["BTC/USD", "ETH/USD", "XRP/USD", "SOL/USD", "AVAX/USD"]:
        sym_clean = sym.replace("/", "_")
        for pattern in [f"{sym_clean}.parquet", f"{sym_clean}.csv"]:
            path = data_dir / pattern
            if path.exists():
                if path.suffix == ".parquet":
                    df = pd.read_parquet(path)
                else:
                    df = pd.read_csv(path, index_col=0, parse_dates=True)
                if "close" in df.columns:
                    close_frames[sym] = df["close"]
                elif "Close" in df.columns:
                    close_frames[sym] = df["Close"]
                break

    if close_frames:
        return pd.DataFrame(close_frames)

    return None


def run_from_start(
    prospective_start: str,
    data_dir: Path,
    state_dir: Path,
    dry_run: bool = False,
) -> None:
    """Process all bars from prospective_start to the latest available data."""
    close = _load_close_data(data_dir)
    if close is None:
        print(f"[INFO] No close data found in {data_dir}. Cannot backfill.")
        return

    start_ts = pd.Timestamp(prospective_start)
    available_bars = close.index[close.index >= start_ts]

    if len(available_bars) == 0:
        print(f"[INFO] No bars found from {prospective_start} onward.")
        return

    print(f"[INFO] Processing {len(available_bars)} bars from {available_bars[0]} to {available_bars[-1]}")

    for bar_ts in available_bars:
        process_bar(
            bar_ts=bar_ts,
            close=close,
            state_dir=state_dir,
            data_dir=data_dir,
            dry_run=dry_run,
        )


def run_latest_bar(
    data_dir: Path,
    state_dir: Path,
    dry_run: bool = False,
) -> dict[str, BarDecision]:
    """Process only the most recent unprocessed bar."""
    close = _load_close_data(data_dir)
    if close is None:
        print(f"[INFO] No close data found in {data_dir}.")
        return {}

    last_processed = None
    for name in FROZEN_CANDIDATES:
        ts = get_last_decision_ts(name, state_dir)
        if ts is not None:
            if last_processed is None or ts < last_processed:
                last_processed = ts

    if last_processed is not None:
        unprocessed = close.index[close.index > last_processed]
    else:
        unprocessed = close.index

    if len(unprocessed) == 0:
        print("[INFO] No unprocessed bars found.")
        return {}

    bar_ts = unprocessed[-1]
    return process_bar(
        bar_ts=bar_ts,
        close=close,
        state_dir=state_dir,
        data_dir=data_dir,
        dry_run=dry_run,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Shadow validation runner for crypto momentum strategy."
    )
    parser.add_argument("--bar", type=str, default=None, help="Process exactly one bar (ISO timestamp)")
    parser.add_argument("--from-start", action="store_true", help="Process all bars since PROSPECTIVE_START")
    parser.add_argument("--start", type=str, default=None, help="Override prospective start for backfill")
    parser.add_argument("--report", action="store_true", help="Generate monitoring report only")
    parser.add_argument("--dry-run", action="store_true", help="Print what would happen, don't update state")
    parser.add_argument("--data-dir", type=str, default="data/local", help="Data directory")
    parser.add_argument("--state-dir", type=str, default="shadow_state", help="Shadow state directory")
    args = parser.parse_args()

    state_dir = Path(args.state_dir)
    data_dir = Path(args.data_dir)

    _ensure_state_dirs(state_dir)

    if args.report:
        output_path = Path("reports/shadow_monitor_report.md")
        report = generate_monitor_report(state_dir, output_path)
        print(report)
        return

    if args.bar:
        bar_ts = pd.Timestamp(args.bar)
        close = _load_close_data(data_dir)
        if close is None:
            print(f"[WARN] No close data found in {data_dir}. Creating synthetic data for dry-run.")
            if args.dry_run:
                import numpy as np
                universe = ["BTC/USD", "ETH/USD", "XRP/USD", "SOL/USD", "AVAX/USD"]
                idx = pd.date_range(end=bar_ts, periods=300, freq="4h", tz="UTC")
                rng = np.random.default_rng(42)
                data = {}
                prices = {"BTC/USD": 65000, "ETH/USD": 3500, "XRP/USD": 0.6, "SOL/USD": 150, "AVAX/USD": 35}
                for sym in universe:
                    p = prices[sym]
                    returns = rng.normal(0.0002, 0.02, len(idx))
                    price_series = p * (1 + returns).cumprod()
                    data[sym] = price_series
                close = pd.DataFrame(data, index=idx)
            else:
                print("[ERROR] No close data and not in dry-run mode. Exiting.")
                sys.exit(1)
        process_bar(
            bar_ts=bar_ts,
            close=close,
            state_dir=state_dir,
            data_dir=data_dir,
            dry_run=args.dry_run,
        )

    elif args.from_start:
        start = args.start if args.start else PROSPECTIVE_START
        run_from_start(
            prospective_start=start,
            data_dir=data_dir,
            state_dir=state_dir,
            dry_run=args.dry_run,
        )

    else:
        print("=== SHADOW VALIDATION FRAMEWORK READY ===")
        print()
        print("Frozen candidate definitions:")
        descs = {
            "control":                "strict top-3 replacement",
            "min_hold_6":             "min 6-rebalance hold (~12 days)",
            "rank_buffer_4":          "retain while rank<=4",
            "combo_buf4_conf2_hold2": "rank_buffer=4 + confirm=2 + hold=2",
            "btc_buyhold":            "BTC benchmark",
            "ewb_buyhold":            "EW5 benchmark",
        }
        for name, h in SPEC_HASHES.items():
            desc = descs.get(name, "")
            print(f"  {name:<28} hash={h}  {desc}")
        print()
        print(f"Proposed prospective start: {PROSPECTIVE_START}")
        print(f"Shadow state directory: {state_dir}/")
        print(f"Report path: reports/shadow_monitor_report.md")
        print()
        print("Commands:")
        print("  Process one bar:")
        print("    .venv/bin/python -m research.shadow.runner --bar 2026-06-09T04:00:00+00:00")
        print("  Backfill from start:")
        print("    .venv/bin/python -m research.shadow.runner --from-start")
        print("  View results:")
        print("    .venv/bin/python -m research.shadow.monitor")
        print("  Verify hashes:")
        print("    .venv/bin/python -m research.shadow.specs")
        print()
        print("No live behavior changed. No Kraken orders. No account state modified.")


if __name__ == "__main__":
    main()
