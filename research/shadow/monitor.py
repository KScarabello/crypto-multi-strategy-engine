"""Monitoring report generator — reads ledger, writes markdown."""
from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from research.shadow.specs import FROZEN_CANDIDATES, SPEC_HASHES, PROSPECTIVE_START
from research.shadow.ledger import load_ledger
from research.shadow.portfolio import load_portfolio


def _max_drawdown(equity_curve: list[float]) -> float:
    if len(equity_curve) < 2:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for v in equity_curve:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _sharpe(returns: list[float], bars_per_year: int = 2190) -> float:
    """Annualised Sharpe ratio assuming 4h bars."""
    if len(returns) < 2:
        return float("nan")
    arr = np.array(returns, dtype=float)
    mean = arr.mean()
    std = arr.std(ddof=1)
    if std == 0:
        return float("nan")
    return float(mean / std * math.sqrt(bars_per_year))


def _candidate_stats(records: list[dict]) -> dict:
    if not records:
        return {}
    equities = [r["new_equity"] for r in records]
    returns_pct = [r["actual_return_pct"] for r in records]
    returns_dec = [r / 100.0 for r in returns_pct]

    total_return = (equities[-1] / equities[0] - 1.0) * 100.0 if equities[0] > 0 else 0.0
    max_dd = _max_drawdown(equities) * 100.0
    sharpe = _sharpe(returns_dec)
    rebalances = sum(1 for r in records if r.get("should_rebalance"))
    all_trades = [t for r in records for t in r.get("hypothetical_trades", [])]
    total_notional = sum(t.get("notional_estimate", 0) for t in all_trades)
    total_fees = sum(t.get("fee_est", 0) + t.get("slip_est", 0) for t in all_trades)
    regime_on = sum(1 for r in records if r.get("btc_regime_state"))

    return {
        "n_bars": len(records),
        "start_equity": equities[0],
        "end_equity": equities[-1],
        "total_return_pct": total_return,
        "max_dd_pct": max_dd,
        "sharpe": sharpe,
        "rebalances": rebalances,
        "total_notional": total_notional,
        "total_cost": total_fees,
        "regime_on_pct": 100.0 * regime_on / len(records) if records else 0.0,
    }


def generate_monitor_report(
    state_dir: Path,
    output_path: Path,
    min_observations_for_sharpe: int = 30,
) -> str:
    """Generate a markdown monitoring report from ledger records."""
    now = datetime.now(timezone.utc).isoformat()
    lines = []
    lines.append("# Shadow Validation Monitor Report")
    lines.append(f"\n**Generated:** {now}")
    lines.append(f"**Prospective start:** {PROSPECTIVE_START}")
    lines.append("")

    all_stats = {}
    for name in FROZEN_CANDIDATES:
        records = load_ledger(name, state_dir)
        stats = _candidate_stats(records)
        all_stats[name] = stats

    # Section 1: Status
    lines.append("## 1. Shadow Validation Status")
    min_bars = min((s.get("n_bars", 0) for s in all_stats.values()), default=0)
    max_bars = max((s.get("n_bars", 0) for s in all_stats.values()), default=0)
    lines.append(f"- Bars processed (min/max across candidates): {min_bars} / {max_bars}")
    if max_bars < min_observations_for_sharpe:
        lines.append(f"\n> ⚠️ **IMMATURE DATA**: Only {max_bars} observations. "
                     f"Need ≥{min_observations_for_sharpe} for reliable statistics.")
    lines.append("")

    # Section 2: Performance table
    lines.append("## 2. Per-Candidate Performance")
    lines.append("")
    header = "| Candidate | Bars | Return% | MaxDD% | Sharpe | Rebalances | Fees+Slip$ | Regime On% |"
    sep = "|-----------|------|---------|--------|--------|------------|------------|------------|"
    lines.append(header)
    lines.append(sep)

    for name, stats in all_stats.items():
        if not stats:
            lines.append(f"| {name} | — | — | — | — | — | — | — |")
            continue
        n = stats["n_bars"]
        ret = f"{stats['total_return_pct']:.2f}%"
        dd = f"{stats['max_dd_pct']:.2f}%"
        sharpe_val = stats["sharpe"]
        if n < min_observations_for_sharpe or math.isnan(sharpe_val):
            sharpe_str = f"*{sharpe_val:.2f}*" if not math.isnan(sharpe_val) else "—"
        else:
            sharpe_str = f"{sharpe_val:.2f}"
        reb = str(stats["rebalances"])
        costs = f"${stats['total_cost']:.2f}"
        reg = f"{stats['regime_on_pct']:.1f}%"
        lines.append(f"| {name} | {n} | {ret} | {dd} | {sharpe_str} | {reb} | {costs} | {reg} |")

    lines.append("")
    if min_bars < min_observations_for_sharpe:
        lines.append(f"> *Sharpe values in italics are statistically immature (< {min_observations_for_sharpe} observations).*")
    lines.append("")

    # Section 3: Regime summary
    lines.append("## 3. Regime State Summary")
    for name in ["control", "min_hold_6", "rank_buffer_4", "combo_buf4_conf2_hold2"]:
        portfolio = load_portfolio(name, state_dir)
        if portfolio:
            gate = "risk-ON" if portfolio.last_gate else "risk-OFF"
            lines.append(f"- **{name}**: current gate={gate}, transitions={portfolio.regime_transitions}, rebalances={portfolio.rebalance_count}")
    lines.append("")

    # Section 4: Delta vs control
    lines.append("## 4. Candidate Delta vs Control")
    ctrl_stats = all_stats.get("control", {})
    if ctrl_stats:
        ctrl_ret = ctrl_stats.get("total_return_pct", 0.0)
        lines.append("")
        lines.append("| Candidate | Return% | Delta vs Control |")
        lines.append("|-----------|---------|-----------------|")
        for name, stats in all_stats.items():
            if name == "control" or not stats:
                continue
            ret = stats.get("total_return_pct", 0.0)
            delta = ret - ctrl_ret
            sign = "+" if delta >= 0 else ""
            lines.append(f"| {name} | {ret:.2f}% | {sign}{delta:.2f}% |")
    lines.append("")

    # Section 5: BTC benchmark comparison
    lines.append("## 5. Benchmark Comparison")
    btc_stats = all_stats.get("btc_buyhold", {})
    ewb_stats = all_stats.get("ewb_buyhold", {})
    if btc_stats:
        lines.append(f"- BTC buy-and-hold: {btc_stats.get('total_return_pct', 0.0):.2f}%")
    if ewb_stats:
        lines.append(f"- EW5 buy-and-hold: {ewb_stats.get('total_return_pct', 0.0):.2f}%")
    lines.append("")

    # Spec hashes
    lines.append("## 6. Frozen Spec Hashes")
    for name, h in SPEC_HASHES.items():
        lines.append(f"- `{name}`: `{h}`")
    lines.append("")

    report = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    return report


if __name__ == "__main__":
    from pathlib import Path
    state_dir = Path("shadow_state")
    output_path = Path("reports/shadow_monitor_report.md")
    report = generate_monitor_report(state_dir, output_path)
    print(report)
