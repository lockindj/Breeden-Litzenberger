"""Executable pipeline from SQL data sourcing to PIT diagram."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# --- Make imports work both as a script and as a package ---------------------
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sql

# Try absolute imports; if they fail, adjust path (already added) and retry.
try:
    from data_loading import (
        load_risk_free_rates,
        load_dividend_yield,
        merge_rates_and_dividends,
    )
    from svi import build_unified_calls_dataframe_external, run_svi_over_all_days
    from analysis import compute_bl_pdfs_for_all_days, prepare_pit_df
except ImportError:
    # If you're running this from a different working dir, ensure project root is on sys.path
    project_root = _THIS_DIR
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from data_loading import (
        load_risk_free_rates,
        load_dividend_yield,
        merge_rates_and_dividends,
    )
    from svi import build_unified_calls_dataframe_external, run_svi_over_all_days
    from analysis import compute_bl_pdfs_for_all_days, prepare_pit_df


def plot_pit_diagram(pit_values: pd.Series) -> None:
    """Plot a PIT diagram from sorted PIT values."""
    pit_sorted = np.sort(pit_values.to_numpy())
    n = len(pit_sorted)
    if n == 0:
        print("No PIT values to plot.", flush=True)
        return
    uniform_q = (np.arange(1, n + 1) - 0.5) / n
    plt.figure(figsize=(6, 6))
    plt.plot(uniform_q, pit_sorted, marker="o", linestyle="", label="Empirical PIT")
    plt.plot([0, 1], [0, 1], "--", color="grey", label="Uniform")
    plt.xlabel("Uniform quantiles")
    plt.ylabel("PIT quantiles")
    plt.title("PIT diagram")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main() -> None:
    """Run the full Breeden–Litzenberger pipeline and plot PIT diagram."""
    # ------------------------------------------------------------------
    # 0) Resolve date ranges from DB (no heavy work at import)
    # ------------------------------------------------------------------
    min_calls, max_calls = sql.get_min_max_dates("calls_table")
    min_puts, max_puts = sql.get_min_max_dates("puts_table")

    if pd.isna(min_calls) or pd.isna(max_calls):
        print("calls_table has no data (min/max are NaT). Aborting.", flush=True)
        return
    if pd.isna(min_puts) or pd.isna(max_puts):
        print("puts_table has no data (min/max are NaT). Aborting.", flush=True)
        return

    # ------------------------------------------------------------------
    # 1) Fetch option snapshots from SQL and combine calls/puts
    # ------------------------------------------------------------------
    calls = sql.fetch_snapshots_calls(min_calls, max_calls)
    print("checkpoint 1", flush=True)

    puts = sql.fetch_snapshots_puts(min_puts, max_puts)
    print("checkpoint 2", flush=True)

    if calls.empty:
        print("No call snapshots fetched. Aborting.", flush=True)
        return
    if puts.empty:
        print("No put snapshots fetched. Aborting.", flush=True)
        return

    # Normalize columns; computing IV in SQL was removed, renames are safe.
    calls = calls.rename(
        columns={
            "c_bid": "bid",
            "c_ask": "ask",
            "c_last": "last",
            "c_iv": "implied_volatility",
        }
    )
    calls["is_call"] = True

    puts = puts.rename(
        columns={
            "p_bid": "bid",
            "p_ask": "ask",
            "p_last": "last",
            "p_iv": "implied_volatility",
        }
    )
    puts["is_call"] = False

    option_df = pd.concat([calls, puts], ignore_index=True)

    # Attach realized prices from call snapshots (one per day)
    realized_map = (
        sql.attach_realized_from_self(calls)[["quote_date", "realized_underlying"]]
        .dropna()
        .drop_duplicates("quote_date")
        .set_index("quote_date")["realized_underlying"]
    )

    # ------------------------------------------------------------------
    # 2) Merge risk-free rates and dividend yields
    # ------------------------------------------------------------------
    rf = load_risk_free_rates("DGS1MO.csv")
    div = load_dividend_yield("enriched_dividend_yield_2010_2023.csv")
    enriched = merge_rates_and_dividends(option_df, rf, div)
    print("checkpoint 3", flush=True)

    if enriched.empty:
        print("Enriched dataframe is empty after merging rates/dividends. Aborting.", flush=True)
        return

    # ------------------------------------------------------------------
    # 3) Build unified call curves and run SVI calibration
    # ------------------------------------------------------------------
    unified_calls = build_unified_calls_dataframe_external(enriched, verbose=False)

    if unified_calls.empty:
        print("Unified calls dataframe is empty. Aborting.", flush=True)
        return

    summary_df, artifacts = run_svi_over_all_days(unified_calls, verbose=False)
    print("checkpoint 4", flush=True)

    if summary_df.empty:
        print("SVI summary is empty. Aborting.", flush=True)
        return

    # ------------------------------------------------------------------
    # 4) Compute BL PDFs and prepare PIT values
    # ------------------------------------------------------------------
    bl_results, mass_summary = compute_bl_pdfs_for_all_days(artifacts, summary_df)

    pit_df = prepare_pit_df(
        bl_results, mass_summary, threshold=0.9, realized_prices=realized_map
    )

    if pit_df.empty:
        print("No PIT values were computed – check data inputs.", flush=True)
        return

    # ------------------------------------------------------------------
    # 5) Plot PIT diagram
    # ------------------------------------------------------------------
    plot_pit_diagram(pit_df["PIT_value"])


if __name__ == "__main__":  # pragma: no cover
    main()
