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
import pymysql

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
    from myutils.implied_vola.adapters.pandas_python import compute_iv_on_dataframe
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
    from myutils.implied_vola.adapters.pandas_python import compute_iv_on_dataframe


def fetch_option_data(
    min_calls: pd.Timestamp,
    max_calls: pd.Timestamp,
    min_puts: pd.Timestamp,
    max_puts: pd.Timestamp,
    chunk_size: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Retrieve option snapshots for calls and puts and attach realized prices.

    Parameters
    ----------
    min_calls, max_calls, min_puts, max_puts:
        Date ranges for calls and puts tables.
    chunk_size:
        Number of rows to request per SQL query. Smaller sizes help avoid
        database timeouts on slow connections.

    Returns
    -------
    option_df:
        Combined DataFrame with call and put snapshots.
    realized_map:
        Series mapping quote_date to realized underlying prices.
    """

    try:
        calls = sql.fetch_snapshots_calls(min_calls, max_calls, chunk_size=chunk_size)
    except pymysql.MySQLError as exc:  # pragma: no cover - network operation
        print(f"Failed to fetch call snapshots: {exc}", flush=True)
        calls = pd.DataFrame()

    try:
        puts = sql.fetch_snapshots_puts(min_puts, max_puts, chunk_size=chunk_size)
    except pymysql.MySQLError as exc:  # pragma: no cover - network operation
        print(f"Failed to fetch put snapshots: {exc}", flush=True)
        puts = pd.DataFrame()

    if calls.empty or puts.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

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

    realized_map = (
        sql.attach_realized_from_self(calls)[["quote_date", "realized_underlying"]]
        .dropna()
        .drop_duplicates("quote_date")
        .set_index("quote_date")["realized_underlying"]
    )

    return option_df, realized_map


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
    chunk_size = int(os.getenv("SQL_CHUNK_SIZE", "50000"))
    option_df, realized_map = fetch_option_data(
        min_calls, max_calls, min_puts, max_puts, chunk_size
    )
    print("checkpoint 1", flush=True)

    if option_df.empty:
        print("No option snapshots fetched. Aborting.", flush=True)
        return

    # ------------------------------------------------------------------
    # 2) Merge risk-free rates and dividend yields
    # ------------------------------------------------------------------
    rf = load_risk_free_rates("DGS1MO.csv")
    div = load_dividend_yield("enriched_dividend_yield_2010_2023.csv")
    enriched = merge_rates_and_dividends(option_df, rf, div)
    print("checkpoint 2", flush=True)

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

    unified_calls = unified_calls.rename(columns={"dte_days": "dte"})
    unified_calls["is_call"] = True
    unified_calls = compute_iv_on_dataframe(
        unified_calls,
        column_map={
            "S": "underlying_last",
            "K": "strike",
            "DTE": "dte",
            "r": "risk_free_1m",
            "q": "div_yield_annual",
            "price": "mid_price",
            "is_call": "is_call",
        },
        dte_unit="days",
        out_iv_col="implied_volatility",
    )

    summary_df, artifacts = run_svi_over_all_days(unified_calls, verbose=False)
    print("checkpoint 3", flush=True)

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