"""Preprocessing utilities for option data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from math import exp, isfinite


def _to_bool_is_call(series: pd.Series) -> pd.Series:
    """Return a boolean option type series.

    Various representations (``'C'``, ``'P'``, ``1``, ``0`` etc.) are coerced to
    ``True`` for calls and ``False`` for puts.  Missing or unrecognized values are
    returned as ``NaN`` and subsequently cast to ``bool``.
    """

    if series.dtype == bool or str(series.dtype) == "boolean":
        return series.astype(bool)

    def cast(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (bool, np.bool_)):
            return bool(x)
        if isinstance(x, (int, np.integer, float, np.floating)):
            return bool(int(x))
        if isinstance(x, str):
            t = x.strip().upper()
            if t in ("C", "CALL", "1", "T", "TRUE"):
                return True
            if t in ("P", "PUT", "0", "F", "FALSE"):
                return False
        return np.nan

    return series.apply(cast).astype("boolean").astype(bool)


def _mid_from_bid_ask_last(
    df: pd.DataFrame, bid_col: str = "bid", ask_col: str = "ask", last_col: str = "last"
) -> np.ndarray:
    """Return mid prices using bid/ask or last as fallback."""

    has_ba = df[[bid_col, ask_col]].notna().all(axis=1)
    return np.where(
        has_ba,
        (pd.to_numeric(df[bid_col], errors="coerce") + pd.to_numeric(df[ask_col], errors="coerce")) / 2.0,
        pd.to_numeric(df.get(last_col, np.nan), errors="coerce"),
    )


def _unified_calls_for_day_external(df_day: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    """Construct a parity-consistent call curve for one day.

    Parameters
    ----------
    df_day:
        DataFrame containing raw option quotes for a single day.  Expected
        columns include ``quote_date``, ``underlying_last``, ``dte``, ``strike``,
        ``bid``, ``ask``, ``last``, ``div_yield_annual`` and ``risk_free_1m``.
    debug:
        If ``True`` print diagnostic information.

    Returns
    -------
    pandas.DataFrame
        Standardised call quotes with columns required by IV calibration.
        The output columns are ``quote_date``, ``underlying_last``,
        ``dte_days``, ``strike``, ``risk_free_1m``, ``div_yield_annual``,
        ``is_call`` and ``mid_price``.
    """

    # Coerce numerics
    num_cols = [
        "underlying_last",
        "dte",
        "strike",
        "bid",
        "ask",
        "last",
        "div_yield_annual",
        "risk_free_1m",
    ]
    for c in num_cols:
        if c in df_day.columns:
            df_day[c] = pd.to_numeric(df_day[c], errors="coerce")

    df_day = df_day.dropna(subset=["quote_date", "underlying_last", "dte", "strike"])
    if df_day.empty:
        if debug:
            print("  -> empty after basic clean (missing quote_date/underlying_last/dte/strike)")
        return pd.DataFrame()

    qdate = pd.to_datetime(df_day["quote_date"].iloc[0]).date()
    S = float(df_day["underlying_last"].iloc[0])
    dte_days = float(np.median(df_day["dte"]))
    T_yrs = dte_days / 365.0
    r = float(np.median(pd.to_numeric(df_day["risk_free_1m"], errors="coerce").dropna()))
    q = float(np.median(pd.to_numeric(df_day["div_yield_annual"], errors="coerce").dropna()))

    if not (isfinite(S) and isfinite(T_yrs) and T_yrs > 0 and isfinite(r) and isfinite(q)):
        if debug:
            print(f"  -> bad snapshot S/T/r/q: S={S}, T_yrs={T_yrs}, r={r}, q={q}")
        return pd.DataFrame()

    if r > 1.0:
        r /= 100.0
    if q > 1.0:
        q /= 100.0

    D = exp(-r * T_yrs)
    F = S * exp((r - q) * T_yrs)

    df_day["is_call"] = _to_bool_is_call(df_day["is_call"])
    calls = df_day.loc[df_day["is_call"] == True].copy()
    puts = df_day.loc[df_day["is_call"] == False].copy()

    calls["c_mid"] = _mid_from_bid_ask_last(calls, "bid", "ask", "last")
    puts["p_mid"] = _mid_from_bid_ask_last(puts, "bid", "ask", "last")

    if debug:
        print(f"  S={S:.4f}, T={T_yrs:.5f}y, r={r:.6f}, q={q:.6f}, F={F:.4f}")
        print(f"  counts: calls={len(calls)}, puts={len(puts)}")
        print(
            f"  mids>0: c_mid={(np.nan_to_num(calls['c_mid'])>0).sum()}, p_mid={(np.nan_to_num(puts['p_mid'])>0).sum()}"
        )

    grid = (
        pd.merge(
            calls[["strike", "c_mid"]],
            puts[["strike", "p_mid"]],
            on="strike",
            how="outer",
        )
        .sort_values("strike")
        .reset_index(drop=True)
    )

    grid["c_synth"] = np.where(
        grid["p_mid"].notna(), grid["p_mid"] + D * (F - grid["strike"]), np.nan
    )

    cond_right = grid["strike"] >= F
    c_best = np.where(cond_right, grid["c_mid"], grid["c_synth"])
    c_best = np.where(np.isnan(c_best), grid["c_mid"], c_best)
    c_best = np.where(np.isnan(c_best), grid["c_synth"], c_best)
    grid["mid_price"] = c_best

    out = grid.dropna(subset=["mid_price"]).copy()
    out = out.loc[out["mid_price"] > 0]

    if debug:
        if not out.empty:
            print(
                f"  unified rows: {len(out)} (min/max strike: {out['strike'].min()} / {out['strike'].max()})"
            )
        else:
            print("  unified rows: 0")

    if out.empty:
        return pd.DataFrame()

    out["quote_date"] = qdate
    out["underlying_last"] = S
    out["dte_days"] = dte_days
    out["risk_free_1m"] = r
    out["div_yield_annual"] = q
    out["is_call"] = True

    return out[
        [
            "quote_date",
            "underlying_last",
            "dte_days",
            "strike",
            "risk_free_1m",
            "div_yield_annual",
            "is_call",
            "mid_price",
        ]
    ]
