"""Data loading helpers for risk-free rates and dividend yields."""

from __future__ import annotations

import pandas as pd


def load_risk_free_rates(path: str) -> pd.DataFrame:
    """Return 1‑month risk-free rates.

    Parameters
    ----------
    path:
        CSV file containing columns ``observation_date`` and ``DGS1MO`` where
        the latter is expressed in percent.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``quote_date`` and ``risk_free_1m`` (as decimal)
        with missing values forward/back filled.
    """

    df = pd.read_csv(path, parse_dates=["observation_date"])
    out = (
        df[["observation_date", "DGS1MO"]]
        .rename(columns={"observation_date": "quote_date", "DGS1MO": "risk_free_1m"})
        .assign(risk_free_1m=lambda x: pd.to_numeric(x["risk_free_1m"], errors="coerce") / 100.0)
        .sort_values("quote_date")
    )
    out["risk_free_1m"] = out["risk_free_1m"].ffill().bfill()
    return out


def load_dividend_yield(path: str) -> pd.DataFrame:
    """Return annualized dividend yield time series.

    Parameters
    ----------
    path:
        CSV file containing columns ``Date`` and ``Annualized_Dividend_Yield``.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``quote_date`` and ``div_yield_annual`` with
        missing values forward/back filled.
    """

    df = pd.read_csv(path, parse_dates=["Date"])
    out = (
        df[["Date", "Annualized_Dividend_Yield"]]
        .rename(columns={"Date": "quote_date", "Annualized_Dividend_Yield": "div_yield_annual"})
        .assign(div_yield_annual=lambda x: pd.to_numeric(x["div_yield_annual"], errors="coerce"))
        .sort_values("quote_date")
    )
    out["div_yield_annual"] = out["div_yield_annual"].ffill().bfill()
    return out


def merge_rates_and_dividends(
    options_df: pd.DataFrame, risk_free_df: pd.DataFrame, dividend_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge risk-free and dividend yield information into an options frame.

    Parameters
    ----------
    options_df:
        DataFrame containing at least a ``quote_date`` column.
    risk_free_df:
        Output of :func:`load_risk_free_rates`.
    dividend_df:
        Output of :func:`load_dividend_yield`.

    Returns
    -------
    pandas.DataFrame
        ``options_df`` enriched with ``risk_free_1m`` and ``div_yield_annual``.
    """

    df = options_df.copy()
    df["quote_date"] = pd.to_datetime(df["quote_date"]).dt.normalize()
    merged = df.merge(risk_free_df, on="quote_date", how="left")
    merged = merged.merge(dividend_df, on="quote_date", how="left")
    return merged
