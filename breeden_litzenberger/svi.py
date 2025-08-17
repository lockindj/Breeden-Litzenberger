"""SVI calibration utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .preprocessing import _unified_calls_for_day_external
from .analysis import bl_density_from_prices


def build_unified_calls_dataframe_external(
    df_enriched: pd.DataFrame, verbose: bool = True, debug_first_n: int = 3
) -> pd.DataFrame:
    """Combine raw option quotes into unified call curves for each day.

    Parameters
    ----------
    df_enriched:
        Option data already merged with ``risk_free_1m`` and
        ``div_yield_annual`` columns.
    verbose:
        Print summary information.
    debug_first_n:
        Print diagnostics for the first *n* days.

    Returns
    -------
    pandas.DataFrame
        Cleaned call quotes ready for implied-volatility computation with
        columns ``quote_date``, ``underlying_last``, ``dte_days``, ``strike``,
        ``risk_free_1m``, ``div_yield_annual``, ``is_call`` and ``mid_price``.
    """

    df = df_enriched.copy()
    df["quote_date"] = pd.to_datetime(df["quote_date"]).dt.date
    dates = pd.Series(df["quote_date"]).sort_values().unique()

    frames, skipped = [], []
    for i, d in enumerate(dates):
        df_day = df.loc[df["quote_date"] == d].copy()
        part = _unified_calls_for_day_external(df_day, debug=(i < debug_first_n))
        if not part.empty:
            frames.append(part)
        else:
            n_calls = int((df_day["is_call"] == True).sum()) if "is_call" in df_day else -1
            n_puts = int((df_day["is_call"] == False).sum()) if "is_call" in df_day else -1
            skipped.append((d, f"no_usable_unified_calls (calls={n_calls}, puts={n_puts})"))

    out = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(
            columns=[
                "quote_date",
                "underlying_last",
                "dte_days",
                "strike",
                "risk_free_1m",
                "div_yield_annual",
                "is_call",
                "mid_price",
            ]
        )
    )
    if verbose:
        print(
            f"Unified call rows: {len(out)} across {out['quote_date'].nunique()} days. Skipped {len(skipped)} days."
        )
        if skipped[:5]:
            print("First few skips:", skipped[:5])
    return out


def _trapz_subset(K, f, a, b):
    """Trapezoidal integral of ``f`` over ``[a,b]`` on the subset of ``K``."""

    K = np.asarray(K, float)
    f = np.asarray(f, float)
    m = (K >= a) & (K <= b)
    if m.sum() < 2:
        return 0.0
    return float(np.trapz(f[m], K[m]))


def run_svi_over_all_days(
    df_IV: pd.DataFrame,
    calib_band=(0.85, 1.15),
    allowed_gap=0.10,
    eval_clamp=(0.75, 1.35),
    nK_eval: int = 1201,
    robust: bool = True,
    verbose: bool = True,
):
    """Run SVI calibration day-by-day with dynamic evaluation bands.

    The function expects external helpers ``fit_svi``, ``price_from_svi`` and
    ``convexity_report`` to be available in the import path.

    Returns
    -------
    summary_df, artifacts
        Calibration diagnostics and per-day artifacts.
    """

    needed = {
        "quote_date",
        "underlying_last",
        "risk_free_1m",
        "div_yield_annual",
        "dte",
        "strike",
        "implied_volatility",
    }
    missing_cols = needed - set(df_IV.columns)
    if missing_cols:
        raise KeyError(f"df_IV missing columns: {sorted(missing_cols)}")

    df = df_IV.copy()
    df["quote_date"] = pd.to_datetime(df["quote_date"]).dt.date
    for c in [
        "underlying_last",
        "risk_free_1m",
        "div_yield_annual",
        "dte",
        "strike",
        "implied_volatility",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    dates = sorted(pd.Series(df["quote_date"]).dropna().unique())

    summary_rows = []
    artifacts = {}

    for d in dates:
        try:
            out = fit_svi(df, d, calib_band=calib_band, robust=robust)

            S, r, q, T = out["S"], out["r"], out["q"], out["T"]
            K_obs = out["K_obs"]
            mny_min_obs = float(np.min(K_obs) / S)
            mny_max_obs = float(np.max(K_obs) / S)

            eval_min = max(mny_min_obs - allowed_gap, eval_clamp[0])
            eval_max = min(mny_max_obs + allowed_gap, eval_clamp[1])

            K_grid, iv_grid, C_grid = price_from_svi(out, band=(eval_min, eval_max), nK=nK_eval)

            rep = convexity_report(K_grid, C_grid)

            f = bl_density_from_prices(K_grid, C_grid, r, T)
            K_min_obs, K_max_obs = float(np.min(K_obs)), float(np.max(K_obs))

            total_eval_mass = _trapz_subset(K_grid, f, K_grid[0], K_grid[-1])
            left_tail_mass = _trapz_subset(K_grid, f, K_grid[0], min(K_min_obs, K_grid[-1]))
            right_tail_mass = _trapz_subset(K_grid, f, max(K_max_obs, K_grid[0]), K_grid[-1])
            core_mass = max(0.0, total_eval_mass - left_tail_mass - right_tail_mass)
            frac_outside_obs = (
                (left_tail_mass + right_tail_mass) / total_eval_mass if total_eval_mass > 1e-10 else np.nan
            )

            lee_ok = bool(out["lee_ok"])
            slope_left = out["slope_left"]
            slope_right = out["slope_right"]

            summary_rows.append(
                {
                    "quote_date": d,
                    "n_obs": int(len(K_obs)),
                    "rmse_w": float(out["rmse_w"]),
                    "mono_viol_inband": int(rep["monotonicity violations"]),
                    "conv_viol_inband": int(rep["convexity violations"]),
                    "lee_bound_violation": (not lee_ok),
                    "slope_left": slope_left,
                    "slope_right": slope_right,
                    "mny_min_obs": mny_min_obs,
                    "mny_max_obs": mny_max_obs,
                    "eval_mny_min": eval_min,
                    "eval_mny_max": eval_max,
                    "extrap_left_mny": max(0.0, mny_min_obs - eval_min),
                    "extrap_right_mny": max(0.0, eval_max - mny_max_obs),
                    "extrap_max_mny": max(
                        max(0.0, mny_min_obs - eval_min), max(0.0, eval_max - mny_max_obs)
                    ),
                    "total_eval_mass": total_eval_mass,
                    "left_tail_mass": left_tail_mass,
                    "right_tail_mass": right_tail_mass,
                    "core_mass": core_mass,
                    "frac_outside_observed": frac_outside_obs,
                    "res_status": int(out["res_status"]),
                    "has_issue": (
                        rep["monotonicity violations"] > 0
                        or rep["convexity violations"] > 0
                        or (not lee_ok)
                    ),
                }
            )

            artifacts[d] = {
                "out": out,
                "K_grid": K_grid,
                "iv_grid": iv_grid,
                "C_grid": C_grid,
                "bl_density": f,
                "convexity_report": rep,
            }

        except Exception as e:
            summary_rows.append(
                {
                    "quote_date": d,
                    "n_obs": 0,
                    "rmse_w": np.nan,
                    "mono_viol_inband": 999,
                    "conv_viol_inband": 999,
                    "lee_bound_violation": True,
                    "slope_left": np.nan,
                    "slope_right": np.nan,
                    "mny_min_obs": np.nan,
                    "mny_max_obs": np.nan,
                    "eval_mny_min": np.nan,
                    "eval_mny_max": np.nan,
                    "extrap_left_mny": np.nan,
                    "extrap_right_mny": np.nan,
                    "extrap_max_mny": np.nan,
                    "total_eval_mass": np.nan,
                    "left_tail_mass": np.nan,
                    "right_tail_mass": np.nan,
                    "core_mass": np.nan,
                    "frac_outside_observed": np.nan,
                    "res_status": 0,
                    "has_issue": True,
                    "notes": f"fit_error:{e}",
                }
            )

    summary_df = pd.DataFrame(summary_rows).sort_values("quote_date").reset_index(drop=True)

    if verbose and not summary_df.empty:
        tot = len(summary_df)
        issues = int(summary_df["has_issue"].fillna(False).sum())
        print(f"Total days: {tot}")
        print(f"Days with in-band issues (mono/convex/Lee): {issues}")
        if np.isfinite(summary_df["frac_outside_observed"]).any():
            print(
                "Median frac_outside_observed:",
                float(np.nanmedian(summary_df["frac_outside_observed"])),
            )
        if np.isfinite(summary_df["total_eval_mass"]).any():
            print(
                "Median total_eval_mass:",
                float(np.nanmedian(summary_df["total_eval_mass"])),
            )

    return summary_df, artifacts
