"""SVI calibration and pricing helpers.

This module provides utilities for fitting Stochastic Volatility Inspired (SVI)
volatility slices and deriving call prices. The implementations are lifted from
the accompanying analysis notebook and kept minimal so that :func:`fit_svi`,
:func:`price_from_svi`, and :func:`convexity_report` are available as plain
functions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from math import log, exp, sqrt, isfinite
from scipy.optimize import least_squares
from scipy.stats import norm

__all__ = [
    "fit_svi",
    "price_from_svi",
    "convexity_report",
    "svi_w",
    "lee_wing_slopes_ok",
]


# ---------------------------------------------------------------------------
# Black–Scholes pricing (fallback) + vectorized call helper
# ---------------------------------------------------------------------------

def _bs_price_fallback(S, K, T, r, q, sig, is_call=True):
    """European Black–Scholes with continuous dividend yield.

    Parameters
    ----------
    S, K : float
        Spot and strike.
    T : float
        Time to expiry in years.
    r, q : float
        Risk-free and dividend yields.
    sig : float
        Volatility.
    is_call : bool
        If ``True`` price a call, else a put.
    """
    if T <= 0:
        if is_call:
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)
    sig = max(sig, 1e-12)
    d1 = (log(S / K) + (r - q + 0.5 * sig * sig) * T) / (sig * sqrt(T))
    d2 = d1 - sig * sqrt(T)
    df_div = exp(-q * T)
    df_rf = exp(-r * T)
    if is_call:
        return df_div * S * norm.cdf(d1) - df_rf * K * norm.cdf(d2)
    else:
        return df_rf * K * norm.cdf(-d2) - df_div * S * norm.cdf(-d1)


# Replace this if you have a preferred Black–Scholes implementation.
bs_price = _bs_price_fallback

# Vectorized call pricer for convenience
bs_call_vec = np.vectorize(lambda S, K, T, r, q, sig: bs_price(S, K, T, r, q, sig, True))


# ---------------------------------------------------------------------------
# SVI (Stochastic Volatility Inspired) parameterization
# ---------------------------------------------------------------------------

def svi_w(k, a, b, rho, m, sigma):
    """Total variance of SVI at log-moneyness ``k``."""
    sigma = np.maximum(sigma, 1e-10)
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


def lee_wing_slopes_ok(b, rho, tol: float = 1e-8):
    """Check Lee's moment formula wing-slope bounds.

    Returns ``(ok, left_slope, right_slope)``.
    """
    left = b * (1.0 - rho)
    right = b * (1.0 + rho)
    ok = (left <= 2.0 + tol) and (right <= 2.0 + tol) and (b >= 0.0)
    return ok, float(left), float(right)


# ---------------------------------------------------------------------------
# Core SVI fit for a single day
# ---------------------------------------------------------------------------

def fit_svi(
    df_IV: pd.DataFrame,
    target_date,
    calib_band=(0.80, 1.15),
    robust: bool = True,
    iv_clip=(1e-4, 5.0),
):
    """Fit one SVI slice on ``target_date`` from rows of ``df_IV``.

    Parameters
    ----------
    df_IV : DataFrame
        Implied-volatility data with required columns.
    target_date : datetime-like
        Day to calibrate.
    calib_band : tuple of float
        Use strikes with ``K/S`` inside this band for fitting.
    robust : bool
        Use a soft-L1 loss for robustness.
    iv_clip : tuple of float
        Clamp implied vols before building total variance.
    """
    d = df_IV.loc[pd.to_datetime(df_IV["quote_date"]).dt.date == pd.to_datetime(target_date).date()].copy()
    if d.empty:
        raise ValueError(f"No rows for quote_date={target_date}")

    for c in ["strike", "implied_volatility", "underlying_last", "div_yield_annual", "risk_free_1m", "dte"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    d = d.dropna(subset=["strike", "implied_volatility", "underlying_last", "div_yield_annual", "risk_free_1m", "dte"])
    if d.empty:
        raise ValueError(f"All rows NA after cleaning for {target_date}")

    S = float(d["underlying_last"].iloc[0])
    r = float(d["risk_free_1m"].median())
    q = float(d["div_yield_annual"].median())
    T = float(d["dte"].median()) / 365.0
    if not (isfinite(S) and isfinite(r) and isfinite(q) and isfinite(T) and T > 0):
        raise ValueError(f"Bad snapshot S/r/q/T for {target_date}")

    F = S * exp((r - q) * T)
    K = d["strike"].to_numpy(float)
    iv = np.clip(d["implied_volatility"].to_numpy(float), iv_clip[0], iv_clip[1])
    w_obs = np.maximum(iv, 1e-10) ** 2 * T

    mny_S = K / S
    sel = (mny_S >= calib_band[0]) & (mny_S <= calib_band[1])
    if sel.sum() < 5:
        sel = np.isfinite(K) & np.isfinite(w_obs)

    K_cal = K[sel]
    w_cal = w_obs[sel]
    k_cal = np.log(K_cal / F)

    a0 = np.clip(np.percentile(w_cal, 10), 1e-6, None)
    span_k = np.percentile(k_cal, 90) - np.percentile(k_cal, 10) + 1e-8
    b0 = np.clip((np.percentile(w_cal, 90) - np.percentile(w_cal, 10)) / span_k, 1e-6, 10.0)
    rho0, m0 = 0.0, 0.0
    sigma0 = np.clip((k_cal.max() - k_cal.min()) / 4.0 if np.isfinite(k_cal.max() - k_cal.min()) else 0.2, 0.05, 2.0)
    x0 = np.array([a0, b0, rho0, m0, sigma0], float)

    lb = np.array([0.0, 1e-9, -0.999, -np.inf, 1e-6], float)
    ub = np.array([np.inf, np.inf, 0.999, np.inf, np.inf], float)

    def resid(x):
        a, b, rho, m, sigma = x
        wf = np.maximum(svi_w(k_cal, a, b, rho, m, sigma), 1e-12)
        return wf - w_cal

    kw = dict(x0=x0, bounds=(lb, ub), jac="2-point", max_nfev=20000)
    if robust:
        fscale = np.median(np.abs(w_cal - np.median(w_cal))) + 1e-6
        res = least_squares(resid, loss="soft_l1", f_scale=fscale, **kw)
    else:
        res = least_squares(resid, **kw)

    a, b, rho, m, sigma = res.x
    rmse_w = float(np.sqrt(np.mean(resid(res.x) ** 2)))

    lee_ok, slope_left, slope_right = lee_wing_slopes_ok(b, rho)

    out = {
        "S": S,
        "r": r,
        "q": q,
        "T": T,
        "F": F,
        "K_obs": K,
        "iv_obs": iv,
        "k_obs": np.log(K / F),
        "mnyS_obs": mny_S,
        "params": {"a": float(a), "b": float(b), "rho": float(rho), "m": float(m), "sigma": float(sigma)},
        "rmse_w": rmse_w,
        "lee_ok": bool(lee_ok),
        "slope_left": float(slope_left),
        "slope_right": float(slope_right),
        "res_status": int(res.status),
    }
    return out


# ---------------------------------------------------------------------------
# Pricing from SVI: build strike grid and price calls
# ---------------------------------------------------------------------------

def price_from_svi(out, band=(0.75, 1.30), nK: int = 1201):
    """Generate call prices and IVs on a strike grid from fitted SVI params."""
    S, r, q, T, F = out["S"], out["r"], out["q"], out["T"], out["F"]
    a, b, rho, m, sig = (out["params"][k] for k in ["a", "b", "rho", "m", "sigma"])

    K = np.linspace(band[0] * S, band[1] * S, nK)
    k = np.log(K / F)

    w = np.maximum(svi_w(k, a, b, rho, m, sig), 1e-12)
    iv = np.sqrt(w / max(T, 1e-12))

    C = bs_call_vec(S, K, T, r, q, iv)
    return K, iv, C


# ---------------------------------------------------------------------------
# No-arbitrage checks in price space
# ---------------------------------------------------------------------------

def convexity_report(K, C):
    """Return discrete monotonicity and convexity diagnostics for call prices."""
    K = np.asarray(K, float)
    C = np.asarray(C, float)
    h = K[1] - K[0]
    dC = np.diff(C) / h
    d2C = np.diff(C, n=2) / (h * h)
    tol = 1e-10
    rep = {
        "min first diff": float(dC.min()),
        "max first diff": float(dC.max()),
        "min second diff": float(d2C.min()),
        "max second diff": float(d2C.max()),
        "monotonicity violations": int((dC > +tol).sum()),
        "convexity violations": int((d2C < -tol).sum()),
    }
    return rep
