"""Analysis helpers: BL density, PIT, plots, and simple models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from math import exp
from typing import Dict, Tuple


def _second_derivative_nonuniform(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Second derivative ``d²y/dx²`` for a monotone, non-uniform grid."""

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    dy_dx = np.gradient(y, x, edge_order=2)
    return np.gradient(dy_dx, x, edge_order=2)


def bl_density_from_prices(K: np.ndarray, C: np.ndarray, r: float, T: float, floor_zero: bool = True) -> np.ndarray:
    """Breeden–Litzenberger density ``q(K) = exp(rT) * d²C/dK²``."""

    d2C = _second_derivative_nonuniform(K, C)
    q = np.exp(r * T) * d2C
    if floor_zero:
        q = np.maximum(q, 0.0)
    return q


def _cumtrapz_grid(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal integral starting at ``x[0]``."""

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    F = np.zeros_like(y, dtype=float)
    if len(x) < 2:
        return F
    dx = np.diff(x)
    area = 0.0
    for i in range(1, len(x)):
        area += 0.5 * (y[i - 1] + y[i]) * dx[i - 1]
        F[i] = area
    return F


def _trapz_subset(x: np.ndarray, y: np.ndarray, a: float, b: float) -> float:
    """Trapezoidal integral of ``y`` over ``[a,b]`` with endpoint interpolation."""

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    lo = max(a, x[0])
    hi = min(b, x[-1])
    if hi <= lo:
        return 0.0
    mask = (x >= lo) & (x <= hi)
    xx, yy = x[mask], y[mask]
    if xx.size == 0 or xx[0] > lo:
        y_lo = np.interp(lo, x, y)
        xx = np.r_[lo, xx]
        yy = np.r_[y_lo, yy]
    if xx[-1] < hi:
        y_hi = np.interp(hi, x, y)
        xx = np.r_[xx, hi]
        yy = np.r_[yy, y_hi]
    return float(np.trapz(yy, xx))


def compute_bl_pdfs_for_all_days(
    artifacts: Dict,
    summary_df: pd.DataFrame | None = None,
    use_float32: bool = False,
) -> Tuple[Dict, pd.DataFrame]:
    """Compute BL PDFs for every day from SVI pricing artifacts."""

    if summary_df is not None and "quote_date" in summary_df.columns:
        dates = list(summary_df["quote_date"])
    else:
        dates = list(artifacts.keys())
    dates = sorted(dates)

    bl_results: Dict = {}
    rows = []

    for d0 in dates:
        if d0 not in artifacts:
            continue
        art = artifacts[d0]
        out = art["out"]
        S, r, q_div, T = out["S"], out["r"], out["q"], out["T"]
        K_grid = np.array(art["K_grid"], dtype=float)
        C_grid = np.array(art["C_grid"], dtype=float)

        f = bl_density_from_prices(K_grid, C_grid, r, T, floor_zero=True)
        F_eval = _cumtrapz_grid(K_grid, f)
        total_eval_mass = float(F_eval[-1]) if F_eval.size else 0.0

        if "K_obs" in art and len(art["K_obs"]) > 0:
            K_obs = np.asarray(art["K_obs"], float)
            Kmin_obs, Kmax_obs = float(np.min(K_obs)), float(np.max(K_obs))
        else:
            lo_idx = int(0.05 * (len(K_grid) - 1))
            hi_idx = int(0.95 * (len(K_grid) - 1))
            Kmin_obs, Kmax_obs = K_grid[lo_idx], K_grid[hi_idx]

        left_tail_mass = _trapz_subset(K_grid, f, K_grid[0], Kmin_obs)
        right_tail_mass = _trapz_subset(K_grid, f, Kmax_obs, K_grid[-1])
        core_mass = max(0.0, total_eval_mass - left_tail_mass - right_tail_mass)

        if use_float32:
            K_stored = K_grid.astype(np.float32)
            f_stored = f.astype(np.float32)
            F_stored = F_eval.astype(np.float32)
        else:
            K_stored, f_stored, F_stored = K_grid, f, F_eval

        bl_results[d0] = {
            "K_grid": K_stored,
            "pdf": f_stored,
            "cdf_eval": F_stored,
            "total_eval_mass": total_eval_mass,
            "left_tail_mass": left_tail_mass,
            "right_tail_mass": right_tail_mass,
            "core_mass": core_mass,
            "S": S,
            "r": r,
            "q": q_div,
            "T": T,
        }

        rows.append(
            {
                "quote_date": d0,
                "total_eval_mass": total_eval_mass,
                "left_tail_mass": left_tail_mass,
                "right_tail_mass": right_tail_mass,
                "core_mass": core_mass,
                "Kmin_obs": Kmin_obs,
                "Kmax_obs": Kmax_obs,
                "Kmin_eval": K_grid[0],
                "Kmax_eval": K_grid[-1],
            }
        )

    mass_summary_df = pd.DataFrame(rows).sort_values("quote_date").reset_index(drop=True)
    return bl_results, mass_summary_df


# ---------------------------------------------------------------------------
# PIT and classification utilities
# ---------------------------------------------------------------------------

def _coerce_realized_to_df(realized_prices_like) -> pd.DataFrame:
    """Normalize realized prices into a tidy two-column DataFrame."""

    if isinstance(realized_prices_like, pd.Series):
        df = realized_prices_like.copy().reset_index()
        if df.shape[1] != 2:
            raise ValueError(
                "Realized Series must become a 2-column DataFrame after reset_index()."
            )
        df.columns = ["quote_date", "realized_price"]
    elif isinstance(realized_prices_like, pd.DataFrame):
        df = realized_prices_like.copy()
        cols = {c.lower(): c for c in df.columns}
        date_col = cols.get("quote_date", df.columns[0])
        price_col = cols.get("realized_price", df.columns[-1])
        df = df[[date_col, price_col]].rename(
            columns={date_col: "quote_date", price_col: "realized_price"}
        )
    else:
        try:
            ser = pd.Series(realized_prices_like)
            df = ser.reset_index()
            df.columns = ["quote_date", "realized_price"]
        except Exception as e:
            raise ValueError(
                "Unsupported realized_prices type; pass a Series or DataFrame."
            ) from e

    df["quote_date"] = pd.to_datetime(df["quote_date"]).dt.normalize()
    df["realized_price"] = pd.to_numeric(df["realized_price"], errors="coerce")
    return df.dropna(subset=["quote_date", "realized_price"]).reset_index(drop=True)


def _build_bl_df(bl_results: Dict) -> pd.DataFrame:
    """Convert ``bl_results`` dict into a tidy DataFrame."""

    rows = []
    for k, rec in bl_results.items():
        d = pd.to_datetime(k).normalize()
        rows.append(
            {
                "quote_date": d,
                "K_grid": rec["K_grid"],
                "pdf": rec["pdf"],
                "S": float(rec["S"]),
                "r": float(rec["r"]),
                "q": float(rec["q"]),
                "T": float(rec["T"]),
                "total_eval_mass": float(rec.get("total_eval_mass", np.nan)),
            }
        )
    return pd.DataFrame(rows).sort_values("quote_date").reset_index(drop=True)


def _rn_up_probability(K: np.ndarray, f: np.ndarray, F: float) -> float:
    """Risk-neutral up probability ``P(K>=F)`` after normalisation."""

    K = np.asarray(K, float)
    f = np.asarray(f, float)
    area = np.trapz(f, K)
    if area <= 0:
        return np.nan
    f = f / area
    mask = K >= F
    if not np.any(mask):
        return 0.0
    return float(np.trapz(f[mask], K[mask]))


def naive_bl_direction_classifier(
    bl_results: Dict,
    realized_prices_like,
    threshold: float = 0.5,
    direction_vs: str = "forward",
):
    """Naive BL-based up/down classifier.

    Parameters
    ----------
    bl_results:
        Output of :func:`compute_bl_pdfs_for_all_days` keyed by ``quote_date``.
    realized_prices_like:
        Series or DataFrame of realized underlying prices.
    threshold:
        Cutoff for classifying an upward move.
    direction_vs:
        ``"forward"`` or ``"spot"`` for the benchmark.

    Returns
    -------
    results_df, metrics
    """

    from sklearn.metrics import confusion_matrix

    df_real = _coerce_realized_to_df(realized_prices_like)
    df_bl = _build_bl_df(bl_results)
    df = df_bl.merge(df_real, on="quote_date", how="inner")
    if df.empty:
        return (
            pd.DataFrame(
                columns=[
                    "quote_date",
                    "realized_price",
                    "S",
                    "F",
                    "P_up",
                    "predicted_label",
                    "realized_label",
                    "correct",
                ]
            ),
            {"accuracy": np.nan, "confusion_matrix": np.array([[0, 0], [0, 0]])},
        )

    F_vals = df["S"] * np.exp((df["r"] - df["q"]) * df["T"])
    df["F"] = F_vals

    df["P_up"] = [
        _rn_up_probability(K, f, F) for K, f, F in zip(df["K_grid"], df["pdf"], df["F"])
    ]
    df["predicted_label"] = (df["P_up"] > threshold).astype(int)

    if direction_vs == "spot":
        df["realized_label"] = (df["realized_price"] > df["S"]).astype(int)
    else:
        df["realized_label"] = (df["realized_price"] > df["F"]).astype(int)

    df["correct"] = (df["predicted_label"] == df["realized_label"]).astype(int)

    if len(df) > 0:
        acc = float(df["correct"].mean())
        cm = confusion_matrix(df["realized_label"], df["predicted_label"], labels=[0, 1])
    else:
        acc = np.nan
        cm = np.array([[0, 0], [0, 0]])

    results_df = (
        df[
            [
                "quote_date",
                "realized_price",
                "S",
                "F",
                "P_up",
                "predicted_label",
                "realized_label",
                "correct",
            ]
        ]
        .sort_values("quote_date")
        .reset_index(drop=True)
    )
    return results_df, {"accuracy": acc, "confusion_matrix": cm}


def prepare_pit_df(
    bl_results: Dict,
    mass_summary: pd.DataFrame,
    threshold: float,
    realized_prices: pd.Series | Dict,
) -> pd.DataFrame:
    """Build a PIT dataframe for all days above a mass threshold."""

    pit_records = []
    valid_days = mass_summary.loc[
        mass_summary["total_eval_mass"] >= threshold, "quote_date"
    ]
    for qd in valid_days:
        if qd not in bl_results or qd not in realized_prices:
            continue
        strikes = bl_results[qd]["K_grid"]
        pdf = bl_results[qd]["pdf"]
        total_mass = bl_results[qd]["total_eval_mass"]
        norm_factor = np.trapz(pdf, strikes)
        if norm_factor <= 0:
            continue
        pdf_norm = pdf / norm_factor
        S_T = realized_prices[qd]
        mask = strikes <= S_T
        if not np.any(mask):
            pit_val = 0.0 if S_T < strikes.min() else 1.0
        else:
            pit_val = np.trapz(pdf_norm[mask], strikes[mask])
        pit_records.append(
            {
                "quote_date": pd.to_datetime(qd),
                "realized": S_T,
                "PIT_value": pit_val,
                "total_eval_mass": total_mass,
            }
        )
    df_pit = pd.DataFrame(pit_records)
    if "quote_date" in df_pit.columns:
        df_pit = df_pit.sort_values("quote_date")
    return df_pit


# ---------------------------------------------------------------------------
# Plotting and modelling
# ---------------------------------------------------------------------------

def plot_roc_curve(y_true, y_score, ax=None):
    """Plot ROC curve and return the matplotlib Axes."""

    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    ax = ax or plt.gca()
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Random guess")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    return ax


def plot_precision_recall_curve(y_true, y_score, ax=None):
    """Plot Precision-Recall curve and return the matplotlib Axes."""

    from sklearn.metrics import precision_recall_curve, average_precision_score
    import matplotlib.pyplot as plt

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    ax = ax or plt.gca()
    ax.plot(recall, precision, color="blue", label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def run_logistic_regression(
    df: pd.DataFrame,
    feature_cols,
    label_col: str,
    test_size: float = 0.3,
):
    """Fit a logistic regression classifier.

    Returns the fitted model and a metrics dictionary containing ROC AUC and a
    ``classification_report`` string.
    """

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, classification_report

    X = df[list(feature_cols)].copy()
    y = df[label_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    log_reg = LogisticRegression(max_iter=5000)
    log_reg.fit(X_train, y_train)
    y_pred_prob = log_reg.predict_proba(X_test)[:, 1]
    y_pred_class = log_reg.predict(X_test)
    metrics = {
        "roc_auc": roc_auc_score(y_test, y_pred_prob),
        "classification_report": classification_report(y_test, y_pred_class),
    }
    return log_reg, metrics
