# sql.py
import pymysql
import pandas as pd
import numpy as np


# -------------------------------
# Core DB helper
# -------------------------------
def SQL_dataframe(SQL_Query: str, timeout: int = 30) -> pd.DataFrame:
    """
    Execute a read-only SQL query and return a pandas DataFrame.
    Uses modest timeouts; chunk at the caller to avoid server timeouts.
    """
    connection = pymysql.connect(
        charset="utf8mb4",
        connect_timeout=timeout,
        cursorclass=pymysql.cursors.DictCursor,
        db="defaultdb",
        host="mysql-db1-spilldjango-starter.e.aivencloud.com",
        password="6_7_I_just_bipped",
        read_timeout=timeout,
        port=13641,
        user="readonly_user",
        write_timeout=timeout,
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute(SQL_Query)
            result = cursor.fetchall()
            return pd.DataFrame(result)
    finally:
        connection.close()


# -------------------------------
# Date range helpers (explicit, no work at import)
# -------------------------------
def get_min_max_dates(table_name: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    q = f"""
        SELECT MIN(quote_date) AS min_date, MAX(quote_date) AS max_date
        FROM {table_name};
    """
    df = SQL_dataframe(q)
    if df.empty or pd.isna(df.loc[0, "min_date"]) or pd.isna(df.loc[0, "max_date"]):
        return (pd.NaT, pd.NaT)
    return (pd.to_datetime(df.loc[0, "min_date"]), pd.to_datetime(df.loc[0, "max_date"]))


# -------------------------------
# Snapshot fetchers (monthly, keyset pagination)
# -------------------------------
def _iter_month_windows(min_date, max_date):
    """Yield (start_date, next_month_start) pairs as date objects."""
    starts = pd.date_range(start=min_date, end=max_date, freq="MS")
    for start in starts:
        next_month = start + pd.DateOffset(months=1)
        yield start.date(), next_month.date()


def fetch_snapshots_calls(min_date, max_date, chunk_size: int = 100000, debug: bool = False) -> pd.DataFrame:
    """
    Fetch dte=30 snapshots from calls_table in monthly windows using keyset pagination.
    Avoids BETWEEN; uses half-open [start, next_month) for better index use in MySQL.
    Computes moneyness in Python to allow covering scans.
    """
    results = []
    for start_date, next_month in _iter_month_windows(min_date, max_date):
        last_id = 0
        while True:
            q = f"""
                SELECT
                    id,
                    quote_date,
                    underlying_last,
                    dte,
                    strike,
                    c_bid,
                    c_ask,
                    c_last,
                    expire_date
                FROM calls_table
                WHERE dte = 30
                  AND quote_date >= '{start_date}'
                  AND quote_date <  '{next_month}'
                  AND id > {last_id}
                ORDER BY id
                LIMIT {chunk_size}
            """
            df = SQL_dataframe(q)
            if df.empty:
                break
            results.append(df)
            last_id = int(df["id"].max())
            if debug:
                print(f"[calls] {start_date}..{next_month}  +{len(df)} rows (last_id={last_id})", flush=True)

    if not results:
        return pd.DataFrame()

    out = pd.concat(results, ignore_index=True)
    # Compute moneyness in Python (faster and index-friendly than in SELECT)
    out["moneyness"] = out["strike"] / out["underlying_last"]
    return out


def fetch_snapshots_puts(min_date, max_date, chunk_size: int = 100000, debug: bool = False) -> pd.DataFrame:
    """
    Fetch dte=30 snapshots from puts_table in monthly windows using keyset pagination.
    Fixes missing AND before quote_date; uses half-open window; computes moneyness in Python.
    """
    results = []
    for start_date, next_month in _iter_month_windows(min_date, max_date):
        last_id = 0
        while True:
            q = f"""
                SELECT
                    id,
                    quote_date,
                    underlying_last,
                    dte,
                    strike,
                    p_bid,
                    p_ask,
                    p_last,
                    expire_date
                FROM puts_table
                WHERE dte = 30
                  AND quote_date >= '{start_date}'
                  AND quote_date <  '{next_month}'
                  AND id > {last_id}
                ORDER BY id
                LIMIT {chunk_size}
            """
            df = SQL_dataframe(q)
            if df.empty:
                break
            results.append(df)
            last_id = int(df["id"].max())
            if debug:
                print(f"[puts]  {start_date}..{next_month}  +{len(df)} rows (last_id={last_id})", flush=True)

    if not results:
        return pd.DataFrame()

    out = pd.concat(results, ignore_index=True)
    out["moneyness"] = out["strike"] / out["underlying_last"]
    return out


# -------------------------------
# Realized (+30d next trading day) helper
# -------------------------------
def attach_realized_from_self(df_snapshots: pd.DataFrame,
                              quote_col: str = "quote_date",
                              ul_col: str = "underlying_last") -> pd.DataFrame:
    """
    For a snapshot DataFrame from a single table:
    - Build calendar of available trading days from `quote_col`
    - For each quote_date, map to the next available trading day >= quote_date + 30d
    - Join realized_underlying from per-day underlying_last (groupby max)
    Returns df with added columns: realized_date, realized_underlying
    """
    if df_snapshots is None or df_snapshots.empty:
        df = df_snapshots.copy() if df_snapshots is not None else pd.DataFrame()
        df[["realized_date", "realized_underlying"]] = np.nan
        return df

    df = df_snapshots.copy()
    df[quote_col] = pd.to_datetime(df[quote_col])

    # Unique trading calendar from the snapshot
    cal = (
        df[[quote_col]]
        .drop_duplicates()
        .sort_values(quote_col)
        .rename(columns={quote_col: "cal_date"})
        .reset_index(drop=True)
    )

    # Targets = quote_date + 30 calendar days
    key = df[[quote_col]].drop_duplicates().sort_values(quote_col).reset_index(drop=True)
    key["target"] = key[quote_col] + pd.Timedelta(days=30)

    # Map to next available trading day using merge_asof
    mapped = pd.merge_asof(
        key.sort_values("target"),
        cal.sort_values("cal_date"),
        left_on="target",
        right_on="cal_date",
        direction="forward",
        allow_exact_matches=True,
    ).rename(columns={"cal_date": "realized_date"})[[quote_col, "realized_date"]]

    # Per-day underlying_last (max is safe if constant intraday)
    ul_by_day = (
        df.groupby(quote_col, as_index=False)[ul_col]
        .max()
        .rename(columns={quote_col: "realized_date", ul_col: "realized_underlying"})
    )

    # Merge realized_date into snapshots, then realized_underlying
    df = df.merge(mapped, on=quote_col, how="left")
    df = df.merge(ul_by_day, on="realized_date", how="left")

    return df


# -------------------------------
# Optional: quick self-test entrypoint (won't run on import)
# -------------------------------
if __name__ == "__main__":
    # Example: peek at date ranges without heavy fetching
    min_calls, max_calls = get_min_max_dates("calls_table")
    min_puts,  max_puts  = get_min_max_dates("puts_table")
    print("calls_table range:", min_calls, "→", max_calls)
    print("puts_table  range:", min_puts,  "→", max_puts)

    # Small smoke test on one month if ranges exist
    if pd.notna(min_calls):
        sample_calls = fetch_snapshots_calls(min_calls, min_calls, chunk_size=20000, debug=True)
        print("sample calls:", sample_calls.shape)
    if pd.notna(min_puts):
        sample_puts = fetch_snapshots_puts(min_puts, min_puts, chunk_size=20000, debug=True)
        print("sample puts:", sample_puts.shape)
