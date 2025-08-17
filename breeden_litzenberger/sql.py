import pymysql

### Methods

# run SQL queries as read only user
def SQL_dataframe(SQL_Query):
    timeout = 10
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
            df = pd.DataFrame(result)
            return df
    finally:
        connection.close()


# actual query 

# ============================================================
# Simple yearly snapshots + realized (+30 next trading day) all in pandas
# - No extra SQL complexity (mirrors your working puts query style)
# - Build the +30 mapping and realized underlying in pandas
# ============================================================
import pandas as pd
import numpy as np

# ---------- 1) Get min/max from each table (split queries to avoid timeout) ----------
date_range_calls = SQL_dataframe("""
    SELECT MIN(quote_date) AS min_date, MAX(quote_date) AS max_date
    FROM calls_table;
""")
min_calls = date_range_calls['min_date'][0]
max_calls = date_range_calls['max_date'][0]

date_range_puts = SQL_dataframe("""
    SELECT MIN(quote_date) AS min_date, MAX(quote_date) AS max_date
    FROM puts_table;
""")
min_puts = date_range_puts['min_date'][0]
max_puts = date_range_puts['max_date'][0]

print("calls_table range:", min_calls, "→", max_calls)
print("puts_table  range:", min_puts,  "→", max_puts)

# ---------- 2) Yearly batching (like your working query) ----------
def fetch_snapshots_calls(min_date, max_date):
    batch_starts = pd.date_range(start=min_date, end=max_date, freq='YS')
    results = []
    for start in batch_starts:
        end = (start + pd.DateOffset(years=1)) - pd.DateOffset(days=1)
        q = f"""
            SELECT 
                id,
                quote_date,
                underlying_last,
                dte,
                strike,
                strike / underlying_last AS moneyness,
                c_bid,
                c_ask,
                c_last,
                expire_date
            FROM calls_table
            WHERE dte = 30
              AND c_bid IS NOT NULL
              AND c_ask IS NOT NULL
              AND c_iv  IS NOT NULL
              AND underlying_last IS NOT NULL
              AND underlying_last > 0
              AND quote_date BETWEEN '{start.date()}' AND '{end.date()}'
        """
        df = SQL_dataframe(q)
        results.append(df)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

def fetch_snapshots_puts(min_date, max_date):
    batch_starts = pd.date_range(start=min_date, end=max_date, freq='YS')
    results = []
    for start in batch_starts:
        end = (start + pd.DateOffset(years=1)) - pd.DateOffset(days=1)
        q = f"""
            SELECT 
                id,
                quote_date,
                underlying_last,
                dte,
                strike,
                strike / underlying_last AS moneyness,
                p_bid,
                p_ask,
                p_last,
                expire_date
            FROM puts_table
            WHERE dte = 30
              AND p_bid IS NOT NULL
              AND p_ask IS NOT NULL
              AND p_iv  IS NOT NULL
              AND underlying_last IS NOT NULL
              AND underlying_last > 0
              AND quote_date BETWEEN '{start.date()}' AND '{end.date()}'
        """
        df = SQL_dataframe(q)
        results.append(df)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

df_snapshots_calls = fetch_snapshots_calls(min_calls, max_calls)
df_snapshots_puts  = fetch_snapshots_puts(min_puts,  max_puts)

print("Calls snapshots:", df_snapshots_calls.shape)
print("Unique call quote_dates:", df_snapshots_calls["quote_date"].nunique() if not df_snapshots_calls.empty else 0)
print("Puts snapshots:", df_snapshots_puts.shape)
print("Unique put quote_dates:", df_snapshots_puts["quote_date"].nunique() if not df_snapshots_puts.empty else 0)

# ---------- 3) Build +30 next-trading-day mapping in pandas ----------
def attach_realized_from_self(df_snapshots, quote_col="quote_date", ul_col="underlying_last"):
    """
    For a snapshot DataFrame from a single table:
    - Build calendar of available trading days from `quote_col`
    - For each quote_date, map to the next available trading day >= quote_date + 30d
    - Join realized_underlying from per-day underlying_last (groupby max)
    Returns df with added columns: realized_date, realized_underlying
    """
    if df_snapshots.empty:
        df_snapshots[["realized_date", "realized_underlying"]] = np.nan
        return df_snapshots

    # Ensure datetime
    df = df_snapshots.copy()
    df[quote_col] = pd.to_datetime(df[quote_col])

    # Unique trading calendar from the snapshot
    cal = (df[[quote_col]]
           .drop_duplicates()
           .sort_values(quote_col)
           .rename(columns={quote_col: "cal_date"})
           .reset_index(drop=True))

    # Target = quote_date + 30 calendar days
    key = df[[quote_col]].drop_duplicates().sort_values(quote_col).reset_index(drop=True)
    key["target"] = key[quote_col] + pd.Timedelta(days=30)

    # Map to next available trading day using merge_asof
    mapped = pd.merge_asof(
        key.sort_values("target"),
        cal.sort_values("cal_date"),
        left_on="target",
        right_on="cal_date",
        direction="forward",
        allow_exact_matches=True
    ).rename(columns={"cal_date": "realized_date"})[[quote_col, "realized_date"]]

    # Per-day underlying_last (max is safe if constant intraday)
    ul_by_day = (df.groupby(quote_col, as_index=False)[ul_col]
                   .max()
                   .rename(columns={quote_col: "realized_date",
                                    ul_col: "realized_underlying"}))

    # Merge realized_date into snapshots, then realized_underlying
    df = df.merge(mapped, on=quote_col, how="left")
    df = df.merge(ul_by_day, on="realized_date", how="left")

    return df

df_snapshots_calls = attach_realized_from_self(df_snapshots_calls,
                                              quote_col="quote_date",
                                              ul_col="underlying_last")

df_snapshots_puts  = attach_realized_from_self(df_snapshots_puts,
                                              quote_col="quote_date",
                                              ul_col="underlying_last")

# ---------- 4) Final checks ----------
print("Calls (with realized):", df_snapshots_calls.shape,
      "| missing realized:", df_snapshots_calls["realized_underlying"].isna().sum())

print("Puts  (with realized):", df_snapshots_puts.shape,
      "| missing realized:", df_snapshots_puts["realized_underlying"].isna().sum())

# Optionally drop rows without a realized trading day found
df_snapshots_calls = df_snapshots_calls.dropna(subset=["realized_date","realized_underlying"])
df_snapshots_puts  = df_snapshots_puts.dropna(subset=["realized_date","realized_underlying"])

print("Calls (after dropna):", df_snapshots_calls.shape)
print("Puts  (after dropna):", df_snapshots_puts.shape)
# 