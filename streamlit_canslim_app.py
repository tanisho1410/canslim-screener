"""
Streamlit-based CANSLIM stock screener.

This module implements a simple Streamlit application to evaluate
individual stocks against a subset of William O'Neil's CANSLIM
criteria. The intent is to provide an interactive front‑end where
users can paste ticker symbols (comma‑ or newline‑separated), run
the screening logic, view the results, and download them as a CSV.

The screening logic is largely adapted from the standalone
`canslim_app.py` module. It relies on yfinance to fetch historical
prices and quarterly/annual earnings data. Because this code will
run client‑side in Streamlit, network access and API limits may
affect performance; consider caching data or integrating a paid
financial API for production use.

To run this app locally:

  pip install streamlit pandas numpy yfinance
  streamlit run streamlit_canslim_app.py

Note: The Google Sheets integration is not implemented here. To
persist results to a spreadsheet, you can adapt the `update_google_sheet`
function from the standalone module and call it after screening.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import math


# -----------------------------
# Screening thresholds (tunable)
# -----------------------------
MIN_REV_YOY = 0.20              # Revenue YoY >= 20%
MIN_EARN_YOY = 0.25             # Earnings YoY >= 25%
MIN_EARN_CAGR_3Y = 0.15         # Earnings CAGR (3Y) >= 15%
NEAR_52W_HIGH_PCT = 0.90        # Price within 90% of 52‑week high
VOLUME_SPIKE_RATIO = 1.40       # Volume spike vs 50‑day average >= +40%
TOP_MOMENTUM_PCT = 0.70         # 6‑month return percentile >= 70%


def fetch_price_block(tickers):
    """Fetch historical price and volume data for each ticker.

    Returns a dict mapping each ticker to a DataFrame with OHLCV,
    moving averages, 52‑week high, and average volume series.
    """
    end = datetime.today()
    start = end - timedelta(days=400 * 1.5)
    data = yf.download(
        tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )
    px = {}
    for t in tickers:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                df = data[t].copy()
            else:
                df = data.copy()
            df = df.dropna()
            df["MA50"] = df["Close"].rolling(50).mean()
            df["MA200"] = df["Close"].rolling(200).mean()
            df["VOL50"] = df["Volume"].rolling(50).mean()
            df["HIGH_252"] = df["High"].rolling(252).max()
            px[t] = df
        except Exception:
            # If data for a ticker can't be fetched, skip it
            continue
    return px


def calc_returns(px):
    """Calculate 6‑month and 3‑month returns for each ticker."""
    ret_6m = {}
    ret_3m = {}
    for t, df in px.items():
        if len(df) < 130:
            continue
        c = df["Close"]
        ret_6m[t] = (c.iloc[-1] / c.iloc[-126] - 1.0) if len(c) >= 126 else np.nan
        ret_3m[t] = (c.iloc[-1] / c.iloc[-63] - 1.0) if len(c) >= 63 else np.nan
    return pd.DataFrame({"ret_6m": ret_6m, "ret_3m": ret_3m}).T


def yoy_growth(current, prev_year):
    """Compute year‑over‑year growth from two values."""
    if pd.isna(current) or pd.isna(prev_year) or prev_year == 0:
        return np.nan
    return (current / prev_year) - 1.0


def cagr(start, end, years):
    """Compute compound annual growth rate (CAGR)."""
    if any(map(pd.isna, [start, end])) or start <= 0:
        return np.nan
    try:
        return (end / start) ** (1 / years) - 1.0
    except Exception:
        return np.nan


def fetch_fundamentals(tickers):
    """Fetch quarterly earnings and annual earnings for each ticker."""
    rows = []
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            q = tk.quarterly_earnings
            a = tk.earnings
            rev_yoy = np.nan
            earn_yoy = np.nan
            if q is not None and isinstance(q, pd.DataFrame) and len(q) >= 5:
                q_sorted = q.sort_index()
                last_rev = q_sorted["Revenue"].iloc[-1]
                last_earn = q_sorted["Earnings"].iloc[-1]
                prev_rev = q_sorted["Revenue"].iloc[-5] if len(q_sorted) >= 5 else np.nan
                prev_earn = q_sorted["Earnings"].iloc[-5] if len(q_sorted) >= 5 else np.nan
                rev_yoy = yoy_growth(last_rev, prev_rev)
                earn_yoy = yoy_growth(last_earn, prev_earn)
            earn_cagr_3y = np.nan
            if a is not None and isinstance(a, pd.DataFrame) and len(a) >= 4:
                a_sorted = a.sort_index()
                start_earn = a_sorted["Earnings"].iloc[-4]
                end_earn = a_sorted["Earnings"].iloc[-1]
                earn_cagr_3y = cagr(start_earn, end_earn, 3)
            rows.append({
                "ticker": t,
                "rev_yoy": rev_yoy,
                "earn_yoy": earn_yoy,
                "earn_cagr_3y": earn_cagr_3y,
            })
        except Exception:
            # If fundamentals can't be fetched, skip the ticker
            continue
    return pd.DataFrame(rows)


def build_snapshot(px):
    """Build a snapshot DataFrame with technical indicators."""
    rows = []
    for t, df in px.items():
        try:
            last = df.iloc[-1]
            close = float(last["Close"])
            ma50 = float(last["MA50"])
            ma200 = float(last["MA200"])
            vol = float(last["Volume"])
            vol50 = float(last["VOL50"])
            hi252 = float(last["HIGH_252"])
            near_52w = close / hi252 if hi252 and not math.isclose(hi252, 0) else np.nan
            vol_spike = vol / vol50 if vol50 and not math.isclose(vol50, 0) else np.nan
            rows.append({
                "ticker": t,
                "close": close,
                "ma50": ma50,
                "ma200": ma200,
                "above_ma50": close > ma50,
                "ma50_gt_ma200": ma50 > ma200,
                "near_52w_high": near_52w,
                "vol_spike_ratio": vol_spike,
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


def screen_tickers(tickers):
    """Run the CANSLIM screening process and return a DataFrame."""
    # Fetch price data and compute snapshots
    px = fetch_price_block(tickers)
    snaps = build_snapshot(px)
    # Compute momentum
    rets = calc_returns(px)
    rets.index.name = "ticker"
    rets = rets.reset_index()
    # Fetch fundamentals
    fnd = fetch_fundamentals(list(snaps["ticker"]))
    # Combine
    df = snaps.merge(rets, on="ticker", how="left").merge(fnd, on="ticker", how="left")
    # 6‑month return percentile
    df["ret_6m_pctile"] = df["ret_6m"].rank(pct=True)
    # Apply screening conditions
    conds = []
    conds.append(df["rev_yoy"] >= MIN_REV_YOY)
    conds.append(df["earn_yoy"] >= MIN_EARN_YOY)
    conds.append(df["earn_cagr_3y"] >= MIN_EARN_CAGR_3Y)
    conds.append(df["near_52w_high"] >= NEAR_52W_HIGH_PCT)
    conds.append(df["above_ma50"])
    conds.append(df["ma50_gt_ma200"])
    conds.append(df["vol_spike_ratio"] >= VOLUME_SPIKE_RATIO)
    conds.append(df["ret_6m_pctile"] >= TOP_MOMENTUM_PCT)
    df["pass"] = np.logical_and.reduce(conds)
    # Order columns for display
    cols = [
        "ticker",
        "pass",
        "close",
        "near_52w_high",
        "above_ma50",
        "ma50_gt_ma200",
        "vol_spike_ratio",
        "ret_3m",
        "ret_6m",
        "ret_6m_pctile",
        "rev_yoy",
        "earn_yoy",
        "earn_cagr_3y",
    ]
    df = df.loc[:, [c for c in cols if c in df.columns]].sort_values(
        ["pass", "ret_6m"], ascending=[False, False]
    )
    return df


def main():
    """Streamlit application entry point."""
    st.title("CANSLIM Stock Screener")
    st.write(
        "This app evaluates stocks against selected CANSLIM criteria and returns a table of results."
    )
    st.write(
        "Enter ticker symbols separated by commas or newlines (e.g. AAPL, MSFT, NVDA)."
    )
    tickers_input = st.text_area("Tickers", value="AAPL, MSFT, NVDA")
    # Parse tickers from input
    raw_tokens = tickers_input.replace("\n", ",").split(",")
    tickers = [t.strip().upper() for t in raw_tokens if t.strip()]
    if st.button("Run Screening"):
        if not tickers:
            st.warning("Please enter at least one ticker symbol.")
        else:
            with st.spinner("Fetching data and screening..."):
                try:
                    df = screen_tickers(tickers)
                    if df.empty:
                        st.info("No data returned. Please check your tickers and try again.")
                    else:
                        num_pass = int(df["pass"].sum())
                        st.success(f"Screening completed. {num_pass} of {len(df)} tickers passed the criteria.")
                        st.dataframe(df)
                        csv_data = df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "Download CSV",
                            csv_data,
                            file_name="canslim_screening_results.csv",
                            mime="text/csv",
                        )
                except Exception as e:
                    st.error(f"An error occurred while screening: {e}")
    st.divider()
    st.write(
        "Note: Google Sheets export is not implemented here. "
        "To save results to a spreadsheet, adapt the `update_google_sheet` "
        "function from the standalone module and invoke it after screening."
    )


if __name__ == "__main__":
    main()