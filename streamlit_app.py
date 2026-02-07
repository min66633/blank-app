import streamlit as st
import pandas as pd
import requests
import math
from datetime import date, timedelta

# ===============================
# Page setup
# ===============================
st.set_page_config(layout="wide")
st.title("📊 Options Greeks & Gamma Exposure Dashboard")
st.write("Last update:", pd.Timestamp.now())


POLYGON_API_KEY = st.secrets["mD0LX0bzkc3sIUH3Hs0lwNucRo90HtML"]

# ===============================
# Math utils (no scipy)
# ===============================
def norm_cdf(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def bs_delta(S, K, T, r, sigma, call=True):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return norm_cdf(d1) if call else norm_cdf(d1) - 1

def bs_gamma(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return math.exp(-d1**2 / 2) / (S * sigma * math.sqrt(2 * math.pi * T))

# ===============================
# Data loaders
# ===============================
def get_price_data(ticker, years=2):
    end = date.today()
    start = end - timedelta(days=365 * years)

    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/"
        f"{start}/{end}?adjusted=true&apiKey={POLYGON_API_KEY}"
    )

    res = requests.get(url).json()
    if "results" not in res:
        return pd.DataFrame()

    df = pd.DataFrame(res["results"])
    df["date"] = pd.to_datetime(df["t"], unit="ms")
    df.rename(columns={"c": "close"}, inplace=True)
    return df[["date", "close"]]

def get_option_chain(ticker):
    url = (
        f"https://api.polygon.io/v3/reference/options/contracts?"
        f"underlying_ticker={ticker}&limit=200&apiKey={POLYGON_API_KEY}"
    )
    res = requests.get(url).json()
    if "results" not in res:
        return pd.DataFrame()

    df = pd.DataFrame(res["results"])
    return df[[
        "strike_price",
        "expiration_date",
        "contract_type",
        "open_interest"
    ]]

# ===============================
# Sidebar inputs
# ===============================
ticker = st.sidebar.text_input("Ticker", "AAPL")
sigma = st.sidebar.slider("Implied Vol (σ)", 0.1, 1.0, 0.3)
risk_free = 0.03
days_to_expiry = st.sidebar.slider("Days to Expiry (proxy)", 7, 180, 30)
T = days_to_expiry / 365

# ===============================
# Load data
# ===============================
price_df = get_price_data(ticker)
if price_df.empty:
    st.error("Price data not available")
    st.stop()

S_now = price_df.iloc[-1]["close"]

option_df = get_option_chain(ticker)

# ===============================
# Greeks time series (ATM 기준)
# ===============================
strike_atm = round(S_now)

price_df["delta"] = price_df["close"].apply(
    lambda S: bs_delta(S, strike_atm, T, risk_free, sigma)
)

price_df["gamma"] = price_df["close"].apply(
    lambda S: bs_gamma(S, strike_atm, T, risk_free, sigma)
)

price_df["delta_exposure"] = price_df["delta"] * S_now
price_df["gamma_exposure"] = price_df["gamma"] * (S_now ** 2)

# ===============================
# Charts – Time Series
# ===============================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Price vs Delta")
    st.line_chart(price_df.set_index("date")[["close", "delta"]])

with col2:
    st.subheader("📈 Price vs Gamma")
    st.line_chart(price_df.set_index("date")[["close", "gamma"]])

col3, col4 = st.columns(2)

with col3:
    st.subheader("📊 Delta Exposure (Time)")
    st.line_chart(price_df.set_index("date")["delta_exposure"])

with col4:
    st.subheader("📊 Gamma Exposure (Time)")
    st.line_chart(price_df.set_index("date")["gamma_exposure"])

# ===============================
# Gamma Exposure by Strike (GEX)
# ===============================
if not option_df.empty:
    option_df["gamma"] = option_df["strike_price"].apply(
        lambda K: bs_gamma(S_now, K, T, risk_free, sigma)
    )

    option_df["gex"] = (
        option_df["gamma"]
        * option_df["open_interest"].fillna(0)
        * (S_now ** 2)
        * 100
    )

    gex_by_strike = option_df.groupby("strike_price")["gex"].sum()

    st.subheader("🔥 Gamma Exposure by Strike")
    st.bar_chart(gex_by_strike)

# ===============================
# Option Chain Table
# ===============================
st.subheader("📋 Option Chain Snapshot")
st.dataframe(option_df.sort_values("strike_price"))

# ===============================
# Custom X–Y Analysis Tool
# ===============================
st.subheader("📐 Custom X–Y Analysis")

numeric_cols = [
    "close",
    "delta",
    "gamma",
    "delta_exposure",
    "gamma_exposure"
]

x_var = st.selectbox("X-axis", numeric_cols, index=0)
y_vars = st.multiselect(
    "Y-axis (multi)",
    numeric_cols,
    default=["delta", "gamma"]
)

custom_df = price_df[[x_var] + y_vars].copy()

normalize = st.checkbox("Normalize (0–1 scaling)", value=True)
if normalize:
    custom_df = (custom_df - custom_df.min()) / (custom_df.max() - custom_df.min())

st.line_chart(custom_df)


