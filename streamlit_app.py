import streamlit as st
import requests
import pandas as pd
import numpy as np
from math import log, sqrt, exp, erf, pi

# =====================
# 기본 설정
# =====================
st.set_page_config(layout="wide", page_title="Options Greeks Dashboard")

API_KEY = st.secrets["POLYGON_API_KEY"]

st.title("📊 Options Greeks Dashboard (Market-based)")

# =====================
# 표준정규분포 함수
# =====================
def norm_cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))

def norm_pdf(x):
    return (1 / sqrt(2 * pi)) * exp(-0.5 * x * x)

# =====================
# Black–Scholes Greeks
# =====================
def greeks(S, K, T, r, sigma, option_type):
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    delta = norm_cdf(d1) if option_type == "call" else norm_cdf(d1) - 1
    gamma = norm_pdf(d1) / (S * sigma * sqrt(T))
    vega = S * norm_pdf(d1) * sqrt(T) / 100
    theta = - (S * norm_pdf(d1) * sigma) / (2 * sqrt(T)) / 365

    vanna = -norm_pdf(d1) * d2 / sigma
    charm = -norm_pdf(d1) * (2 * r * T - d2 * sigma * sqrt(T)) / (2 * T * sigma * sqrt(T))

    return delta, gamma, vega, theta, vanna, charm

# =====================
# 입력 UI
# =====================
c1, c2, c3, c4 = st.columns(4)

with c1:
    ticker = st.text_input("Ticker", "AAPL").upper()

with c2:
    option_type = st.selectbox("Option Type", ["call", "put"])

with c3:
    K = st.number_input("Strike", value=100.0)

with c4:
    T = st.number_input("Time to Expiry (years)", value=0.5)

r = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 2.0) / 100
sigma = st.slider("Implied Volatility (%)", 5.0, 100.0, 25.0) / 100

# =====================
# 기초자산 가격 불러오기 (Polygon)
# =====================
from datetime import date, timedelta

end_date = date.today()
start_date = end_date - timedelta(days=365 * 5)  # 최근 5년

price_url = (
    f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/"
    f"{start_date}/{end_date}"
)

res = requests.get(price_url, params={"apiKey": API_KEY}).json()

if "results" not in res:
    st.error("기초자산 가격을 불러올 수 없습니다.")
    st.stop()

price_df = pd.DataFrame(res["results"])
price_df["date"] = pd.to_datetime(price_df["t"], unit="ms")
S = price_df.iloc[-1]["c"]

st.subheader(f"📈 {ticker} Price")
st.metric("Last Price", f"{S:.2f}")
st.line_chart(price_df.set_index("date")["c"])

# =====================
# Greeks 계산
# =====================
delta, gamma, vega, theta, vanna, charm = greeks(S, K, T, r, sigma, option_type)

st.subheader("📌 Option Greeks")

g1, g2, g3, g4, g5, g6 = st.columns(6)

g1.metric("Delta", f"{delta:.4f}")
g2.metric("Gamma", f"{gamma:.6f}")
g3.metric("Vega", f"{vega:.4f}")
g4.metric("Theta (per day)", f"{theta:.4f}")
g5.metric("Vanna", f"{vanna:.4f}")
g6.metric("Charm", f"{charm:.4f}")

st.caption("Market price from Polygon | Greeks calculated (Black–Scholes)")

import numpy as np

def bs_delta(S, K, T, r, sigma, option="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option == "call":
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1
price_df = pd.DataFrame(res["results"])

price_df = price_df.rename(columns={
    "c": "close",
    "o": "open",
    "h": "high",
    "l": "low",
    "v": "volume"
})

price_df["date"] = pd.to_datetime(price_df["t"], unit="ms")

price_df["delta"] = price_df["close"].apply(
    lambda S: bs_delta(S, strike, T, r, sigma, option_type)
)

st.subheader("Delta Trend (Price-driven)")
st.line_chart(price_df.set_index("date")["delta"])

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.scatter(price_df["close"], price_df["delta"], alpha=0.5)
ax.set_xlabel("Underlying Price")
ax.set_ylabel("Delta")

st.pyplot(fig)

strikes = np.arange(
    price_df["close"].min() * 0.8,
    price_df["close"].max() * 1.2,
    5
)

def bs_gamma(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

current_price = price_df["close"].iloc[-1]

gamma_values = [
    bs_gamma(current_price, K, T, r, sigma) for K in strikes
]

fig, ax = plt.subplots()
ax.plot(strikes, gamma_values)
ax.axvline(current_price, linestyle="--")
ax.set_xlabel("Strike")
ax.set_ylabel("Gamma")

st.pyplot(fig)

