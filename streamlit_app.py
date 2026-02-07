import streamlit as st
import pandas as pd
import yfinance as yf  # 추가됨
import math
from datetime import date, timedelta

# ===============================
# Page setup
# ===============================
st.set_page_config(layout="wide", page_title="Options Greeks Dashboard")
st.title("📊 Options Greeks & Gamma Exposure Dashboard")

POLYGON_API_KEY = "mD0LX0bzkc3sIUH3Hs0lwNucRo90HtML"

# ===============================
# Math utils (Black-Scholes)
# ===============================
def norm_cdf(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def bs_delta(S, K, T, r, sigma, call=True):
    if T <= 0: return 0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return norm_cdf(d1) if call else norm_cdf(d1) - 1

def bs_gamma(S, K, T, r, sigma):
    if T <= 0: return 0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return math.exp(-d1**2 / 2) / (S * sigma * math.sqrt(2 * math.pi * T))

# ===============================
# Data loaders (yfinance 활용)
# ===============================
@st.cache_data(ttl=3600)  # 1시간 동안 데이터 캐싱 (API 호출 절약)
def get_price_data_yf(ticker, years=2):
    try:
        # yfinance를 사용하여 데이터 다운로드
        data = yf.download(ticker, period=f"{years}y", interval="1d")
        if data.empty:
            return pd.DataFrame()
        
        df = data.reset_index()
        # yfinance는 컬럼이 MultiIndex일 수 있어 단순화
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        df.rename(columns={"Date": "date", "Close": "close"}, inplace=True)
        return df[["date", "close"]]
    except Exception as e:
        st.error(f"yfinance 에러: {e}")
        return pd.DataFrame()

import requests
def get_option_chain(ticker):
    url = (
        f"https://api.polygon.io/v3/reference/options/contracts?"
        f"underlying_ticker={ticker}&limit=200&apiKey={POLYGON_API_KEY}"
    )
    try:
        res = requests.get(url).json()
        if "results" not in res:
            return pd.DataFrame()

        df = pd.DataFrame(res["results"])
        
        # 존재하는 컬럼만 안전하게 추출
        desired = ["strike_price", "expiration_date", "contract_type", "open_interest"]
        available = [c for c in desired if c in df.columns]
        
        final_df = df[available].copy()
        if "open_interest" not in final_df.columns:
            final_df["open_interest"] = 0
            
        return final_df
    except:
        return pd.DataFrame()

# ===============================
# Sidebar & Logic
# ===============================
ticker = st.sidebar.text_input("Ticker", "AAPL").upper()
sigma = st.sidebar.slider("Implied Vol (σ)", 0.1, 1.0, 0.3)
risk_free = 0.03
days_to_expiry = st.sidebar.slider("Days to Expiry (proxy)", 7, 180, 30)
T = days_to_expiry / 365

# 데이터 로드
price_df = get_price_data_yf(ticker)

if not price_df.empty:
    S_now = float(price_df.iloc[-1]["close"])
    st.metric(f"{ticker} Current Price", f"${S_now:.2f}")

    # Greeks 계산
    strike_atm = round(S_now)
    price_df["delta"] = price_df["close"].apply(lambda S: bs_delta(S, strike_atm, T, risk_free, sigma))
    price_df["gamma"] = price_df["close"].apply(lambda S: bs_gamma(S, strike_atm, T, risk_free, sigma))
    
    # Charts
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Price vs Delta")
        st.line_chart(price_df.set_index("date")[["close", "delta"]])
    with col2:
        st.subheader("📈 Price vs Gamma")
        st.line_chart(price_df.set_index("date")[["close", "gamma"]])

    # Option Chain & GEX
    option_df = get_option_chain(ticker)
    if not option_df.empty:
        st.subheader("🔥 Gamma Exposure by Strike")
        option_df["gamma"] = option_df["strike_price"].apply(lambda K: bs_gamma(S_now, K, T, risk_free, sigma))
        option_df["gex"] = option_df["gamma"] * option_df["open_interest"].fillna(0) * (S_now**2) * 100
        
        gex_chart = option_df.groupby("strike_price")["gex"].sum()
        st.bar_chart(gex_chart)
        
        st.subheader("📋 Option Chain Snapshot")
        st.dataframe(option_df)
else:
    st.error("데이터를 불러올 수 없습니다. 티커가 올바른지 확인해주세요.")


