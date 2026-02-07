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
def get_option_chain_yf(ticker):
    try:
        tk = yf.Ticker(ticker)
        # 가장 가까운 만기일 하나를 선택
        exps = tk.options
        if not exps:
            return pd.DataFrame()
        
        # 첫 번째 만기일 데이터 가져오기 (가장 활발함)
        opt = tk.option_chain(exps[0])
        calls = opt.calls
        puts = opt.puts
        
        # 콜/풋 합치기
        calls['contract_type'] = 'call'
        puts['contract_type'] = 'put'
        df = pd.concat([calls, puts])
        
        # 컬럼명 통일 (Polygon 스타일 -> yfinance 스타일)
        df.rename(columns={
            "strike": "strike_price",
            "openInterest": "open_interest"
        }, inplace=True)
        
        return df[["strike_price", "open_interest", "contract_type"]]
    except Exception as e:
        st.error(f"Option 데이터 에러: {e}")
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
# 2. Gamma Exposure 차트 로직 수정
    option_df = get_option_chain_yf(ticker)
    
    if not option_df.empty:
        st.subheader(f"🔥 Dealer Gamma Exposure (Expiry: {expiry_date})")
        
        # Gamma 계산
        option_df["gamma"] = option_df["strike_price"].apply(lambda K: bs_gamma(S_now, K, T, risk_free, sigma))
        
        # GEX 계산 로직 변경 (Call은 +, Put은 -)
        def calculate_gex(row):
            # 딜러 입장에서 헷지하는 노출도 계산 (단위: 백만)
            flip = 1 if row['contract_type'] == 'call' else -1
            # 공식: Gamma * OI * (S^2) * 100(계약단위) / 10^6 (Million 단위)
            return flip * row['gamma'] * row['open_interest'] * (S_now**2) * 0.01 / 10**6

        option_df["gex_scaled"] = option_df.apply(calculate_gex, axis=1)
        
        # 현재가 근처 범위 필터링 (±15%)
        mask = (option_df["strike_price"] > S_now * 0.85) & (option_df["strike_price"] < S_now * 1.15)
        gex_chart = option_df[mask].groupby("strike_price")["gex_scaled"].sum()
        
        # 차트 출력
        st.bar_chart(gex_chart)
        st.caption("Y-축 단위: Million USD per 1% move (상대적 지표)")
        
        st.subheader("📋 Option Chain Snapshot")
        st.dataframe(option_df)
else:
    st.error("데이터를 불러올 수 없습니다. 티커가 올바른지 확인해주세요.")


