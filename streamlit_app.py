import streamlit as st
import pandas as pd
import yfinance as yf
import math
from datetime import date, timedelta

# ===============================
# Page setup
# ===============================
st.set_page_config(layout="wide", page_title="Advanced Options Greeks Dashboard")
st.title("📊 Options Greeks & Gamma Exposure Dashboard")

# ===============================
# Math utils (Black-Scholes)
# ===============================
def norm_cdf(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def bs_delta(S, K, T, r, sigma, call=True):
    if T <= 0 or sigma <= 0: return 0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return norm_cdf(d1) if call else norm_cdf(d1) - 1

def bs_gamma(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0: return 0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return math.exp(-d1**2 / 2) / (S * sigma * math.sqrt(2 * math.pi * T))

# ===============================
# Data loaders (yfinance 활용)
# ===============================
@st.cache_data(ttl=3600)
def get_full_data(ticker):
    # 가격 데이터 (2년치)
    data = yf.download(ticker, period="2y", interval="1d")
    if data.empty:
        return pd.DataFrame(), pd.DataFrame(), None
    
    price_df = data.reset_index()
    price_df.columns = [col[0] if isinstance(col, tuple) else col for col in price_df.columns]
    price_df.rename(columns={"Date": "date", "Close": "close"}, inplace=True)
    
    # 옵션 데이터 (가장 가까운 만기일)
    try:
        tk = yf.Ticker(ticker)
        exps = tk.options
        if not exps:
            return price_df, pd.DataFrame(), None
        
        target_expiry = exps[0]
        opt = tk.option_chain(target_expiry)
        calls, puts = opt.calls.copy(), opt.puts.copy()
        calls['contract_type'], puts['contract_type'] = 'call', 'put'
        
        option_df = pd.concat([calls, puts])
        option_df.rename(columns={"strike": "strike_price", "openInterest": "open_interest"}, inplace=True)
        return price_df, option_df, target_expiry
    except:
        return price_df, pd.DataFrame(), None

# ===============================
# Sidebar inputs
# ===============================
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Ticker", "AAPL").upper()
sigma = st.sidebar.slider("Implied Vol (σ)", 0.1, 1.0, 0.25)
risk_free = 0.04
days_to_expiry = st.sidebar.slider("Days to Expiry (T)", 1, 180, 30)
T_const = days_to_expiry / 365

# ===============================
# Load & Process Data
# ===============================
price_df, option_df, expiry_date = get_full_data(ticker)

if not price_df.empty:
    S_now = float(price_df.iloc[-1]["close"])
    st.metric(f"{ticker} Current Price", f"${S_now:.2f}")

    # --- 1. 시계열 Greeks 계산 (ATM 기준 역사적 추이) ---
    strike_atm = round(S_now)
    price_df["delta"] = price_df["close"].apply(lambda S: bs_delta(S, strike_atm, T_const, risk_free, sigma))
    price_df["gamma"] = price_df["close"].apply(lambda S: bs_gamma(S, strike_atm, T_const, risk_free, sigma))
    price_df["delta_exposure"] = price_df["delta"] * S_now
    price_df["gamma_exposure"] = price_df["gamma"] * (S_now ** 2)

    # --- 2. Charts – Time Series ---
    st.markdown("---")
    st.subheader("📈 Historical Greeks Analysis (ATM Proxy)")
    col1, col2 = st.columns(2)
    with col1:
        st.line_chart(price_df.set_index("date")[["close", "delta"]], use_container_width=True)
        st.caption("Price vs Delta History")
    with col2:
        st.line_chart(price_df.set_index("date")[["close", "gamma"]], use_container_width=True)
        st.caption("Price vs Gamma History")

    col3, col4 = st.columns(2)
    with col3:
        st.line_chart(price_df.set_index("date")["delta_exposure"])
        st.caption("Delta Exposure over Time")
    with col4:
        st.line_chart(price_df.set_index("date")["gamma_exposure"])
        st.caption("Gamma Exposure over Time")

    # --- 3. Gamma Exposure by Strike (GEX) ---
    st.markdown("---")
    if not option_df.empty:
        st.subheader(f"🔥 Dealer Gamma Exposure (Expiry: {expiry_date})")
        
        # GEX 계산 로직 (백만 달러 단위 + 풋 옵션 음수 처리)
        option_df["gamma_val"] = option_df["strike_price"].apply(lambda K: bs_gamma(S_now, K, T_const, risk_free, sigma))
        
        def calc_gex(row):
            flip = 1 if row['contract_type'] == 'call' else -1
            # Gamma * OI * S^2 * 0.01 (1% move) / 1,000,000
            return flip * row['gamma_val'] * row['open_interest'].fillna(0) * (S_now**2) * 0.01 / 10**6

        option_df["gex_mil"] = option_df.apply(calc_gex, axis=1)
        
        # 시각화 범위 필터링 (현재가 ±15%)
        mask = (option_df["strike_price"] > S_now * 0.85) & (option_df["strike_price"] < S_now * 1.15)
        gex_by_strike = option_df[mask].groupby("strike_price")["gex_mil"].sum()
        
        st.bar_chart(gex_by_strike)
        st.info("💡 Positive (Call) Gamma stabilizes price, Negative (Put) Gamma accelerates volatility.")

        # --- 4. Option Chain Table ---
        with st.expander("📋 View Full Option Chain Snapshot"):
            st.dataframe(option_df.sort_values("strike_price"), use_container_width=True)
    else:
        st.warning("옵션 데이터를 찾을 수 없어 GEX 차트를 표시할 수 없습니다.")

    # --- 5. Custom X–Y Analysis Tool ---
    st.markdown("---")
    st.subheader("📐 Custom X–Y Analysis")
    
    numeric_cols = ["close", "delta", "gamma", "delta_exposure", "gamma_exposure"]
    x_var = st.selectbox("X-axis (Independent)", numeric_cols, index=0)
    y_vars = st.multiselect("Y-axis (Dependent)", numeric_cols, default=["delta", "gamma"])

    if y_vars:
        custom_df = price_df[[x_var] + y_vars].copy()
        
        normalize = st.checkbox("Normalize (0–1 scaling for comparison)", value=True)
        if normalize:
            for col in custom_df.columns:
                c_min, c_max = custom_df[col].min(), custom_df[col].max()
                if c_max != c_min:
                    custom_df[col] = (custom_df[col] - c_min) / (c_max - c_min)
        
        st.line_chart(custom_df.set_index(x_var) if x_var in custom_df.columns else custom_df)

else:
    st.error("데이터를 로드할 수 없습니다. 티커를 확인해 주세요.")


