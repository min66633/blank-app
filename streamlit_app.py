import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta
from scipy.stats import norm

# ===============================
# 기본 설정
# ===============================
st.set_page_config(layout="wide")
st.title("Options Greeks Dashboard (Price-driven)")

POLYGON_API_KEY = "YOUR_POLYGON_API_KEY"  # 이미 등록돼 있으면 써도 됨

# ===============================
# 사용자 입력
# ===============================
ticker = st.text_input("Ticker", value="AAPL").upper()
option_type = st.selectbox("Option Type", ["call", "put"])

T_days = st.slider("Days to Expiration", 7, 180, 30)
T = T_days / 365

r = st.slider("Risk-free Rate (%)", 0.0, 5.0, 3.0) / 100
sigma = st.slider("Implied Volatility (%)", 5.0, 100.0, 25.0) / 100

# ===============================
# Polygon 가격 데이터 로드
# ===============================
end_date = date.today()
start_date = end_date - timedelta(days=365 * 2)  # 무료 플랜 안전 범위

price_url = (
    f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/"
    f"{start_date}/{end_date}?adjusted=true&apiKey={POLYGON_API_KEY}"
)

res = requests.get(price_url).json()

if "results" not in res:
    st.error("가격 데이터를 불러올 수 없습니다.")
    st.stop()

price_df = pd.DataFrame(res["results"])

# ===============================
# 컬럼 정리 (🔥 중요)
# ===============================
price_df = price_df.rename(columns={
    "c": "close",
    "o": "open",
    "h": "high",
    "l": "low",
    "v": "volume"
})

price_df["date"] = pd.to_datetime(price_df["t"], unit="ms")
price_df = price_df.sort_values("date")

# 안전 체크
assert "close" in price_df.columns

# ===============================
# 옵션 기준값 정의
# ===============================
S_current = price_df["close"].iloc[-1]
strike = S_current  # ATM 가정

# ===============================
# Black-Scholes Greeks 함수
# ===============================
def bs_d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def bs_delta(S, K, T, r, sigma, option):
    d1 = bs_d1(S, K, T, r, sigma)
    return norm.cdf(d1) if option == "call" else norm.cdf(d1) - 1

def bs_gamma(S, K, T, r, sigma):
    d1 = bs_d1(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

# ===============================
# Greeks 계산 (가격 기반)
# ===============================
price_df["delta"] = price_df["close"].apply(
    lambda S: bs_delta(S, strike, T, r, sigma, option_type)
)

price_df["gamma"] = price_df["close"].apply(
    lambda S: bs_gamma(S, strike, T, r, sigma)
)

# ===============================
# 1️⃣ Greeks 추이 차트
# ===============================
st.subheader("Delta & Gamma Trend (Price-driven)")
st.line_chart(
    price_df.set_index("date")[["delta", "gamma"]]
)

# ===============================
# 2️⃣ 가격 vs Delta
# ===============================
st.subheader("Price vs Delta")

fig1, ax1 = plt.subplots()
ax1.scatter(price_df["close"], price_df["delta"], alpha=0.5)
ax1.set_xlabel("Underlying Price")
ax1.set_ylabel("Delta")
st.pyplot(fig1)

# ===============================
# 3️⃣ Gamma Exposure (Strike Sweep)
# ===============================
st.subheader("Gamma Exposure by Strike")

strikes = np.arange(
    S_current * 0.8,
    S_current * 1.2,
    S_current * 0.02
)

gamma_by_strike = [
    bs_gamma(S_current, K, T, r, sigma) for K in strikes
]

fig2, ax2 = plt.subplots()
ax2.plot(strikes, gamma_by_strike)
ax2.axvline(S_current, linestyle="--", label="Current Price")
ax2.set_xlabel("Strike")
ax2.set_ylabel("Gamma")
ax2.legend()

st.pyplot(fig2)

# ===============================
# 4️⃣ Gamma 집중 구간 해석
# ===============================
max_gamma_strike = strikes[np.argmax(gamma_by_strike)]

st.markdown(
    f"""
### 🔥 Gamma 집중 구간
- **최대 Gamma Strike:** `{max_gamma_strike:.2f}`
- 현재 가격이 이 구간에 가까울수록  
  → **가격 변동성 확대 가능성**
"""
)

# ===============================
# 끝
# ===============================
st.success("대시보드 로드 완료")


