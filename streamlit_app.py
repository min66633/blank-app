import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import date, timedelta
import math

# =====================================
# 페이지 설정
# =====================================
st.set_page_config(layout="wide")
st.title("Options Greeks Dashboard (Price-driven, No SciPy)")

# =====================================
# 사용자 입력
# =====================================
ticker = st.text_input("Ticker", value="AAPL").upper()
option_type = st.selectbox("Option Type", ["call", "put"])

T_days = st.slider("Days to Expiration", 7, 180, 30)
T = T_days / 365

r = st.slider("Risk-free Rate (%)", 0.0, 5.0, 3.0) / 100
sigma = st.slider("Implied Volatility (%)", 5.0, 100.0, 25.0) / 100

POLYGON_API_KEY = "mD0LX0bzkc3sIUH3Hs0lwNucRo90HtML"

# =====================================
# Polygon 가격 데이터
# =====================================
end_date = date.today()
start_date = end_date - timedelta(days=365 * 2)

url = (
    f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/"
    f"{start_date}/{end_date}?adjusted=true&apiKey={POLYGON_API_KEY}"
)

res = requests.get(url).json()

if "results" not in res:
    st.error("가격 데이터를 불러올 수 없습니다.")
    st.stop()

price_df = pd.DataFrame(res["results"])

price_df = price_df.rename(columns={
    "c": "close",
    "o": "open",
    "h": "high",
    "l": "low",
    "v": "volume"
})

price_df["date"] = pd.to_datetime(price_df["t"], unit="ms")
price_df = price_df.sort_values("date")

# =====================================
# Black-Scholes (표준 라이브러리만 사용)
# =====================================
def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def norm_pdf(x):
    return (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x)

def bs_d1(S, K, T, r, sigma):
    return (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))

def bs_delta(S, K, T, r, sigma, option):
    d1 = bs_d1(S, K, T, r, sigma)
    if option == "call":
        return norm_cdf(d1)
    else:
        return norm_cdf(d1) - 1

def bs_gamma(S, K, T, r, sigma):
    d1 = bs_d1(S, K, T, r, sigma)
    return norm_pdf(d1) / (S * sigma * math.sqrt(T))

# =====================================
# 옵션 기준값
# =====================================
S_current = price_df["close"].iloc[-1]
strike = S_current  # ATM 가정

# =====================================
# Greeks 계산 (가격 기반)
# =====================================
price_df["delta"] = price_df["close"].apply(
    lambda S: bs_delta(S, strike, T, r, sigma, option_type)
)

price_df["gamma"] = price_df["close"].apply(
    lambda S: bs_gamma(S, strike, T, r, sigma)
)

# =====================================
# 1️⃣ Greeks 추이 차트
# =====================================
st.subheader("Delta & Gamma Trend (Price-driven)")
st.line_chart(
    price_df.set_index("date")[["delta", "gamma"]]
)

# =====================================
# 2️⃣ 가격 vs Delta
# =====================================
st.subheader("Price vs Delta")
st.scatter_chart(
    price_df,
    x="close",
    y="delta"
)

# =====================================
# 3️⃣ Gamma Exposure (Strike Sweep)
# =====================================
st.subheader("Gamma Exposure by Strike")

strikes = np.arange(
    S_current * 0.8,
    S_current * 1.2,
    S_current * 0.02
)

gamma_values = [
    bs_gamma(S_current, K, T, r, sigma) for K in strikes
]

gamma_df = pd.DataFrame({
    "strike": strikes,
    "gamma": gamma_values
})

st.line_chart(
    gamma_df.set_index("strike")
)

# =====================================
# 4️⃣ Gamma 집중 구간 해석
# =====================================
max_gamma_strike = gamma_df.loc[gamma_df["gamma"].idxmax(), "strike"]

st.markdown(
    f"""
### 🔥 Gamma 집중 구간
- **최대 Gamma Strike:** `{max_gamma_strike:.2f}`
- 현재 가격: `{S_current:.2f}`  
- 이 구간 근처에서는 가격 변동이 **가속**되거나  
  **강하게 눌릴 가능성**이 있습니다.
"""
)

st.success("대시보드 로드 완료 (표준 라이브러리만 사용)")



