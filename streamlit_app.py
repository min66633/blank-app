import streamlit as st
import requests
import pandas as pd

st.set_page_config(layout="wide", page_title="Options Greeks (Polygon)")

API_KEY = st.secrets["POLYGON_API_KEY"]

st.title("📊 Options Greeks Monitor (Polygon)")

# --- 입력 ---
c1, c2, c3 = st.columns(3)
with c1:
    ticker = st.text_input("Ticker", "AAPL").upper()
with c2:
    option_type = st.selectbox("Option Type", ["call", "put"])
with c3:
    expiration = st.text_input("Expiration (YYYY-MM-DD)", "2026-03-20")

# --- 옵션 체인 스냅샷 ---
url = f"https://api.polygon.io/v3/snapshot/options/{ticker}"
params = {
    "expiration_date": expiration,
    "contract_type": option_type,
    "apiKey": API_KEY
}

res = requests.get(url, params=params).json()

if "results" not in res:
    st.error("옵션 데이터를 불러올 수 없습니다.")
    st.stop()

options = res["results"]

df = pd.json_normalize(options)
df = df.dropna(subset=["greeks.delta"])

strike = st.selectbox("Strike", sorted(df["details.strike_price"].unique()))
row = df[df["details.strike_price"] == strike].iloc[0]

# --- Greeks 카드 ---
g1, g2, g3, g4 = st.columns(4)

g1.metric("Delta", f"{row['greeks.delta']:.4f}")
g2.metric("Gamma", f"{row['greeks.gamma']:.6f}")
g3.metric("Theta", f"{row['greeks.theta']:.4f}")
g4.metric("Vega", f"{row['greeks.vega']:.4f}")

st.subheader("💵 Option Price")
st.write({
    "Last": row["last_trade.p"],
    "Bid": row["quote.bid"],
    "Ask": row["quote.ask"]
})

# --- 기초자산 가격 차트 ---
st.subheader("📈 Underlying Price (30D)")

price_url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/2025-01-01/2025-02-08"
price_res = requests.get(price_url, params={"apiKey": API_KEY}).json()

if "results" in price_res:
    price_df = pd.DataFrame(price_res["results"])
    price_df["date"] = pd.to_datetime(price_df["t"], unit="ms")
    st.line_chart(price_df.set_index("date")["c"])
