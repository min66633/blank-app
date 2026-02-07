import streamlit as st
import numpy as np
from math import log, sqrt, exp, erf, pi

st.set_page_config(page_title="Options Greeks Monitor", layout="wide")

st.title("📊 Options Greeks Monitor (Black–Scholes)")

# --- 표준정규분포 함수 (scipy 대체) ---
def norm_cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))

def norm_pdf(x):
    return (1 / sqrt(2 * pi)) * exp(-0.5 * x * x)

# --- 입력 영역 ---
col1, col2, col3 = st.columns(3)

with col1:
    S = st.number_input("Underlying Price", value=100.0)

with col2:
    K = st.number_input("Strike Price", value=100.0)

with col3:
    T = st.number_input("Time to Expiry (years)", value=0.5)

r = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 2.0) / 100
sigma = st.slider("Implied Volatility (%)", 1.0, 100.0, 20.0) / 100

# --- Black–Scholes 계산 ---
d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
d2 = d1 - sigma * sqrt(T)

delta = norm_cdf(d1)
gamma = norm_pdf(d1) / (S * sigma * sqrt(T))
vanna = -norm_pdf(d1) * d2 / sigma
charm = -norm_pdf(d1) * (2*r*T - d2*sigma*sqrt(T)) / (2*T*sigma*sqrt(T))

# --- 출력 ---
st.subheader("📌 Greeks")

g1, g2, g3, g4 = st.columns(4)

g1.metric("Delta", f"{delta:.4f}")
g2.metric("Gamma", f"{gamma:.6f}")
g3.metric("Vanna", f"{vanna:.6f}")
g4.metric("Charm", f"{charm:.6f}")

st.caption("Model: Black–Scholes | No external libraries | Monitoring use")
