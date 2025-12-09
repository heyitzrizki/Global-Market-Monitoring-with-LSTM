import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from tensorflow.keras.models import load_model
import joblib
import os

st.set_page_config(page_title="Market Terminal", layout="wide")

# --------------------------
# Custom UI
# --------------------------
css = """
<style>
body { background-color: #0A0F1F; color: #F4F4F6; }
.sidebar .sidebar-content { background-color: #111A2C !important; }
h1, h2, h3 { color: #F4F4F6; }
.metric-card {
    background-color: #111A2C;
    padding: 15px;
    border-radius: 8px;
    border: 1px solid #1A2538;
    color: #F4F4F6;
}
.stButton>button {
    background-color: #2DD881;
    color: black;
    border-radius: 6px;
    padding: 0.6rem 1.2rem;
}
.stButton>button:hover {
    background-color: #6FEDB7;
    color: black;
}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

DATA_PATH = "data/global_market_master.csv"
MODEL_PATH = "models/lstm_model.keras"
SCALER_PATH = "models/scaler.pkl"

# --------------------------
# Load Data
# --------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"], index_col="Date")
    return df

df = load_data()

# --------------------------
# Sidebar Navigation
# --------------------------
page = st.sidebar.radio("", [
    "Overview",
    "Market Dashboard",
    "Correlation Map",
    "Forecasting",
    "Risk Scenario Simulator"
])

# --------------------------
# Page 1: Overview
# --------------------------
if page == "Overview":
    st.title("Market Overview")

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("USD/KRW", f"{latest['USD/KRW']:.2f}", f"{latest['USD/KRW']-prev['USD/KRW']:.2f}")
    c2.metric("VIX", f"{latest['VIX']:.2f}", f"{latest['VIX']-prev['VIX']:.2f}")
    c3.metric("Oil WTI", f"{latest['OIL_WTI']:.2f}", f"{latest['OIL_WTI']-prev['OIL_WTI']:.2f}")
    c4.metric("Gold", f"{latest['GOLD']:.2f}", f"{latest['GOLD']-prev['GOLD']:.2f}")
    c5.metric("KRW 30D Vol", f"{df['USD/KRW'].pct_change().rolling(30).std().iloc[-1]*100:.2f}%")

    st.write("Latest snapshot:")
    st.dataframe(df.tail())

# --------------------------
# Page 2: Market Dashboard
# --------------------------
elif page == "Market Dashboard":
    st.title("Market Dashboard")

    st.markdown("### FX Trend Panel")
    fx_cols = ["USD/KRW", "USD/JPY", "USD/CNY", "USD/IDR"]

    cc1, cc2 = st.columns(2)
    cc3, cc4 = st.columns(2)
    slots = [cc1, cc2, cc3, cc4]

    for col, box in zip(fx_cols, slots):
        data = df[col].tail(100)
        fig = px.line(data, template="plotly_dark", title=col)
        fig.update_traces(line_color="#2DD881")
        box.plotly_chart(fig, use_container_width=True)

# --------------------------
# Page 3: Correlation Map
# --------------------------
elif page == "Correlation Map":
    st.title("Correlation Map")

    corr = df.corr()
    fig = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        aspect="auto",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

# --------------------------
# Page 4: Forecasting
# --------------------------
elif page == "Forecasting":
    st.title("LSTM Forecasting")

    if not os.path.exists(MODEL_PATH):
        st.error("Model file missing.")
        st.stop()

    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    horizon = st.slider("Forecast horizon (days)", 1, 14, 7)

    if st.button("Run Forecast"):
        arr = df["USD/KRW"].values.reshape(-1,1)
        scaled = scaler.transform(arr)
        seq = scaled[-30:].reshape(1,30,1)

        preds_scaled = []
        for _ in range(horizon):
            p = model.predict(seq)[0][0]
            preds_scaled.append(p)
            seq = np.append(seq[:,1:,:], [[[p]]], axis=1)

        preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1,1)).flatten()
        dates = pd.date_range(df.index[-1], periods=horizon+1)[1:]

        forecast_df = pd.DataFrame({"Forecast": preds}, index=dates)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index[-120:], y=df["USD/KRW"].tail(120),
            mode="lines", line=dict(color="#4C8BF5", width=2),
            name="Recent"
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df.index, y=forecast_df["Forecast"],
            mode="lines+markers",
            line=dict(color="#2DD881", width=2),
            marker=dict(color="#00E6AC", size=8),
            name="Forecast"
        ))
        fig.update_layout(template="plotly_dark", title="USD/KRW Forecast")
        st.plotly_chart(fig, use_container_width=True)
        st.write(forecast_df)

# --------------------------
# Page 5: Risk Scenario Simulator
# --------------------------
elif page == "Risk Scenario Simulator":
    st.title("Risk Scenario Simulator")
    st.markdown("Simulate market shocks and estimate the impact on USD/KRW.")

    df_ret = df.pct_change().dropna()

    X = df_ret[["VIX", "OIL_WTI", "GOLD", "USD/JPY"]]
    y = df_ret["USD/KRW"]

    coef = np.linalg.lstsq(X.values, y.values, rcond=None)[0]
    beta_vix, beta_oil, beta_gold, beta_usdjpy = coef

    st.subheader("Manual Scenario Builder")

    c1, c2 = st.columns(2)
    vix_shock = c1.slider("VIX Shock (%)", -30, 50, 0)
    oil_shock = c2.slider("Oil Shock (%)", -30, 30, 0)
    gold_shock = c1.slider("Gold Shock (%)", -10, 10, 0)
    jpy_shock = c2.slider("USD/JPY Shock (%)", -5, 5, 0)

    vix_r = vix_shock / 100
    oil_r = oil_shock / 100
    gold_r = gold_shock / 100
    jpy_r = jpy_shock / 100

    predicted_move = (
        beta_vix * vix_r +
        beta_oil * oil_r +
        beta_gold * gold_r +
        beta_usdjpy * jpy_r
    )

    current_krw = df["USD/KRW"].iloc[-1]
    projected_krw = current_krw * (1 + predicted_move)

    st.metric("Estimated USD/KRW Change", f"{predicted_move*100:.2f}%", f"{projected_krw-current_krw:.2f}")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Predicted Move"],
        y=[predicted_move * 100],
        marker_color="#2DD881"
    ))
    fig.update_layout(template="plotly_dark", title="USD/KRW Estimated % Move")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Preset Scenarios")

    preset = st.selectbox("Choose a scenario:", [
        "None",
        "Risk-Off Panic",
        "Oil Crash",
        "Geopolitical Stress",
        "KRW Strengthening"
    ])

    if preset != "None":
        presets = {
            "Risk-Off Panic": (30, -5, 3, 1),
            "Oil Crash": (10, -12, 2, 0),
            "Geopolitical Stress": (15, 0, 5, 0),
            "KRW Strengthening": (-10, 2, -3, -2)
        }

        vix_s, oil_s, gold_s, jpy_s = presets[preset]

        predicted = (
            beta_vix * (vix_s/100) +
            beta_oil * (oil_s/100) +
            beta_gold * (gold_s/100) +
            beta_usdjpy * (jpy_s/100)
        )

        proj = current_krw * (1 + predicted)

        st.metric(f"{preset} Impact", f"{predicted*100:.2f}%")
        st.write(f"Projected USD/KRW: **{proj:.2f}**")

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=["KRW Reaction"],
            y=[predicted * 100],
            marker_color="#6FEDB7"
        ))
        fig2.update_layout(template="plotly_dark", title=f"{preset} Scenario Impact")
        st.plotly_chart(fig2, use_container_width=True)
