import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import joblib
import os

st.set_page_config(page_title="Market Dashboard", layout="wide")

# --------------------------
# Custom UI Styling
# --------------------------
custom_css = """
<style>

html, body, [class*="css"]  {
    background-color: #0A0F1F !important;
    color: #F4F4F6 !important;
}

.sidebar .sidebar-content {
    background-color: #111A2C !important;
}

h1, h2, h3, h4 {
    color: #F4F4F6 !important;
    letter-spacing: 0.5px;
}

.metric-card {
    padding: 18px;
    border-radius: 8px;
    background-color: #111A2C;
    color: #F4F4F6;
    margin-bottom: 10px;
    border: 1px solid #1A2538;
}

.stButton>button {
    background-color: #2DD881;
    color: black;
    border-radius: 6px;
    padding: 0.6rem 1.2rem;
    border: none;
}

.stButton>button:hover {
    background-color: #6FEDB7;
    color: black;
}

</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

DATA_PATH = "data/global_market_master.csv"
MODEL_PATH = "models/lstm_model.keras"
SCALER_PATH = "models/scaler.pkl"

@st.cache_data
def load_market_data():
    if not os.path.exists(DATA_PATH):
        st.error("Data file missing.")
        st.stop()
    return pd.read_csv(DATA_PATH, parse_dates=["Date"], index_col="Date")

def load_lstm():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        st.error("Model files missing.")
        st.stop()
    return load_model(MODEL_PATH), joblib.load(SCALER_PATH)

def prepare_sequences(df_scaled, lookback=30):
    seq = df_scaled[-lookback:].values
    return np.expand_dims(seq, axis=0)

def forecast_steps(model, scaler, df, steps=7, lookback=30, target="USD/KRW"):
    scaled = scaler.transform(df)
    scaled_df = pd.DataFrame(scaled, columns=df.columns, index=df.index)
    window = prepare_sequences(scaled_df, lookback)
    preds_scaled = []

    for _ in range(steps):
        pred = model.predict(window)[0][0]
        preds_scaled.append(pred)

        new_row = window[0][-1].copy()
        new_row[df.columns.get_loc(target)] = pred
        updated = np.vstack([window[0][1:], new_row])
        window = np.expand_dims(updated, 0)

    restore = np.zeros((steps, df.shape[1]))
    restore[:, df.columns.get_loc(target)] = preds_scaled
    final = scaler.inverse_transform(restore)[:, df.columns.get_loc(target)]
    return final


df = load_market_data()
cols = df.columns.tolist()

st.sidebar.title("Menu")
page = st.sidebar.radio("", ["Overview", "Market Dashboard", "Correlation Map", "Forecasting"])

# -------------------------------------
# Overview
# -------------------------------------
if page == "Overview":
    st.title("Market Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric(label="USD/KRW", value=f"{df['USD/KRW'].iloc[-1]:,.2f}")
    c2.metric(label="VIX", value=f"{df['VIX'].iloc[-1]:,.2f}")
    c3.metric(label="OIL WTI", value=f"{df['OIL_WTI'].iloc[-1]:,.2f}")

    st.write("Latest snapshot:")
    st.dataframe(df.tail())

# -------------------------------------
# Market Dashboard
# -------------------------------------
elif page == "Market Dashboard":
    st.title("Market Dashboard")

    selected = st.selectbox("Select indicator:", cols)

    fig = px.line(
        df, y=selected, 
        title=selected,
        template="plotly_dark",
        color_discrete_sequence=["#2DD881"]
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------
# Correlation Map
# -------------------------------------
elif page == "Correlation Map":
    st.title("Correlation Map")

    corr = df.corr()
    fig = px.imshow(
        corr, text_auto=True,
        color_continuous_scale="RdBu_r",
        aspect="auto",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------
# Forecasting
# -------------------------------------
elif page == "Forecasting":
    st.title("LSTM Forecasting")

    model, scaler = load_lstm()
    horizon = st.slider("Forecast horizon (days)", 1, 14, 7)

    if st.button("Run Forecast"):
        preds = forecast_steps(model, scaler, df, steps=horizon)
        dates = pd.date_range(df.index[-1], periods=horizon+1, freq="D")[1:]
        result = pd.DataFrame({"Forecast": preds}, index=dates)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index[-90:], y=df["USD/KRW"].iloc[-90:],
            mode="lines",
            line=dict(color="#4C8BF5", width=2),
            name="Recent Trend"
        ))
        fig.add_trace(go.Scatter(
            x=result.index, y=result["Forecast"],
            mode="lines+markers",
            line=dict(color="#2DD881", width=2),
            marker=dict(color="#00E6AC", size=8),
            name="Forecast"
        ))

        fig.update_layout(
            template="plotly_dark",
            title="USD/KRW Forecast",
            hovermode="x unified"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.write(result)
