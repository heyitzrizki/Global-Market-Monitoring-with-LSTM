import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import joblib
import os

st.set_page_config(
    page_title="Global Market Monitoring",
    layout="wide"
)

DATA_PATH = "data/global_market_master.csv"
MODEL_PATH = "models/lstm_model.keras"
SCALER_PATH = "models/scaler.pkl"


@st.cache_data
def load_market_data():
    if not os.path.exists(DATA_PATH):
        st.error(f"Data file not found at: {DATA_PATH}")
        st.stop()
    return pd.read_csv(DATA_PATH, parse_dates=["Date"], index_col="Date")

def load_lstm():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        st.error("Model or scaler file missing. Please check /model folder.")
        st.stop()
    return load_model(MODEL_PATH), joblib.load(SCALER_PATH)

def prepare_sequences(df_scaled, lookback=30):
    window = df_scaled[-lookback:].values
    return np.expand_dims(window, axis=0)

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
page = st.sidebar.radio(
    "", 
    ["Overview", "Market Dashboard", "Correlation Explorer", "Forecasting"]
)

if page == "Overview":
    st.title("Global Market Monitoring Dashboard")

    c1, c2, c3 = st.columns(3)
    c1.metric("USD/KRW", f"{df['USD/KRW'].iloc[-1]:,.2f}")
    c2.metric("OIL WTI", f"{df['OIL_WTI'].iloc[-1]:,.2f}")
    c3.metric("VIX Index", f"{df['VIX'].iloc[-1]:,.2f}")

    st.markdown("Latest market dataset:")
    st.dataframe(df.tail())

elif page == "Market Dashboard":
    st.title("Market Dashboard")

    selected = st.selectbox("Select indicator:", cols)
    fig = px.line(df, y=selected, title=selected, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

elif page == "Correlation Explorer":
    st.title("Correlation Explorer")

    corr = df.corr()
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        aspect="auto",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

elif page == "Forecasting":
    st.title("LSTM Forecasting")

    model, scaler = load_lstm()
    horizon = st.slider("Forecast horizon (days)", 1, 14, 7)

    if st.button("Run Forecast"):
        preds = forecast_steps(model, scaler, df, steps=horizon)
        dates = pd.date_range(df.index[-1], periods=horizon+1, freq="D")[1:]

        res = pd.DataFrame({"Forecast": preds}, index=dates)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index[-60:], y=df["USD/KRW"].iloc[-60:], 
            mode="lines", name="Recent Trend"
        ))
        fig.add_trace(go.Scatter(
            x=res.index, y=res["Forecast"],
            mode="lines+markers", name="Forecast", line=dict(color="#00e6ac")
        ))
        fig.update_layout(
            template="plotly_dark",
            title="USD/KRW Forecast",
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        c1.write(res)
        c2.markdown(
            """
            **Interpretation:**
            - The model reacts to changes in FX, commodities, and volatility.
            - Useful for directional signals rather than exact numeric targets.
            - Forecast uncertainty increases as the horizon extends.
            """
        )
