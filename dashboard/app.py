
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="SentinelAI SOC",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ SentinelAI SOC")
st.subheader("AI-Powered Threat Detection Dashboard")

st.caption(
    "Synthetic Windows authentication data | "
    "Random Forest detection results"
)

uploaded_file = st.file_uploader(
    "Upload soc_predictions.csv",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Upload the prediction CSV to start.")
    st.stop()

df = pd.read_csv(uploaded_file)

required = [
    "window_start_utc", "source_ip",
    "failed_logins", "prediction", "risk_score"
]

if not set(required).issubset(df.columns):
    st.error("The uploaded CSV has missing columns.")
    st.stop()

df["window_start_utc"] = pd.to_datetime(
    df["window_start_utc"],
    utc=True,
    errors="coerce"
)

alerts = df[df["prediction"] == 1].copy()

col1, col2, col3 = st.columns(3)

col1.metric("Analyzed Windows", len(df))
col2.metric("Suspicious Windows", len(alerts))
col3.metric("Normal Windows", len(df) - len(alerts))

st.divider()

st.subheader("Top Suspicious Source IPs")
top_ips = alerts["source_ip"].value_counts().head(10)
st.bar_chart(top_ips)

st.subheader("Detection Timeline")
timeline = (
    alerts.dropna(subset=["window_start_utc"])
    .set_index("window_start_utc")
    .resample("1D")
    .size()
)
st.line_chart(timeline)

st.subheader("SOC Security Alerts")
st.dataframe(
    alerts.sort_values("risk_score", ascending=False),
    use_container_width=True
)

st.caption(
    "Predictions are experimental. Risk scores "
    "are not calibrated attack probabilities."
)
