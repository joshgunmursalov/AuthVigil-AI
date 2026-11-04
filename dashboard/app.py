
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

# SentinelAI SOC - Advanced SOC Filtering

df["prediction"] = pd.to_numeric(
    df["prediction"], errors="coerce"
)

df["risk_score"] = pd.to_numeric(
    df["risk_score"], errors="coerce"
)

st.sidebar.header("SOC Investigation Filters")

# 1. Source IP Filter
ip_options = sorted(
    df["source_ip"].dropna().astype(str).unique()
)

selected_ips = st.sidebar.multiselect(
    "Select Source IPs",
    options=ip_options
)

# 2. Alert Status Filter
selected_status = st.sidebar.selectbox(
    "Alert Status",
    ["All", "Suspicious", "Normal"]
)

# 3. Date Range Filter
valid_dates = df["window_start_utc"].dropna()

if valid_dates.empty:
    st.error("No valid timestamps found.")
    st.stop()

min_date = valid_dates.min().date()
max_date = valid_dates.max().date()

date_range = st.sidebar.date_input(
    "Date Range (UTC)",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# 4. Apply Filters
filtered = df.copy()

if selected_ips:
    filtered = filtered[
        filtered["source_ip"].isin(selected_ips)
    ]

if selected_status == "Suspicious":
    filtered = filtered[
        filtered["prediction"] == 1
    ]

elif selected_status == "Normal":
    filtered = filtered[
        filtered["prediction"] == 0
    ]

if len(date_range) == 2:
    start_date, end_date = date_range

    filtered = filtered[
        filtered["window_start_utc"].dt.date.between(
            start_date, end_date
        )
    ]

# 5. SOC Metrics
alerts = filtered[
    filtered["prediction"] == 1
].copy()

st.subheader("SOC Detection Overview")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Analyzed Windows",
    len(filtered)
)

col2.metric(
    "Suspicious Windows",
    len(alerts)
)

col3.metric(
    "Normal Windows",
    int((filtered["prediction"] == 0).sum())
)

st.divider()

# 6. Top Suspicious IPs
st.subheader("Top Suspicious Source IPs")

if not alerts.empty:
    top_ips = (
        alerts["source_ip"]
        .value_counts()
        .head(10)
    )

    st.bar_chart(top_ips)
else:
    st.info("No suspicious IPs for these filters.")

# 7. Detection Timeline
st.subheader("Detection Timeline")

if not alerts.empty:
    timeline = (
        alerts.dropna(subset=["window_start_utc"])
        .set_index("window_start_utc")
        .resample("1D")
        .size()
    )

    st.line_chart(timeline)
else:
    st.info("No alerts in selected date range.")

# 8. SOC Alert Table
st.subheader("SOC Security Alerts")

st.dataframe(
    alerts.sort_values(
        "risk_score",
        ascending=False
    ),
    use_container_width=True,
    hide_index=True
)

st.caption(
    "Synthetic dataset | Experimental predictions | "
    "Risk scores are not calibrated probabilities."
)


# SentinelAI SOC - Alert Prioritization

st.divider()
st.subheader("SOC Alert Prioritization")

priority_alerts = alerts.copy()

def assign_priority(score):
    if pd.isna(score):
        return "Unknown"
    if score >= 0.90:
        return "High"
    elif score >= 0.70:
        return "Medium"
    return "Low"

priority_alerts["review_priority"] = (
    priority_alerts["risk_score"].apply(assign_priority)
)

priority_counts = (
    priority_alerts["review_priority"]
    .value_counts()
    .reindex(
        ["High", "Medium", "Low", "Unknown"],
        fill_value=0
    )
)

col1, col2, col3 = st.columns(3)

col1.metric("High Priority", priority_counts["High"])
col2.metric("Medium Priority", priority_counts["Medium"])
col3.metric("Low Priority", priority_counts["Low"])

st.subheader("Alerts by Review Priority")

st.bar_chart(
    priority_counts[["High", "Medium", "Low"]]
)

st.subheader("Prioritized SOC Alerts")

st.dataframe(
    priority_alerts.sort_values(
        "risk_score",
        ascending=False
    ),
    use_container_width=True,
    hide_index=True
)

st.download_button(
    label="Download Prioritized Alerts",
    data=priority_alerts.to_csv(index=False),
    file_name="sentinelai_prioritized_alerts.csv",
    mime="text/csv"
)

st.caption(
    "Review priority uses experimental score "
    "thresholds, not verified incident severity."
)

