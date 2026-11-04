
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


# SentinelAI SOC - AI Prediction Integration

from pathlib import Path
import joblib
import numpy as np

@st.cache_resource
def load_rf_assets():
    model_dir = Path(__file__).resolve().parent

    model = joblib.load(
        model_dir / "random_forest.joblib"
    )
    config = joblib.load(
        model_dir / "model_config.joblib"
    )

    features = list(config["feature_columns"])

    if len(features) != model.n_features_in_:
        raise ValueError("Model feature count mismatch.")

    if hasattr(model, "feature_names_in_"):
        if list(model.feature_names_in_) != features:
            raise ValueError("Model feature order mismatch.")

    return model, features


data_mode = st.radio(
    "Data Source",
    [
        "Existing Predictions",
        "New Feature CSV - Run AI"
    ],
    horizontal=True
)

if data_mode == "Existing Predictions":

    uploaded_file = st.file_uploader(
        "Upload soc_predictions.csv",
        type=["csv"],
        key="predictions_upload"
    )

    if uploaded_file is None:
        st.info("Upload the prediction CSV to start.")
        st.stop()

    df = pd.read_csv(uploaded_file)

else:

    new_file = st.file_uploader(
        "Upload 15-minute Feature CSV",
        type=["csv"],
        key="features_upload"
    )

    if new_file is None:
        st.info("Upload a feature CSV for AI analysis.")
        st.stop()

    try:
        model, feature_columns = load_rf_assets()
    except Exception as error:
        st.error(f"Model loading failed: {error}")
        st.stop()

    raw_df = pd.read_csv(new_file)

    if raw_df.empty:
        st.error("The uploaded CSV is empty.")
        st.stop()

    # Use test records for the original labeled dataset.
    if (
        "split" in raw_df.columns
        and raw_df["split"].eq("test").any()
    ):
        test_only = st.checkbox(
            "Use only test windows",
            value=True
        )

        if test_only:
            raw_df = raw_df[
                raw_df["split"] == "test"
            ].copy()

    required_columns = [
        "window_start_utc",
        "source_ip"
    ] + feature_columns

    missing = [
        col for col in required_columns
        if col not in raw_df.columns
    ]

    if missing:
        st.error(f"Missing features: {missing}")
        st.stop()

    features_df = raw_df[
        feature_columns
    ].apply(pd.to_numeric, errors="coerce")

    if not np.isfinite(
        features_df.to_numpy(dtype=float)
    ).all():
        st.error(
            "Feature columns contain missing "
            "or invalid numeric values."
        )
        st.stop()

    # Prevent test labels from being used as inputs.
    df = raw_df.drop(
        columns=["is_attack", "scenario", "split"],
        errors="ignore"
    ).copy()

    # Actual Random Forest inference
    df["prediction"] = model.predict(
        features_df
    ).astype(int)

    attack_class_index = list(
        model.classes_
    ).index(1)

    df["risk_score"] = model.predict_proba(
        features_df
    )[:, attack_class_index]

    df["alert_status"] = np.where(
        df["prediction"] == 1,
        "Suspicious",
        "Normal"
    )

    st.success(
        f"AI analysis completed: {len(df)} windows."
    )

    st.metric(
        "AI Detected Suspicious Windows",
        int(df["prediction"].sum())
    )

    st.download_button(
        "Download AI Predictions",
        data=df.to_csv(index=False),
        file_name="sentinelai_ai_predictions.csv",
        mime="text/csv"
    )

    st.caption(
        "Random Forest scores are not "
        "calibrated incident probabilities."
    )


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


# SentinelAI SOC - Incident Investigation Panel

st.divider()
st.subheader("SOC Incident Investigation")

if priority_alerts.empty:
    st.info("No alerts available for investigation.")

else:
    investigation_df = (
        priority_alerts
        .sort_values("risk_score", ascending=False)
        .reset_index(drop=True)
    )

    selected_id = st.selectbox(
        "Select an alert to investigate",
        options=list(range(len(investigation_df))),
        format_func=lambda i: (
            f"{investigation_df.loc[i, 'source_ip']} | "
            f"{investigation_df.loc[i, 'window_start_utc']} | "
            f"{investigation_df.loc[i, 'review_priority']}"
        )
    )

    incident = investigation_df.iloc[selected_id]

    st.markdown("### Selected Alert Details")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Failed Logins",
        int(incident["failed_logins"])
    )

    col2.metric(
        "Model Score",
        f"{incident['risk_score']:.4f}"
    )

    col3.metric(
        "Review Priority",
        incident["review_priority"]
    )

    st.markdown("### Authentication Evidence")

    evidence_columns = [
        "window_start_utc",
        "source_ip",
        "event_count",
        "failed_logins",
        "failure_rate",
        "unique_usernames",
        "rdp_logon_count",
        "off_hours_count",
        "risk_score",
        "review_priority"
    ]

    available_columns = [
        c for c in evidence_columns
        if c in investigation_df.columns
    ]

    st.dataframe(
        incident[available_columns]
        .to_frame(name="Value"),
        use_container_width=True
    )

    st.warning(
        "This alert is an experimental model "
        "prediction. Further investigation is "
        "required before confirming an incident."
    )


# SentinelAI SOC - Alert Triage Management

if not priority_alerts.empty:

    st.divider()
    st.subheader("SOC Alert Triage")

    case_key = (
        f"{incident['source_ip']}|"
        f"{incident['window_start_utc']}"
    )

    if "case_records" not in st.session_state:
        st.session_state["case_records"] = {}

    records = st.session_state["case_records"]
    previous = records.get(case_key, {})

    status_options = [
        "New",
        "Investigating",
        "Benign",
        "Escalated"
    ]

    status = st.selectbox(
        "Investigation Status",
        status_options,
        index=status_options.index(
            previous.get("status", "New")
        ),
        key=f"triage_status_{case_key}"
    )

    notes = st.text_area(
        "Analyst Investigation Notes",
        value=previous.get("notes", ""),
        placeholder="Enter investigation findings...",
        key=f"triage_notes_{case_key}"
    )

    if st.button("Save Investigation"):
        records[case_key] = {
            "source_ip": incident["source_ip"],
            "timestamp": str(
                incident["window_start_utc"]
            ),
            "model_score": float(
                incident["risk_score"]
            ),
            "review_priority": incident[
                "review_priority"
            ],
            "status": status,
            "notes": notes
        }

        st.success("Investigation saved in this session!")

    if records:
        st.subheader("Saved Investigations")

        case_df = pd.DataFrame(
            list(records.values())
        )

        st.dataframe(
            case_df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Export Investigation Report",
            data=case_df.to_csv(index=False),
            file_name="sentinelai_investigations.csv",
            mime="text/csv"
        )

    st.caption(
        "Investigation records are temporary. "
        "Export the CSV to preserve your work."
    )


# SentinelAI SOC - Automated Detection Rules

st.divider()
st.subheader("Automated Detection Rules")

if not priority_alerts.empty:

    def numeric_feature(name):
        value = pd.to_numeric(
            incident.get(name, 0),
            errors="coerce"
        )
        return float(value) if pd.notna(value) else 0.0

    failed = numeric_feature("failed_logins")
    failure_rate = numeric_feature("failure_rate")
    unique_users = numeric_feature("unique_usernames")
    rdp = numeric_feature("rdp_logon_count")
    off_hours = numeric_feature("off_hours_count")

    detected_rules = []

    # Rule 1: Excessive failed logins
    if failed >= 10 and failure_rate >= 0.70:
        detected_rules.append({
            "rule": "Excessive Failed Logins",
            "reason": (
                f"{int(failed)} failed logins; "
                f"failure rate {failure_rate:.1%}"
            ),
            "recommendation":
                "Review authentication failures and "
                "their source, timing, and affected accounts."
        })

    # Rule 2: Possible password spraying
    if unique_users >= 5 and failed >= 5:
        detected_rules.append({
            "rule": "Possible Password Spraying",
            "reason": (
                f"{int(unique_users)} unique usernames "
                f"and {int(failed)} failed logins"
            ),
            "recommendation":
                "Correlate failed logins across accounts "
                "and verify whether attempts share a source."
        })

    # Rule 3: Off-hours remote access
    if rdp >= 1 and off_hours >= 1:
        detected_rules.append({
            "rule": "Off-Hours Remote Access",
            "reason": (
                f"{int(rdp)} RDP logons and "
                f"{int(off_hours)} off-hours events"
            ),
            "recommendation":
                "Verify authorized remote access, "
                "user identity, and session timing."
        })

    if detected_rules:

        st.metric(
            "Matched Detection Rules",
            len(detected_rules)
        )

        for item in detected_rules:
            with st.expander(
                item["rule"],
                expanded=True
            ):
                st.write("Evidence:", item["reason"])
                st.write(
                    "Investigation guidance:",
                    item["recommendation"]
                )

    else:
        st.info(
            "No illustrative detection rules matched. "
            "The ML alert still requires investigation."
        )

    st.caption(
        "Rules are experimental heuristics, "
        "not confirmed incident classifications."
    )


# SentinelAI SOC - Detection Rule Coverage Analysis

st.divider()
st.subheader("Detection Rule Coverage Analysis")

if not priority_alerts.empty:

    rule_df = priority_alerts.copy()

    numeric_columns = [
        "failed_logins",
        "failure_rate",
        "unique_usernames",
        "rdp_logon_count",
        "off_hours_count"
    ]

    for col in numeric_columns:
        rule_df[col] = pd.to_numeric(
            rule_df[col], errors="coerce"
        ).fillna(0)

    # Rule 1
    rule_df["excessive_failures"] = (
        (rule_df["failed_logins"] >= 10) &
        (rule_df["failure_rate"] >= 0.70)
    )

    # Rule 2
    rule_df["possible_spraying"] = (
        (rule_df["unique_usernames"] >= 5) &
        (rule_df["failed_logins"] >= 5)
    )

    # Rule 3
    rule_df["off_hours_remote"] = (
        (rule_df["rdp_logon_count"] >= 1) &
        (rule_df["off_hours_count"] >= 1)
    )

    rule_columns = [
        "excessive_failures",
        "possible_spraying",
        "off_hours_remote"
    ]

    rule_df["matched_rule_count"] = (
        rule_df[rule_columns].sum(axis=1)
    )

    rule_names = {
        "excessive_failures": "Excessive Failed Logins",
        "possible_spraying": "Possible Password Spraying",
        "off_hours_remote": "Off-Hours Remote Access"
    }

    rule_df["matched_rules"] = rule_df.apply(
        lambda row: ", ".join(
            name for col, name in rule_names.items()
            if row[col]
        ) or "No rule matched",
        axis=1
    )

    total = len(rule_df)
    covered = int(
        (rule_df["matched_rule_count"] > 0).sum()
    )
    uncovered = total - covered

    c1, c2, c3 = st.columns(3)

    c1.metric("ML Alerts", total)
    c2.metric("Rule-Matched Alerts", covered)
    c3.metric("No Rule Matched", uncovered)

    st.metric(
        "Rule Coverage",
        f"{covered / total * 100:.1f}%"
    )

    st.subheader("Detection Rule Distribution")

    rule_counts = (
        rule_df[rule_columns].sum()
        .rename(index=rule_names)
    )

    st.bar_chart(rule_counts)

    st.subheader("Alert Coverage Details")

    st.dataframe(
        rule_df[
            [
                "window_start_utc",
                "source_ip",
                "risk_score",
                "review_priority",
                "matched_rule_count",
                "matched_rules"
            ]
        ].sort_values(
            "risk_score", ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Export Detection Rule Analysis",
        data=rule_df.to_csv(index=False),
        file_name="sentinelai_rule_coverage.csv",
        mime="text/csv"
    )

    st.caption(
        "Rule coverage measures agreement with "
        "experimental heuristics, not detection accuracy."
    )

else:
    st.info("No suspicious alerts to analyze.")


# SentinelAI SOC - Hybrid Detection Engine

st.divider()
st.subheader("Hybrid Detection Engine")

# Hazirda secilmis filterlere esasen butun
# ML normal ve suspicious pencerelerini yoxlayiriq
hybrid_df = filtered.copy()

required_features = [
    "prediction",
    "failed_logins",
    "failure_rate",
    "unique_usernames",
    "rdp_logon_count",
    "off_hours_count"
]

missing_features = [
    col for col in required_features
    if col not in hybrid_df.columns
]

if missing_features:
    st.error(
        f"Missing columns: {missing_features}"
    )

elif hybrid_df.empty:
    st.info("No windows match the current filters.")

else:
    for col in required_features:
        hybrid_df[col] = pd.to_numeric(
            hybrid_df[col], errors="coerce"
        )

    # Machine Learning Detection
    hybrid_df["ml_alert"] = (
        hybrid_df["prediction"] == 1
    )

    # Rule-Based Detection
    hybrid_df["rule_excessive_failures"] = (
        (hybrid_df["failed_logins"] >= 10) &
        (hybrid_df["failure_rate"] >= 0.70)
    )

    hybrid_df["rule_password_spraying"] = (
        (hybrid_df["unique_usernames"] >= 5) &
        (hybrid_df["failed_logins"] >= 5)
    )

    hybrid_df["rule_off_hours_remote"] = (
        (hybrid_df["rdp_logon_count"] >= 1) &
        (hybrid_df["off_hours_count"] >= 1)
    )

    rule_cols = [
        "rule_excessive_failures",
        "rule_password_spraying",
        "rule_off_hours_remote"
    ]

    hybrid_df["rule_alert"] = (
        hybrid_df[rule_cols].any(axis=1)
    )

    # Hybrid OR: ML OR Rules
    hybrid_df["hybrid_or"] = (
        hybrid_df["ml_alert"] |
        hybrid_df["rule_alert"]
    )

    # Hybrid AND: ML AND Rules
    hybrid_df["hybrid_and"] = (
        hybrid_df["ml_alert"] &
        hybrid_df["rule_alert"]
    )

    # Detection Statistics
    ml_count = int(hybrid_df["ml_alert"].sum())
    rule_count = int(hybrid_df["rule_alert"].sum())
    or_count = int(hybrid_df["hybrid_or"].sum())
    and_count = int(hybrid_df["hybrid_and"].sum())

    st.subheader("Detection Mode Comparison")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("ML Only", ml_count)
    c2.metric("Rules Only", rule_count)
    c3.metric("Hybrid OR", or_count)
    c4.metric("Hybrid AND", and_count)

    # Detection comparison chart
    comparison_df = pd.DataFrame({
        "Mode": [
            "ML Only",
            "Rules Only",
            "Hybrid OR",
            "Hybrid AND"
        ],
        "Alert Count": [
            ml_count,
            rule_count,
            or_count,
            and_count
        ]
    })

    st.bar_chart(
        comparison_df.set_index("Mode")
    )

    # Overlap Analysis
    st.subheader("ML and Rule Agreement")

    ml = hybrid_df["ml_alert"]
    rule = hybrid_df["rule_alert"]

    agreement = pd.DataFrame({
        "Rule matched": [
            int((ml & rule).sum()),
            int((~ml & rule).sum())
        ],
        "No rule matched": [
            int((ml & ~rule).sum()),
            int((~ml & ~rule).sum())
        ]
    }, index=["ML Suspicious", "ML Normal"])

    st.dataframe(agreement)

    # Interactive detection mode selector
    st.subheader("Select Detection Mode")

    detection_mode = st.selectbox(
        "Detection Strategy",
        [
            "ML Only",
            "Rules Only",
            "Hybrid OR",
            "Hybrid AND"
        ]
    )

    mode_columns = {
        "ML Only": "ml_alert",
        "Rules Only": "rule_alert",
        "Hybrid OR": "hybrid_or",
        "Hybrid AND": "hybrid_and"
    }

    selected_col = mode_columns[detection_mode]

    selected_alerts = hybrid_df[
        hybrid_df[selected_col]
    ].copy()

    st.metric(
        "Selected Mode Alerts",
        len(selected_alerts)
    )

    st.subheader("Hybrid Detection Results")

    display_columns = [
        "window_start_utc",
        "source_ip",
        "prediction",
        "risk_score",
        "ml_alert",
        "rule_alert",
        "hybrid_or",
        "hybrid_and"
    ]

    st.dataframe(
        selected_alerts[display_columns],
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Export Hybrid Detection Results",
        data=selected_alerts.to_csv(index=False),
        file_name="sentinelai_hybrid_results.csv",
        mime="text/csv"
    )

    st.caption(
        "Experimental comparison only. "
        "Alert counts do not measure accuracy. "
        "Ground-truth labels are required for "
        "precision, recall and F1 evaluation."
    )


# SentinelAI SOC - Model Performance Comparison

from pathlib import Path

st.divider()
st.subheader("Model Performance Comparison")

performance_path = (
    Path(__file__).resolve().parent
    / "sentinelai_model_performance.csv"
)

if performance_path.exists():

    perf = pd.read_csv(performance_path)

    st.caption(
        "Evaluation: 2,001 synthetic test windows. "
        "Results are fixed and independent of "
        "dashboard filters."
    )

    # Performance comparison table
    st.markdown("### Detection Performance Metrics")

    metric_cols = [
        "Precision", "Recall", "F1_score",
        "FPR", "Accuracy"
    ]

    display_perf = perf.copy()
    display_perf[metric_cols] = (
        display_perf[metric_cols] * 100
    ).round(2)

    st.dataframe(
        display_perf[
            ["Model"] + metric_cols
        ],
        use_container_width=True,
        hide_index=True
    )

    # Model comparison chart
    st.markdown("### Model Comparison Chart")

    chart_data = (
        perf.set_index("Model")[
            ["Precision", "Recall", "F1_score"]
        ] * 100
    )

    st.bar_chart(
    chart_data,
    stack=False,
    y_label="Performance (%)"
)

    # Individual model investigation
    st.markdown("### Individual Model Evaluation")

    selected_model = st.selectbox(
        "Select a model",
        perf["Model"].tolist(),
        key="performance_model"
    )

    model_row = perf[
        perf["Model"] == selected_model
    ].iloc[0]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Precision",
        f"{model_row['Precision']:.2%}"
    )
    c2.metric(
        "Recall",
        f"{model_row['Recall']:.2%}"
    )
    c3.metric(
        "F1-score",
        f"{model_row['F1_score']:.2%}"
    )
    c4.metric(
        "False Positive Rate",
        f"{model_row['FPR']:.2%}"
    )

    # Confusion Matrix
    st.markdown("### Confusion Matrix")

    cm = pd.DataFrame(
        [
            [int(model_row["TN"]),
             int(model_row["FP"])],
            [int(model_row["FN"]),
             int(model_row["TP"])]
        ],
        index=["Actual Normal", "Actual Attack"],
        columns=[
            "Predicted Normal",
            "Predicted Attack"
        ]
    )

    st.dataframe(cm, use_container_width=True)

    # Error statistics
    st.markdown("### Detection Error Summary")

    c1, c2 = st.columns(2)

    c1.metric(
        "Missed Attacks",
        int(model_row["FN"])
    )

    c2.metric(
        "False Alerts",
        int(model_row["FP"])
    )

    st.warning(
        "Evaluation uses synthetic data. "
        "Hybrid rules were developed after "
        "examining test results, so their "
        "comparison is exploratory."
    )

else:
    st.error("Model performance CSV not found.")
