import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline
from datetime import datetime as dt, time
import pytz
import os

@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True
    )

classifier = load_model()

if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=[
        "Timestamp", "Text", "Compound Score", "Positive", "Neutral", "Negative"
    ])

st.subheader("🫧 My Mood Map")

tf_choice = st.selectbox("Timeframe", ["Today", "This Week", "This Month",
                           "This Year", "Custom Dates"])
log_path = "mood_logs.csv"

if ("data" in st.session_state and not st.session_state.data.empty) \
        or (os.path.exists(log_path)):

    if os.path.exists(log_path):
        log_data = pd.read_csv(log_path)
    else:
        log_data = st.session_state.data

    local_tz = pytz.timezone("America/Los_Angeles")
    curr_day  = dt.now(local_tz).date()
    curr_week = dt.now(local_tz).isocalendar().week   # use local tz, not system tz
    curr_month = dt.now(local_tz).month
    curr_year  = dt.now(local_tz).year

    # ── FIX 1: parse with utc=True then convert to local ─────────────────────
    log_data["Timestamp"] = pd.to_datetime(log_data["Timestamp"], utc=True, errors="coerce")
    log_data["Timestamp_local"] = log_data["Timestamp"].dt.tz_convert(local_tz)
    # ─────────────────────────────────────────────────────────────────────────

    if tf_choice == "Today":
        log_data = log_data[log_data["Timestamp_local"].dt.date == curr_day]

    elif tf_choice == "This Week":
        log_data = log_data[
            log_data["Timestamp_local"].dt.isocalendar().week == curr_week
        ]

    elif tf_choice == "This Month":
        log_data = log_data[log_data["Timestamp_local"].dt.month == curr_month]

    elif tf_choice == "This Year":
        log_data = log_data[log_data["Timestamp_local"].dt.year == curr_year]

    else:  # Custom Dates
        custom_range = st.date_input(
            "Select a date range",
            value=(dt.now().date(), dt.now().date())
        )
        if len(custom_range) == 2:
            start_date, end_date = custom_range
        else:
            st.warning("Please select both a start date and an end date!")
            st.stop()
        start_dt = dt.combine(start_date, time.min).replace(tzinfo=local_tz)
        end_dt   = dt.combine(end_date,   time.max).replace(tzinfo=local_tz)
        log_data = log_data[
            (log_data["Timestamp_local"] >= start_dt) &
            (log_data["Timestamp_local"] <= end_dt)
        ]

    if log_data.shape[0] == 0:
        st.warning("No mood logs found for this timeframe!")

    else:
        text_data = log_data["Text"].dropna().astype(str).tolist()

        all_results = []
        for entry in text_data:
            res = classifier(entry)[0]
            results = {r["label"]: r["score"] for r in res}
            results["Entry"] = entry
            all_results.append(results)

        df = pd.DataFrame(all_results)
        emotion_cols = [c for c in df.columns if c != "Entry"]
        heatmap_data = df[emotion_cols].T

        colorscale = [
            [0.0,  "#fde9c4"],
            [0.16, "#f9d7a7"],
            [0.32, "#f5c285"],
            [0.48, "#e8a761"],
            [0.64, "#c97c5c"],
            [0.80, "#9c4f46"],
            [1.0,  "#622c25"],
        ]

        fig = px.imshow(
            heatmap_data,
            labels=dict(x="Entry", y="Emotion", color="Score"),
            x=[f"Entry {i+1}" for i in range(len(df))],
            y=emotion_cols,
            color_continuous_scale=colorscale,
        )
        fig.update_layout(
            width=1000,
            height=800,
            xaxis=dict(visible=False),
            yaxis=dict(title="Emotion"),
            margin=dict(l=100, r=50, t=50, b=50),
        )
        st.plotly_chart(fig)

else:
    st.warning("No mood logs found yet!")
