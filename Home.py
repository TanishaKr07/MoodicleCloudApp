# Home.py — Moodicle landing page
import os
import streamlit as st
import pandas as pd
from datetime import datetime as dt, time
import pytz
import plotly.graph_objects as go
import plotly.express as px
from transformers import pipeline

st.set_page_config(page_title="Moodicle", page_icon="🌿", layout="wide")

# ── Theme ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Comfortaa:wght@400;600;700&display=swap');
[data-testid="stSidebar"] { background-color: #302b28 !important; }
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .css-17eq0hr { color: #E7DFCF !important; font-weight: 600; }
[data-testid="stSidebar"] .stRadio > div > label:hover {
    background-color: #3d3530 !important; border-radius: 8px;
}
div[data-testid="stAppViewContainer"] * { font-family: 'Comfortaa', cursive !important; }
</style>
""", unsafe_allow_html=True)

local_tz = pytz.timezone("America/Los_Angeles")

# ── Cached models ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_emotion_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True
    )

classifier = load_emotion_model()

# ── Helpers ────────────────────────────────────────────────────────────────────
def load_logs():
    path = "mood_logs.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
        df["Timestamp_local"] = df["Timestamp"].dt.tz_convert(local_tz)
        return df
    return pd.DataFrame(columns=["Timestamp", "Timestamp_local",
                                  "Text", "Positive", "Neutral", "Negative"])

def compute_streak(df):
    if df.empty:
        return 0
    dates = sorted(df["Timestamp_local"].dt.date.unique(), reverse=True)
    today = dt.now(local_tz).date()
    streak, expected = 0, today
    for d in dates:
        if d == expected:
            streak += 1
            expected = (pd.Timestamp(expected) - pd.Timedelta(days=1)).date()
        elif d < expected:
            break
    return streak

def compute_habits_today():
    path = "moodbloom.csv"
    if not os.path.exists(path):
        return 0, 12
    df = pd.read_csv(path, index_col="date")
    today_key = dt.today().date().strftime("d_%Y-%m-%d")
    if today_key not in df.index:
        return 0, 12
    row = df.loc[today_key]
    return int(row.sum()), len(row)

def build_heatmap(subset):
    """Run emotion classifier on subset texts, return a plotly heatmap figure."""
    texts = subset["Text"].dropna().astype(str).tolist()
    all_results = []
    for entry in texts:
        res = classifier(entry, top_k=None)[0]
        row = {r["label"]: r["score"] for r in res}
        row["Entry"] = entry
        all_results.append(row)
    df = pd.DataFrame(all_results)
    emotion_cols = [c for c in df.columns if c != "Entry"]
    heatmap_data = df[emotion_cols].T

    colorscale = [
        [0.0,  "#fde9c4"], [0.16, "#f9d7a7"], [0.32, "#f5c285"],
        [0.48, "#e8a761"], [0.64, "#c97c5c"], [0.80, "#9c4f46"],
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
        xaxis=dict(visible=False),
        yaxis=dict(title=""),
        margin=dict(l=60, r=20, t=20, b=20),
        font=dict(color="white"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig

# ── Greeting ───────────────────────────────────────────────────────────────────
now = dt.now(local_tz)
hour = now.hour
if hour < 12:
    greeting = "Good morning ☀️"
elif hour < 17:
    greeting = "Good afternoon 🌤️"
else:
    greeting = "Good evening 🌙"

st.markdown(f"### {greeting}")
st.markdown(
    f"<p style='color:#8a7a6a;font-size:13px;margin-top:-10px;'>"
    f"{now.strftime('%A, %B')} {now.day}</p>",
    unsafe_allow_html=True
)
st.write("")

# ── Stat cards ─────────────────────────────────────────────────────────────────
log_data = load_logs()
streak      = compute_streak(log_data)
today_count = (log_data[log_data["Timestamp_local"].dt.date == now.date()].shape[0]
               if not log_data.empty else 0)
habits_done, habits_total = compute_habits_today()

card_style = ("background:#2a2320;border-radius:12px;padding:14px 18px;"
              "border:0.5px solid #3d3530;text-align:center;")

def stat_card(col, label, value, sub):
    col.markdown(
        f"<div style='{card_style}'>"
        f"<div style='font-size:11px;color:#7a6a5a;text-transform:uppercase;"
        f"letter-spacing:0.08em;margin-bottom:4px;'>{label}</div>"
        f"<div style='font-size:24px;font-weight:700;color:#e8ac54;'>{value}</div>"
        f"<div style='font-size:11px;color:#7a6a5a;margin-top:2px;'>{sub}</div>"
        f"</div>",
        unsafe_allow_html=True
    )

c1, c2, c3 = st.columns(3)
stat_card(c1, "Streak",       streak,                        "consecutive days")
stat_card(c2, "Today's logs", today_count,                   "entries")
stat_card(c3, "Habits done",  f"{habits_done}/{habits_total}", "today")

st.write("")

# ── Shared timeframe selector ──────────────────────────────────────────────────
st.subheader("Your Mood Trends")

tf_choice = st.selectbox(
    "Timeframe",
    ["Today", "This Week", "This Month", "This Year", "Custom Dates"]
)

subset    = pd.DataFrame()
empty_msg = ""

if not log_data.empty:
    curr_day   = now.date()
    curr_week  = now.isocalendar().week
    curr_month = now.month
    curr_year  = now.year

    if tf_choice == "Today":
        subset    = log_data[log_data["Timestamp_local"].dt.date == curr_day]
        empty_msg = "No log entries today!"
    elif tf_choice == "This Week":
        subset    = log_data[log_data["Timestamp_local"].dt.isocalendar().week == curr_week]
        empty_msg = "No log entries this week!"
    elif tf_choice == "This Month":
        subset    = log_data[log_data["Timestamp_local"].dt.month == curr_month]
        empty_msg = "No log entries this month!"
    elif tf_choice == "This Year":
        subset    = log_data[log_data["Timestamp_local"].dt.year == curr_year]
        empty_msg = "No log entries this year!"
    else:
        custom_range = st.date_input("Select a date range",
                                     value=(now.date(), now.date()))
        if len(custom_range) == 2:
            start_date, end_date = custom_range
        else:
            st.warning("Please select both a start and end date!")
            st.stop()
        start_dt = dt.combine(start_date, time.min).replace(tzinfo=local_tz)
        end_dt   = dt.combine(end_date,   time.max).replace(tzinfo=local_tz)
        subset    = log_data[(log_data["Timestamp_local"] >= start_dt) &
                             (log_data["Timestamp_local"] <= end_dt)]
        empty_msg = f"No log entries between {start_date} and {end_date}!"
else:
    st.info("Add a log entry to see your mood trends here!")

# ── Donut + heatmap side by side ───────────────────────────────────────────────
if not subset.empty:
    col_donut, col_heat = st.columns(2)

    # Donut
    with col_donut:
        avg_pos = subset["Positive"].mean()
        avg_neu = subset["Neutral"].mean()
        avg_neg = subset["Negative"].mean()
        total   = avg_pos + avg_neu + avg_neg
        fig_donut = go.Figure(go.Pie(
            values=[avg_pos / total, avg_neu / total, avg_neg / total],
            labels=["Positive", "Neutral", "Negative"],
            hole=0.6,
            marker_colors=["#e8ac54", "#d8cbb4", "#d27b6f"],
            textinfo="label+percent",
            hoverinfo="skip",
            textfont_size=13,
        ))
        fig_donut.update_layout(
            height=400,
            margin=dict(t=20, b=20, l=20, r=20),
            font=dict(color="white"),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # Emotion heatmap
    with col_heat:
        with st.spinner("Analysing emotions..."):
            fig_heat = build_heatmap(subset)
        st.plotly_chart(fig_heat, use_container_width=True)

elif empty_msg:
    st.info(empty_msg)

st.write("")

# ── Explore cards — 3 remaining pages ─────────────────────────────────────────
st.subheader("Explore")

def explore_card(icon, name, desc):
    return (
        f"<div style='background:#2a2320;border:0.5px solid #3d3530;"
        f"border-radius:12px;padding:16px 18px;'>"
        f"<div style='font-size:18px;margin-bottom:6px;'>{icon}</div>"
        f"<div style='font-size:14px;font-weight:700;color:#e8dfd0;margin-bottom:4px;'>{name}</div>"
        f"<div style='font-size:12px;color:#7a6a5a;line-height:1.5;'>{desc}</div>"
        f"</div>"
    )

ec1, ec2, ec3 = st.columns(3)

with ec1:
    st.markdown(explore_card("📝", "MoodLogs",
        "Write or upload mood entries and see your sentiment scores."),
        unsafe_allow_html=True)
    st.page_link("pages/MoodLogs.py", label="Click to Explore →")

with ec2:
    st.markdown(explore_card("🌱", "MoodBloom",
        "Check off daily habits and watch your plant grow."),
        unsafe_allow_html=True)
    st.page_link("pages/MoodBloom.py", label="Click to Explore →")

with ec3:
    st.markdown(explore_card("🧘", "MoodBot",
        "Chat with Zen, your personal mental wellness companion."),
        unsafe_allow_html=True)
    st.page_link("pages/MoodBot.py", label="Click to Explore →")
