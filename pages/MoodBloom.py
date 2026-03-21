import streamlit as st
import pandas as pd
from streamlit_lottie import st_lottie
import json
import os
from datetime import datetime as dt

#st.set_page_config(page_title="Mood Bloom", layout="centered")
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

seeds = load_lottiefile("assets/animations/seeds.json")
seedling = load_lottiefile("assets/animations/seedling.json")
plant = load_lottiefile("assets/animations/plant.json")
flower = load_lottiefile("assets/animations/flower.json")

path = "moodbloom.csv"
checks = ["workout", "sleep", "water", "food", "walk", "meditation",
          "screentime", "gratitude", "connect", "nature", "song",
          "creative"]
today_date = dt.today().date().strftime("d_%Y-%m-%d")
if os.path.exists(path):
    moodbloom_df = pd.DataFrame(pd.read_csv(path,index_col="date"))
    if today_date not in list(moodbloom_df.index):
        today_habits={checks[i]: 0 for i in range(len(checks))}
    else:
        today_habits = moodbloom_df.loc[today_date]
    habit_vals = {checks[i]:today_habits[checks[i]] for i in range(len(checks))}
else:
    moodbloom_df = pd.DataFrame(columns=["date"] + checks).set_index("date")
    habit_vals = {checks[i]: 0 for i in range(len(checks))}
with st.container():
    st.subheader("Daily Checklist 🌱")
    st.write("Track your daily mood-lifting habits and watch your plant grow!")
    st.write("")

    col1,col2 = st.columns([1,1.5])

    # pre-load today's saved values into session state ONCE on first load,
    # then let session state alone drive the checkboxes — we must NOT also
    # pass value= to st.checkbox when a key is already in session_state,
    # otherwise Streamlit throws the "widget set via Session State API" warning
    for habit in checks:
        if habit not in st.session_state:
            st.session_state[habit] = bool(habit_vals[habit])

    with col1:
        habits = {
            "workout":    st.checkbox("🏋️ Did a short workout",                               key="workout"),
            "sleep":      st.checkbox("🛌 Got 7+ hours of sleep last night",                  key="sleep"),
            "water":      st.checkbox("💧 Drank 8+ glasses of water",                         key="water"),
            "food":       st.checkbox("🍛 Ate a balanced meal",                               key="food"),
            "walk":       st.checkbox("🚶 Went for a 15+ minute walk or stretching",          key="walk"),
            "meditation": st.checkbox("🧘 Practiced 5+ minutes of mindfulness or meditation", key="meditation"),
            "screentime": st.checkbox("📱 Limited social media use",                          key="screentime"),
            "gratitude":  st.checkbox("🙏 Expressed gratitude",                               key="gratitude"),
            "connect":    st.checkbox("🤗 Connected with a friend",                           key="connect"),
            "nature":     st.checkbox("🌻 Spent time in nature, sunlight, or with pets",      key="nature"),
            "song":       st.checkbox("🎶 Listened to a favorite song or podcast",            key="song"),
            "creative":   st.checkbox("🎨 Engaged in a hobby or tried something new",         key="creative"),
        }
        for habit in habit_vals:
            habit_vals[habit]=habits[habit]
        moodbloom_df.loc[today_date]=habit_vals
        moodbloom_df.to_csv(path, index=True)

    with col2:
        habits_checked = sum(habits.values())
        if habits_checked <= 3:
            st_lottie(seeds, height=300, key="seeds")
        elif habits_checked <= 6:
            st_lottie(seedling, height=300, key="seedling")
        elif habits_checked <= 9:
            st_lottie(plant, height=300, key="plant")
        else:
            st_lottie(flower, height=300, key="flower")

        
        st.markdown(
        f"<div style='text-align: center; padding-top: 10px;'>"
        f"Progress: {habits_checked}/{len(habits)} habits completed 🌟"
        f"</div>",
        unsafe_allow_html=True
    )
# ── Habit completion calendar heatmap (moved from My Stats) ───────────────────
import plotly.express as px

st.write("")
st.subheader("Habit History 📅")

month_map = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December"
}
weekday_map = {
    0: "Monday", 1: "Tuesday", 2: "Wednesday",
    3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"
}

if os.path.exists(path):
    user_month = st.selectbox("Month", list(month_map.values()),
                              index=dt.today().month - 1)
    cal_data = pd.read_csv(path)
    cal_data["date"] = cal_data["date"].apply(lambda x: pd.to_datetime(x[2:]))
    cal_data["month"] = cal_data["date"].dt.month.apply(lambda x: month_map[x])
    cal_data = cal_data[cal_data["month"] == user_month].drop(columns="month")

    if cal_data["date"].dropna().empty:
        st.info("No habit data for this month yet!")
    else:
        dates = pd.date_range(cal_data["date"].min(), cal_data["date"].max(), freq="D")
        falses = [False] * (len(cal_data.columns) - 1)
        for d in dates:
            if d not in list(cal_data["date"]):
                cal_data.loc[len(cal_data)] = [d] + falses

        cal_data = cal_data.set_index("date")
        cal_data["completion"] = cal_data.sum(axis=1) / cal_data.shape[1]
        cal_data = cal_data.reset_index()

        cal_data["week"] = cal_data["date"].dt.day.apply(
            lambda x: "Week 1" if x in range(1, 8) else (
                "Week 2" if x in range(8, 15) else (
                    "Week 3" if x in range(15, 22) else "Week 4"
                )
            )
        )
        days_order = ["Monday", "Tuesday", "Wednesday", "Thursday",
                      "Friday", "Saturday", "Sunday"]
        cal_data["weekday"] = pd.Categorical(
            cal_data["date"].dt.day_of_week.apply(lambda x: weekday_map[x]),
            categories=days_order, ordered=True
        )
        pivot = cal_data.pivot_table(columns="weekday", values="completion", index="week")
        pivot = pivot.fillna(0)
        pivot.columns = [str(c) for c in pivot.columns]

        mood_warm = [
            [0.0, "#fde9c4"], [0.16, "#f9d7a7"], [0.32, "#f5c285"],
            [0.48, "#e8a761"], [0.64, "#c97c5c"], [0.8,  "#9c4f46"],
            [1.0, "#622c25"]
        ]
        fig_pivot = px.imshow(
            pivot,
            labels=dict(x="Day of week", y="Week", color="Completion"),
            color_continuous_scale=mood_warm,
            aspect="auto"
        )
        fig_pivot.update_layout(
            title=f"{user_month} Habit Completion",
            xaxis_title="Day of Week",
            yaxis_title="Week",
            height=400,
            width=800
        )
        st.plotly_chart(fig_pivot)
else:
    st.info("Complete some habits above to see your history here!")
