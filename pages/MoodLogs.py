import os
import streamlit as st
import pandas as pd
from datetime import datetime as dt
import pytz
from mood_utils import analyze_text
import fitz  # PyMuPDF

local_tz = pytz.timezone("America/Los_Angeles")

if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(
        columns=["Timestamp", "Text", "Positive", "Neutral", "Negative"]
    )

st.title("📝 My Logs")
st.subheader("New Log")

log_method = st.radio("Choose log method...", ["Type something", "Upload a file"])

if log_method == "Type something":
    user_input = st.text_area("Type something...")
else:
    log_file = st.file_uploader("Upload a text file", type=["txt", "pdf"])
    if log_file is not None and ".txt" in log_file.name:
        parse_log = log_file.read().decode("utf-8")
        user_input = st.text_area("Preview", parse_log, height=100)
    elif log_file is not None and ".pdf" in log_file.name:
        parse_pdf = fitz.open(stream=log_file.read(), filetype="pdf")
        extracted_text = ""
        for pg in parse_pdf:
            extracted_text += pg.get_text()
        parse_pdf.close()
        user_input = st.text_area("Preview", extracted_text, height=100)
    else:
        user_input = ""

if st.button("Log Data"):
    if len(user_input.strip()) > 0:
        senti_scores = analyze_text(user_input)
        data_log = pd.DataFrame({
            "Timestamp": [dt.now(local_tz)],
            "Text":      [user_input],
            "Positive":  senti_scores["pos"],
            "Neutral":   senti_scores["neu"],
            "Negative":  senti_scores["neg"]
        })
        logs = "mood_logs.csv"
        if os.path.exists(logs):
            all_logs = pd.concat(
                [pd.read_csv(logs), data_log], axis=0, ignore_index=True
            )
            all_logs.to_csv(logs, index=False)
        else:
            data_log.to_csv(logs, index=False)
        st.session_state.data = pd.concat(
            [st.session_state.data, data_log], ignore_index=True
        )
        st.success("✅ Your entry has been logged successfully!")
        st.dataframe(data_log, use_container_width=True)
    else:
        st.warning("⚠️ Please check your log data and try again")

st.subheader("Past Logs")
logs = "mood_logs.csv"
if os.path.exists(logs):
    all_logs = pd.read_csv(logs)
    st.dataframe(all_logs, use_container_width=True)
elif not st.session_state.data.empty:
    st.dataframe(st.session_state.data, use_container_width=True)
else:
    st.info("No entries yet")

if st.button("Clear"):
    if os.path.exists(logs):
        os.remove(logs)
    st.session_state.data = pd.DataFrame(
        columns=["Timestamp", "Text", "Positive", "Neutral", "Negative"]
    )
    st.rerun()
