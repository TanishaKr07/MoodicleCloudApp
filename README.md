# Moodicle 🌿

Moodicle is a personal mood tracking and mental wellness app born at the intersection of my two disciplines - cognitive behavioral neuroscience and data science. I love journaling, but I wanted to go one step further than just express my feelings; I wanted to explore them - see how the emotional tone changed over time, if any trends emerged, track my habits that may influence how I feel. In addition, I thought, wouldn't it be great to have a personal wellness-centered chatbot that could not only suggest wellbeing tips but also inform them based on the patterns in my recent entries? 

And that is how Moodicle was born.
 
---

## Features

### 📝 MoodLogs
Write or upload a journal entry and Moodicle runs it through a fine-tuned RoBERTa emotion classifier — `j-hartmann/emotion-english-distilroberta-base` - returning scores across 7 emotions: joy, sadness, anger, fear, disgust, surprise, and neutral. Those scores are aggregated into positive, neutral, and negative sentiment and saved over time so you can track your emotional tone across days, weeks, and months.

### 📊 Mood Trends
The home dashboard visualises your mood data as an interactive donut chart showing the breakdown of positive, neutral, and negative sentiment, alongside an emotion heatmap that shows which specific emotions dominated your entries across any timeframe you choose.

### 🌱 MoodBloom
A daily checklist of 12 mood-lifting habits — from sleep and hydration to mindfulness and creative expression. Check them off and a Lottie plant animation grows with every 4 habits completed. Your habit history is visualised as a weekly heatmap so you can see consistency over time.

### 🧘 MoodBot (Zen)
Zen is a custom AI therapist persona built on Llama 3.3 70B via the Groq API. Before the conversation begins, Zen reads your 5 most recent mood entries and uses them as context — so responses are grounded in your actual emotional patterns, not generic advice.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Framework | Streamlit |
| Language | Python |
| Emotion model | HuggingFace Transformers — distilRoBERTa |
| AI companion | Groq API — Llama 3.3 70B |
| Visualisation | Plotly |
| Data | Pandas |
| Animations | Streamlit Lottie |
| PDF parsing | PyMuPDF |

---
