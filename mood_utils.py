#importing the required modules
import streamlit as st
st.set_page_config(page_title="senTIME", page_icon="🧠", layout="wide")
from transformers import pipeline

#Load emotion model (cached) so it doesn't reload every run
@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base", 
        return_all_scores=True
    )
classifier = load_model()
def analyze_text(text):
    res = classifier(text)[0]
    results = {r["label"]: r["score"] for r in res}
    pos = 0
    pos_num = 0
    neg = 0
    neg_num = 0
    neu = 0
    neu_num = 0
    for result in results.keys():
        if result in ["joy", "surprise", "trust", "anticipation"]:
            pos_num+=1
            pos+=results[result]
        elif result in ["sadness", "anger", "fear", "disgust"]:
            neg_num+=1
            neg+=results[result]
        elif result in ["calm", "neutral"]:
            neu_num+=1
            neu+=results[result]
    pos_avg = pos/pos_num
    neg_avg = neg/neg_num
    neu_avg = neu/neu_num
    senti_scores = {"pos":pos_avg, "neg":neg_avg, "neu":neu_avg}
    return senti_scores
