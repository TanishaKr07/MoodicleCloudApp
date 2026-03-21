import os
import json
import streamlit as st
import pandas as pd
import requests
import regex as re
import pytz
from datetime import datetime as dt

#applying the same font and styling as the rest of the app
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Comfortaa:wght@400;600;700&display=swap');
div[data-testid="stAppViewContainer"] * { font-family: 'Comfortaa', cursive !important; }
div[data-testid="stChatMessage"] { background-color: #2a2320 !important; border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

#loading Zen's character profile from ryo.json, same as your original launch.py
with open("character_profiles/ryo.json", "r") as reader:
    ryo = json.load(reader)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
local_tz = pytz.timezone("America/Los_Angeles") #using LA timezone, same as the rest of the app

def load_mood_context(): #reads the user's recent mood logs to give Zen some context before the chat begins
    path = "mood_logs.csv"
    if not os.path.exists(path): #if the user hasn't logged anything yet, tell Zen that
        return "The user has not logged any mood entries yet."
    df = pd.read_csv(path)
    if df.empty:
        return "The user has not logged any mood entries yet."
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    df["Timestamp_local"] = df["Timestamp"].dt.tz_convert(local_tz)
    recent = df.sort_values("Timestamp_local", ascending=False).head(5) #get the 5 most recent entries
    lines = []
    for _, row in recent.iterrows():
        ts   = row["Timestamp_local"].strftime("%b %d, %I:%M %p") #human-readable timestamp
        text = str(row["Text"])[:200] #truncate long entries so the prompt doesn't get too long
        pos  = round(float(row["Positive"]) * 100, 1)
        neu  = round(float(row["Neutral"])  * 100, 1)
        neg  = round(float(row["Negative"]) * 100, 1)
        lines.append(
            f'- [{ts}] "{text}" '
            f'(positive: {pos}%, neutral: {neu}%, negative: {neg}%)'
        )
    return "The user's 5 most recent mood log entries:\n" + "\n".join(lines)
    #this gets injected into the system prompt so Zen can reference the user's actual mood data

def build_prompt(user_input, character, chat):
    #this function is taken directly from your original prompt_engine.py — unchanged
    role = character["role"]
    personality = ", ".join(character["personality"])
    quirks = ", ".join(character["quirks"])
    instructions = "You are Zen, the user's personal trained therapist. \
        You reflect on the user's entries in an introspective and thoughtful way. \
            You are not just another chatbot, you are the user's companion, and someone \
                who helps the user make sense of their own emotions. Encourage the user to \
                    open up, ask questions to the user, offer advice if you can.\
                        Whatever you intend on saying to the user, put it in double quotations always.\
                            This is the most important part to distinguish your thinking from what you \
                                actually end up saying to the user.\n\n"
    
    #adding mood context on top of your original prompt so Zen is aware of the user's recent logs
    mood_context = load_mood_context()

    system_prompt = (
        f"{role}"
        f"Your personality is {personality}. "
        f"Your quirks: {quirks}. "
        f"{instructions}"
        f"Here is some context about the user's recent mood: {mood_context}\n\n"
        f"You have had the following conversation with the user till now: {chat}\
            "
    ) #concatenates all the multiple individual strings in this string-only tuple into a single, long string

    chat_prompts = (
        f"system: {system_prompt}\n"
        f"Reply only as Zen to the user's message: {user_input}\n"
    )
    #DeepSeek expects the prompt to be a single string, so we concatenate all parts into one string.
    #This is different from how OpenAI expects the prompt to be a list of messages.
    #Use OpenAI's format if you want to use a gpt-3.5-turbo model.
    return chat_prompts

def deepseek_call(user_input, chat): #user_input = user's msg to chatbot/ Zen; chat = current convo log
    #this function is taken directly from your original app.py — unchanged
    chat_prompt = build_prompt(user_input, ryo, chat)
    # chat_prompt is a string containing instructions to the system to behave like Zen (system_prompt),
    # the user's input message (user_input)
    response = requests.post( #API call to the TogetherAI API
        "https://api.groq.com/openai/v1/chat/completions", #switched to Groq — genuinely free, fast, and reliable
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        json={ #JSON object containing the parameters for the API call
            #Together AI's new endpoint uses OpenAI-style message format instead of a single prompt string
            #so we split the prompt into system and user roles
            "model": "llama-3.3-70b-versatile", #Llama 3.3 70B on Groq — free tier, no credits needed
            #if you get rate limited, try swapping the model for "mistralai/mistral-7b-instruct:free" or "google/gemma-2-9b-it:free"
            "messages": [
                {"role": "system", "content": chat_prompt}, #system prompt contains Zen's personality + chat history
                {"role": "user",   "content": user_input}   #the user's latest message
            ],
            "max_tokens": 1000, #maximum number of tokens in the response
            "temperature": 0.7, #controls the randomness of the response
            #lower temperature means the response is more predictable and focused,
            #while higher temperature means the response is more random and creative
        }
    )

    raw = response.json()
    #the new endpoint returns content inside choices[0]["message"]["content"] instead of output["choices"][0]["text"]
    if "choices" not in raw: #if the API returns an error, show it clearly instead of crashing silently
        return chat_prompt, f"API error: {raw}"
    full_response = raw["choices"][0]["message"]["content"]

    #extract Zen's reply from the full response
    all_options = re.findall(r'"[^"]+"', full_response) #must be double quotes at the start and end of the string
    #between those quotes, there must be at least (hence, +) one character, which can be anything
    #except a double quote (hence, [^"]).
    #the character class that we have defined over here is called a negated double-quotes character class.
    #that is, the class contains any character except a double quote.
    if len(all_options) >= 1 and all_options[-1][1:-1] != user_input:
        zen_reply = all_options[-1][1:-1] #just the first line
    else:
        zen_reply = "It seems like I am having trouble generating a response.\
                Please refresh this message to try again. Alternatively, please provide \
                    me a new message to respond to."
    return chat_prompt, zen_reply #both are strings

#the UI below replaces the Gradio interface from launch.py with native Streamlit components
#the chat logic above is untouched — this is purely a different window for the same brain

st.title("Chat with Zen 🧘")
st.markdown(
    "<p style='color:#8a7a6a;font-size:13px;margin-top:-12px;'>"
    "Your personal mental wellness companion</p>",
    unsafe_allow_html=True,
)

if not GROQ_API_KEY: #show a clear warning if the key isn't set up yet
    st.warning(
        "API key not configured. Add groq_apik() to apik.py to activate Zen.",
        icon="⚠️",
    )

if "zen_messages" not in st.session_state: #zen_messages stores the full conversation history
    st.session_state.zen_messages = [] #starts empty at the beginning of each session

if "zen_chat_log" not in st.session_state: #zen_chat_log is the plain string passed to deepseek_call
    st.session_state.zen_chat_log = "" #this mirrors the chat variable in your original new_chat()

#render all previous messages in the conversation so the user can scroll back through them
for msg in st.session_state.zen_messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

#show Zen's opening message the first time the page loads, before the user has said anything
if not st.session_state.zen_messages:
    opening = (
        "Hello 🌿 I'm Zen. I've had a look at your recent mood entries — "
        "I'm here whenever you're ready to talk. How are you feeling right now?"
    )
    with st.chat_message("assistant"):
        st.write(opening)
    st.session_state.zen_messages.append({"role": "assistant", "content": opening})
    #append the opening message to history so it doesn't re-render on the next rerun

if user_input := st.chat_input("Talk to Zen..."): #st.chat_input is Streamlit's equivalent of Gradio's ChatInterface input box
    with st.chat_message("user"):
        st.write(user_input) #display the user's message immediately
    st.session_state.zen_messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Zen is reflecting..."): #show a spinner while waiting for the API response
            try:
                chat_prompt, zen_reply = deepseek_call(user_input, st.session_state.zen_chat_log)
                #deepseek_call returns a tuple of chat_prompt and zen_reply, same as your original app.py
            except Exception as e:
                zen_reply = f"Something went wrong: {str(e)}"
                chat_prompt = ""
        st.write(zen_reply)
    st.session_state.zen_messages.append({"role": "assistant", "content": zen_reply})

    #append the interaction to the chat log, same as your original new_chat() function
    interaction = "user: " + user_input + "\n" + \
                  "zen: " + zen_reply + "\n"
    st.session_state.zen_chat_log += interaction
    #zen_chat_log is essentially the memory of the conversation so far, and is updated
    #in the system_prompt for each interaction to provide context for Zen's responses

if len(st.session_state.zen_messages) > 1: #only show the clear button once the conversation has started
    st.write("")
    if st.button("Start a new conversation", type="secondary"):
        st.session_state.zen_messages = [] #wipe the conversation history
        st.session_state.zen_chat_log = "" #wipe the chat log too so Zen starts fresh
        st.rerun()
