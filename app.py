"""
This UI module is meant to runs inference and post-processing on serial number part images.
"""
import streamlit as st
from pathlib import Path
from utils.api_utils import LanguageModel

################################################
# Required in every Clarifai streamlit app
################################################

st.set_page_config(layout="wide")

if "model_loaded" not in st.session_state:
    st.session_state["model_loaded"] = False

# Display main app name and information
st.image(
    "https://i.imgur.com/so0nwml.png",
    width=300,
)
st.title("AI Text Detector")
st.write("This is a Streamlit app that detects if the text is generated by AI or not.")

# Get configuration parameters
model_name = st.selectbox("Select language model", ["", "gpt2", "gpt2-medium", "gpt2-large"])
text = st.text_area("Enter text here")
st.warning("Longerexts produce better results. Try to aim for 400-500 words")

if "model_name" not in st.session_state:
    st.session_state["model_name"] = model_name
elif st.session_state["model_name"] != model_name:
    st.session_state["model_name"] = model_name
    st.session_state["model_loaded"] = False

model_loading_state = st.empty()
if not st.session_state["model_loaded"] and model_name:
    model_loading_state.warning("Loading model...")
    st.session_state["language_model"] = LanguageModel(model_name_or_path=st.session_state["model_name"])
    model_loading_state.success("Model Loaded")
    st.session_state["model_loaded"] = True

if text and st.session_state["model_name"]:
    st.session_state["language_model"].check_if_ai_generated(text)
