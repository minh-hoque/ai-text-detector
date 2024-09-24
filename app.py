"""
This UI Module is used to detect if the text is generated by AI or not.
"""
import gc
import streamlit as st
from pathlib import Path
from utils.api_utils import LanguageModel, word_counter, check_password


################################################
# Required in every Clarifai streamlit app
################################################

st.set_page_config(layout="wide")

if check_password():
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

    # If the model name is not in session state, add it
    if "model_name" not in st.session_state:
        st.session_state["model_name"] = model_name
    # If the model name is changed, reload the model
    elif st.session_state["model_name"] != model_name:
        st.session_state["model_name"] = model_name
        st.session_state["model_loaded"] = False

    # Placeholder for model loading
    model_loading_state = st.empty()
    if not st.session_state["model_loaded"] and model_name != "":
        # Delete the model from session state and run garbage collection
        if "language_model" in st.session_state:
            del st.session_state["language_model"]
            gc.collect()

        # Load the model
        model_loading_state.warning("Loading model...")
        st.session_state["language_model"] = LanguageModel(model_name_or_path=st.session_state["model_name"])
        model_loading_state.success("Model Loaded")
        st.session_state["model_loaded"] = True

    text = st.text_area("Enter text here")
    st.warning(
        "**Minimum # words: 100 - Maximum # words: 750**. Longer texts produce better results. Try to aim for **400-500 words**",
        icon="🤖",
    )

    if st.button("Detect"):
        if st.session_state["model_name"] == "":
            st.warning("Please select a language model")
        else:
            word_length = word_counter(text)
            # Check if the text is generated by AI
            st.info("Your text has " + str(word_length) + " words !")
            if word_length >= 100 and word_length <= 750:
                st.session_state["language_model"].check_if_ai_generated(text)
            elif word_length < 100:
                st.error("Minimun word length required = 100", icon="🚨")
            else:
                st.error("Max word length = 750", icon="🚨")
