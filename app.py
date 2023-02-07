"""
This UI Module is used to detect if the text is generated by AI or not.
"""
import gc
import streamlit as st
from pathlib import Path
from utils.api_utils import LanguageModel

################################################
# Required in every Clarifai streamlit app
################################################

st.set_page_config(layout="wide")

# https://docs.streamlit.io/knowledge-base/deploy/authentication-without-sso
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True


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
    text = st.text_area("Enter text here")
    st.warning("Longer texts produce better results. Try to aim for 400-500 words")

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

    # Check if the text is generated by AI
    if text and st.session_state["model_name"]:
        st.session_state["language_model"].check_if_ai_generated(text)
