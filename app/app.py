import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import pandas as pd
import os

# Set page title and sidebar
st.set_page_config(page_title="Kinyarwanda Hate Speech Detection") #Urubuga rwo Gusuzuma Imvugo y’Urwango
st.sidebar.title("Kinyarwanda Hate Speech Detection") #Urubuga rwo Gusuzuma Imvugo y’Urwango
st.sidebar.markdown("---")
st.sidebar.button("Dashboard", disabled=True)  
st.sidebar.button("Statistics")     # Ibyerekeye Amakuru
st.sidebar.button("Settings")          # Igenamiterere

# Main header
st.title("Kinyarwanda Hate Speech Detection")

# Load model and tokenizer
@st.cache_resource
def load_model():
    # Get absolute path to model
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "checkpoint-best"))
    #st.write(f"Loading model from: {model_path}")
    if not os.path.exists(model_path):
        st.error(f"Model path does not exist: {model_path}")
        raise FileNotFoundError(f"Model path {model_path} not found")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    return model, tokenizer

model, tokenizer = load_model()


# Initialize session state for predictions and feedback
if "predictions" not in st.session_state:
    st.session_state.predictions = []
if "feedback" not in st.session_state:
    st.session_state.feedback = []

# Input section
st.subheader("Detect Hate Speech")
st.write("")
text = st.text_area("", placeholder="Type a message...", height=150)

# Detect and Clear buttons
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("Detect"):  # Detect
        if text:
            # Tokenize and predict
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                prediction = torch.argmax(probs, dim=-1).item()
                confidence = probs[0][prediction].item()
            
            # Map prediction to label (matches LabelEncoder: negative=0, neutral=1, positive=2)
            label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
            label = label_map[prediction]
            
            # Store prediction
            st.session_state.predictions.insert(0, {
                "Text": text,
                "Prediction": f"{label} ({confidence:.2%})"
            })
        else:
            st.warning("Andika ubutumwa bwa mbere!")

with col2:
    if st.button("Clear"):
        text = ""
        st.experimental_rerun()

# Display recent predictions
st.subheader("Recent Predictions")
if st.session_state.predictions:
    df = pd.DataFrame(st.session_state.predictions)
    st.table(df)
else:
    st.write("No recent predictions")

# Feedback section
st.subheader("Give Feedback")
feedback = st.text_input("Igitekerezo cyawe (si itegeko)", placeholder="Andika igitekerezo...")
if st.button("Tanga"):
    if feedback:
        st.success("Murakoze ku igitekerezo cyanyu!")
    else:
        st.warning("Andika igitekerezo bwa mbere!")

# Display feedback (for testing)
if st.session_state.feedback:
    st.subheader("Ibitekerezo Byakiriwe")
    feedback_df = pd.DataFrame(st.session_state.feedback)
    st.table(feedback_df)