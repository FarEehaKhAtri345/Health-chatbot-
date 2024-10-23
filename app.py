import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

def get_prediction(query):
    inputs = tokenizer(query, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    return logits

# Streamlit app
st.title("Health Symptom Chatbot")

# User input
user_input = st.text_input("Describe your symptoms:")

# Predict button
if st.button("Predict"):
    if user_input:
        prediction = get_prediction(user_input)
        st.write(f"Model output: {prediction}")
    else:
        st.write("Please enter your symptoms.")
