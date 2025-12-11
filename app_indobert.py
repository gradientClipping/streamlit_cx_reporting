import streamlit as st
import pandas as pd
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    recall_score,
    precision_score
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

st.set_page_config(
    page_title="IndoBERT Classifier",
    layout="wide"
)

DATA_PATH = "raw_data_classifier_full.csv" 
FEEDBACK_FILE = "new_training_data.csv"
REMOTE_MODEL_ID = "juangwijaya/indobert-2-neg-cx"
RANDOM_STATE = 42

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

@st.cache_data
def load_and_prep_data(filepath):
    if not os.path.exists(filepath):
        return None
    try:
        df = pd.read_csv(filepath)
        if "label" in df.columns:
            df["label"] = df["label"].astype(str).str.strip().str.lower()
            label_map = {"yes": 1, "no": 0, "1": 1, "0": 0}
            df["label"] = df["label"].map(label_map)
            df = df.dropna(subset=["label"])
            df["label"] = df["label"].astype(int)
        df["message"] = df["message"].astype(str)
        return df
    except Exception:
        return None

def get_splits(df):
    if df is None: return None, None
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        random_state=RANDOM_STATE
    )
    return train_df, val_df

@st.cache_resource
def load_model_pipeline():
    try:
        tokenizer = AutoTokenizer.from_pretrained(REMOTE_MODEL_ID)
        model = AutoModelForSequenceClassification.from_pretrained(REMOTE_MODEL_ID)
        model.to(DEVICE)
        model.eval()
        return tokenizer, model
    except Exception as e:
        return None, None

def predict_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding="max_length")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    with torch.no_grad():
        logits = model(**inputs).logits
    
    probs = torch.nn.functional.softmax(logits, dim=-1)
    pred_label = torch.argmax(probs, dim=-1).item()
    confidence = probs[0][pred_label].item()
    
    return pred_label, confidence

st.title("IndoBERT Classification Dashboard")

if 'history' not in st.session_state:
    st.session_state['history'] = []

raw_df = load_and_prep_data(DATA_PATH)
train_df, val_df = get_splits(raw_df)
tokenizer, model = load_model_pipeline()

with st.sidebar:
    st.header("System Status")
    if model:
        st.success(f"Model Active: Hugging Face Hub")
    else:
        st.error("Model Error: Could not download from Hugging Face.")
    
    st.caption(f"Device: {DEVICE.upper()}")
    st.divider()

    st.header("Prediction")
    user_input = st.text_area("Input text:")
    
    if st.button("Analyze", type="primary"):
        if model and user_input:
            label, conf = predict_text(user_input, tokenizer, model)
            label_str = "YES" if label == 1 else "NO"
            st.session_state['history'].insert(0, {
                "Input Text": user_input,
                "Prediction": label_str,
                "Confidence": f"{conf:.2%}",
                "Label Code": label
            })
        elif not user_input:
            st.warning("Please enter text.")
        else:
            st.error("Model not ready.")

    st.divider()
    st.header("Feedback")
    correct_label_input = st.radio("Actual Label", ["YES (1)", "NO (0)"], horizontal=True)
    if st.button("Save Feedback"):
        if user_input:
            final_lbl = 1 if "YES" in correct_label_input else 0
            new_row = pd.DataFrame([[user_input, final_lbl]], columns=['message', 'label'])
            new_row.to_csv(FEEDBACK_FILE, mode='a', header=not os.path.exists(FEEDBACK_FILE), index=False)
            st.success("Saved.")

if st.session_state['history']:
    latest = st.session_state['history'][0]
    c1, c2 = st.columns([1, 3])
    with c1:
        st.subheader("Result")
        if latest["Prediction"] == "YES":
            st.info(f"YES ({latest['Confidence']})")
        else:
            st.error(f"NO ({latest['Confidence']})")
    with c2:
        st.subheader("Input")
        st.write(latest["Input Text"])
    st.divider()

if raw_df is not None:
    tab1, tab2, tab3, tab4 = st.tabs(["History", "Metrics", "Error Analysis", "Dataset"])
else:
    tab1 = st.tabs(["History"])[0]
    tab2, tab3, tab4 = None, None, None

with tab1:
    if st.session_state['history']:
        st.dataframe(pd.DataFrame(st.session_state['history']), use_container_width=True)
        if st.button("Clear Log"):
            st.session_state['history'] = []
            st.rerun()
    else:
        st.write("No session history.")

if raw_df is not None:
    with tab2:
        st.subheader("Validation Metrics")
        if st.button("Calculate Metrics"):
            if model:
                with st.spinner("Calculating..."):
                    val_texts = val_df["message"].tolist()
                    val_labels = val_df["label"].tolist()
                    preds = []
                    batch_size = 16
                    for i in range(0, len(val_texts), batch_size):
                        batch_texts = val_texts[i:i+batch_size]
                        batch_inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
                        batch_inputs = {k: v.to(DEVICE) for k, v in batch_inputs.items()}
                        with torch.no_grad():
                            logits = model(**batch_inputs).logits
                        preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
                    
                    acc = accuracy_score(val_labels, preds)
                    f1 = f1_score(val_labels, preds)
                    c1, c2 = st.columns(2)
                    c1.metric("Accuracy", f"{acc:.2%}")
                    c2.metric("F1 Score", f"{f1:.2%}")
                    st.text(classification_report(val_labels, preds))
                    
                    val_df["Predicted"] = preds
                    st.session_state['val_preds'] = val_df

    with tab3:
        if 'val_preds' in st.session_state:
            df_res = st.session_state['val_preds']
            opt = st.radio("Filter", ["False Negatives", "False Positives"], horizontal=True)
            if "Negative" in opt:
                errs = df_res[(df_res["label"] == 1) & (df_res["Predicted"] == 0)]
            else:
                errs = df_res[(df_res["label"] == 0) & (df_res["Predicted"] == 1)]
            st.write(f"Errors: {len(errs)}")
            st.dataframe(errs, use_container_width=True)
        else:
            st.info("Run metrics first.")

    with tab4:
        c1, c2 = st.columns(2)
        with c1: st.bar_chart(raw_df["label"].value_counts())
        with c2: 
            raw_df["len"] = raw_df["message"].apply(len)
            fig, ax = plt.subplots(figsize=(6,3))
            sns.histplot(data=raw_df, x="len", hue="label", kde=True, ax=ax)
            st.pyplot(fig)
