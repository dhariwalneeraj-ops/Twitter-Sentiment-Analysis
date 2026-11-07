"""
Twitter Sentiment Analysis - Streamlit Main Dashboard Entry Point
AI-Powered Twitter Sentiment Analysis | Python, Scikit-learn, NLP, LIME, Streamlit, Generative AI
"""

import os
import sys
from pathlib import Path
import streamlit as st
import pandas as pd

# Add root folder to sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.train import train_and_save_models
from src.prediction import SentimentPredictor

# Streamlit Page Configuration
st.set_page_config(
    page_title="Twitter Sentiment Analysis AI",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1DA1F2;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 5px solid #1DA1F2;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .feature-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 2px;
        background-color: #e1f5fe;
        color: #0288d1;
    }
</style>
""", unsafe_allow_html=True)


def check_and_initialize_system():
    """Verifies pre-trained models exist; if not, triggers initial training."""
    models_dir = root_dir / "models"
    required = ["logistic_model.pkl", "naive_bayes.pkl", "svm_model.pkl", "tfidf_vectorizer.pkl"]
    missing = [f for f in required if not (models_dir / f).exists()]

    if missing:
        with st.spinner("Initializing models and dataset on first run..."):
            train_and_save_models(
                dataset_path=str(root_dir / "dataset" / "twitter.csv"),
                models_dir=str(models_dir),
                reports_dir=str(root_dir / "reports")
            )
        st.success("✅ Models trained and initialized successfully!")


check_and_initialize_system()

# Sidebar Setup
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3256/3256013.png", width=64)
    st.title("🐦 Control Panel")
    st.caption("AI-Powered Twitter Sentiment Engine")

    st.markdown("---")
    st.subheader("🔑 Generative AI Configuration")
    user_api_key = st.text_input(
        "Google Gemini API Key (Optional)",
        type="password",
        value=st.session_state.get("gemini_api_key", ""),
        help="Enter your Gemini API key to enable live cloud LLM sentiment summaries. If left blank, high-accuracy offline heuristic NLP summaries will be used."
    )
    if user_api_key:
        st.session_state["gemini_api_key"] = user_api_key
        st.success("Gemini API key active")
    else:
        st.info("Operating in Offline Heuristic NLP mode")

    st.markdown("---")
    st.subheader("⚙️ System Status")
    try:
        predictor = SentimentPredictor(models_dir=root_dir / "models")
        st.success(f"Models Ready: {len(predictor.models)} / 3 Active")
        st.caption("• Logistic Regression\n• Naive Bayes\n• Support Vector Machine")
    except Exception as e:
        st.error(f"Models offline: {e}")

    if st.button("🔄 Retrain All Models"):
        with st.spinner("Retraining on full dataset..."):
            _, comp_df = train_and_save_models(
                dataset_path=str(root_dir / "dataset" / "twitter.csv"),
                models_dir=str(root_dir / "models"),
                reports_dir=str(root_dir / "reports")
            )
            st.success("Models retrained successfully!")
            st.dataframe(comp_df, hide_index=True)


# Main Content Area
st.markdown('<div class="main-header">🐦 AI-Powered Twitter Sentiment Analysis Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">End-to-End NLP Sentiment Classification, Explainable AI (LIME), Aspect Analytics & Generative Summaries</div>', unsafe_allow_html=True)

# Top Metrics Row
col1, col2, col3, col4 = st.columns(4)

with col1:
    dataset_file = root_dir / "dataset" / "twitter.csv"
    num_samples = len(pd.read_csv(dataset_file)) if dataset_file.exists() else 0
    st.metric("Curated Tweets", f"{num_samples:,}", "Multi-domain")

with col2:
    st.metric("ML Classifiers", "3 Models", "LR, NB, SVM")

with col3:
    st.metric("XAI Explainer", "LIME Enabled", "Word Contributions")

with col4:
    st.metric("GenAI Summarizer", "Gemini + Fallback", "Executive Briefing")

st.markdown("---")

# Quick Sentiment Playground
st.subheader("⚡ Quick Sentiment Prediction Playground")
st.write("Test tweet classification instantly using our pre-trained ensemble.")

quick_input = st.text_input(
    "Enter a tweet or customer review:",
    value="The OLED screen is absolutely gorgeous, but customer support took 5 days to respond."
)

col_q1, col_q2 = st.columns([1, 2])

if quick_input:
    try:
        predictor = SentimentPredictor(models_dir=root_dir / "models")
        res = predictor.predict(quick_input, model_key="logistic")

        sentiment_color = {
            "positive": "#28a745",
            "negative": "#dc3545",
            "neutral": "#ffc107"
        }.get(res["sentiment"], "#17a2b8")

        with col_q1:
            st.markdown(f"""
            <div style="background-color: {sentiment_color}; padding: 1.2rem; border-radius: 8px; color: white; text-align: center;">
                <h4 style="margin:0; text-transform: uppercase;">{res['sentiment']}</h4>
                <p style="margin:0; font-size: 1.4rem; font-weight: bold;">{res['confidence'] * 100:.1f}% Confidence</p>
                <small>Model: {res['model_name']}</small>
            </div>
            """, unsafe_allow_html=True)

        with col_q2:
            st.write("**Class Probability Distribution:**")
            prob_df = pd.DataFrame(list(res["probabilities"].items()), columns=["Sentiment", "Probability"])
            st.bar_chart(prob_df.set_index("Sentiment"), color="#1DA1F2")

    except Exception as e:
        st.warning(f"Could not run quick prediction: {e}")

st.markdown("---")

# Feature Highlights Grid
st.subheader("🎯 Core Demonstrated Capabilities")

cap_col1, cap_col2 = st.columns(2)

with cap_col1:
    st.markdown("""
    #### 1. 📊 Model Comparison with Actual Metrics
    - Evaluates **Logistic Regression**, **Naive Bayes**, and **SVM** on Stratified Test Sets.
    - Displays real Accuracy, Precision, Recall, Macro F1, Confusion Matrices, and ROC-AUC curves.
    - *Navigate to `3_Model_Comparison` in sidebar.*

    #### 2. ⚡ Live Tweet Sentiment Prediction
    - Interactive real-time prediction for custom tweets with confidence scoring.
    - Multi-model simultaneous comparison on identical inputs.
    - Batch CSV file upload and instantaneous prediction download.
    - *Navigate to `2_Live_Prediction` in sidebar.*

    #### 3. 🔍 LIME Explainable AI (XAI)
    - Interprets black-box ML decisions word-by-word using **LIME**.
    - Highlights influential positive and negative sentiment tokens with color-coded attributions.
    - *Navigate to `5_Explainable_AI` in sidebar.*
    """)

with cap_col2:
    st.markdown("""
    #### 4. 🏷️ Aspect-Based Sentiment Analysis (ABSA)
    - Identifies granular product attributes: *Battery, Camera, Screen, Support, Audio, Performance, etc.*
    - Clause-level sentiment extraction (e.g. contrastive tweets with both praise and complaint).
    - Aggregated sentiment polarity breakdown per aspect across the dataset.
    - *Navigate to `4_Aspect_Analysis` in sidebar.*

    #### 5. 🤖 Generative AI Sentiment Summaries
    - Generates executive-level management briefings, key takeaways, and root-cause analyses.
    - Powered by Google Gemini API with automated fallback to built-in heuristic NLP synthesis.
    - Interactive Q&A against tweet sentiment patterns.
    - *Navigate to `6_AI_Insights` in sidebar.*
    """)

st.info("👈 Use the sidebar on the left to navigate across all dashboard pages.")
