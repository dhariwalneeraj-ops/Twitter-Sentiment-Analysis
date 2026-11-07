"""
Page 1: Project Overview & Dataset Analytics
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.preprocessing import clean_text

st.title("📖 Project Overview & Dataset Analytics")
st.markdown("""
This platform is an enterprise-grade NLP and Explainable AI sentiment intelligence system built using 
**Python**, **Scikit-learn**, **LIME**, **Streamlit**, and **Google Gemini Generative AI**.
""")

# Architecture Pipeline Card
st.subheader("🏗️ System Architecture & Workflow")

st.markdown("""
```mermaid
flowchart LR
    A["Raw Twitter Stream"] --> B["NLP Preprocessing & Cleaning"]
    B --> C["TF-IDF Feature Extraction"]
    C --> D1["Logistic Regression"]
    C --> D2["Naive Bayes"]
    C --> D3["Support Vector Machine"]
    D1 & D2 & D3 --> E["Comparative Evaluation"]
    D1 --> F["LIME Explainable AI"]
    B --> G["Aspect-Based Sentiment (ABSA)"]
    E & F & G --> H["Generative AI Executive Briefing"]
```
""", unsafe_allow_html=True)

st.markdown("---")

# Load Dataset
dataset_path = root_dir / "dataset" / "twitter.csv"
if not dataset_path.exists():
    st.error(f"Dataset file not found at {dataset_path}")
    st.stop()

df = pd.read_csv(dataset_path)

st.subheader("📊 Corpus Statistics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Sample Tweets", len(df))
with col2:
    pos_count = (df["sentiment"] == "positive").sum()
    st.metric("Positive Tweets", f"{pos_count} ({pos_count/len(df)*100:.1f}%)")
with col3:
    neg_count = (df["sentiment"] == "negative").sum()
    st.metric("Negative Tweets", f"{neg_count} ({neg_count/len(df)*100:.1f}%)")
with col4:
    neu_count = (df["sentiment"] == "neutral").sum()
    st.metric("Neutral Tweets", f"{neu_count} ({neu_count/len(df)*100:.1f}%)")

# Sentiment Distribution Plots
col_plot1, col_plot2 = st.columns(2)

with col_plot1:
    st.write("**Sentiment Distribution**")
    sent_counts = df["sentiment"].value_counts()
    fig_pie, ax_pie = plt.subplots(figsize=(5, 4))
    colors = ["#28a745", "#dc3545", "#ffc107"]
    ax_pie.pie(
        sent_counts.values,
        labels=[s.capitalize() for s in sent_counts.index],
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        wedgeprops=dict(width=0.45, edgecolor="w")
    )
    ax_pie.set_title("Sentiment Class Balance", fontweight="bold")
    st.pyplot(fig_pie)

with col_plot2:
    st.write("**Tweet Length Distribution (Word Count)**")
    df["word_count"] = df["text"].astype(str).apply(lambda x: len(x.split()))
    fig_hist, ax_hist = plt.subplots(figsize=(6, 4))
    sns.histplot(df["word_count"], kde=True, bins=20, color="#1DA1F2", ax=ax_hist)
    ax_hist.set_title("Tweet Length Histogram", fontweight="bold")
    ax_hist.set_xlabel("Words per Tweet")
    ax_hist.set_ylabel("Tweet Frequency")
    st.pyplot(fig_hist)

st.markdown("---")

# Preprocessing Demonstration
st.subheader("🧹 Preprocessing Transformation Pipeline")
st.write("Examine how raw social media tweets are transformed into normalized, negation-preserved tokens.")

sample_tweet = st.selectbox(
    "Choose a sample tweet to test preprocessing:",
    options=df["text"].head(8).tolist()
)

col_raw, col_clean = st.columns(2)
with col_raw:
    st.info(f"**Original Raw Tweet:**\n\n{sample_tweet}")
with col_clean:
    cleaned = clean_text(sample_tweet)
    st.success(f"**Cleaned & Tokenized Output:**\n\n`{cleaned}`")

st.markdown("---")

# Interactive Dataset Viewer
st.subheader("🔍 Interactive Dataset Explorer")
filter_col1, filter_col2 = st.columns(2)

with filter_col1:
    selected_sentiment = st.multiselect(
        "Filter by Sentiment:",
        options=sorted(df["sentiment"].unique()),
        default=sorted(df["sentiment"].unique())
    )

with filter_col2:
    if "aspect" in df.columns:
        available_aspects = sorted(df["aspect"].dropna().unique())
        selected_aspect = st.multiselect(
            "Filter by Product Aspect:",
            options=available_aspects,
            default=available_aspects
        )
    else:
        selected_aspect = None

filtered_df = df[df["sentiment"].isin(selected_sentiment)]
if selected_aspect and "aspect" in df.columns:
    filtered_df = filtered_df[filtered_df["aspect"].isin(selected_aspect)]

st.write(f"Showing **{len(filtered_df)}** matching tweets:")
st.dataframe(filtered_df, use_container_width=True)
