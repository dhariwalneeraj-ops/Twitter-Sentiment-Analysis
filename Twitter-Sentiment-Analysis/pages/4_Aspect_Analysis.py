"""
Page 4: Aspect-Based Sentiment Analysis (ABSA)
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.prediction import SentimentPredictor
from src.aspect_analysis import (
    AspectSentimentAnalyzer,
    ASPECT_TAXONOMY,
    plot_aspect_distribution
)

st.title("🏷️ Aspect-Based Sentiment Analysis (ABSA)")
st.markdown("""
Pinpoint granular customer feedback on individual product features—such as 
**Battery**, **Camera**, **Customer Support**, **Display**, and **Performance**—even 
when a single tweet contains conflicting sentiments.
""")

# Initialize Analyzer
predictor = SentimentPredictor(models_dir=root_dir / "models")
analyzer = AspectSentimentAnalyzer(predictor=predictor)

tab_single, tab_agg = st.tabs(["🔍 Interactive Aspect Extraction", "📈 Corpus Aspect Analytics"])

# TAB 1: Single Tweet Aspect Breakdown
with tab_single:
    st.subheader("Clause-Level Aspect Decomposition")
    st.write("Notice how conflicting sentiments inside one review are automatically separated:")

    preset_tweet = st.selectbox(
        "Choose an example mixed-sentiment review:",
        options=[
            "The camera takes stunning 4K photos, but the battery drains ridiculously fast.",
            "Super fast delivery and great packaging, but customer support hung up on me.",
            "Display 120Hz refresh rate is silky smooth, though the price is way too expensive.",
            "Custom review input..."
        ]
    )

    if preset_tweet == "Custom review input...":
        input_tweet = st.text_area(
            "Enter your text:",
            value="Great audio quality and punchy bass, but the software update keeps crashing.",
            height=90
        )
    else:
        input_tweet = preset_tweet

    if st.button("🔎 Detect Aspects & Local Sentiments", type="primary"):
        aspects = analyzer.analyze_tweet(input_tweet)

        st.markdown("---")
        st.write(f"**Identified {len(aspects)} Distinct Aspect Mentions:**")

        for item in aspects:
            aspect_name = item["aspect"].replace("_", " ").title()
            sentiment = item["sentiment"]
            clause = item["clause"]
            conf = item["confidence"]
            score = item["polarity_score"]

            color_map = {
                "positive": ("#d4edda", "#155724", "🟢 POSITIVE"),
                "negative": ("#f8d7da", "#721c24", "🔴 NEGATIVE"),
                "neutral": ("#fff3cd", "#856404", "🟡 NEUTRAL")
            }
            bg, fg, badge_txt = color_map.get(sentiment, ("#e2e3e5", "#383d41", "NEUTRAL"))

            st.markdown(f"""
            <div style="background-color: {bg}; border-left: 6px solid {fg}; padding: 12px 16px; border-radius: 6px; margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <strong style="font-size: 1.1rem; color: {fg};">Feature: {aspect_name}</strong>
                    <span style="font-weight: bold; background: white; padding: 2px 8px; border-radius: 4px; color: {fg};">
                        {badge_txt} ({conf*100:.1f}%)
                    </span>
                </div>
                <p style="margin: 6px 0 0 0; color: #333333;"><strong>Target Clause:</strong> "<em>{clause}</em>"</p>
                <small style="color: #666666;">Polarity Score: {score:+.2f} [-1.0 = Highly Negative, +1.0 = Highly Positive]</small>
            </div>
            """, unsafe_allow_html=True)


# TAB 2: Aggregate Dataset Aspect Insights
with tab_agg:
    st.subheader("Corpus-Wide Aspect Intelligence")
    st.write("Aggregated distribution of customer feedback across key product dimensions:")

    dataset_path = root_dir / "dataset" / "twitter.csv"
    if dataset_path.exists():
        df_tweets = pd.read_csv(dataset_path)
        with st.spinner("Analyzing aspects across corpus..."):
            agg_results = analyzer.analyze_dataset(df_tweets, text_col="text")
            summary_df = agg_results["summary_df"]

        if not summary_df.empty:
            col_chart, col_rank = st.columns([1.5, 1])

            with col_chart:
                fig_dist = plot_aspect_distribution(summary_df)
                st.pyplot(fig_dist)

            with col_rank:
                st.markdown("#### 🏆 Net Satisfaction Index")
                st.caption("(Positive - Negative) / Total Mentions")

                ranked_df = summary_df[["aspect", "total_mentions", "sentiment_ratio"]].copy()
                ranked_df["aspect"] = ranked_df["aspect"].str.replace("_", " ").str.title()
                ranked_df.rename(columns={
                    "aspect": "Aspect",
                    "total_mentions": "Mentions",
                    "sentiment_ratio": "Net Score"
                }, inplace=True)
                ranked_df.sort_values(by="Net Score", ascending=False, inplace=True)

                st.dataframe(
                    ranked_df.style.background_gradient(
                        subset=["Net Score"],
                        cmap="RdYlGn",
                        vmin=-1.0,
                        vmax=1.0
                    ),
                    use_container_width=True,
                    hide_index=True
                )

            st.markdown("---")
            st.subheader("📋 Aspect Mentions Summary Table")
            display_summary = summary_df.copy()
            display_summary["aspect"] = display_summary["aspect"].str.replace("_", " ").str.title()
            display_summary.columns = [
                "Product Aspect", "Total Mentions", "Positive Count",
                "Negative Count", "Neutral Count", "Avg Polarity", "Net Sentiment Ratio"
            ]
            st.dataframe(display_summary, use_container_width=True, hide_index=True)

    else:
        st.warning("Corpus file not found.")
