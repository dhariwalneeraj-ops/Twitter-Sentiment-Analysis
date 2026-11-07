"""
Page 2: Live Tweet Sentiment Prediction
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

st.title("⚡ Live Tweet Sentiment Prediction")
st.markdown("Classify customer sentiment in real time using your choice of trained machine learning models.")

# Initialize Predictor
try:
    predictor = SentimentPredictor(models_dir=root_dir / "models")
except Exception as e:
    st.error(f"Failed to load trained models: {e}. Please ensure models are trained.")
    st.stop()

tab_single, tab_batch = st.tabs(["📝 Single Tweet Prediction", "📁 Batch CSV Prediction"])

# TAB 1: Single Prediction
with tab_single:
    st.subheader("Interactive Tweet Analysis")

    # Example tweet presets
    preset = st.selectbox(
        "Choose a sample preset or write your own below:",
        options=[
            "Custom Input",
            "Loving the new battery life! Lasts 2 full days without needing a charger #happy",
            "Customer service was atrocious, waited 50 minutes and got disconnected. Furious!",
            "Tracking number shows the shipment was dispatched from regional sorting facility.",
            "Display is gorgeous and vivid, but the audio volume is terribly quiet and muffled."
        ]
    )

    default_text = "" if preset == "Custom Input" else preset
    user_tweet = st.text_area(
        "Enter Tweet Text:",
        value=default_text,
        height=100,
        placeholder="Type or paste any tweet here..."
    )

    col_model, col_run = st.columns([2, 1])
    with col_model:
        model_choice = st.selectbox(
            "Select ML Classifier:",
            options=["logistic", "naive_bayes", "svm"],
            format_func=lambda x: {
                "logistic": "Logistic Regression (TF-IDF)",
                "naive_bayes": "Multinomial Naive Bayes (TF-IDF)",
                "svm": "Support Vector Machine (Linear SVC)"
            }.get(x, x)
        )

    predict_btn = st.button("🚀 Analyze Sentiment", type="primary")

    if (predict_btn or user_tweet) and user_tweet.strip():
        # Run prediction on selected model
        res = predictor.predict(user_tweet, model_key=model_choice)

        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 1.5])

        sentiment_colors = {
            "positive": ("#d4edda", "#155724", "🟢 POSITIVE"),
            "negative": ("#f8d7da", "#721c24", "🔴 NEGATIVE"),
            "neutral": ("#fff3cd", "#856404", "🟡 NEUTRAL")
        }
        bg, fg, label_txt = sentiment_colors.get(res["sentiment"], ("#e2e3e5", "#383d41", res["sentiment"].upper()))

        with res_col1:
            st.markdown(f"""
            <div style="background-color: {bg}; color: {fg}; padding: 1.5rem; border-radius: 10px; border: 1px solid {fg}; text-align: center;">
                <h2 style="margin: 0; font-size: 1.8rem;">{label_txt}</h2>
                <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; font-weight: bold;">
                    Confidence: {res['confidence'] * 100:.1f}%
                </p>
                <small style="color: {fg};">Classifier: {res['model_name']}</small>
            </div>
            """, unsafe_allow_html=True)

            st.write(f"**Cleaned Tokens:** `{res['cleaned_text']}`")

        with res_col2:
            st.write("**Prediction Probabilities Across Classes:**")
            probs_df = pd.DataFrame(
                list(res["probabilities"].items()),
                columns=["Sentiment", "Probability"]
            )
            probs_df["Percentage"] = (probs_df["Probability"] * 100).round(2).astype(str) + "%"

            st.dataframe(probs_df, use_container_width=True, hide_index=True)

            fig_p, ax_p = plt.subplots(figsize=(6, 2.8))
            ax_p.barh(
                list(res["probabilities"].keys()),
                list(res["probabilities"].values()),
                color=["#dc3545", "#ffc107", "#28a745"]
            )
            ax_p.set_xlim(0, 1.0)
            ax_p.set_xlabel("Probability")
            ax_p.set_title("Probability Distribution")
            st.pyplot(fig_p)

        st.markdown("---")

        # Multi-Model Consensus View
        st.subheader("🤝 Multi-Model Consensus Comparison")
        st.write("Cross-evaluating this tweet against all 3 algorithms simultaneously:")

        all_preds = predictor.predict_all_models(user_tweet)
        c1, c2, c3 = st.columns(3)

        cols = [c1, c2, c3]
        for idx, (mkey, mres) in enumerate(all_preds.items()):
            with cols[idx]:
                st.markdown(f"**{mres['model_name']}**")
                s_color = {"positive": "green", "negative": "red", "neutral": "orange"}.get(mres["sentiment"], "gray")
                st.markdown(f"Prediction: :{s_color}[**{mres['sentiment'].upper()}**]")
                st.progress(float(mres["confidence"]))
                st.caption(f"Confidence: {mres['confidence']*100:.1f}%")


# TAB 2: Batch CSV Prediction
with tab_batch:
    st.subheader("Batch CSV Sentiment Inference")
    st.write("Upload a CSV file with a `text` or `tweet` column to process high volumes of tweets at once.")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded dataset:")
            st.dataframe(batch_df.head(5), use_container_width=True)

            # Detect text column
            possible_cols = [c for c in batch_df.columns if c.lower() in ["text", "tweet", "content", "tweet_text"]]
            chosen_col = st.selectbox(
                "Select column containing tweet text:",
                options=batch_df.columns,
                index=batch_df.columns.get_loc(possible_cols[0]) if possible_cols else 0
            )

            batch_model = st.selectbox("Batch Classifier:", ["logistic", "naive_bayes", "svm"])

            if st.button("⚡ Run Batch Predictions", type="primary"):
                with st.spinner("Processing batch predictions..."):
                    texts = batch_df[chosen_col].astype(str).tolist()
                    preds_df = predictor.predict_batch(texts, model_key=batch_model)

                    # Merge results
                    merged_df = pd.concat([batch_df.reset_index(drop=True), preds_df.drop(columns=["text"])], axis=1)

                st.success(f"Successfully processed {len(merged_df)} tweets!")

                # Batch metrics
                bcol1, bcol2 = st.columns(2)
                with bcol1:
                    st.write("**Predicted Sentiment Distribution**")
                    st.dataframe(merged_df["predicted_sentiment"].value_counts())
                with bcol2:
                    fig_b, ax_b = plt.subplots(figsize=(5, 3.5))
                    merged_df["predicted_sentiment"].value_counts().plot(
                        kind="bar",
                        color=["#28a745", "#dc3545", "#ffc107"],
                        ax=ax_b
                    )
                    ax_b.set_title("Batch Sentiment Distribution")
                    st.pyplot(fig_b)

                st.dataframe(merged_df, use_container_width=True)

                csv_data = merged_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Enriched Predictions CSV",
                    data=csv_data,
                    file_name="sentiment_predictions_output.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Error processing CSV: {e}")
