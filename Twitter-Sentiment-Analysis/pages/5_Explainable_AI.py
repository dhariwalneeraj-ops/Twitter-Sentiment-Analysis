"""
Page 5: Explainable AI (XAI) using LIME
"""

import sys
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.prediction import SentimentPredictor
from src.explainability import SentimentExplainer, plot_word_importance

st.title("🔍 Explainable AI (XAI) with LIME")
st.markdown("""
Understand **why** machine learning models make specific sentiment predictions. 
**LIME (Local Interpretable Model-agnostic Explanations)** perturbs the input tweet and fits a local 
interpretable model to quantify the exact mathematical contribution of every token.
""")

# Initialize predictor and explainer
predictor = SentimentPredictor(models_dir=root_dir / "models")
explainer = SentimentExplainer(predictor=predictor)

# Tweet Input Section
preset_xai = st.selectbox(
    "Select an example tweet to explain:",
    options=[
        "The battery life is exceptional and the 120Hz display is gorgeous!",
        "Terrible experience, device freezes constantly and customer service was totally unhelpful.",
        "Delivered safely on Tuesday, standard retail packaging with manual.",
        "Custom tweet..."
    ]
)

if preset_xai == "Custom tweet...":
    tweet_to_explain = st.text_area(
        "Enter text to explain:",
        value="Loving the camera quality, but the battery drains so fast after the update.",
        height=85
    )
else:
    tweet_to_explain = preset_xai

col_opt1, col_opt2, col_opt3 = st.columns(3)
with col_opt1:
    model_choice = st.selectbox(
        "Classifier to Explain:",
        options=["logistic", "naive_bayes", "svm"],
        format_func=lambda x: predictor.MODEL_NAMES.get(x, x)
    )
with col_opt2:
    num_features = st.slider("Top Influential Tokens:", min_value=4, max_value=12, value=8)
with col_opt3:
    num_samples = st.slider("LIME Perturbation Samples:", min_value=200, max_value=800, value=400, step=100)

if st.button("🔮 Generate LIME Explanation", type="primary"):
    with st.spinner("Generating LIME local perturbations & computing token weights..."):
        exp_res = explainer.explain(
            text=tweet_to_explain,
            model_key=model_choice,
            num_features=num_features,
            num_samples=num_samples
        )

    st.markdown("---")

    # Prediction Summary Badge
    pred_label = exp_res["predicted_label"]
    conf = exp_res["confidence"]

    sent_colors = {
        "positive": ("#28a745", "POSITIVE"),
        "negative": ("#dc3545", "NEGATIVE"),
        "neutral": ("#ffc107", "NEUTRAL")
    }
    hex_color, label_str = sent_colors.get(pred_label, ("#17a2b8", pred_label.upper()))

    st.markdown(f"""
    <div style="background-color: #f8f9fa; border-left: 6px solid {hex_color}; padding: 14px 20px; border-radius: 6px;">
        <span style="font-size: 1.2rem; font-weight: bold; color: {hex_color};">
            Predicted Sentiment: {label_str} ({conf*100:.1f}% Confidence)
        </span>
        <br>
        <span style="color: #555555;">Model: {exp_res['model_name']}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 Token Influence Breakdown")
    col_chart, col_table = st.columns([1.5, 1])

    with col_chart:
        fig_importance = plot_word_importance(
            exp_res["word_weights"],
            target_class=pred_label,
            max_words=num_features
        )
        st.pyplot(fig_importance)

    with col_table:
        st.write("**Local Token Weights Table**")
        df_weights = pd.DataFrame(exp_res["word_weights"], columns=["Word / Token", "Weight"])
        df_weights["Direction"] = df_weights["Weight"].apply(
            lambda w: f"Pulls towards {pred_label.upper()}" if w > 0 else f"Pushes away from {pred_label.upper()}"
        )
        st.dataframe(df_weights, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Native LIME Interactive Visualization
    st.subheader("🌐 Interactive LIME Visualization Canvas")
    st.caption("Embedded interactive LIME HTML view showing text token highlights and prediction prob margins:")

    if exp_res.get("html_body"):
        components.html(exp_res["html_body"], height=380, scrolling=True)

    st.markdown("""
    #### 💡 How LIME Works in this System:
    1. **Perturbation:** LIME creates hundreds of synthetic variations of the tweet by randomly removing words.
    2. **Probability Probing:** The sentiment classifier computes class probabilities for every variation.
    3. **Distance Weighting:** Variations closer to the original text receive higher sample weights via exponential kernel.
    4. **Interpretable Fit:** A sparse linear regression model is fit on these weighted samples to assign exact contribution coefficients to individual tokens.
    """)
