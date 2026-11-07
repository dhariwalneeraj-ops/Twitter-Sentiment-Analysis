"""
Page 3: Model Comparison & Evaluation Metrics
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.train import load_and_preprocess_data, train_and_save_models
from src.evaluation import plot_confusion_matrix, plot_multiclass_roc
from src.prediction import SentimentPredictor

st.title("📊 Model Comparison & Comprehensive Evaluation")
st.markdown("""
Compare the actual performance of **Logistic Regression**, **Multinomial Naive Bayes**, 
and **Support Vector Machine (Linear SVM)** on stratified held-out test data.
""")

# Load or generate evaluation metrics
comparison_csv = root_dir / "reports" / "model_comparison.csv"
if not comparison_csv.exists():
    st.info("Generating evaluation metrics on test dataset...")
    train_and_save_models(
        dataset_path=str(root_dir / "dataset" / "twitter.csv"),
        models_dir=str(root_dir / "models"),
        reports_dir=str(root_dir / "reports")
    )

comparison_df = pd.read_csv(comparison_csv)

st.subheader("🏆 Model Leaderboard & Core Metrics")
st.dataframe(
    comparison_df.style.highlight_max(
        subset=["Accuracy", "Precision (Macro)", "Recall (Macro)", "F1-Score (Macro)"],
        color="#d4edda"
    ),
    use_container_width=True,
    hide_index=True
)

st.markdown("---")

# Metrics Visualization
st.subheader("📈 Cross-Model Metric Benchmarks")

col_bench, col_insights = st.columns([1.5, 1])

with col_bench:
    metrics_to_plot = ["Accuracy", "Precision (Macro)", "Recall (Macro)", "F1-Score (Macro)"]
    plot_df = comparison_df.set_index("Model")[metrics_to_plot].T

    fig_bar, ax_bar = plt.subplots(figsize=(7, 4.2))
    plot_df.plot(kind="bar", ax=ax_bar, colormap="tab10", width=0.75)
    ax_bar.set_ylim(0, 1.05)
    ax_bar.set_ylabel("Score (0.0 to 1.0)", fontweight="bold")
    ax_bar.set_title("Performance Metrics by Algorithm", fontsize=12, fontweight="bold", pad=12)
    ax_bar.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=0)
    plt.legend(frameon=True, loc="lower right")
    plt.tight_layout()
    st.pyplot(fig_bar)

with col_insights:
    st.markdown("""
    #### 💡 Algorithmic Insights:
    
    * **Logistic Regression (TF-IDF):**
      * Provides continuous, well-calibrated class posterior probabilities.
      * Excellent baseline for sparse n-gram spaces and pairs naturally with **LIME**.
    
    * **Multinomial Naive Bayes:**
      * Highly efficient frequency-based classifier assuming feature conditional independence.
      * Fast inference, though sometimes yields overconfident probability extremes.
    
    * **Support Vector Machine (Linear Kernel):**
      * Effective at finding maximum-margin separating hyperplanes in high-dimensional text spaces.
      * Enabled with Platt scaling (`probability=True`) for full explainability support.
    """)

st.markdown("---")

# Live Confusion Matrix & ROC Visualizer
st.subheader("🎯 Confusion Matrices & ROC Curves")

try:
    predictor = SentimentPredictor(models_dir=root_dir / "models")
    df_data = load_and_preprocess_data(str(root_dir / "dataset" / "twitter.csv"))
    
    # Train-test split for plotting curves
    from sklearn.model_selection import train_test_split
    X_arr = np.array(df_data["cleaned_text"].tolist(), dtype=str)
    y_arr = np.array(df_data["sentiment"].tolist(), dtype=str)
    X_train, X_test, y_train, y_test = train_test_split(
        X_arr,
        y_arr,
        test_size=0.25,
        random_state=42,
        stratify=y_arr
    )
    X_test_vec = predictor.vectorizer.transform(X_test)
    classes = list(np.unique(y_test))

    selected_eval_model = st.selectbox(
        "Select Model to Inspect Confusion Matrix & ROC Curves:",
        options=list(predictor.models.keys()),
        format_func=lambda x: predictor.MODEL_NAMES.get(x, x)
    )

    model_obj = predictor.models[selected_eval_model]
    y_pred = model_obj.predict(X_test_vec)
    y_proba = model_obj.predict_proba(X_test_vec) if hasattr(model_obj, "predict_proba") else None

    col_cm, col_roc = st.columns(2)

    with col_cm:
        cm_fig = plot_confusion_matrix(
            y_true=y_test,
            y_pred=y_pred,
            labels=classes,
            title=f"Confusion Matrix: {predictor.MODEL_NAMES.get(selected_eval_model, selected_eval_model)}"
        )
        st.pyplot(cm_fig)

    with col_roc:
        if y_proba is not None:
            roc_fig = plot_multiclass_roc(
                y_true=y_test,
                y_proba=y_proba,
                classes=classes,
                model_name=predictor.MODEL_NAMES.get(selected_eval_model, selected_eval_model)
            )
            st.pyplot(roc_fig)
        else:
            st.info("Probability estimation not available for this model configuration.")

except Exception as e:
    st.warning(f"Could not render detailed confusion matrices: {e}")
