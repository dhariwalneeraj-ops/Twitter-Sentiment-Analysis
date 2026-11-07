"""
Explainable AI (XAI) Module using LIME
Identifies and visualizes influential words behind sentiment predictions,
providing local feature attribution, HTML explanations, and contribution plots.
"""

from typing import Dict, Any, List, Tuple, Callable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

from src.preprocessing import clean_text


class SentimentExplainer:
    """
    Explains individual sentiment model predictions using LIME
    (Local Interpretable Model-agnostic Explanations).
    """

    def __init__(self, predictor):
        self.predictor = predictor
        self.vectorizer = predictor.vectorizer

    def _get_predict_proba_fn(self, model_key: str) -> Callable[[List[str]], np.ndarray]:
        """
        Builds a predict_proba function compatible with LimeTextExplainer.
        Accepts a list of raw text strings and returns an (N, n_classes) probability array.
        """
        model = self.predictor.models[model_key]

        def predict_fn(texts: List[str]) -> np.ndarray:
            cleaned_texts = [clean_text(t) for t in texts]
            # Handle empty texts by filling a dummy placeholder to keep matrix shape
            cleaned_texts = [t if t.strip() else "neutral" for t in cleaned_texts]
            features = self.vectorizer.transform(cleaned_texts)
            if hasattr(model, "predict_proba"):
                return model.predict_proba(features)
            else:
                # Fallback for models without predict_proba (e.g. standard SVC)
                preds = model.predict(features)
                classes = model.classes_
                res = np.zeros((len(preds), len(classes)))
                for i, p in enumerate(preds):
                    idx = np.where(classes == p)[0][0]
                    res[i, idx] = 1.0
                return res

        return predict_fn

    def explain(
        self,
        text: str,
        model_key: str = "logistic",
        num_features: int = 8,
        num_samples: int = 500
    ) -> Dict[str, Any]:
        """
        Generates LIME explanation for a single tweet.
        
        Returns:
            dict containing:
                - predicted_label: str
                - confidence: float
                - word_weights: list of (word, weight) tuples
                - html_body: str (for Streamlit HTML embedding)
                - top_positive_words: list of (word, weight)
                - top_negative_words: list of (word, weight)
        """
        if model_key not in self.predictor.models:
            model_key = list(self.predictor.models.keys())[0]

        model = self.predictor.models[model_key]
        classes = list(model.classes_)
        predict_fn = self._get_predict_proba_fn(model_key)

        # Baseline prediction
        base_pred = self.predictor.predict(text, model_key=model_key)
        pred_label = base_pred["sentiment"]
        pred_idx = classes.index(pred_label)

        if LIME_AVAILABLE:
            try:
                explainer = LimeTextExplainer(
                    class_names=classes,
                    random_state=42
                )
                exp = explainer.explain_instance(
                    text_instance=text,
                    classifier_fn=predict_fn,
                    labels=[pred_idx],
                    num_features=num_features,
                    num_samples=num_samples
                )
                word_weights = exp.as_list(label=pred_idx)
                html_body = exp.as_html(labels=[pred_idx])
            except Exception as e:
                # Graceful heuristic fallback if LIME perturbation encounters empty samples
                word_weights, html_body = self._fallback_explanation(text, model_key, pred_label)
        else:
            word_weights, html_body = self._fallback_explanation(text, model_key, pred_label)

        # Separate positive contributors vs negative contributors
        positive_contributors = [(w, wt) for w, wt in word_weights if wt > 0]
        negative_contributors = [(w, wt) for w, wt in word_weights if wt < 0]

        # Sort by absolute weight
        word_weights.sort(key=lambda x: abs(x[1]), reverse=True)

        return {
            "text": text,
            "model_key": model_key,
            "model_name": self.predictor.MODEL_NAMES.get(model_key, model_key),
            "predicted_label": pred_label,
            "confidence": base_pred["confidence"],
            "probabilities": base_pred["probabilities"],
            "word_weights": word_weights,
            "positive_contributors": positive_contributors,
            "negative_contributors": negative_contributors,
            "html_body": html_body
        }

    def _fallback_explanation(
        self,
        text: str,
        model_key: str,
        pred_label: str
    ) -> Tuple[List[Tuple[str, float]], str]:
        """
        Robust statistical attribution fallback using TF-IDF token weights
        and model coefficients or perturbation.
        """
        cleaned = clean_text(text)
        tokens = list(set(cleaned.split()))
        if not tokens:
            return [], "<p>No significant sentiment tokens detected.</p>"

        base_pred = self.predictor.predict(text, model_key=model_key)
        base_prob = base_pred["probabilities"].get(pred_label, 0.5)

        word_weights = []
        for token in tokens:
            # Leave-one-out perturbation
            perturbed = " ".join([t for t in tokens if t != token])
            p_pred = self.predictor.predict(perturbed, model_key=model_key)
            new_prob = p_pred["probabilities"].get(pred_label, 0.5)
            # Impact: how much prob drops when word is removed
            impact = base_prob - new_prob
            word_weights.append((token, round(float(impact), 4)))

        word_weights.sort(key=lambda x: abs(x[1]), reverse=True)

        # Generate simple HTML
        tags = []
        for word, wt in word_weights:
            color = "#28a745" if wt > 0 else "#dc3545"
            tags.append(
                f"<span style='background-color:{color}; color:white; padding:3px 6px; "
                f"margin:2px; border-radius:4px;'>{word} ({wt:+.3f})</span>"
            )
        html_body = "<div>" + " ".join(tags) + "</div>"

        return word_weights, html_body


def plot_word_importance(
    word_weights: List[Tuple[str, float]],
    target_class: str = "Sentiment",
    max_words: int = 10
) -> plt.Figure:
    """
    Renders horizontal bar chart of LIME feature weights.
    Positive weights (green) pull towards target class;
    Negative weights (red) pull away from target class.
    """
    if not word_weights:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No influential tokens found", ha="center", va="center")
        ax.axis("off")
        return fig

    # Top words
    top = word_weights[:max_words]
    words = [x[0] for x in top][::-1]
    weights = [x[1] for x in top][::-1]
    colors = ["#28a745" if w > 0 else "#dc3545" for w in weights]

    fig, ax = plt.subplots(figsize=(7, max(3.5, len(words) * 0.45)))
    bars = ax.barh(words, weights, color=colors, edgecolor="#333333", alpha=0.85)

    ax.axvline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("LIME Feature Weight (Contribution to Class)", fontsize=10, fontweight="bold")
    ax.set_title(f"Top Word Contributions for: '{target_class.upper()}'", fontsize=12, fontweight="bold", pad=12)
    ax.grid(axis="x", alpha=0.3)

    # Annotate value labels
    for bar in bars:
        val = bar.get_width()
        offset = 0.005 if val >= 0 else -0.015
        ha = "left" if val >= 0 else "right"
        ax.text(
            val + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}",
            va="center",
            ha=ha,
            fontsize=9,
            fontweight="bold"
        )

    plt.tight_layout()
    return fig
