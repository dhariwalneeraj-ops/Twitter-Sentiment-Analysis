"""
Prediction Engine Module
Loads pre-trained models and TF-IDF vectorizer, and provides single-tweet,
multi-model comparative, and batch CSV sentiment inferences.
"""

import os
import joblib
from pathlib import Path
from typing import Dict, Any, List, Union
import numpy as np
import pandas as pd

from src.preprocessing import clean_text


class SentimentPredictor:
    """
    Manages loading and real-time inference across all 3 trained sentiment models.
    """

    MODEL_FILES = {
        "logistic": "logistic_model.pkl",
        "naive_bayes": "naive_bayes.pkl",
        "svm": "svm_model.pkl"
    }

    MODEL_NAMES = {
        "logistic": "Logistic Regression",
        "naive_bayes": "Naive Bayes",
        "svm": "Support Vector Machine"
    }

    def __init__(self, models_dir: Union[str, Path] = None):
        if models_dir is None:
            base_dir = Path(__file__).resolve().parent.parent
            self.models_dir = base_dir / "models"
        else:
            self.models_dir = Path(models_dir)

        self.vectorizer = None
        self.models: Dict[str, Any] = {}
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Loads TF-IDF vectorizer and all trained models into memory."""
        vec_path = self.models_dir / "tfidf_vectorizer.pkl"
        if not vec_path.exists():
            raise FileNotFoundError(
                f"TF-IDF vectorizer not found at {vec_path}. Please train models first using src/train.py."
            )

        self.vectorizer = joblib.load(vec_path)

        for key, fname in self.MODEL_FILES.items():
            model_path = self.models_dir / fname
            if model_path.exists():
                self.models[key] = joblib.load(model_path)
            else:
                print(f"[!] Warning: Model file {fname} not found in {self.models_dir}")

    def is_ready(self) -> bool:
        """Checks if vectorizer and at least one model are loaded."""
        return self.vectorizer is not None and len(self.models) > 0

    def predict(
        self,
        text: str,
        model_key: str = "logistic"
    ) -> Dict[str, Any]:
        """
        Predicts sentiment for a single tweet string.
        
        Returns:
            dict containing:
                - text: input text
                - cleaned_text: preprocessed text
                - sentiment: predicted label (e.g. positive, negative, neutral)
                - confidence: highest class probability (0-1)
                - probabilities: dictionary of {label: prob}
                - model_key: model identifier
                - model_name: human readable model name
        """
        if not self.is_ready():
            raise RuntimeError("Predictor is not ready: models are not loaded.")

        if model_key not in self.models:
            fallback = list(self.models.keys())[0]
            print(f"[!] Model '{model_key}' not found. Falling back to '{fallback}'.")
            model_key = fallback

        model = self.models[model_key]
        cleaned = clean_text(text)

        # Handle empty string after cleaning
        if not cleaned.strip():
            classes = list(model.classes_)
            uniform_prob = round(1.0 / len(classes), 4)
            return {
                "text": text,
                "cleaned_text": "",
                "sentiment": "neutral" if "neutral" in classes else classes[0],
                "confidence": uniform_prob,
                "probabilities": {c: uniform_prob for c in classes},
                "model_key": model_key,
                "model_name": self.MODEL_NAMES.get(model_key, model_key)
            }

        # Vectorize
        features = self.vectorizer.transform([cleaned])
        pred_label = model.predict(features)[0]

        # Extract probabilities
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features)[0]
            classes = model.classes_
            prob_dict = {cls_name: round(float(p), 4) for cls_name, p in zip(classes, probs)}
            confidence = round(float(np.max(probs)), 4)
        else:
            # Fallback for models without predict_proba
            prob_dict = {pred_label: 1.0}
            confidence = 1.0

        return {
            "text": text,
            "cleaned_text": cleaned,
            "sentiment": str(pred_label),
            "confidence": confidence,
            "probabilities": prob_dict,
            "model_key": model_key,
            "model_name": self.MODEL_NAMES.get(model_key, model_key)
        }

    def predict_all_models(self, text: str) -> Dict[str, Dict[str, Any]]:
        """
        Executes sentiment prediction across all loaded models for comparison.
        """
        results = {}
        for key in self.models:
            results[key] = self.predict(text, model_key=key)
        return results

    def predict_batch(
        self,
        texts: List[str],
        model_key: str = "logistic"
    ) -> pd.DataFrame:
        """
        Generates predictions for a list of texts and returns a structured DataFrame.
        """
        records = []
        for t in texts:
            pred = self.predict(t, model_key=model_key)
            records.append({
                "text": pred["text"],
                "cleaned_text": pred["cleaned_text"],
                "predicted_sentiment": pred["sentiment"],
                "confidence": pred["confidence"],
                **{f"prob_{k}": v for k, v in pred["probabilities"].items()}
            })
        return pd.DataFrame(records)
