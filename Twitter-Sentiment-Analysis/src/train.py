"""
Model Training Module
Loads Twitter dataset, performs text preprocessing, fits TF-IDF vectorizer,
trains Logistic Regression, Naive Bayes, and SVM models, evaluates them,
and serializes artifacts for inference and explainability.
"""

import os
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from src.preprocessing import clean_text
from src.evaluation import compute_metrics, generate_comparison_table


def load_and_preprocess_data(dataset_path: str) -> pd.DataFrame:
    """Loads CSV dataset and applies text cleaning."""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at: {dataset_path}")

    df = pd.read_csv(dataset_path)
    if "text" not in df.columns or "sentiment" not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'sentiment' columns.")

    # Drop missing values
    df = df.dropna(subset=["text", "sentiment"]).copy()
    df["cleaned_text"] = df["text"].apply(clean_text)
    # Ensure there is non-empty cleaned text
    df = df[df["cleaned_text"].str.strip().str.len() > 0].copy()
    return df


def train_and_save_models(
    dataset_path: str = None,
    models_dir: str = None,
    reports_dir: str = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Executes full training pipeline:
    - Preprocessing
    - Train-test split
    - TF-IDF extraction
    - Training 3 distinct classifiers: Logistic Regression, Naive Bayes, SVM
    - Serializing model files & comparison reports
    """
    base_dir = Path(__file__).resolve().parent.parent
    if dataset_path is None:
        dataset_path = str(base_dir / "dataset" / "twitter.csv")
    if models_dir is None:
        models_dir = str(base_dir / "models")
    if reports_dir is None:
        reports_dir = str(base_dir / "reports")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    print(f"[*] Loading and preprocessing dataset from {dataset_path}...")
    df = load_and_preprocess_data(dataset_path)
    print(f"[+] Loaded {len(df)} samples across classes: {df['sentiment'].value_counts().to_dict()}")

    X = np.array(df["cleaned_text"].tolist(), dtype=str)
    y = np.array(df["sentiment"].tolist(), dtype=str)

    # Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"[*] Fitting TF-IDF Vectorizer (ngram=(1,2), max_features=5000)...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Initialize Classifiers
    models = {
        "Logistic Regression": LogisticRegression(
            C=1.5,
            max_iter=1000,
            random_state=random_state
        ),
        "Naive Bayes": MultinomialNB(
            alpha=0.3
        ),
        "Support Vector Machine": SVC(
            C=1.0,
            kernel="linear",
            probability=True,
            random_state=random_state
        )
    }

    eval_results = []
    trained_models = {}

    print("[*] Training models...")
    for name, model in models.items():
        print(f"  -> Training {name}...")
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        y_proba = model.predict_proba(X_test_vec)

        metrics = compute_metrics(y_test, y_pred, model_name=name)
        metrics["y_test"] = y_test
        metrics["y_pred"] = y_pred
        metrics["y_proba"] = y_proba
        metrics["classes"] = model.classes_
        eval_results.append(metrics)
        trained_models[name] = model

        print(f"     Accuracy: {metrics['accuracy']:.4f} | F1-Macro: {metrics['f1_macro']:.4f}")

    # Generate Model Comparison DataFrame
    comparison_df = generate_comparison_table(eval_results)
    comparison_csv_path = os.path.join(reports_dir, "model_comparison.csv")
    comparison_df.to_csv(comparison_csv_path, index=False)
    print(f"[+] Saved comparison report to {comparison_csv_path}")

    # Save artifacts
    joblib.dump(vectorizer, os.path.join(models_dir, "tfidf_vectorizer.pkl"))
    joblib.dump(trained_models["Logistic Regression"], os.path.join(models_dir, "logistic_model.pkl"))
    joblib.dump(trained_models["Naive Bayes"], os.path.join(models_dir, "naive_bayes.pkl"))
    joblib.dump(trained_models["Support Vector Machine"], os.path.join(models_dir, "svm_model.pkl"))
    print(f"[+] Successfully saved all models and vectorizer to {models_dir}")

    return {
        "eval_results": eval_results,
        "models": trained_models,
        "vectorizer": vectorizer,
        "classes": sorted(list(set(y)))
    }, comparison_df


if __name__ == "__main__":
    train_and_save_models()
