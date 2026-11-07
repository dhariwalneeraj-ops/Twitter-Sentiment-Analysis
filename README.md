# 🐦 AI-Powered Twitter Sentiment Analysis

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red.svg)](https://streamlit.io/)
[![LIME](https://img.shields.io/badge/XAI-LIME-green.svg)](https://github.com/marcotcr/lime)
[![Generative AI](https://img.shields.io/badge/GenAI-Google%20Gemini-purple.svg)](https://ai.google.dev/)

An enterprise-grade Natural Language Processing (NLP) sentiment analysis system featuring **Explainable AI (LIME)**, **Aspect-Based Sentiment Analysis (ABSA)**, **Multi-Model Machine Learning Classifiers**, and **Generative AI Summarization** packaged into an interactive multi-page **Streamlit** dashboard.

---

## 🌟 Key Highlights & The 5 Core Demonstrations

1. **📊 Model Comparison with Actual Evaluation Metrics:**
   - Benchmarks **Logistic Regression**, **Multinomial Naive Bayes**, and **Support Vector Machine (Linear SVM)** on stratified test splits.
   - Computes real Accuracy, Precision (Macro/Weighted), Recall (Macro/Weighted), and F1-Score (Macro/Weighted).
   - Generates interactive **Confusion Matrix Heatmaps** and **One-vs-Rest ROC Curves** with AUC metrics.

2. **⚡ Live Tweet Sentiment Prediction:**
   - Real-time sentiment prediction (Positive, Negative, Neutral) with confidence probability percentages.
   - Multi-model consensus comparison: tests single tweets across all 3 classifiers side-by-side.
   - Batch CSV processor: upload CSVs with tweet columns and download predictions with confidence scores.

3. **🔍 LIME Explanations for Individual Predictions:**
   - Implements Local Interpretable Model-agnostic Explanations (**LIME**) to illuminate the model's inner decision-making.
   - Displays feature attribution charts (green positive contributors vs red negative contributors).
   - Embeds interactive LIME HTML widgets directly into the dashboard.

4. **🏷️ Aspect-Level Sentiment Detection (ABSA):**
   - Automatically detects key product features: *Battery, Camera, Customer Support, Delivery, Screen/Display, Price, Performance, UI/UX, Audio, and Software*.
   - Isolates clauses within compound/contrastive sentences (e.g., *"The camera is gorgeous, but the battery life is terrible"*) to score each attribute independently.
   - Visualizes corpus-wide aspect sentiment distributions and Net Satisfaction Index rankings.

5. **🤖 Generative AI-Generated Sentiment Summaries:**
   - Integrates **Google Gemini API** (`google-genai` SDK) to synthesize thousands of tweets into executive briefings.
   - Automatically detects customer praise drivers, pain-point root causes, and strategic engineering recommendations.
   - Includes a built-in offline heuristic NLP synthesizer as a seamless fallback if no API key is provided.

---

## 📁 Repository Structure

```
Twitter-Sentiment-Analysis/
│
├── app.py                      # Main Streamlit dashboard & landing controller
│
├── pages/                      # Multi-page Streamlit application
│   ├── 1_Overview.py           # Architecture, methodology & corpus analytics
│   ├── 2_Live_Prediction.py    # Real-time single & batch tweet prediction
│   ├── 3_Model_Comparison.py   # Metrics comparison, confusion matrix & ROC
│   ├── 4_Aspect_Analysis.py    # Aspect extraction & sentiment breakdown
│   ├── 5_Explainable_AI.py     # LIME word attribution & visualization
│   └── 6_AI_Insights.py        # GenAI executive briefings & Q&A
│
├── src/                        # Modular production Python package
│   ├── preprocessing.py        # Text cleaning, emoji mapping & negation preservation
│   ├── train.py                # Model training, vectorization & report export
│   ├── prediction.py           # Inference engine across LR, NB, SVM
│   ├── evaluation.py           # Evaluation metrics, ROC & confusion matrix plots
│   ├── explainability.py       # LIME text explainer & word attribution plots
│   ├── aspect_analysis.py      # Aspect taxonomy & clause decomposition
│   └── genai.py                # Gemini API integration & offline heuristic fallback
│
├── models/                     # Serialized model artifacts
│   ├── logistic_model.pkl
│   ├── naive_bayes.pkl
│   ├── svm_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── dataset/
│   └── twitter.csv             # Labeled multi-domain tweet dataset
│
├── reports/
│   └── model_comparison.csv    # Exported evaluation metrics benchmark
│
├── tests/                      # Automated unit test suite
│   ├── test_preprocessing.py   # Cleaning, tokenization & negation tests
│   └── test_prediction.py      # Inference, aspect & LIME pipeline tests
│
├── requirements.txt            # Python dependencies
├── .gitignore
└── README.md                   # Comprehensive documentation
```

---

## 🚀 Quickstart & Setup Guide

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, 3.13, or 3.14
- Recommended: Virtual environment (`venv` or `conda`)

### 2. Installation
Clone the repository and install the dependencies:
```bash
cd Twitter-Sentiment-Analysis
pip install -r requirements.txt
```

### 3. Model Training
Train the TF-IDF vectorizer and the 3 ML models:
```bash
python -m src.train
```
*Note: The models are pre-trained and serialized in `models/`, but running the script allows fresh retraining at any time.*

### 4. Running the Streamlit Application
Launch the multi-page dashboard:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 5. Running the Automated Test Suite
Execute the unit tests to verify NLP preprocessing and ML pipelines:
```bash
python -m unittest discover -s tests
```

---

## 🧠 Machine Learning & NLP Architecture

### Text Preprocessing & Negation Safeguard
Standard stopword filtering removes words like *"not"*, turning *"not good"* into *"good"*, causing severe false positive errors. Our preprocessing engine in [`src/preprocessing.py`](file:///src/preprocessing.py):
- Strips Twitter handles (`@user`), URLs (`https://...`), and symbols.
- Unescapes HTML entities and normalizes contractions (*"don't"* -> *"do not"*).
- Maps emotional emojis (e.g. ❤️, 🔥, 😡, 😭) to descriptive sentiment tokens.
- **Negation Preservation:** Whitelists negation tokens (`not`, `never`, `hardly`, `no`, `without`) to preserve true sentiment semantics.

### Feature Extraction
- **TF-IDF Vectorizer:** Captures unigrams and bigrams (`ngram_range=(1, 2)`) with sublinear term-frequency scaling to down-weight frequent dominant words.

### Classifier Suite
1. **Logistic Regression:** Regularized linear classification (`C=1.5`) providing calibrated class posterior probabilities.
2. **Multinomial Naive Bayes:** Laplace-smoothed (`alpha=0.3`) probabilistic classifier optimized for text frequencies.
3. **Support Vector Machine (Linear SVC):** Finds maximal-margin hyperplanes with Platt probability scaling (`probability=True`) for explainability support.

### Explainable AI (LIME)
In [`src/explainability.py`](file:///src/explainability.py), LIME draws sample perturbations around the input text, probes model predictions, and trains an interpretable linear surrogate with exponential distance weighting to compute precise token coefficients.

### Aspect-Based Sentiment Analysis (ABSA)
In [`src/aspect_analysis.py`](file:///src/aspect_analysis.py), rule-guided clause parsing splits complex sentences at conjunctions (*"but"*, *"however"*, *"although"*), tags aspect keywords according to domain taxonomy, and computes localized sentiment for each feature.

---

## 🔑 Generative AI Configuration (Google Gemini)

You can run the app completely **offline** using the built-in heuristic NLP summarizer.
To enable live Gemini LLM generative summaries:
1. Obtain an API key from [Google AI Studio](https://aistudio.google.com/).
2. Either enter it in the **Streamlit sidebar** input box, or export it in your environment:
   ```bash
   export GEMINI_API_KEY="your-api-key-here"     # Linux/macOS
   $env:GEMINI_API_KEY="your-api-key-here"       # Windows PowerShell
   ```

---

## 📄 License
MIT License. Open for educational, portfolio, and commercial use.
