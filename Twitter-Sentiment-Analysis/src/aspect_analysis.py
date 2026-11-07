"""
Aspect-Based Sentiment Analysis (ABSA) Module
Detects domain-specific product/service attributes and extracts localized
sentiment for each identified aspect within sentences and clauses.
"""

import re
from typing import Dict, Any, List, Optional
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Aspect category keyword lexicon
ASPECT_TAXONOMY = {
    "battery": [
        "battery", "charge", "charging", "charger", "mah", "drain", "draining",
        "standby", "battery life", "overheating", "power bank"
    ],
    "camera": [
        "camera", "lens", "photo", "photos", "picture", "pictures", "video",
        "portrait", "zoom", "sensor", "night mode", "bokeh", "macro", "selfie",
        "shutter", "telephoto", "ultrawide", "optical"
    ],
    "customer_support": [
        "support", "customer service", "agent", "rep", "representative", "helpdesk",
        "ticket", "call center", "refund", "warranty", "care", "helpline", "live chat"
    ],
    "delivery": [
        "delivery", "shipping", "shipped", "courier", "package", "parcel",
        "shipment", "dispatch", "order", "tracking", "arrived", "doorstep"
    ],
    "screen": [
        "screen", "display", "oled", "amoled", "lcd", "panel", "hz", "120hz",
        "brightness", "bezel", "colors", "gorilla glass", "resolution", "dead pixel"
    ],
    "price": [
        "price", "pricing", "cost", "expensive", "cheap", "affordable", "value",
        "rip off", "rip-off", "bargain", "discount", "fee", "subscription", "worth"
    ],
    "performance": [
        "performance", "speed", "fast", "slow", "lag", "lagging", "stutter",
        "stuttering", "processor", "cpu", "gpu", "ram", "multitasking", "gaming",
        "freeze", "freezing", "throttling", "thermal", "fps"
    ],
    "ui_design": [
        "ui", "ux", "design", "interface", "layout", "aesthetic", "minimalist",
        "navigation", "menu", "icons", "gestures", "typography", "ergonomics"
    ],
    "audio": [
        "audio", "sound", "speaker", "speakers", "volume", "bass", "treble",
        "mic", "microphone", "earbuds", "headphones", "spatial audio", "noise cancellation"
    ],
    "software": [
        "software", "update", "os", "firmware", "patch", "bug", "bugs", "glitch",
        "crash", "crashing", "sync", "bluetooth", "wifi", "ecosystem"
    ]
}

# Positive & negative sentiment indicators for rule-based clause scoring
POSITIVE_SEEDS = {
    "great", "amazing", "love", "loved", "loving", "awesome", "excellent", "superb",
    "fantastic", "best", "good", "impressed", "impressive", "smooth", "crisp",
    "flawless", "stunning", "gem", "phenomenal", "perfection", "top", "fast",
    "reliable", "worth", "bargain", "pleasure", "kudos", "brilliant", "steal"
}

NEGATIVE_SEEDS = {
    "bad", "terrible", "horrible", "awful", "worst", "hate", "hated", "poor",
    "disappointed", "disappointing", "slow", "lag", "crash", "broken", "garbage",
    "useless", "drain", "unacceptable", "overpriced", "annoying", "defect",
    "defective", "flicker", "burn", "crack", "ugly", "rude", "fail", "failed"
}


def segment_into_clauses(text: str) -> List[str]:
    """
    Splits compound sentences into clauses using coordinating conjunctions
    (but, however, although, while, yet) and clause punctuation (;, .).
    This enables contrasting aspect sentiments (e.g., 'Great screen, but terrible battery').
    """
    pattern = r"(?<=[.!?])\s+|;\s*|\s+(?:but|however|although|though|yet|while|whereas)\s+"
    parts = re.split(pattern, text, flags=re.IGNORECASE)
    return [p.strip() for p in parts if p.strip()]


def detect_aspects_in_text(text: str) -> List[str]:
    """Identifies which aspect categories are present in the text."""
    lower_text = text.lower()
    matched_aspects = []
    for aspect, keywords in ASPECT_TAXONOMY.items():
        for kw in keywords:
            # Word boundary match for exact keywords
            if re.search(r"\b" + re.escape(kw) + r"\b", lower_text):
                matched_aspects.append(aspect)
                break
    return matched_aspects


class AspectSentimentAnalyzer:
    """
    Performs Aspect-Based Sentiment Analysis (ABSA) by localizing sentiments
    to specific detected product attributes.
    """

    def __init__(self, predictor=None):
        self.predictor = predictor

    def analyze_tweet(self, text: str) -> List[Dict[str, Any]]:
        """
        Analyzes a single tweet, identifies all aspects and resolves their specific sentiment.
        
        Returns:
            List of dicts:
                - aspect: str (e.g. 'battery', 'camera')
                - clause: str (the text span discussing this aspect)
                - sentiment: 'positive', 'negative', or 'neutral'
                - confidence: float
                - polarity_score: float (-1.0 to +1.0)
        """
        clauses = segment_into_clauses(text)
        results = []
        overall_aspects = detect_aspects_in_text(text)

        if not overall_aspects:
            # General fallback aspect
            sent, conf, score = self._score_text_span(text)
            return [{
                "aspect": "general",
                "clause": text,
                "sentiment": sent,
                "confidence": conf,
                "polarity_score": score
            }]

        for aspect in overall_aspects:
            # Find the most relevant clause mentioning this aspect
            relevant_clauses = []
            for clause in clauses:
                if any(re.search(r"\b" + re.escape(kw) + r"\b", clause.lower()) for kw in ASPECT_TAXONOMY[aspect]):
                    relevant_clauses.append(clause)

            target_span = " ".join(relevant_clauses) if relevant_clauses else text
            sent, conf, score = self._score_text_span(target_span)

            results.append({
                "aspect": aspect,
                "clause": target_span,
                "sentiment": sent,
                "confidence": conf,
                "polarity_score": score
            })

        return results

    def _score_text_span(self, span: str) -> (str, float, float):
        """Scores a text clause using predictor probabilities or lexicon rules."""
        if self.predictor and self.predictor.is_ready():
            try:
                pred = self.predictor.predict(span)
                label = pred["sentiment"]
                conf = pred["confidence"]
                if label == "positive":
                    score = conf
                elif label == "negative":
                    score = -conf
                else:
                    score = 0.0
                return label, conf, round(score, 3)
            except Exception:
                pass

        # Lexicon scoring fallback
        words = re.findall(r"\w+", span.lower())
        pos_count = sum(1 for w in words if w in POSITIVE_SEEDS)
        neg_count = sum(1 for w in words if w in NEGATIVE_SEEDS)

        if pos_count > neg_count:
            return "positive", 0.75, 0.75
        elif neg_count > pos_count:
            return "negative", 0.75, -0.75
        else:
            return "neutral", 0.50, 0.0

    def analyze_dataset(self, df: pd.DataFrame, text_col: str = "text") -> Dict[str, Any]:
        """
        Runs aspect analysis across a collection of tweets, compiling aggregated stats.
        """
        aspect_records = []
        for idx, row in df.iterrows():
            tweet_text = str(row[text_col])
            aspects = self.analyze_tweet(tweet_text)
            for item in aspects:
                aspect_records.append({
                    "tweet_id": row.get("tweet_id", idx),
                    "aspect": item["aspect"],
                    "sentiment": item["sentiment"],
                    "polarity_score": item["polarity_score"]
                })

        adf = pd.DataFrame(aspect_records)
        if adf.empty:
            return {"summary_df": pd.DataFrame(), "records_df": pd.DataFrame()}

        # Summary by aspect
        summary = adf.groupby("aspect").agg(
            total_mentions=("sentiment", "count"),
            positive_count=("sentiment", lambda s: (s == "positive").sum()),
            negative_count=("sentiment", lambda s: (s == "negative").sum()),
            neutral_count=("sentiment", lambda s: (s == "neutral").sum()),
            avg_polarity=("polarity_score", "mean")
        ).reset_index()

        summary["sentiment_ratio"] = (
            (summary["positive_count"] - summary["negative_count"]) / summary["total_mentions"]
        ).round(3)
        summary.sort_values(by="total_mentions", ascending=False, inplace=True)

        return {
            "summary_df": summary,
            "records_df": adf
        }


def plot_aspect_distribution(summary_df: pd.DataFrame) -> plt.Figure:
    """
    Generates a stacked bar chart showing sentiment breakdown per aspect.
    """
    if summary_df.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No Aspect Data Available", ha="center", va="center")
        ax.axis("off")
        return fig

    aspects = summary_df["aspect"].tolist()
    pos = summary_df["positive_count"].tolist()
    neu = summary_df["neutral_count"].tolist()
    neg = summary_df["negative_count"].tolist()

    y_pos = np.arange(len(aspects))
    fig, ax = plt.subplots(figsize=(8, max(4.0, len(aspects) * 0.45)))

    ax.barh(y_pos, pos, color="#28a745", label="Positive", alpha=0.85)
    ax.barh(y_pos, neu, left=pos, color="#ffc107", label="Neutral", alpha=0.85)
    left_neg = np.array(pos) + np.array(neu)
    ax.barh(y_pos, neg, left=left_neg, color="#dc3545", label="Negative", alpha=0.85)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([a.replace("_", " ").title() for a in aspects], fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("Mention Count", fontweight="bold")
    ax.set_title("Aspect-Level Sentiment Breakdown", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return fig
