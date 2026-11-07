"""
Unit tests for Sentiment Prediction, Aspect Analysis, and Explainability.
"""

import unittest
from pathlib import Path
import pandas as pd

from src.train import train_and_save_models
from src.prediction import SentimentPredictor
from src.aspect_analysis import AspectSentimentAnalyzer, detect_aspects_in_text
from src.explainability import SentimentExplainer


class TestPredictionPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Ensure models are trained and saved
        base_dir = Path(__file__).resolve().parent.parent
        data_path = str(base_dir / "dataset" / "twitter.csv")
        models_dir = str(base_dir / "models")
        reports_dir = str(base_dir / "reports")

        # Train if not already done
        if not (Path(models_dir) / "logistic_model.pkl").exists():
            train_and_save_models(data_path, models_dir, reports_dir)

        cls.predictor = SentimentPredictor(models_dir=models_dir)
        cls.aspect_analyzer = AspectSentimentAnalyzer(predictor=cls.predictor)
        cls.explainer = SentimentExplainer(predictor=cls.predictor)

    def test_predictor_readiness(self):
        self.assertTrue(self.predictor.is_ready())
        self.assertIn("logistic", self.predictor.models)
        self.assertIn("naive_bayes", self.predictor.models)
        self.assertIn("svm", self.predictor.models)

    def test_single_prediction_structure(self):
        tweet = "Absolutely in love with the battery life and gorgeous screen!"
        res = self.predictor.predict(tweet, model_key="logistic")
        self.assertIn("sentiment", res)
        self.assertIn("confidence", res)
        self.assertIn("probabilities", res)
        self.assertTrue(0.0 <= res["confidence"] <= 1.0)
        self.assertEqual(res["sentiment"], "positive")

    def test_negative_prediction(self):
        tweet = "Worst phone purchase ever, battery dies instantly and screen flickers."
        res = self.predictor.predict(tweet, model_key="logistic")
        self.assertEqual(res["sentiment"], "negative")

    def test_predict_all_models(self):
        tweet = "Arrived on time and in great condition."
        all_res = self.predictor.predict_all_models(tweet)
        self.assertIn("logistic", all_res)
        self.assertIn("naive_bayes", all_res)
        self.assertIn("svm", all_res)

    def test_batch_prediction(self):
        tweets = [
            "Super fast delivery!",
            "Terrible sound quality from speakers.",
            "Normal retail packaging."
        ]
        df = self.predictor.predict_batch(tweets, model_key="logistic")
        self.assertEqual(len(df), 3)
        self.assertIn("predicted_sentiment", df.columns)
        self.assertIn("confidence", df.columns)

    def test_aspect_detection(self):
        tweet = "The camera takes stunning shots but battery drains rapidly."
        aspects = detect_aspects_in_text(tweet)
        self.assertIn("camera", aspects)
        self.assertIn("battery", aspects)

    def test_aspect_sentiment_analysis(self):
        tweet = "The camera is fabulous, but customer support was totally unhelpful."
        results = self.aspect_analyzer.analyze_tweet(tweet)
        self.assertTrue(len(results) >= 2)
        aspect_names = [r["aspect"] for r in results]
        self.assertIn("camera", aspect_names)
        self.assertIn("customer_support", aspect_names)

    def test_explainability(self):
        tweet = "Awesome battery performance and lightning fast charging!"
        exp = self.explainer.explain(tweet, model_key="logistic", num_features=5)
        self.assertIn("word_weights", exp)
        self.assertTrue(len(exp["word_weights"]) > 0)
        self.assertIn("html_body", exp)


if __name__ == "__main__":
    unittest.main()
