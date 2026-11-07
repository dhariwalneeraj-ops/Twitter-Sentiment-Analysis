"""
Unit tests for NLP tweet preprocessing module.
"""

import unittest
import pandas as pd
from src.preprocessing import clean_text, tokenize, preprocess_series, translate_emojis


class TestPreprocessing(unittest.TestCase):

    def test_url_removal(self):
        raw = "Check out this phone at https://example.com/phone and http://store.org"
        cleaned = clean_text(raw)
        self.assertNotIn("https", cleaned)
        self.assertNotIn("example.com", cleaned)
        self.assertIn("phone", cleaned)

    def test_user_mention_removal(self):
        raw = "Hey @Apple @TechSupport, my device is awesome!"
        cleaned = clean_text(raw)
        self.assertNotIn("@apple", cleaned)
        self.assertNotIn("@techsupport", cleaned)
        self.assertIn("device", cleaned)
        self.assertIn("awesome", cleaned)

    def test_hashtag_expansion(self):
        raw = "Loving the new display #iPhone #OLED"
        cleaned = clean_text(raw)
        self.assertNotIn("#", cleaned)
        self.assertIn("iphone", cleaned)
        self.assertIn("oled", cleaned)

    def test_emoji_translation(self):
        raw = "The camera is pure ❤️🔥"
        translated = translate_emojis(raw)
        self.assertIn("love", translated)
        self.assertIn("fire", translated)

    def test_negation_preservation(self):
        raw = "The phone is not good, I don't recommend it."
        cleaned = clean_text(raw, preserve_negations=True)
        # Verify negation word 'not' is not dropped as a stopword
        self.assertIn("not", cleaned)

    def test_tokenize(self):
        raw = "Fast delivery and great battery life!"
        tokens = tokenize(raw)
        self.assertIsInstance(tokens, list)
        self.assertTrue(len(tokens) > 0)
        self.assertIn("battery", tokens)

    def test_preprocess_series(self):
        series = pd.Series([
            "Amazing camera quality! 🔥",
            "Terrible customer service @store",
            None
        ])
        res = preprocess_series(series)
        self.assertEqual(len(res), 3)
        self.assertIn("camera", res.iloc[0])
        self.assertIn("service", res.iloc[1])
        self.assertEqual(res.iloc[2], "")


if __name__ == "__main__":
    unittest.main()
