"""
Tweet Preprocessing Module
Provides robust text cleaning, tokenization, emoji handling, and negation preservation
for sentiment analysis and explainability pipelines.
"""

import re
import html
from typing import List, Union
import pandas as pd

# Comprehensive negation tokens that must NOT be removed as stopwords
NEGATION_TOKENS = {
    "not", "no", "never", "none", "neither", "nor", "nowhere", "hardly",
    "scarcely", "barely", "doesn't", "doesnt", "don't", "dont", "didn't",
    "didnt", "won't", "wont", "wouldn't", "wouldnt", "can't", "cant",
    "cannot", "couldn't", "couldnt", "shouldn't", "shouldnt", "isn't",
    "isnt", "aren't", "arent", "wasn't", "wasnt", "weren't", "werent",
    "haven't", "havent", "hasn't", "hasnt", "hadn't", "hadnt"
}

# Standard English stopwords (excluding negations)
BASE_STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "only",
    "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just",
    "should", "now"
}

# Common emoji translation mapping for sentiment enrichment
EMOJI_DICT = {
    "❤️": " love ",
    "💖": " love ",
    "😍": " adore ",
    "😊": " happy ",
    "😃": " joy ",
    "😁": " happy ",
    "🔥": " fire awesome ",
    "👍": " good thumbs up ",
    "👌": " perfect ",
    "🎉": " celebrate ",
    "✨": " amazing ",
    "🌟": " stellar ",
    "👏": " applause bravo ",
    "🚀": " fast great ",
    "💯": " perfection ",
    "💔": " heartbreak bad ",
    "😡": " angry terrible ",
    "😠": " mad ",
    "🤬": " furious ",
    "😭": " crying sad ",
    "😢": " sad ",
    "😞": " disappointed ",
    "👎": " bad thumbs down ",
    "🤮": " disgusting awful ",
    "💩": " crap terrible ",
    "⚠️": " warning problem ",
    "❌": " failure wrong ",
    "🚫": " forbidden bad ",
    "💀": " dead awful ",
    "🤡": " clown joke ",
}


def translate_emojis(text: str) -> str:
    """Translates known emojis into sentiment-carrying descriptive words."""
    for emoji_char, replacement in EMOJI_DICT.items():
        if emoji_char in text:
            text = text.replace(emoji_char, replacement)
    return text


def clean_text(
    text: Union[str, float],
    remove_stopwords: bool = True,
    preserve_negations: bool = True,
    expand_hashtags: bool = True
) -> str:
    """
    Cleans raw tweet text through comprehensive NLP preprocessing stages.
    
    Stages:
    1. Handle NaN / null values.
    2. Unescape HTML entities (&amp; -> &).
    3. Translate sentiment emojis to words.
    4. Strip URLs (http://, https://, www.*).
    5. Strip Twitter handles (@username).
    6. Process hashtags (#awesome -> awesome).
    7. Normalize contractions (don't -> do not).
    8. Remove punctuation and unwanted special characters.
    9. Stopword removal while safeguarding negations.
    10. Whitespace normalization.
    """
    if not isinstance(text, str):
        return ""

    # 1. Decode HTML
    cleaned = html.unescape(text)

    # 2. Translate emojis
    cleaned = translate_emojis(cleaned)

    # 3. Remove URLs
    cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned)

    # 4. Remove User Mentions (@user)
    cleaned = re.sub(r"@\w+", " ", cleaned)

    # 5. Handle Hashtags (#tech -> tech)
    if expand_hashtags:
        cleaned = re.sub(r"#(\w+)", r" \1 ", cleaned)
    else:
        cleaned = re.sub(r"#\w+", " ", cleaned)

    # 6. Normalize common contractions to assist negation detection
    contractions = {
        r"\bcan\'t\b": "can not",
        r"\bcant\b": "can not",
        r"\bwon\'t\b": "will not",
        r"\bwont\b": "will not",
        r"\bdon\'t\b": "do not",
        r"\bdont\b": "do not",
        r"\bdoesn\'t\b": "does not",
        r"\bdoesnt\b": "does not",
        r"\bdidn\'t\b": "did not",
        r"\bdidnt\b": "did not",
        r"\bisn\'t\b": "is not",
        r"\bisnt\b": "is not",
        r"\baren\'t\b": "are not",
        r"\barent\b": "are not",
        r"\bwasn\'t\b": "was not",
        r"\bwasnt\b": "was not",
        r"\bweren\'t\b": "were not",
        r"\bwerent\b": "were not",
        r"\bhaven\'t\b": "have not",
        r"\bhavent\b": "have not",
        r"\bhasn\'t\b": "has not",
        r"\bhasnt\b": "has not",
        r"\bhadn\'t\b": "had not",
        r"\bhadnt\b": "had not",
        r"\bwouldn\'t\b": "would not",
        r"\bwouldnt\b": "would not",
        r"\bshouldn\'t\b": "should not",
        r"\bshouldnt\b": "should not",
        r"\bcouldn\'t\b": "could not",
        r"\bcouldnt\b": "could not",
    }
    for pattern, repl in contractions.items():
        cleaned = re.sub(pattern, repl, cleaned, flags=re.IGNORECASE)

    # 7. Lowercase
    cleaned = cleaned.lower()

    # 8. Remove non-alphabetical characters (retain letters and spaces)
    cleaned = re.sub(r"[^a-z\s]", " ", cleaned)

    # 9. Tokenize & Stopword Filtering
    tokens = cleaned.split()
    if remove_stopwords:
        filtered = []
        for token in tokens:
            if preserve_negations and token in NEGATION_TOKENS:
                filtered.append(token)
            elif token not in BASE_STOPWORDS and len(token) > 1:
                filtered.append(token)
        tokens = filtered

    # 10. Rejoin and normalize spaces
    result = " ".join(tokens)
    return result


def tokenize(text: str) -> List[str]:
    """Tokenizes preprocessed string into word list."""
    cleaned = clean_text(text)
    return cleaned.split()


def preprocess_series(series: pd.Series) -> pd.Series:
    """Preprocesses an entire pandas Series of tweet texts."""
    return series.fillna("").apply(clean_text)
