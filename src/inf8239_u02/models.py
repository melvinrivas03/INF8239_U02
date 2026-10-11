"""Baseline y dos pipelines comparables: misma vectorización TF-IDF, distinto clasificador."""
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

TFIDF_PARAMS = dict(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True)


def build_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(**TFIDF_PARAMS)


def build_models() -> dict:
    return {
        "baseline_dummy": Pipeline([("tfidf", build_tfidf()),
                                    ("model", DummyClassifier(strategy="most_frequent"))]),
        "tfidf_nb": Pipeline([("tfidf", build_tfidf()), ("model", MultinomialNB(alpha=0.1))]),
        "tfidf_logreg": Pipeline([("tfidf", build_tfidf()),
                                  ("model", LogisticRegression(C=10, class_weight="balanced",
                                                               max_iter=2000))]),
    }
