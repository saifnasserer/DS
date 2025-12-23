import joblib
import os
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Tuple, Optional


def load_model(model_dir: str = "ml_model") -> Tuple[BaseEstimator, TfidfVectorizer]:
    model_path = os.path.join(model_dir, "model.pkl")
    vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"Model files not found in {model_dir}")
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer


def predict(text: str, model: Optional[BaseEstimator] = None, 
            vectorizer: Optional[TfidfVectorizer] = None,
            model_dir: str = "ml_model") -> Tuple[int, float]:
    if model is None or vectorizer is None:
        model, vectorizer = load_model(model_dir)
    
    from src.preprocess import clean_text, preprocess_text
    
    processed = preprocess_text(clean_text(text))
    vec = vectorizer.transform([processed])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    
    return int(pred), float(proba[pred])

