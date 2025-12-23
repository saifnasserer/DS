import sys
import os
import logging
from typing import Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import load_data, clean_dataframe, clean_text, preprocess_text
from src.modeling import train_models
from src.evaluate import evaluate

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
import joblib
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train(data_path: str, model_dir: str = "ml_model") -> Tuple[BaseEstimator, TfidfVectorizer]:
    logger.info("Starting training pipeline")
    
    try:
        logger.info(f"Loading data from {data_path}")
        df = load_data(data_path)
        logger.info(f"Loaded {len(df)} rows")

        logger.info("Cleaning and preprocessing data")
        df = clean_dataframe(df)
        df['text'] = df['text'].apply(clean_text)
        df['text'] = df['text'].apply(preprocess_text)
        logger.info(f"After cleaning: {len(df)} rows")

        logger.info("Extracting TF-IDF features")
        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(df['text'])
        y = df['target']
        logger.info(f"Feature matrix shape: {X.shape}")

        logger.info("Splitting data into train and test sets")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        logger.info(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
        
        logger.info("Training models")
        models = train_models(X_train, y_train)
        logger.info("Trained models: Logistic Regression, SVM, Naive Bayes")
        
        model = models['lr']
        
        logger.info("Evaluating models")
        for name, model_instance in models.items():
            logger.info(f"\n{name.upper()} Performance:")
            report = evaluate(model_instance, X_test, y_test)
            logger.info(report)
        
        logger.info("Saving models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "model.pkl")
        vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")
        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vectorizer_path)
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Vectorizer saved to: {vectorizer_path}")
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average='weighted')),
            "recall": float(recall_score(y_test, y_pred, average='weighted')),
            "f1_score": float(f1_score(y_test, y_pred, average='weighted')),
            "model_type": "Logistic Regression"
        }
        
        import json
        from datetime import datetime
        metrics["training_date"] = datetime.now().isoformat()
        metrics_path = os.path.join(model_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to: {metrics_path}")

        logger.info("Training pipeline completed successfully")
        return model, vectorizer
        
    except FileNotFoundError:
        logger.error(f"Data file not found: {data_path}")
        raise
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, "data", "raw", "train.csv")
    
    if not os.path.exists(data_path):
        logger.error(f"Training data not found at {data_path}")
        sys.exit(1)
    
    train(data_path)

