from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.base import BaseEstimator
from typing import Dict
from scipy.sparse import csr_matrix
import numpy as np


def train_models(X_train: csr_matrix, y_train: np.ndarray) -> Dict[str, BaseEstimator]:
    models = {
        "lr": LogisticRegression(max_iter=500, random_state=42),
        "svm": LinearSVC(random_state=42, max_iter=1000),
        "nb": MultinomialNB()
    }
    
    for model in models.values():
        model.fit(X_train, y_train)
    
    return models
