from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator
from scipy.sparse import csr_matrix
import numpy as np


def evaluate(model: BaseEstimator, X_test: csr_matrix, y_test: np.ndarray) -> str:
    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred, output_dict=False)

