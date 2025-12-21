from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

def train_models(X_train, y_train):
    models = {
        "lr": LogisticRegression(max_iter=500),
        "svm": LinearSVC(),
        "nb": MultinomialNB()
    }
    for model in models.values():
        model.fit(X_train, y_train)
    return models
