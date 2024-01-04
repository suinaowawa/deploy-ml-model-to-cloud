import numpy as np
from sklearn.ensemble import RandomForestClassifier
from starter.ml.model import train_model, compute_model_metrics, inference


def test_train_model():
    np.random.seed(42)
    X_train = np.random.rand(100, 5)
    y_train = np.random.randint(2, size=100)

    # Test if the function returns a trained RandomForestClassifier
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)


def test_compute_model_metrics():
    np.random.seed(42)
    y_true = np.random.randint(2, size=100)
    y_pred = np.random.randint(2, size=100)

    # Test if the function returns valid precision, recall, and fbeta scores
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)


def test_inference():
    np.random.seed(42)
    X_test = np.random.rand(50, 5)
    model = RandomForestClassifier(random_state=42)

    # Fit the model
    y_train = np.random.randint(2, size=50)
    model.fit(X_test, y_train)

    # Test if the function returns predictions as numpy array
    predictions = inference(model, X_test)
    assert isinstance(predictions, np.ndarray)
