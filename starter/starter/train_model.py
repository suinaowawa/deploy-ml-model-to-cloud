# Script to train machine learning model.

import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer

# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics


def compute_metrics_on_slice(
    slice: pd.DataFrame,
    model: RandomForestClassifier,
    cat_cols: list,
    label: str,
    encoder: OneHotEncoder,
    lb: LabelBinarizer,
):
    """
    Computes precision, recall, and Fbeta metrics for a slice of data using a trained machine learning model.

    Parameters
    ----------
    slice : pd.DataFrame
        Slice of data for which metrics are computed.
    model : RandomForestClassifier
        Trained machine learning model.
    cat_cols : list
        List of categorical columns.
    label : str
        Name of the label column in the data.
    encoder : OneHotEncoder
        Trained OneHotEncoder for categorical features.
    lb : LabelBinarizer
        Trained LabelBinarizer for label encoding.

    Returns
    -------
    precision : float
        Precision score for the slice.
    recall : float
        Recall score for the slice.
    fbeta : float
        Fbeta score for the slice.
    """
    X_slice, y_slice, _, _ = process_data(
        slice, cat_cols, label, training=False, encoder=encoder, lb=lb
    )
    preds = inference(model, X_slice)
    precision, recall, fbeta = compute_model_metrics(y_slice, preds)
    return precision, recall, fbeta


# Add code to load in the data.
data = pd.read_csv("data/census.csv")
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=42)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# Train and save a model.
model = train_model(X_train, y_train)

model_path = "model/trained_model.pkl"
with open(model_path, "wb") as model_file:
    pickle.dump(model, model_file)

print(f"Trained model saved to {model_path}")

# Run inference on model.
y_preds = inference(model, X_test)

# Compute metrics on test data
precision, recall, fbeta = compute_model_metrics(y_test, y_preds)
print("Model Metrics: precision, recall, fbeta:", precision, recall, fbeta)

# Compute metrics on slices of education
unique_values = data["education"].unique()

with open("model/slice_output.txt", "w") as f:
    for val in unique_values:
        slice_data = data[data["education"] == val]
        precision, recall, fbeta = compute_metrics_on_slice(
            slice_data,
            model,
            cat_cols=cat_features,
            label="salary",
            encoder=encoder,
            lb=lb,
        )
        print(
            f"Slicing on column: education, value: {val} | Precision: {precision} | Recall: {recall} | Fbeta: {fbeta}"
        )
        f.write(
            f"Slicing on column: education, value: {val} | Precision: {precision} | Recall: {recall} | Fbeta: {fbeta}\n"
        )
