# Put the code for your API here.
import pickle
from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel

import uvicorn

from starter.ml.data import process_data

app = FastAPI()

# Load the trained model
with open("model/trained_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("model/encoder.pkl", "rb") as encoder_file:
    encoder = pickle.load(encoder_file)

with open("model/lb.pkl", "rb") as lb_file:
    lb = pickle.load(lb_file)

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


# Pydantic model for input validation
class InputData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        schema_extra = {
            "examples": [
                {
                    "age": 39,
                    "workclass": "State-gov",
                    "fnlgt": 77516,
                    "education": "Bachelors",
                    "education_num": 13,
                    "marital_status": "Never-married",
                    "occupation": "Adm-clerical",
                    "relationship": "Not-in-family",
                    "race": "White",
                    "sex": "Male",
                    "capital_gain": 2174,
                    "capital_loss": 0,
                    "hours_per_week": 40,
                    "native_country": "United-States",
                }
            ]
        }


# Mock model inference function
def predict_salary(data: InputData) -> str:
    data_dict = data.dict()
    df = pd.DataFrame([data_dict])
    column_mapping = {
        "education_num": "education-num",
        "marital_status": "marital-status",
        "capital_gain": "capital-gain",
        "capital_loss": "capital-loss",
        "hours_per_week": "hours-per-week",
        "native_country": "native-country",
    }
    df.rename(columns=column_mapping, inplace=True)

    x, _, _, _ = process_data(
        df,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb,
    )
    salary = model.predict(x)
    return ">50K" if salary == 1 else "<=50K"


# Welcome message at the root endpoint
@app.get("/")
def welcome():
    return {"message": "Welcome to the Salary Prediction API"}


# Model inference endpoint using POST
@app.post("/predict")
def predict(data: InputData):
    prediction = predict_salary(data)
    return {"prediction": prediction}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, log_level="debug")
