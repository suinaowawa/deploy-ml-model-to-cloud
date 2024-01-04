import requests

url = "https://salary-prediction-endpoint.onrender.com/predict/"

# Data to be sent in the request body
data = {
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

# Make a POST request
response = requests.post(url, json=data)

# Print the response
print(f"Status Code: {response.status_code}")
print("Response Body:")
print(response.text)
