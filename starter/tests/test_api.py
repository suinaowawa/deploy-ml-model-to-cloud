from main import app


def test_welcome():
    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Salary Prediction API"}


def test_predict_salary_lte_50():
    from fastapi.testclient import TestClient

    client = TestClient(app)
    data = {
        "age": 35,
        "workclass": "Private",
        "fnlgt": 200000,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 5000,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States",
    }
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert response.json() == {"prediction": "<=50K"}


def test_predict_salary_gt_50():
    from fastapi.testclient import TestClient

    client = TestClient(app)
    data = {
        "age": 40,
        "workclass": "Private",
        "fnlgt": 216237,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Divorced",
        "occupation": "Exec-managerial",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 45,
        "native_country": "United-States",
    }
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert response.json() == {"prediction": ">50K"}
