import json

import pytest

from app.app import app


@pytest.fixture
def client():
    with app.test_client() as client:
        yield client


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_ping(client):
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_echo_valid(client):
    payload = {"message": "hello"}
    response = client.post("/echo", json=payload)
    assert response.status_code == 200
    result = response.get_json()
    assert result["original"] == "hello"
    assert result["processed"] == "olleh"


def test_echo_missing_message(client):
    response = client.post("/echo", json={})
    assert response.status_code == 400


def test_predict_valid(client):
    payload = {"text": "hello model"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    result = response.get_json()
    assert "prediction" in result


def test_predict_missing_text(client):
    response = client.post("/predict", json={})
    assert response.status_code == 400


def test_predictwithXGB_missing_body(client):
    response = client.post("/predictwithXGB", json={})
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_model1_loads():
    from app.app import load_model1

    model = load_model1()
    assert model is not None


def test_predictwithXGB1_valid(client):
    # Example taken from your expected input
    valid_input = {
        "pclass": 3,
        "sex": "male",
        "age": 22.0,
        "sibsp": 1,
        "parch": 0,
        "fare": 7.25,
        "embarked": "S",
        "class": "Third",
        "who": "man",
        "adult_male": True,
        "deck": "A",
        "embark_town": "Southampton",
        "alive": "no",
        "alone": False,
    }

    response = client.post("/predictwithXGB1", json=valid_input)
    assert response.status_code == 200
    assert "prediction" in response.get_json()
    assert response.status_code == 200
    assert "prediction" in response.get_json()
