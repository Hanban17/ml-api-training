import pytest

from app.app import app  # adjust if your app is in app/__init__.py


@pytest.fixture
def client():
    with app.test_client() as client:
        yield client


def test_addition():
    assert 2 + 2 == 4
    print("Sucessful")


def test_predict(client):
    response = client.post("/predict", json={"text": "hello"})
    assert response.status_code == 200
    assert "prediction" in response.get_json()
