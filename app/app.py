import logging
import os
import pickle

import joblib
import pandas as pd
from flask import Flask, jsonify, request

from model.model import dummy_predict

# Must import this so joblib can find the function during unpickling
from model.preprocessing import AgeImputer, EmbarkDeckImputer, drop_columns_fn

logging.basicConfig(level=logging.INFO)


app = Flask(__name__)  # Initialize the Flask application


@app.route("/")  # Define a route for the root URL
def index():
    return "Hello, again!"


@app.route("/ping")
def ping():
    return jsonify({"status": "ok"})


@app.route("/echo", methods=["POST"])
def echo():
    data = request.get_json()  # Parse JSON from the request body
    if not data or "message" not in data:
        # If no JSON or 'message' field is missing, return a 400 Bad Request
        return jsonify({"error": "No 'message' provided"}), 400
    original_msg = data["message"]
    # Process the message (for example, reverse the string)
    processed_msg = original_msg[::-1]  # reverses the string
    app.logger.info("Received /echo request with data: %s", data)
    # Return the result as JSON
    return jsonify({"original": original_msg, "processed": processed_msg})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data or "text" not in data:
            app.logger.warning("Bad request to /predict: %s", data)
            return jsonify({"error": "Missing 'text' in request body"}), 400
        input_text = data["text"]
        app.logger.info("Received /predict request with text: %s", input_text)
        if not isinstance(input_text, str):
            return jsonify({"error": "'text' must be a string"}), 400
        result = dummy_predict(input_text)
        app.logger.info("Prediction result: %s", result)

        return (
            jsonify(
                {
                    "input": input_text,
                    "prediction": result["prediction"],
                    "model": "dummy_v1",
                    "success": True,
                }
            ),
            200,
        )
    except Exception as e:
        app.logger.exception("Error during prediction")
        return jsonify({"error": "Internal server error"}, e), 500


# Load model once at startup
model_path = os.path.join(
    os.path.dirname(__file__), "..", "model", "titanic_xgboost_pipeline.joblib"
)
model = joblib.load(model_path)


EXPECTED_COLUMNS = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked",
    "class",
    "who",
    "adult_male",
    "deck",
    "embark_town",
    "alive",
    "alone",
]


@app.route("/predictwithXGB", methods=["POST"])
def predictwithXGB():
    data = request.get_json()

    if not data:
        return (
            jsonify(
                {
                    "error": "Missing JSON body.",
                    "example_format": {
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
                        "deck": None,
                        "embark_town": "Southampton",
                        "alive": "no",
                        "alone": False,
                    },
                }
            ),
            400,
        )

    # Check for missing fields
    missing_fields = [col for col in EXPECTED_COLUMNS if col not in data]
    if missing_fields:
        return (
            jsonify(
                {
                    "error": f"Missing required fields: {missing_fields}",
                    "expected_fields": EXPECTED_COLUMNS,
                }
            ),
            400,
        )

    try:
        # Convert input JSON into a one-row DataFrame
        input_df = pd.DataFrame([data])

        # Predict using the pipeline model
        prediction = model1.predict(input_df)

        return (
            jsonify({"input": data, "prediction": int(prediction[0]), "success": True}),
            200,
        )

    except Exception as e:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500


model1 = pickle.load(open("model/titanic_pipeline.pkl", "rb"))


@app.route("/predictwithXGB1", methods=["POST"])
def predictwithXGB1():
    data = request.get_json()

    if not data:
        return (
            jsonify(
                {
                    "error": "Missing JSON body.",
                    "example_format": {
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
                        "deck": None,
                        "embark_town": "Southampton",
                        "alive": "no",
                        "alone": False,
                    },
                }
            ),
            400,
        )

    # Check for missing fields
    missing_fields = [col for col in EXPECTED_COLUMNS if col not in data]
    if missing_fields:
        return (
            jsonify(
                {
                    "error": f"Missing required fields: {missing_fields}",
                    "expected_fields": EXPECTED_COLUMNS,
                }
            ),
            400,
        )

    try:
        # Convert input JSON into a one-row DataFrame
        input_df = pd.DataFrame([data])

        # Predict using the pipeline model
        prediction = model1.predict(input_df)

        return (
            jsonify({"input": data, "prediction": int(prediction[0]), "success": True}),
            200,
        )

    except Exception as e:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
