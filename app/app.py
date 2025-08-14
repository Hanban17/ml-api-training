# Must be imported before loading the model
import logging
import os
import pickle

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flask_marshmallow import Marshmallow
from marshmallow import fields

from model.model import dummy_predict

# Must import this so joblib can find the function during unpickling
from model.preprocessing import AgeImputer, DropColumnsTransformer, EmbarkDeckImputer

logging.basicConfig(level=logging.INFO)


app = Flask(__name__)  # Initialize the Flask application
ma = Marshmallow(app)


class InputSchema(ma.Schema):
    pclass = fields.Integer(required=True)
    sex = fields.String(required=True)
    age = fields.Float(required=True)
    sibsp = fields.Integer(required=True)
    parch = fields.Integer(required=True)
    fare = fields.Float(required=True)
    embarked = fields.String()
    class_ = fields.String(data_key="class")
    who = fields.String(required=True)
    adult_male = fields.Boolean(required=True)
    deck = fields.String()
    embark_town = fields.String(required=True)
    alive = fields.String(required=True)
    alone = fields.Boolean(required=True)


schema = InputSchema()  # ✅ Move this ABOVE the route

DEFAULT_XGB = os.path.join(
    os.path.dirname(__file__), "..", "model", "titanic_xgboost_pipeline.joblib"
)
DEFAULT_PKL = os.path.join(
    os.path.dirname(__file__), "..", "model", "titanic_pipeline.pkl"
)
# --- Globals (start as None; fill on first use) ---
_xgb_model = None
_pkl_model = None


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "GREAT"})


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
def load_model():
    global _xgb_model
    if _xgb_model is None:
        model_path = os.getenv("XGB_MODEL_PATH", DEFAULT_XGB)
        app.logger.info("Loading XGB model from %s", model_path)
        _xgb_model = joblib.load(model_path)
    return _xgb_model


def load_model1():
    global _pkl_model
    if _pkl_model is None:
        model_path = os.getenv("PKL_MODEL_PATH", DEFAULT_PKL)
        app.logger.info("Loading PKL model from %s", model_path)
        with open(model_path, "rb") as f:
            _pkl_model = pickle.load(f)
    return _pkl_model


model = None

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
    global model
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
        model = load_model()
        prediction = model1.predict(input_df)

        return (
            jsonify({"input": data, "prediction": int(prediction[0]), "success": True}),
            200,
        )

    except Exception as e:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500


@app.route("/predictwithXGB1", methods=["POST"])
def predictwithXGB1():
    global model1
    json_data = request.get_json()
    errors = schema.validate(json_data)
    if errors:
        return jsonify(errors), 400

    df = pd.DataFrame([schema.load(json_data)])
    df.rename(columns={"class_": "class"}, inplace=True)
    model1 = load_model1()
    prediction = model1.predict(df)
    return jsonify({"prediction": int(prediction[0])}), 200


if __name__ == "__main__":
    model = load_model()
    model1 = load_model1()
    app.run(host="0.0.0.0", port=5000, debug=True)
