import logging

from flask import Flask, jsonify, request
from model.model import dummy_predict

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


if __name__ == "__main__":
    app.run(debug=True)
