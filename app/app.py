import logging

from flask import Flask, jsonify, request

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


if __name__ == "__main__":
    app.run(debug=True)
if __name__ == "__main__":
    app.run(debug=True)
