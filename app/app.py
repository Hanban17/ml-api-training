from flask import Flask, jsonify, request

app = Flask(__name__)  # Initialize the Flask application


@app.route("/")  # Define a route for the root URL
def index():
    return "Hello, World!"


if __name__ == "__main__":
    app.run(debug=True)
