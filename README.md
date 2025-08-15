# Flask ML API Project

## Project Overview

This is a simple Flask-based web API intended for serving a machine learning model. So far, it includes:

- Basic API routes (`/`, `/ping`, and `/echo`)
- Testing setup with `pytest`
- Code formatting/linting tools: `black`, `flake8`, `isort`
- Virtual environment setup
- Project structure ready for expansion

---

## Folder Structure


flask-ml-api-project/
├── app/                     # Application package
│   └── __init__.py          # Initializes Flask app and defines routes
├── model/                   # (For ML model files or related code)
├── tests/                   # Test suite for the application
│   └── test_sample.py       # Basic test to verify API endpoints
├── .gitignore               # Git ignore file (e.g. venv, __pycache__, etc.)
├── requirements.txt         # Python dependencies for the project
└── README.md                # Project documentation (this file)
Note: The app/ directory will also contain other modules (like routes, config, etc.) as the project grows. The model/ directory is currently a placeholder for any machine learning model files or code that will be added in later stages. The .gitignore is configured to ignore virtual environment folders, compiled bytecode, and other unnecessary files.

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Hanban17/ml-api-training.git
cd ml-api-training

### 2. Set up a virtual enironment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1

### 3. Install dependencies

```bash
pip install -r requirements.txt

## Running app

with debug:
python app\app.py

without debug:
$env:FLASK_APP = "app.app"
flask run

##API Endpoints

| Method | Endpoint | Description              |
| ------ | -------- | ------------------------ |
| GET    | `/`      | Welcome message          |
| GET    | `/ping`  | Health check             |
| POST   | `/echo`  | Echo back a JSON message |

Example using cURL:
Invoke-WebRequest -Uri http://localhost:5000/echo `
-Method POST `
-Body '{"message":"Hello"}' `
-ContentType "application/json"

Or via Postman

## /predict Endpoint
URL:
POST /predict

Description:
Accepts a JSON payload with a "text" field and returns a prediction.
(Currently uses a dummy model that returns the length of the input text.)

### Example request
POST /predict
Content-Type: application/json

{
  "text": "hello"
}

### Example Response
{
  "prediction": 5
}
### Example Bad Request
POST /predict
Content-Type: application/json

{}
### Example Bad Request Response
{
  "error": "Missing 'text' in request body"
}

🧠 Model Training and Preprocessing
We extended this project to include a fully reproducible machine learning pipeline using the Titanic dataset. This pipeline includes preprocessing, model training, and serialization steps.

model/preprocessing.py: Contains custom preprocessing classes and functions, including:

EmbarkDeckImputer: Handles missing values in embark_town and deck.

AgeImputer: Imputes missing age values based on grouped means by sex, pclass, and alone.

drop_columns_fn: Drops irrelevant columns for model training.

model/titanic_model.py: This script:

Loads and preprocesses the Titanic dataset.

Builds a pipeline combining preprocessing and an XGBoostClassifier.

Trains the model using train_test_split.

Saves the pipeline in two formats:

titanic_pipeline.pkl

titanic_xgboost_pipeline.joblib

These model artifacts are used to simulate loading a trained model in production.

📡 API Endpoints for Prediction
Two new prediction endpoints were added to demonstrate the use of both pickle and joblib serialized pipelines:

POST /predictwithXGB
Description: Uses the pipeline stored in titanic_xgboost_pipeline.joblib to make predictions.

Input Format:
A JSON object representing one Titanic passenger, with the full feature set expected by the model:
{
  "pclass": 3,
  "sex": "male",
  "age": 22.0,
  "sibsp": 1,
  "parch": 0,
  "fare": 7.25,
  "embarked": "S",
  "class": "Third",
  "who": "man",
  "adult_male": true,
  "deck": null,
  "embark_town": "Southampton",
  "alive": "no",
  "alone": false
}
Output:
{
    "input": {
        "adult_male": true,
        "age": 22.0,
        "alive": "no",
        "alone": false,
        "class": "Third",
        "deck": null,
        "embark_town": "Southampton",
        "embarked": "S",
        "fare": 7.25,
        "parch": 0,
        "pclass": 3,
        "sex": "male",
        "sibsp": 1,
        "who": "man"
    },
    "prediction": 0,
    "success": true
}
POST /predictwithXGB1
Description: Uses the titanic_pipeline.pkl model to demonstrate loading and using a Pickle-serialized pipeline.

Functionality is the same as /predictwithXGB.

✅ Input Validation with Marshmallow
The Flask API uses Marshmallow to validate incoming JSON payloads. This ensures that data passed to the model is complete, structured, and correctly typed.

We define an InputSchema using Marshmallow to enforce the expected format for prediction endpoints like /predictwithXGB1.

Example Schema Definition:
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

This schema:

Validates input types and required fields

Renames the class_ field to class to avoid Python keyword conflicts

Helps return meaningful validation errors on bad input

Example Validation Error (from /predictwithXGB1):
{
  "age": ["Missing data for required field."],
  "fare": ["Not a valid number."]
}

🧪 Testing the API (test_app.py)
A test suite has been added using Pytest to ensure reliability and correctness of the Flask application. The test file tests/test_app.py covers:

Basic health and ping endpoints

Schema validation

Titanic prediction endpoints (including /predictwithXGB1)

Error handling and edge cases

Example Test Case:
def test_predictwithXGB1_valid(client):
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

# Running the App with Docker
docker build -t flask-ml-app:1.2 .
docker run -p 5000:5000 flask-ml-app:1.2

Flask will log something like:
* Running on all addresses (0.0.0.0)
* Running on http://127.0.0.1:5000
* Running on http://172.17.0.2:5000

Here’s what each means:

127.0.0.1:5000 → Loopback address inside the container.
172.17.0.2:5000 → The container’s internal IP on Docker’s bridge network (not directly accessible from your host machine).
0.0.0.0:5000 → The app is listening on all interfaces inside the container.

How to Access the API
From your host machine (Windows/Mac/Linux), use:
http://localhost:5000
or 
http://127.0.0.1:5000

#Environment Variables & Model Paths

This application uses environment variables to configure the location of trained model files at runtime.
Two variables are used:

Variable Name	Purpose	Default Value (if not set)
XGB_MODEL_PATH	Path to the XGBoost pipeline .joblib model file	model/titanic_xgboost_pipeline.joblib
PKL_MODEL_PATH	Path to the Scikit-learn pipeline .pkl model file	model/titanic_pipeline.pkl

##Local Development
When running locally, you can either:
Set these environment variables manually
Windows PowerShell
$env:XGB_MODEL_PATH = "model\titanic_xgboost_pipeline.joblib"
$env:PKL_MODEL_PATH = "model\titanic_pipeline.pkl"
python app/app.py

##Docker Usage

> docker run -d `
>>   -p 5000:5000 `
>>   --name test-api-env `
>>   -v "$PWD\model:/app/model" `
>>   -e PKL_MODEL_PATH=/app/model/titanic_pipeline_b.pkl `
>>   -e XGB_MODEL_PATH=/app/model/titanic_xgboost_pipeline.joblib `
>>   flask-ml-app:env_1

or
docker run -d -p 5000:5000 \
  -e XGB_MODEL_PATH="/app/model/titanic_xgboost_pipeline.joblib" \
  -e PKL_MODEL_PATH="/app/model/titanic_pipeline.pkl" \
  hanhyk/ml-api-training:env_1

  Note: The container's paths should match where the models are copied inside the Docker image (usually /app/model/...).



## Running Tests
pytest

## Code Quality Tools
black .
isort .

##Lint code
flake8 .


## Notes
Always activate your virtual environment before working.

Keep requirements.txt updated with:
pip freeze > requirements.txt

Additional Files (if needed): If your logic is complex, you might create new modules to keep app.py clean. For instance, a model/utils.py for model loading or preprocessing functions, or a app/schemas.py for Pydantic/Marshmellow models. 

We should push to a seperate branch, that will then push to main when it succeeds
