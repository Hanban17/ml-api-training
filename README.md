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
