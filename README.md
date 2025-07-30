


# Step 1: Create a virtual environment
python -m venv venv

# Step 2: Activate the virtual environment (PowerShell)
.\venv\Scripts\Activate.ps1

# Step 3: Upgrade pip (optional but recommended)
python -m pip install --upgrade pip

# Step 4: Install project dependencies
pip install -r requirements.txt

# Run Flask without debug
$env:FLASK_APP = "app.app"
flask run

# Run Flask with debug
python app\app.py