# sign2text-api
API microservice for sign2text backend.

# Prerequisites
1. Python
2. pip

# Running the app
```
# Create a python virtual environment
python -m venv .venv

# Run the environment
.venv\Scripts\Activate.ps1

# Validate env running
Get-Command python

# Installing requirements
pip install -r requirements.txt

# Start the server (within the environment)
fastapi dev main.py
```

# Documentation
```
# After server started
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/redoc
```