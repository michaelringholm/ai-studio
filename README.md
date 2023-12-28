# ai-studio
A UI for training AI models

# Upgrade pip
python -m pip install --upgrade pip

# Create python virtual environment for dependencies
python -m venv venv

# Activate virtual environment
.\venv\scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run UI
streamlit run app.py

# Creating custom callback
https://keras.io/guides/writing_your_own_callbacks/