@echo off
cd /d %~dp0

if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
)

call .venv\Scripts\activate

echo Installing requirements...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo Starting Streamlit app...
streamlit run streamlit_app.py
pause