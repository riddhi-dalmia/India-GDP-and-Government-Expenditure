@echo off
echo ====================================================
echo  XAI Project Setup - Government Spending and GDP
echo ====================================================
echo.
echo Step 1: Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.10 or higher from python.org
    pause
    exit /b 1
)

echo.
echo Step 2: Installing required packages...
call venv\Scripts\activate
pip install -r requirements.txt

echo.
echo Step 3: Testing data loader...
python src\01_data_loader.py

echo.
echo ====================================================
echo  Setup complete! Run the dashboard with:
echo     venv\Scripts\activate
echo     streamlit run app.py
echo ====================================================
pause
