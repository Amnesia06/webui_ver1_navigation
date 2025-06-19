@echo off
REM Setup script for Farm Robot Path Planner Web UI (Windows)

echo 🤖 Setting up Farm Robot Path Planner Web UI...

REM Create virtual environment
python -m venv farm_planner_env
call farm_planner_env\Scripts\activate.bat

REM Install requirements
pip install -r requirements.txt

echo ✅ Setup complete!
echo.
echo To run the application:
echo 1. Activate virtual environment: farm_planner_env\Scripts\activate.bat
echo 2. Run server: python run_server.py
echo 3. Open browser: http://localhost:5000

pause
