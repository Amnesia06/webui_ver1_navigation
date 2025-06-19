#!/bin/bash
# Setup script for Farm Robot Path Planner Web UI

echo "🤖 Setting up Farm Robot Path Planner Web UI..."

# Create virtual environment
python3 -m venv farm_planner_env
source farm_planner_env/bin/activate

# Install requirements
pip install -r requirements.txt

echo "✅ Setup complete!"
echo ""
echo "To run the application:"
echo "1. Activate virtual environment: source farm_planner_env/bin/activate"
echo "2. Run server: python run_server.py"
echo "3. Open browser: http://localhost:5000"
