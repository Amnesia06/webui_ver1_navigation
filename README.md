
A small Python web UI and navigation utility for generating and visualizing navigation/path data. 
Includes a simple web server and path-related utilities for local testing and development.

Key features
[1] Lightweight web interface to view navigation data (templates/index.html).
[2] Simple path utilities and runner scripts (path.py, run_server.py, web_app.py).
[3] Setup scripts for Windows and Unix-like environments (setup.bat, setup.sh).

Repository layout
[1] path.py — Core path/navigation utilities used by the app.
[2] web_app.py — Main web application (Flask or similar) that serves the UI.
[3] run_server.py — Helper to run the web server (entry point for local testing).
[4] requrements.html — (note: spelled "requrements") HTML file containing dependency or requirements notes — check it for dependency hints.
[5] setup.bat / setup.sh — Setup scripts for Windows and Unix environments.
[6] templates/index.html — Frontend template for the web UI.
[7] __pycache__/ — Python bytecode cache (ignored in VCS).

Requirements:

Flask==2.3.3
matplotlib==3.7.2
numpy==1.24.3
Pillow==10.0.0
