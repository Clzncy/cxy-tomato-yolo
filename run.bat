@echo off
cd /d "%~dp0"
where pythonw >nul 2>&1
if %errorlevel%==0 (
    start "" pythonw yolo_trainer_app.py
) else (
    start "" python yolo_trainer_app.py
)
exit /b
