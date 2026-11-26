@echo off
rem Starts Piano Typer. Double-click me.
rem The Python environment lives in %USERPROFILE%\.pianotyper\venv because Windows
rem can't load the audio driver from a very long folder path.
cd /d "%~dp0"
set VENV=%USERPROFILE%\.pianotyper\venv
if not exist "%VENV%\Scripts\python.exe" (
  echo First run: setting up Python packages...
  python -m venv "%VENV%" || goto :err
  "%VENV%\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt || goto :err
)
if not exist layout.json "%VENV%\Scripts\python.exe" layout.py
"%VENV%\Scripts\python.exe" server.py %*
goto :eof
:err
echo Setup failed. Is Python installed? (python --version)
pause
