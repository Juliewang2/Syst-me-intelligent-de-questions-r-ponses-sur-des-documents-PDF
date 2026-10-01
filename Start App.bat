@echo off
setlocal EnableDelayedExpansion
title PDF Chat - Startup

echo ================================================================
echo   PDF Chat - Windows Startup Script
echo ================================================================
echo.

REM ---------------------------------------------------------------
REM Move to the folder this script lives in, so it works no matter
REM where the user double-clicks it from.
REM ---------------------------------------------------------------
cd /d "%~dp0"
echo [INFO] Working directory: %cd%
echo.

REM ---------------------------------------------------------------
REM 1. Verify Python is installed
REM ---------------------------------------------------------------
echo [STEP 1/6] Checking for Python...
REM The pinned dependencies (numpy, faiss, onnxruntime) need Python 3.12;
REM prefer it via the "py" launcher even if a newer Python is the default.
set "PYTHON_CMD=python"
py -3.12 --version >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_CMD=py -3.12"
) else (
    where python >nul 2>nul
    if errorlevel 1 (
        echo.
        echo [ERROR] Python was not found on your system, or it is not on PATH.
        echo         Please install Python 3.12 from https://www.python.org/downloads/
        echo         and make sure to check "Add python.exe to PATH" during setup.
        echo.
        pause
        exit /b 1
    )
)
for /f "tokens=2" %%v in ('%PYTHON_CMD% --version 2^>^&1') do set PYVER=%%v
echo [OK] Found Python %PYVER% ^(%PYTHON_CMD%^)
echo %PYVER% | findstr /b "3.12." >nul
if errorlevel 1 (
    echo [WARN] This project is tested with Python 3.12. With %PYVER% some
    echo        dependencies may fail to install. Install Python 3.12 if they do.
)
echo.

REM ---------------------------------------------------------------
REM 2. Create a virtual environment if one doesn't already exist
REM ---------------------------------------------------------------
echo [STEP 2/6] Checking for virtual environment...
if not exist "venv\Scripts\activate.bat" (
    echo [INFO] No virtual environment found. Creating one now...
    %PYTHON_CMD% -m venv venv
    if errorlevel 1 (
        echo.
        echo [ERROR] Failed to create the virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
) else (
    echo [OK] Virtual environment already exists.
)
echo.

REM ---------------------------------------------------------------
REM 3. Activate the virtual environment
REM ---------------------------------------------------------------
echo [STEP 3/6] Activating virtual environment...
call "venv\Scripts\activate.bat"
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to activate the virtual environment.
    pause
    exit /b 1
)
echo [OK] Virtual environment activated.
echo.

REM ---------------------------------------------------------------
REM 4. Install missing dependencies
REM ---------------------------------------------------------------
echo [STEP 4/6] Checking / installing dependencies (this may take a minute)...
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to install dependencies from requirements.txt.
    echo         Check your internet connection and try again.
    echo.
    pause
    exit /b 1
)
echo [OK] Dependencies are up to date.
echo.

REM ---------------------------------------------------------------
REM 5. Verify the .env file exists
REM ---------------------------------------------------------------
echo [STEP 5/6] Checking for .env configuration file...
if not exist ".env" (
    if exist ".env.example" (
        echo [WARN] No .env file found. Creating one from .env.example...
        copy /y ".env.example" ".env" >nul
        echo [WARN] IMPORTANT: Open the new .env file and set your OPENAI_API_KEY
        echo        before using chat/upload features. See INSTRUCTION.md for help.
    ) else (
        echo [ERROR] Neither .env nor .env.example was found. Cannot continue safely.
        pause
        exit /b 1
    )
) else (
    echo [OK] .env file found.
    findstr /C:"OPENAI_API_KEY=sk-your-openai-api-key-here" ".env" >nul
    if not errorlevel 1 (
        echo [WARN] Your .env file still has the placeholder OPENAI_API_KEY.
        echo        Chat and upload features will fail until you set a real key.
        echo        See INSTRUCTION.md, Section 11, for how to get one.
    )
)
echo.

REM ---------------------------------------------------------------
REM 6. Launch the application
REM ---------------------------------------------------------------
echo [STEP 6/6] Starting PDF Chat...
echo ================================================================
echo   The app will open at:  http://127.0.0.1:8000
echo   API docs available at: http://127.0.0.1:8000/docs
echo   Press CTRL+C in this window to stop the server.
echo ================================================================
echo.

start "" http://127.0.0.1:8000

uvicorn main:app --host 127.0.0.1 --port 8000 --reload

REM ---------------------------------------------------------------
REM If uvicorn exits (crash or normal stop), keep the window open
REM so the user can read any error messages.
REM ---------------------------------------------------------------
echo.
echo ================================================================
echo   The server has stopped.
echo ================================================================
pause
