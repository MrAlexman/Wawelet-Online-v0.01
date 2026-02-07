@echo off
setlocal ENABLEEXTENSIONS

REM ==========================================================
REM Перейти в папку, где лежит этот bat (корень проекта)
REM ==========================================================
cd /d "%~dp0"

echo [INFO] Project root: %cd%

REM ==========================================================
REM Проверка main.py
REM ==========================================================
if not exist "main.py" (
    echo [ERROR] main.py not found in project root!
    pause
    exit /b 1
)

REM ==========================================================
REM Определяем python
REM ==========================================================
set "PYTHON=python"

REM Если есть виртуальное окружение — используем его
if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
    echo [INFO] Using virtual environment: .venv
) else (
    echo [INFO] Using system Python
)

REM ==========================================================
REM Проверка python
REM ==========================================================
"%PYTHON%" --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found or not executable
    pause
    exit /b 1
)

REM ==========================================================
REM Установка PYTHONPATH (на всякий случай)
REM ==========================================================
set "PYTHONPATH=%cd%"

REM ==========================================================
REM Запуск приложения
REM ==========================================================
echo.
echo [INFO] Starting application...
echo.

"%PYTHON%" "main.py"

REM ==========================================================
REM Обработка ошибки
REM ==========================================================
if errorlevel 1 (
    echo.
    echo [ERROR] Application exited with error code %errorlevel%
    pause
)

endlocal
