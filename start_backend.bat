@echo off
echo.
echo  EduLens - AI Classroom Intelligence
echo  Backend Server (FastAPI)
echo.

cd /d "%~dp0"

echo [1/4] Clearing Python bytecode cache...
for /d /r backend %%d in (__pycache__) do (
    if exist "%%d" rmdir /s /q "%%d" 2>nul
)
echo  Done.

echo.
echo [2/4] Installing backend dependencies...
pip install fastapi "uvicorn[standard]" python-multipart pydantic fpdf2 --quiet 2>nul
if errorlevel 1 (
    echo  Warning: pip install had issues - continuing anyway
)

echo.
echo [3/4] Checking model weights...
if exist "weights\best_clip_model.pth" (
    echo  [OK] Engagement model found
) else (
    echo  [INFO] Engagement model not found - will use demo mode
)
if exist "weights\best_collab_group_fresh.npz" (
    echo  [OK] Collaboration model found
) else (
    echo  [INFO] Collaboration model not found - will use demo mode
)

echo.
echo [4/4] Starting backend on http://localhost:8000
echo       Swagger docs: http://localhost:8000/docs
echo.
set PYTHONDONTWRITEBYTECODE=1
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
pause
