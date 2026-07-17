@echo off
echo.
echo  ╔══════════════════════════════════════════╗
echo  ║   EduLens — AI Classroom Intelligence   ║
echo  ║   Frontend Server (Next.js)             ║
echo  ╚══════════════════════════════════════════╝
echo.

cd /d "%~dp0\frontend"

echo [1/2] Installing Node dependencies...
call npm install

echo.
echo [2/2] Starting frontend on http://localhost:3000
echo.
call npm run dev
pause
