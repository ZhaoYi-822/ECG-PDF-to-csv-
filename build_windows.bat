@echo off
setlocal
python -m pip install -r requirements.txt pyinstaller
python -m PyInstaller --noconfirm --clean --windowed --name SamsungECG --collect-all matplotlib app.py
echo.
echo Build complete: dist\SamsungECG\SamsungECG.exe
pause
