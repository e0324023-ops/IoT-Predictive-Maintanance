@echo off
echo =====================================================================
echo Launching IoT Predictive Maintenance Dashboard...
echo =====================================================================
cd /d "%~dp0"
python -m streamlit run app.py
pause
