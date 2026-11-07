
@echo off
cd /d D:\mars_app

if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

python -m streamlit run app.py

pause
