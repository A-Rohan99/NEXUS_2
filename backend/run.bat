@echo off
cd /d "%~dp0"
echo NEXUS Backend — starting on http://localhost:8000
echo.
if not exist "nexus_vault" mkdir nexus_vault
if not exist "nexus_vault\chats" mkdir nexus_vault\chats
if not exist "nexus_vault\records" mkdir nexus_vault\records
if not exist "nexus_vault\prescriptions" mkdir nexus_vault\prescriptions
if not exist "nexus_vault\embeddings" mkdir nexus_vault\embeddings
python -c "import fastapi, uvicorn" 2>nul || (
  echo Installing dependencies...
  pip install -r requirements.txt
)
uvicorn main:app --reload --host 0.0.0.0 --port 8000
