#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
echo "NEXUS Backend — starting on http://localhost:8000"
echo ""
mkdir -p nexus_vault/chats nexus_vault/records nexus_vault/prescriptions nexus_vault/embeddings
if ! python -c "import fastapi, uvicorn" 2>/dev/null; then
  echo "Installing dependencies..."
  pip install -r requirements.txt
fi
exec uvicorn main:app --reload --host 0.0.0.0 --port 8000
