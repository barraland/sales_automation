#!/bin/bash

# Definisci i percorsi (usa quelli assoluti per sicurezza)
VENV_PATH="/home/sommojames/langgraph_env"
PROJECT_PATH="/home/sommojames/sales_automation"
MAIN_FILE="$PROJECT_PATH/main.py"

echo "🚀 Attivazione ambiente virtuale..."
source $VENV_PATH/bin/activate

echo "🕵️  Avvio debugpy in ascolto sulla porta 5678..."
echo "⏳ In attesa del debugger di VS Code (premi F5)..."

# PYTHONPATH include sia la root che la cartella src per evitare errori di import
export PYTHONPATH=$PROJECT_PATH:$PROJECT_PATH/src

# Lancia il processo
python3 -m debugpy --listen 0.0.0.0:5678 --wait-for-client $MAIN_FILE