#!/bin/bash

# Porta definita nel tuo main.py
PORT=9999

echo "🚀 Avvio del tunnel Ngrok sulla porta $PORT..."
echo "------------------------------------------------"
echo "👉 Ricorda di copiare l'URL .ngrok-free.app"
echo "👉 Incollalo nella dashboard Meta con /whatsapp alla fine"
echo "------------------------------------------------"

# Avvia ngrok
ngrok http $PORT