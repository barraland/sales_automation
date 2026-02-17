## main.py
import uvicorn

import sys
import os
# FORZA IL PATH DEL VENV NEL SISTEMA
sys.path.append("/home/sommojames/langgraph_env/lib/python3.10/site-packages")
print(f"🚀 EXECUTABLE: {sys.executable}")

# Aggiungiamo la directory corrente al path di sistema 
# per evitare errori di importazione dei moduli in 'src'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("🚀 Avvio Beverage Sales Agent...")
    print("📡 Webhook WhatsApp pronto su: http://localhost:9999/whatsapp")
    print("📖 Documentazione API (Swagger): http://localhost:9999/docs")

    uvicorn.run("src.api.endpoint:app", host="0.0.0.0", port=9999, reload=False)