### src.api.endpoint.py
import uvicorn
import requests
import json
import os
from fastapi import FastAPI, Request, Response, BackgroundTasks
from fastapi.responses import PlainTextResponse
from langchain_core.messages import HumanMessage, AIMessage
from src.graph.graph_app import create_graph

app = FastAPI(title="Beverage Agent API")
beverage_agent = create_graph()

# --- CONFIGURAZIONE ---
TOKEN = "EAAedUO2ZA8XQBQjkSkaL5MPUlhpIPiX3ZCUfRCyQjfABI0DiN0X4ZBN5m8RP2PiZAoZBNEG6pNaCQfw6uTYZC3bqZArQz6JJFZBpDiiuTWDwdLRNRR4xHbIaEpXybjval2lZCZBP3DZCfhTsZAcVpsHA4jZA4WNLZAPeT4Jp5T0gkY5ZBkop08JFsiGuVDfuDxxCz0SrwrfpGkGypZBHWsEEjVWZA44k0CuCGJkyLyx8i77P4lQJZCKtd89le0e7JH5fwAcYT0h6wSLAbCaFx440fV6KZAYswZDZD"
PHONE_NUMBER_ID = "1043497188838302"
VERIFY_TOKEN = "my_verify_token_123"

# --- MAPPING AGENTI (Il cuore della soluzione) ---
# In produzione qui potresti interrogare una tabella SQL Agenti
AGENT_MAPPING = {
    "393755116724": "AG001",
    "393441234567": "AG002"
}

def process_and_respond(sender_id: str, user_text: str):
    print("\n" + "="*40)
    print(f"📩 NUOVO MESSAGGIO DA: {sender_id}")
    print(f"💬 TESTO: {user_text}")
    print("="*40)


    agent_code = AGENT_MAPPING.get(sender_id, "AG001")
    config = {"configurable": {"thread_id": sender_id}}

    # ✅ Recupera stato corrente (snapshot)
    current_state = beverage_agent.get_state(config)
    state_values = current_state.values if current_state.values else {}

    # ✅ Leggi campi già presenti nello snapshot
    history = state_values.get("chat_history", []) if isinstance(state_values, dict) else getattr(state_values, "chat_history", [])

    # ✅ Prepara input completo per invoke — include reset is_finished e question
    inputs = {
        "question": user_text,
        "chat_history": history,
        "observations": {},        # reset observations a ogni nuova richiesta
        "agent_code": agent_code,
        "is_finished": False,
        "next_tasks": [],          # reset task del turno precedente
    }

    # ✅ Invoca il grafo
    result = beverage_agent.invoke(inputs, config=config)

    answer = result.get("final_answer", "Non ho trovato informazioni specifiche.")
    history_text = answer.text if hasattr(answer, 'text') else str(answer)
    new_history = history + [HumanMessage(content=user_text), AIMessage(content=history_text)]

    # ✅ Aggiorna lo stato con la chat aggiornata (campi dichiarati in AgentState)
    beverage_agent.update_state(config, {
        "chat_history": new_history,
    })

    send_whatsapp_message(sender_id, answer)

def send_whatsapp_message(to: str, content):
    """
    Funzione universale Meta Cloud API.
    """
    url = f"https://graph.facebook.com/v22.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {TOKEN}", 
        "Content-Type": "application/json"
    }
    
    # 1. CASO LISTA INTERATTIVA
    if hasattr(content, 'use_interactive_list') and content.use_interactive_list:
        rows = []
        for item in content.items[:10]:
            rows.append({
                "id": item.id,
                "title": item.title[:24], 
                "description": (item.description[:72] if item.description else "")
            })
        
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "interactive",
            "interactive": {
                "type": "list",
                "header": {"type": "text", "text": "Selezione Prodotti"},
                "body": {"text": content.text},
                "footer": {"text": "Tocca il bottone per scegliere"},
                "action": {
                    "button": content.list_button_text[:20],
                    "sections": [{"title": "Risultati Ricerca", "rows": rows}]
                }
            }
        }
    
    # 2. CASO TESTO SEMPLICE
    else:
        text_body = content.text if hasattr(content, 'text') else str(content)
        # Protezione caratteri per Meta
        if len(text_body) > 4000:
            text_body = text_body[:3997] + "..."
            
        payload = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "text",
            "text": {"body": text_body}
        }

    try:
        r = requests.post(url, json=payload, headers=headers)
        if r.status_code == 200:
            print(f"✅ Messaggio inviato a {to}")
        else:
            print(f"❌ Errore Meta: {r.text}")
    except Exception as e:
        print(f"❌ Errore connessione: {e}")

@app.post("/whatsapp")
async def webhook(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    try:
        for entry in data.get("entry", []):
            for change in entry.get("changes", []):
                value = change.get("value", {})
                if "messages" in value:
                    for msg in value["messages"]:
                        sender = msg["from"]
                        
                        if msg["type"] == "text":
                            text = msg["text"]["body"]
                            background_tasks.add_task(process_and_respond, sender, text)
                        
                        elif msg["type"] == "interactive":
                            # Se l'utente seleziona un cliente dalla lista
                            selection_id = msg["interactive"]["list_reply"]["id"]
                            selection_title = msg["interactive"]["list_reply"]["title"]
                            fake_text = f"Ho selezionato: {selection_title} (ID: {selection_id})"
                            background_tasks.add_task(process_and_respond, sender, fake_text)
                            
    except Exception as e:
        print(f"⚠️ Errore parsing webhook: {e}")
    return {"status": "ok"}

@app.get("/whatsapp")
async def verify(request: Request):
    p = request.query_params
    if p.get("hub.verify_token") == VERIFY_TOKEN:
        return PlainTextResponse(content=p.get("hub.challenge"))
    return Response(content="Forbidden", status_code=403)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9999)