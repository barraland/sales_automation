### src.api.endpoint.py
import uvicorn
import requests
import json
import os
import tempfile
from datetime import datetime
from fastapi import FastAPI, Request, Response, BackgroundTasks
from fastapi.responses import PlainTextResponse
from langchain_core.messages import HumanMessage, AIMessage
from openai import OpenAI
from src.graph.graph_app import create_graph

app = FastAPI(title="Beverage Agent API")
beverage_agent = create_graph()

# --- CONFIGURAZIONE ---
TOKEN = "EAAedUO2ZA8XQBQvvBXZBWRjwQUMBZBLsb0h0XRzkZCC424oucLXWfAI7AmG0w1dFx91rZCsVBMt0tr7x5i9MyNsOjaGo4nsO2sEPUO4Vr3l4KLr9Uwusur56Un17OJDrFYSjsojqvhRtVmLbevuJbEU9zzZAF445ZCNUnLhfLGlwZAMR56hO06j2l4VZAIkM2XQZDZD"
PHONE_NUMBER_ID = "1043497188838302"
VERIFY_TOKEN = "my_verify_token_123"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# --- MAPPING AGENTI (Il cuore della soluzione) ---
# In produzione qui potresti interrogare una tabella SQL Agenti
AGENT_MAPPING = {
    "393755116724": "AG001",
    "393441234567": "AG002"
}

def transcribe_audio(media_id: str) -> str | None:
    """
    Scarica il vocale WhatsApp (OGG/Opus) e lo trascrive con OpenAI Whisper.
    Restituisce il testo trascritto, o None in caso di errore.
    """
    headers = {"Authorization": f"Bearer {TOKEN}"}

    # Step 1: ottieni URL di download dal Graph API
    try:
        r = requests.get(
            f"https://graph.facebook.com/v22.0/{media_id}",
            headers=headers
        )
        if r.status_code != 200:
            print(f"❌ Errore recupero URL media: {r.text}")
            return None
        media_url = r.json().get("url")
        if not media_url:
            print("❌ URL media assente nella risposta Meta")
            return None
    except Exception as e:
        print(f"❌ Errore Meta media API: {e}")
        return None

    # Step 2: scarica il file audio
    try:
        audio_resp = requests.get(media_url, headers=headers)
        if audio_resp.status_code != 200:
            print(f"❌ Errore download audio: {audio_resp.status_code}")
            return None
        audio_bytes = audio_resp.content
    except Exception as e:
        print(f"❌ Errore download audio: {e}")
        return None

    # Step 3: trascrivi con Whisper
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        with open(tmp_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language="it",
            )
        os.unlink(tmp_path)
        text = transcript.text.strip()
        print(f"🎙️ Trascritto: {text}")
        return text
    except Exception as e:
        print(f"❌ Errore Whisper: {e}")
        return None


def transcribe_and_process(sender_id: str, media_id: str):
    """Trascrive un vocale WhatsApp e lo passa alla pipeline normale."""
    print(f"\n🎙️ Vocale ricevuto da {sender_id} — trascrizione in corso...")
    text = transcribe_audio(media_id)
    if text:
        process_and_respond(sender_id, text)
    else:
        # Notifica l'utente che la trascrizione è fallita
        class _Msg:
            use_interactive_list = False
            text = "❌ Non sono riuscito a trascrivere il messaggio vocale. Puoi riscriverlo?"
        send_whatsapp_message(sender_id, _Msg())


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
        "current_datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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

                        elif msg["type"] == "audio":
                            media_id = msg["audio"]["id"]
                            background_tasks.add_task(transcribe_and_process, sender, media_id)
                            
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