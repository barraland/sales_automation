### src.api.endpoint.py
import uvicorn
import requests
import csv
import json
import os
import sqlite3
import tempfile
from datetime import datetime
from fastapi import FastAPI, Request, Response, BackgroundTasks
from fastapi.responses import PlainTextResponse, JSONResponse
from langchain_core.messages import HumanMessage, AIMessage
from openai import OpenAI
from src.graph.graph_app import create_graph, agent_name_map

app = FastAPI(title="Beverage Agent API")
beverage_agent = create_graph()

# --- CONFIGURAZIONE ---
TOKEN = "EAAedUO2ZA8XQBQvvBXZBWRjwQUMBZBLsb0h0XRzkZCC424oucLXWfAI7AmG0w1dFx91rZCsVBMt0tr7x5i9MyNsOjaGo4nsO2sEPUO4Vr3l4KLr9Uwusur56Un17OJDrFYSjsojqvhRtVmLbevuJbEU9zzZAF445ZCNUnLhfLGlwZAMR56hO06j2l4VZAIkM2XQZDZD"
PHONE_NUMBER_ID = "1043497188838302"
VERIFY_TOKEN = "my_verify_token_123"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# --- MAPPING AGENTI caricato da data/agenti.csv (telefono → codice) ---
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_agenti_csv = os.path.join(_project_root, "data", "agenti.csv")
AGENT_MAPPING: dict = {}
if os.path.exists(_agenti_csv):
    with open(_agenti_csv, newline="", encoding="utf-8") as _f:
        for _row in csv.DictReader(_f):
            AGENT_MAPPING[_row["telefono"]] = _row["codice"]
else:
    print(f"⚠️ File agenti non trovato: {_agenti_csv} — nessun agente mappato")

# =============================================================================
# LOGGING SU SQLITE
# =============================================================================
_LOG_DB = os.path.join(_project_root, "sql_lite", "db", "database_ordini.db")

def _init_log_table():
    conn = sqlite3.connect(_LOG_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversation_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          TEXT    NOT NULL,
            sender_id   TEXT    NOT NULL,
            agent_code  TEXT    NOT NULL,
            user_msg    TEXT,
            plan_json   TEXT,
            obs_json    TEXT,
            response    TEXT,
            duration_ms INTEGER
        )
    """)
    conn.commit()
    conn.close()

_init_log_table()


def _log_turn(
    sender_id: str,
    agent_code: str,
    user_msg: str,
    result: dict,
    answer,
    duration_ms: int,
):
    """Scrive un record di log per ogni turno di conversazione."""
    try:
        # Nuovo: pending_call al posto di next_tasks/observations
        pending = result.get("pending_call")
        plan_list = [pending] if pending else []
        obs = {}

        conn = sqlite3.connect(_LOG_DB)
        conn.execute(
            """
            INSERT INTO conversation_log
                (ts, sender_id, agent_code, user_msg, plan_json, obs_json, response, duration_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                sender_id,
                agent_code,
                user_msg,
                json.dumps(plan_list, ensure_ascii=False, default=str),
                json.dumps(obs,       ensure_ascii=False, default=str),
                answer.text if hasattr(answer, "text") else str(answer),
                duration_ms,
            ),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"⚠️ Errore log DB: {e}")


# =============================================================================
# TRASCRIZIONE AUDIO
# =============================================================================
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
        class _Msg:
            use_interactive_list = False
            text = "❌ Non sono riuscito a trascrivere il messaggio vocale. Puoi riscriverlo?"
        send_whatsapp_message(sender_id, _Msg())


# =============================================================================
# PIPELINE PRINCIPALE
# =============================================================================
def run_graph(sender_id: str, user_text: str):
    """
    Cuore della pipeline: esegue il grafo e restituisce l'oggetto FinalAnswer.
    Usato sia da process_and_respond (WhatsApp) sia da /test/chat (test locale).
    """
    agent_code = AGENT_MAPPING.get(sender_id, "AG001")
    config = {"configurable": {"thread_id": sender_id}}

    current_state = beverage_agent.get_state(config)
    state_values = current_state.values if current_state.values else {}
    history = (
        state_values.get("chat_history", [])
        if isinstance(state_values, dict)
        else getattr(state_values, "chat_history", [])
    )

    agent_info = agent_name_map.get(agent_code, {})
    inputs = {
        "question": user_text,
        "chat_history": history,
        "agent_code": agent_code,
        "agent_nome": agent_info.get("nome", ""),
        "agent_cognome": agent_info.get("cognome", ""),
        "current_datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    t0 = datetime.now()
    result = beverage_agent.invoke(inputs, config=config)
    duration_ms = int((datetime.now() - t0).total_seconds() * 1000)

    answer = result.get("final_answer", "Non ho trovato informazioni specifiche.")
    history_text = answer.text if hasattr(answer, "text") else str(answer)
    new_history = history + [HumanMessage(content=user_text), AIMessage(content=history_text)]

    beverage_agent.update_state(config, {"chat_history": new_history, "final_answer": None})

    _log_turn(sender_id, agent_code, user_text, result, answer, duration_ms)

    return answer


def process_and_respond(sender_id: str, user_text: str):
    print("\n" + "="*40)
    print(f"📩 NUOVO MESSAGGIO DA: {sender_id}")
    print(f"💬 TESTO: {user_text}")
    print("="*40)

    answer = run_graph(sender_id, user_text)
    send_whatsapp_message(sender_id, answer)


# =============================================================================
# WHATSAPP SENDER
# =============================================================================
def send_whatsapp_message(to: str, content):
    """Funzione universale Meta Cloud API."""
    url = f"https://graph.facebook.com/v22.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json"
    }

    if hasattr(content, 'use_interactive_list') and content.use_interactive_list:
        # Sezioni per città (clienti) oppure lista piatta (prodotti)
        raw_sections = getattr(content, 'sections', [])
        if raw_sections:
            # Multi-sezione: clienti raggruppati per città
            wa_sections = []
            for sec in raw_sections[:10]:
                rows = [
                    {
                        "id": item.id,
                        "title": item.title[:24],
                        "description": (item.description[:72] if item.description else "")
                    }
                    for item in sec.items[:10]
                ]
                if rows:
                    wa_sections.append({"title": sec.title[:24], "rows": rows})
            header_text = "Seleziona Cliente"
        else:
            # Lista piatta: prodotti
            rows = [
                {
                    "id": item.id,
                    "title": item.title[:24],
                    "description": (item.description[:72] if item.description else "")
                }
                for item in content.items[:10]
            ]
            wa_sections = [{"title": "Risultati Ricerca", "rows": rows}]
            header_text = "Selezione Prodotti"

        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "interactive",
            "interactive": {
                "type": "list",
                "header": {"type": "text", "text": header_text},
                "body": {"text": content.text},
                "footer": {"text": "Tocca il bottone per scegliere"},
                "action": {
                    "button": content.list_button_text[:20],
                    "sections": wa_sections
                }
            }
        }
    else:
        text_body = content.text if hasattr(content, 'text') else str(content)
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


# =============================================================================
# WEBHOOK WHATSAPP
# =============================================================================
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


# =============================================================================
# ENDPOINT DI TEST
# =============================================================================
@app.post("/test/chat")
async def test_chat(request: Request):
    body = await request.json()
    text = body.get("text", "").strip()
    if not text:
        return {"error": "Campo 'text' obbligatorio"}

    default_sender = next(iter(AGENT_MAPPING))
    sender_id = body.get("sender_id", default_sender)

    answer = run_graph(sender_id, text)

    response = {"text": answer.text if hasattr(answer, "text") else str(answer)}
    if getattr(answer, "use_interactive_list", False):
        raw_sections = getattr(answer, "sections", [])
        if raw_sections:
            response["sections"] = [
                {
                    "title": sec.title,
                    "items": [
                        {"id": item.id, "title": item.title, "description": getattr(item, "description", "")}
                        for item in sec.items
                    ],
                }
                for sec in raw_sections
            ]
        else:
            response["list"] = [
                {"id": item.id, "title": item.title, "description": getattr(item, "description", "")}
                for item in answer.items
            ]
    return response


@app.delete("/test/chat")
async def test_reset(request: Request):
    body = await request.json()
    default_sender = next(iter(AGENT_MAPPING))
    sender_id = body.get("sender_id", default_sender)
    config = {"configurable": {"thread_id": sender_id}}
    beverage_agent.update_state(config, {"chat_history": [], "pending_call": None, "final_answer": None})
    return {"reset": True, "sender_id": sender_id}


# =============================================================================
# ENDPOINT ADMIN LOG
# =============================================================================
@app.get("/admin/logs")
async def get_logs(
    agent: str = None,
    sender: str = None,
    date: str = None,
    limit: int = 50,
    offset: int = 0,
):
    """
    Legge i log di conversazione dal DB.

    Query params:
      agent=AG001          filtra per codice agente
      sender=393755116724  filtra per numero di telefono
      date=2026-02-20      filtra per data (prefisso su ts)
      limit=50             numero massimo di righe (max 200)
      offset=0             paginazione
    """
    limit = min(limit, 200)

    conditions = []
    params = []
    if agent:
        conditions.append("agent_code = ?")
        params.append(agent)
    if sender:
        conditions.append("sender_id = ?")
        params.append(sender)
    if date:
        conditions.append("ts LIKE ?")
        params.append(f"{date}%")

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    params += [limit, offset]

    try:
        conn = sqlite3.connect(_LOG_DB)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"""
            SELECT id, ts, sender_id, agent_code, user_msg,
                   plan_json, obs_json, response, duration_ms
            FROM conversation_log
            {where}
            ORDER BY id DESC
            LIMIT ? OFFSET ?
            """,
            params,
        ).fetchall()
        conn.close()
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    result = []
    for r in rows:
        entry = dict(r)
        # Deserializza JSON inline per renderli navigabili
        for field in ("plan_json", "obs_json"):
            try:
                entry[field] = json.loads(entry[field]) if entry[field] else None
            except Exception:
                pass
        result.append(entry)

    return {"total": len(result), "offset": offset, "logs": result}


@app.delete("/admin/logs")
async def clear_logs(agent: str = None):
    """Svuota i log. Se agent= specificato, solo quell'agente."""
    try:
        conn = sqlite3.connect(_LOG_DB)
        if agent:
            conn.execute("DELETE FROM conversation_log WHERE agent_code = ?", (agent,))
        else:
            conn.execute("DELETE FROM conversation_log")
        conn.commit()
        conn.close()
        return {"cleared": True, "agent": agent or "all"}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# =============================================================================
# ENDPOINT RESET COMPLETO
# =============================================================================
@app.post("/admin/reset")
async def admin_reset():
    """
    Reset completo del sistema:
    - Cancella e ricrea tutte le tabelle di database_ordini.db (ricarica clienti da CSV)
    - Svuota i checkpoint LangGraph (checkpoints.db)

    Dopo il reset ogni conversazione riparte da zero.
    """
    import csv as _csv

    # ── 1. Reset database_ordini.db ──────────────────────────────────────────
    try:
        conn = sqlite3.connect(_LOG_DB)
        cur = conn.cursor()
        cur.execute("PRAGMA foreign_keys = OFF")

        # Drop tutte le tabelle esistenti
        for tbl in ["cart_item", "order_item", "conversation_log"]:
            cur.execute(f"DROP TABLE IF EXISTS {tbl}")
        cur.execute("DROP TABLE IF EXISTS [order]")
        cur.execute("DROP TABLE IF EXISTS clienti_fts")

        # Ricrea clienti_fts (FTS5)
        cur.execute("""
            CREATE VIRTUAL TABLE clienti_fts USING fts5(
                client_id  UNINDEXED,
                ragione_sociale,
                alias,
                agent_id   UNINDEXED,
                indirizzo,
                citta,
                tokenize='unicode61'
            )
        """)

        # Ricrea tabelle transazionali
        cur.execute("""
            CREATE TABLE [order] (
                order_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                client_id    TEXT    NOT NULL,
                agent_id     TEXT    NOT NULL,
                status       TEXT    DEFAULT 'RECEIVED',
                total_amount REAL    DEFAULT 0.0,
                created_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
                note         TEXT
            )
        """)
        cur.execute("CREATE INDEX idx_order_created_at ON [order](created_at)")
        cur.execute("""
            CREATE TABLE order_item (
                item_id       INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id      INTEGER NOT NULL,
                sku           TEXT    NOT NULL,
                description   TEXT,
                quantity      INTEGER NOT NULL,
                price_at_order REAL,
                FOREIGN KEY(order_id) REFERENCES [order](order_id) ON DELETE CASCADE
            )
        """)
        cur.execute("""
            CREATE TABLE cart_item (
                cart_item_id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id     TEXT NOT NULL,
                client_id    TEXT NOT NULL,
                sku          TEXT NOT NULL,
                description  TEXT,
                quantity     INTEGER DEFAULT 0,
                price        REAL,
                added_at     DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(agent_id, client_id, sku)
            )
        """)
        cur.execute("""
            CREATE TABLE conversation_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          TEXT    NOT NULL,
                sender_id   TEXT    NOT NULL,
                agent_code  TEXT    NOT NULL,
                user_msg    TEXT,
                plan_json   TEXT,
                obs_json    TEXT,
                response    TEXT,
                duration_ms INTEGER
            )
        """)

        # Ricarica clienti da CSV
        csv_path = os.path.join(_project_root, "data", "clienti.csv")
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            clienti = [
                (row["client_id"], row["ragione_sociale"], row["alias"],
                 row["agent_id"], row["indirizzo"], row["citta"])
                for row in reader
            ]
        cur.executemany("INSERT INTO clienti_fts VALUES (?, ?, ?, ?, ?, ?)", clienti)
        cur.execute("PRAGMA foreign_keys = ON")
        conn.commit()
        conn.close()
        n_clienti = len(clienti)
    except Exception as e:
        return JSONResponse({"error": f"Errore reset DB ordini: {e}"}, status_code=500)

    # ── 2. Svuota checkpoints LangGraph ──────────────────────────────────────
    _checkpoint_db = os.path.join(_project_root, "checkpoints.db")
    try:
        cp_conn = sqlite3.connect(_checkpoint_db)
        for tbl in ["checkpoint_writes", "checkpoint_blobs", "checkpoints"]:
            try:
                cp_conn.execute(f"DELETE FROM {tbl}")
            except sqlite3.OperationalError:
                pass  # tabella non ancora creata, ignorabile
        cp_conn.commit()
        cp_conn.close()
    except Exception as e:
        return JSONResponse({"error": f"Errore reset checkpoint: {e}"}, status_code=500)

    print("🔄 RESET COMPLETO eseguito — DB ordini ricreato, checkpoint svuotati.")
    return {
        "reset": True,
        "clienti_caricati": n_clienti,
        "db_ordini": "ricreato",
        "checkpoints": "svuotati",
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9999)
