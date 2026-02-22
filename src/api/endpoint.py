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

    # Se la risposta ha una lista interattiva, appende items numerati (con ID/SKU) al testo
    # della chat_history così il dispatcher può risolvere "la 1 ed il 4" al turno successivo
    # senza ri-fare RAG. Il contatore è globale attraverso tutte le sezioni.
    _sections   = getattr(answer, "sections", None) or []
    _loose      = getattr(answer, "items", None) or []
    if not _sections and _loose:
        from src.graph.shared import WhatsAppSection
        _sections = [WhatsAppSection(title="", items=_loose)]
    if _sections and any(s.items for s in _sections):
        _lines, _n = [], 1
        for _sec in _sections:
            for _item in (_sec.items or []):
                _id   = f" [{_item.id}]" if _item.id else ""
                _desc = f" — {_item.description}" if _item.description else ""
                _lines.append(f"  {_n}. {_item.title}{_id}{_desc}")
                _n += 1
        if _lines:
            history_text = history_text + "\n" + "\n".join(_lines)

    new_history = history + [HumanMessage(content=user_text), AIMessage(content=history_text)]

    beverage_agent.update_state(config, {"chat_history": new_history, "final_answer": None})

    _log_turn(sender_id, agent_code, user_text, result, answer, duration_ms)

    return answer


def process_and_respond(sender_id: str, user_text: str):
    print("\n" + "="*40)
    print(f"📩 NUOVO MESSAGGIO DA: {sender_id}")
    print(f"💬 TESTO: {user_text}")
    print("="*40)

    agent_code = AGENT_MAPPING.get(sender_id, "AG001")
    answer = run_graph(sender_id, user_text)
    send_whatsapp_message(sender_id, answer, agent_code=agent_code)


# =============================================================================
# OVERFLOW EMAIL (risultati troppo lunghi per WhatsApp)
# =============================================================================
def send_overflow_email(agent_code: str, subject: str, full_text: str, raw_data: list = None):
    """Invia il risultato completo via email con eventuale allegato Excel."""
    from src.graph.shared import agent_email_map
    gmail_from     = os.getenv("GMAIL_FROM", "")
    gmail_password = os.getenv("GMAIL_APP_PASSWORD", "")
    if not gmail_from or not gmail_password:
        print("⚠️ [OVERFLOW EMAIL] Credenziali Gmail non configurate — email non inviata")
        return
    to_addr = agent_email_map.get(agent_code, gmail_from)

    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.base import MIMEBase
    from email import encoders
    import smtplib
    import io

    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"]    = gmail_from
    msg["To"]      = to_addr
    msg.attach(MIMEText(full_text, "plain", "utf-8"))

    if raw_data:
        try:
            import openpyxl
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.append(list(raw_data[0].keys()))
            for row in raw_data:
                ws.append(list(row.values()))
            buf = io.BytesIO()
            wb.save(buf)
            buf.seek(0)
            part = MIMEBase("application", "vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            part.set_payload(buf.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", 'attachment; filename="risultato.xlsx"')
            msg.attach(part)
        except ImportError:
            import csv as _csv
            buf = io.StringIO()
            writer = _csv.DictWriter(buf, fieldnames=list(raw_data[0].keys()))
            writer.writeheader()
            writer.writerows(raw_data)
            csv_part = MIMEText(buf.getvalue(), "plain", "utf-8")
            csv_part.add_header("Content-Disposition", 'attachment; filename="risultato.csv"')
            msg.attach(csv_part)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(gmail_from, gmail_password)
            s.send_message(msg)
        print(f"📧 [OVERFLOW EMAIL] Inviata a {to_addr}")
    except Exception as e:
        print(f"⚠️ [OVERFLOW EMAIL ERROR]: {e}")


# =============================================================================
# WHATSAPP SENDER
# =============================================================================
_WA_BODY_INTERACTIVE = 900   # max char body lista interattiva prima di overflow email
_WA_BODY_TEXT        = 3500  # max char testo semplice prima di overflow email
_WA_MAX_ROWS         = 10    # max righe totali lista interattiva


def send_whatsapp_message(to: str, content, agent_code: str = ""):
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
            # Multi-sezione: clienti raggruppati per città — cap a _WA_MAX_ROWS righe totali
            all_items_count = sum(len(sec.items) for sec in raw_sections)
            wa_sections = []
            total_rows = 0
            for sec in raw_sections[:10]:
                if total_rows >= _WA_MAX_ROWS:
                    break
                rows = []
                for item in sec.items:
                    if total_rows >= _WA_MAX_ROWS:
                        break
                    rows.append({
                        "id": item.id,
                        "title": item.title[:24],
                        "description": (item.description[:72] if item.description else "")
                    })
                    total_rows += 1
                if rows:
                    wa_sections.append({"title": sec.title[:24], "rows": rows})
            header_text = "Seleziona Cliente"

            # Overflow per righe troncate: invia email con lista completa
            if all_items_count > _WA_MAX_ROWS:
                print(f"   📋 [OVERFLOW ROWS] {all_items_count} clienti > max {_WA_MAX_ROWS} — invio email a agent_code={agent_code!r}")
                send_overflow_email(
                    agent_code,
                    f"Lista clienti completa ({all_items_count} totali)",
                    "\n".join(
                        f"{r.get('alias') or r.get('ragione_sociale') or r.get('client_id','')} "
                        f"— {r.get('citta','')} ({r.get('client_id','')})"
                        for r in (getattr(content, "raw_data", None) or [])
                    ),
                    getattr(content, "raw_data", None),
                )
        else:
            # Lista piatta: prodotti — max _WA_MAX_ROWS
            rows = [
                {
                    "id": item.id,
                    "title": item.title[:24],
                    "description": (item.description[:72] if item.description else "")
                }
                for item in content.items[:_WA_MAX_ROWS]
            ]
            wa_sections = [{"title": "Risultati Ricerca", "rows": rows}]
            header_text = "Selezione Prodotti"
            all_items_count = len(content.items)

        # Overflow: body troppo lungo per WhatsApp
        body_text = content.text or " "
        rows_truncated = raw_sections and all_items_count > _WA_MAX_ROWS
        if rows_truncated:
            body_text = body_text.rstrip() + f"\n⚠️ Mostrati {_WA_MAX_ROWS}/{all_items_count} — lista completa inviata via email."
        elif len(body_text) > _WA_BODY_INTERACTIVE:
            full_text = body_text
            body_text = body_text[:_WA_BODY_INTERACTIVE - 40] + "…\n📧 Risultato completo inviato via email."
            send_overflow_email(
                agent_code, "Risultato completo", full_text,
                getattr(content, "raw_data", None)
            )
        if not body_text.strip():
            body_text = " "

        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "interactive",
            "interactive": {
                "type": "list",
                "header": {"type": "text", "text": header_text},
                "body": {"text": body_text},
                "footer": {"text": "Tocca il bottone per scegliere"},
                "action": {
                    "button": content.list_button_text[:20],
                    "sections": wa_sections
                }
            }
        }
    else:
        text_body = content.text if hasattr(content, 'text') else str(content)
        if len(text_body) > _WA_BODY_TEXT:
            full_text = text_body
            text_body = text_body[:_WA_BODY_TEXT - 40] + "…\n📧 Risultato completo inviato via email."
            send_overflow_email(
                agent_code, "Risultato completo", full_text,
                getattr(content, "raw_data", None)
            )
        elif len(text_body) > 4000:
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


@app.delete("/test/reset_chat")
async def test_reset_chat(request: Request):
    """Resetta chat history e pending_call per un sender_id."""
    body = await request.json()
    default_sender = next(iter(AGENT_MAPPING))
    sender_id = body.get("sender_id", default_sender)
    config = {"configurable": {"thread_id": sender_id}}
    beverage_agent.update_state(config, {"chat_history": [], "pending_call": None, "final_answer": None})
    print(f"🔄 RESET CHAT per {sender_id}")
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
@app.post("/admin/reset_db")
async def admin_reset_db():
    """
    Reset completo del sistema:
    - Droppa e ricrea tutte le tabelle di database_ordini.db via setup_full_database()
    - Svuota i checkpoint LangGraph (checkpoints.db)

    Dopo il reset ogni conversazione riparte da zero.
    """
    import importlib.util
    _script = os.path.join(_project_root, "sql_lite", "scripts",
                           "create_database_ordini._anagrafica_cli.py")
    spec = importlib.util.spec_from_file_location("_setup_db", _script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # ── 1. Ricrea database_ordini.db ─────────────────────────────────────────
    try:
        mod.setup_full_database()
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

    print("🔄 RESET DB COMPLETO — database ricreato, checkpoint svuotati.")
    return {
        "reset": True,
        "db_ordini": "ricreato",
        "checkpoints": "svuotati",
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9999)
