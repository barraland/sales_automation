import os
import csv
import json
import logging
import sqlite3
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional, Literal, Union, TypedDict
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic import TypeAdapter
from typing import Union
from pydantic import BaseModel, Field
from typing_extensions import Annotated
       
# Import Provider
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

# Import client e tool (Assicurati che i path siano corretti)
from src.tools.rag_tool import search_product_smart, get_catalog_facets
from src.tools.rag_tool_anagrafica_clienti import search_client_smart

load_dotenv()
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("graph_app")

# =============================================================================
# FACTORY PER I MODELLI LLM
# =============================================================================
def get_model(role: Literal["planner", "generic"], structured_schema: Any = None, method: str = "structured_output"):
    provider = os.getenv(f"{role.upper()}_PROVIDER", "google").lower()

    if provider == "google":
        model_name = os.getenv(f"GEMINI_MODEL_{role.upper()}")
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0,
            convert_system_message_to_human=True
        )
    else:
        model_name = os.getenv(f"OPENAI_MODEL_{role.upper()}")
        llm = ChatOpenAI(
            model=model_name,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0
        )

    if structured_schema:
        if method == "function_calling":
            return llm.with_structured_output(structured_schema, method="function_calling")
        return llm.with_structured_output(structured_schema)

    return llm


# =============================================================================
# SINCRONIZZAZIONE VALORI DISTINTI DI CATEGORIA E BRAND DA ANAGRAFICA PRODOTTO
# =============================================================================
facets = get_catalog_facets()
brand_values = facets.get("brand", [])
categoria_values = facets.get("categoria", [])

# =============================================================================
# ANAGRAFICA AGENTI (caricata da data/agenti.csv)
# =============================================================================
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_agenti_csv = os.path.join(_project_root, "data", "agenti.csv")
agent_email_map: Dict[str, str] = {}          # codice → email
agent_name_map: Dict[str, Dict[str, str]] = {} # codice → {nome, cognome}
if os.path.exists(_agenti_csv):
    with open(_agenti_csv, newline="", encoding="utf-8") as _f:
        for _row in csv.DictReader(_f):
            codice = _row["codice"]
            agent_email_map[codice] = _row["email"]
            agent_name_map[codice] = {
                "nome":    _row.get("nome", ""),
                "cognome": _row.get("cognome", ""),
            }
    print(f"✅ Agenti caricati: {list(agent_email_map.keys())}")
else:
    print(f"⚠️ File agenti non trovato: {_agenti_csv}")

# =============================================================================
# SCHEMI E STATO
# =============================================================================
from typing import List, Union, Optional, Literal
from pydantic import BaseModel, Field

# =============================================================================
# ARGOMENTI TIPIZZATI PER I TOOL
# =============================================================================
class SearchClientArgs(BaseModel):
    placeholder: str                         # es. "PLACEHOLDER_CLIENT_1"
    query_text: Optional[str] = None         # testo originale da risolvere

class CartArgs(BaseModel):
    action: Literal["add", "remove", "view", "clear"]
    client_id: str                            # può essere un ID reale o un placeholder
    sku: Optional[str] = None                # può essere SKU reale o placeholder
    quantity: Optional[int] = None           # DEVE essere esplicitata dall'utente per "add"
    price: Optional[float] = None            # prezzo unitario, popolato dall'executor
    description: Optional[str] = None        # brand + formato, popolato dall'executor

class SearchProductArgs(BaseModel):
    placeholder: str
    query: str
    filters_json: str = ""
    top_k: int = 10
    info_only: bool = False  # True per domande informative sul catalogo (non aggiunta al carrello)

class OrderArgs(BaseModel):
    action: Literal["insert_order", "list_orders"]
    client_id: Optional[str] = None      # obbligatorio per insert_order
    date_from: Optional[str] = None      # ISO datetime: "YYYY-MM-DD HH:MM:SS"
    date_to: Optional[str] = None        # ISO datetime: "YYYY-MM-DD HH:MM:SS"
    status_filter: Optional[str] = None  # es. "RECEIVED"

class CatalogArgs(BaseModel):
    action: Literal["list_brands", "list_categories", "list_brands_by_category"]
    categoria: Optional[str] = None      # obbligatorio per list_brands_by_category

class ClarifyArgs(BaseModel):
    intent: Literal["explain_capabilities", "out_of_scope"]
    detail: Optional[str] = None  # cosa ha chiesto l'utente che è fuori scope

# =============================================================================
# TASK E PIANO
# =============================================================================
class Task(BaseModel):
    id: str
    tool: Literal[
        "search_client",
        "search_product",
        "manage_cart",
        "manage_orders",
        "manage_catalog",
        "clarify",
    ]
    args: Union[SearchClientArgs, CartArgs, SearchProductArgs, OrderArgs, CatalogArgs, ClarifyArgs]
    deps: List[str] = Field(default_factory=list)  # ID task da completare prima
    status: Literal["pending", "success", "failed"] = "pending"

class Plan(BaseModel):
    tasks: List[Task] = Field(default_factory=list)
    final_answer: Optional[str] = None

class WhatsAppListItem(BaseModel):
    id: str
    title: str
    description: Optional[str]

class FinalResponse(BaseModel):
    text: str
    use_interactive_list: bool = False
    list_button_text: str = "Vedi opzioni"
    items: List[WhatsAppListItem] = []

class AgentState(BaseModel):
    agent_code: str
    agent_nome: str = ""
    agent_cognome: str = ""
    known_clients: Dict[str, Any] = {}
    known_products: Dict[str, Any] = {}
    cart: Dict[str, Any] = {}
    chat_history: List[Any] = []
    question: Optional[str] = None
    is_finished: bool = False
    next_tasks: List[Any] = Field(default_factory=list)
    observations: Dict[str, Any] = {}
    iteration: int = 0
    final_answer: Optional[Any] = None
    current_datetime: Optional[str] = None  # popolato dall'endpoint ad ogni messaggio

class ExecutionContext(BaseModel):
    """
    Contiene tutti i dati temporanei della conversazione o del planner,
    NON persistenti. Deve vivere solo durante la richiesta.
    """

    # 🔹 Piano corrente generato dall'LLM
    plan: Optional[Any] = None

    # 🔹 Osservazioni runtime dei task, risultati parziali ecc.
    observations: Dict[str, Any] = {}

    # 🔹 Placeholder map temporanea
    placeholder_map_clients: Dict[str, str] = {}    # placeholder -> client_id
    placeholder_map_products: Dict[str, str] = {}   # placeholder -> sku

    # 🔹 Eventuali risultati temporanei dei tool
    tool_results: Dict[str, Any] = {}

    # 🔹 Lista di task correnti in esecuzione
    active_tasks: List[str] = []

    # 🔹 Log runtime specifico per debugging
    runtime_logs: List[str] = []

    class Config:
        # Evita serializzazione nei checkpoint
        underscore_attrs_are_private = True
        arbitrary_types_allowed = True

# =============================================================================
# EMAIL CONFERMA ORDINE
# =============================================================================
def send_order_email(order_result: dict, known_clients: dict, agent_code: str = "") -> bool:
    """
    Invia una email di conferma ordine via Gmail SMTP (App Password).
    Il mittente è GMAIL_FROM (.env); il destinatario è l'email dell'agente
    letta da data/agenti.csv tramite agent_email_map.
    """
    gmail_from = os.getenv("GMAIL_FROM", "")
    gmail_password = os.getenv("GMAIL_APP_PASSWORD", "")

    if not gmail_from or not gmail_password:
        print("⚠️ GMAIL_FROM o GMAIL_APP_PASSWORD non configurati — email non inviata")
        return False

    agent_to = agent_email_map.get(agent_code, gmail_from)

    order_id = order_result.get("order_id")
    client_id = order_result.get("client_id", "")
    total = order_result.get("total", 0.0)
    items = order_result.get("items", [])

    client_info = known_clients.get(client_id, {})
    client_name = client_info.get("ragione_sociale") or client_info.get("alias") or client_id

    items_lines = "\n".join(
        "  • {desc} x{qty} — €{tot:.2f}".format(
            desc=r.get("description") or r.get("sku", ""),
            qty=r.get("quantity", ""),
            tot=(r.get("price") or 0.0) * (r.get("quantity") or 0),
        )
        for r in items
    )

    body = (
        f"Nuovo ordine confermato dal sistema Sales Bot.\n\n"
        f"Ordine: #{order_id}\n"
        f"Cliente: {client_name} ({client_id})\n"
        f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Stato: RECEIVED\n\n"
        f"Prodotti:\n{items_lines if items_lines else '  (nessun dettaglio)'}\n\n"
        f"Totale: €{total:.2f}\n"
        f"---\nSales Automation Bot\n"
    )

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = f"Ordine #{order_id} confermato — {client_name}"
    msg["From"] = gmail_from
    msg["To"] = agent_to

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(gmail_from, gmail_password)
            smtp.sendmail(gmail_from, [agent_to], msg.as_string())
        print(f"✅ Email ordine #{order_id} inviata a {agent_to} (agente {agent_code})")
        return True
    except Exception as e:
        print(f"❌ Errore invio email: {e}")
        return False


# =============================================================================
# GESTIONE CARRELLO
# =============================================================================
def sql_manage_cart(agent_code: str, args: CartArgs):
    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
    db_path = os.path.join(project_root, "sql_lite", "db", "database_ordini.db")

    if not os.path.exists(db_path):
        return f"Errore: Il file database non esiste in {db_path}"

    conn = sqlite3.connect(db_path)
    conn.set_trace_callback(print)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        cursor.execute("PRAGMA foreign_keys = ON;")

        if args.action == "add":
            print(f"📝 [DB ACCESS] Agent {agent_code} is INSERTING for Client {args.client_id}")
            cursor.execute("""
                INSERT INTO cart_item (agent_id, client_id, sku, description, quantity, price)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(agent_id, client_id, sku)
                DO UPDATE SET
                    quantity = quantity + excluded.quantity,
                    description = COALESCE(excluded.description, cart_item.description),
                    price = COALESCE(excluded.price, cart_item.price)
            """, (agent_code, args.client_id, args.sku, args.description, args.quantity or 1, args.price))

            res = f"Aggiunto {args.quantity or 1}x SKU {args.sku} al carrello per cliente {args.client_id}."

        elif args.action == "remove":
            cursor.execute(
                "DELETE FROM cart_item WHERE agent_id = ? AND client_id = ? AND sku = ?",
                (agent_code, args.client_id, args.sku)
            )
            res = f"Rimosso SKU {args.sku} dal carrello per cliente {args.client_id}."

        elif args.action == "view":
            cursor.execute("""
                SELECT sku, description, quantity, price
                FROM cart_item
                WHERE agent_id = ? AND client_id = ?
            """, (agent_code, args.client_id))
            items = cursor.fetchall()
            res = [dict(row) for row in items] if items else "Il carrello è vuoto."

        elif args.action == "clear":
            cursor.execute(
                "DELETE FROM cart_item WHERE agent_id = ? AND client_id = ?",
                (agent_code, args.client_id)
            )
            res = "Carrello svuotato."

        else:
            res = f"Azione '{args.action}' non riconosciuta."

        conn.commit()
        return res

    except Exception as e:
        print(f"💥 [SQL ERROR]: {e}")
        return f"Errore DB: {e}"

    finally:
        conn.close()


def _get_db_path():
    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
    return os.path.join(project_root, "sql_lite", "db", "database_ordini.db")


def sql_insert_order(agent_code: str, args: "OrderArgs"):
    """
    Converte il carrello (cart_item) di un cliente in un ordine confermato.
    Inserisce la testata in [order] e le righe in order_item, poi svuota cart_item.
    Restituisce un dict con order_id e riepilogo.
    """
    db_path = _get_db_path()
    if not os.path.exists(db_path):
        return {"error": f"Database non trovato in {db_path}"}

    client_id = args.client_id
    if not client_id:
        return {"error": "client_id obbligatorio per insert_order"}

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        cursor.execute("PRAGMA foreign_keys = ON;")

        # Leggi il carrello
        cursor.execute("""
            SELECT sku, description, quantity, price
            FROM cart_item
            WHERE agent_id = ? AND client_id = ?
        """, (agent_code, client_id))
        items = cursor.fetchall()

        if not items:
            return {"error": "Carrello vuoto — nessun ordine creato"}

        items = [dict(r) for r in items]
        total = sum(
            (r["quantity"] or 0) * (r["price"] or 0.0)
            for r in items
        )

        # Inserisci testata ordine
        cursor.execute("""
            INSERT INTO [order] (client_id, agent_id, status, total_amount)
            VALUES (?, ?, 'RECEIVED', ?)
        """, (client_id, agent_code, round(total, 2)))
        order_id = cursor.lastrowid

        # Inserisci righe ordine
        for r in items:
            cursor.execute("""
                INSERT INTO order_item (order_id, sku, description, quantity, price_at_order)
                VALUES (?, ?, ?, ?, ?)
            """, (order_id, r["sku"], r["description"], r["quantity"], r["price"]))

        # Svuota carrello
        cursor.execute(
            "DELETE FROM cart_item WHERE agent_id = ? AND client_id = ?",
            (agent_code, client_id)
        )

        conn.commit()
        print(f"✅ Ordine #{order_id} creato per cliente {client_id} | Totale €{total:.2f}")
        return {
            "order_id": order_id,
            "client_id": client_id,
            "total": round(total, 2),
            "items_count": len(items),
            "items": items,
            "status": "RECEIVED",
        }

    except Exception as e:
        conn.rollback()
        print(f"💥 [SQL ERROR insert_order]: {e}")
        return {"error": str(e)}
    finally:
        conn.close()


def sql_list_orders(agent_code: str, args: "OrderArgs"):
    """
    Restituisce gli ordini dell'agente con filtri opzionali su cliente,
    stato e intervallo di date.
    """
    db_path = _get_db_path()
    if not os.path.exists(db_path):
        return {"error": f"Database non trovato in {db_path}"}

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        query = """
            SELECT o.order_id, o.client_id, o.status, o.total_amount, o.created_at
            FROM [order] o
            WHERE o.agent_id = ?
        """
        params = [agent_code]

        if args.client_id:
            query += " AND o.client_id = ?"
            params.append(args.client_id)
        if args.status_filter:
            query += " AND o.status = ?"
            params.append(args.status_filter)
        if args.date_from:
            query += " AND o.created_at >= ?"
            params.append(args.date_from)
        if args.date_to:
            query += " AND o.created_at <= ?"
            params.append(args.date_to)

        query += " ORDER BY o.created_at DESC LIMIT 20"

        cursor.execute(query, params)
        orders = [dict(r) for r in cursor.fetchall()]

        # Per ogni ordine carica le righe
        for order in orders:
            cursor.execute("""
                SELECT sku, description, quantity, price_at_order
                FROM order_item WHERE order_id = ?
            """, (order["order_id"],))
            order["items"] = [dict(r) for r in cursor.fetchall()]

        return orders if orders else []

    except Exception as e:
        print(f"💥 [SQL ERROR list_orders]: {e}")
        return {"error": str(e)}
    finally:
        conn.close()


# =============================================================================
# PLANNER NODE
# =============================================================================
def planner_node(state: AgentState):
    #llm = get_model("planner", structured_schema=Plan)  # rimosso 'method' se get_model non lo supporta
    llm = get_model("planner", structured_schema=Plan, method="function_calling")

    agent_code = getattr(state, "agent_code", "AG001")
    known_clients = getattr(state, "known_clients", {})
    known_products = getattr(state, "known_products", {})
    current_datetime = getattr(state, "current_datetime", "N/D")
    is_first_message = len(getattr(state, "chat_history", [])) == 0

    system_prompt = f"""
Sei un assistente virtuale per agenti commerciali che servono e riforniscono clienti per il settore Horeca.
Il tuo compito è generare UN PIANO COMPLETO E DEFINITIVO per soddisfare la richiesta dell'agente umano {agent_code}.

----------------------------------------------------------------------
📌 REGOLA PRIORITARIA — LEGGI PRIMA DI TUTTO IL RESTO
----------------------------------------------------------------------

Prima di pianificare qualsiasi task, controlla sempre nell'ordine:

1. **ID espliciti nel messaggio**: Se il messaggio contiene `(ID: XXX)`, estrai XXX e verifica:
   - Se XXX è un client_id in "Lista clienti già risolti" → usalo direttamente in manage_cart, SENZA search_client.
   - Se XXX è uno sku in "Lista prodotti già risolti" → usalo direttamente in manage_cart, SENZA search_product.

2. **Clienti già noti**: Se il cliente menzionato corrisponde (nome, alias, ragione sociale, città) a un entry in "Lista clienti già risolti", usa quel client_id DIRETTAMENTE. NON creare search_client.

3. **Prodotti già noti**: Se il prodotto menzionato corrisponde a un entry in "Lista prodotti già risolti", usa quello sku DIRETTAMENTE. NON creare search_product.

4. **Operazione incompleta**: Se dalla chat history risulta un'operazione in corso rimasta in attesa (cliente o prodotto mancante), e il messaggio attuale fornisce l'informazione mancante, ricostruisci l'operazione completa con manage_cart usando i dati ora disponibili.

5. **Cliente implicito dalla conversazione**: Se il messaggio attuale NON menziona un cliente
   ma dalla chat history recente è chiaro su quale cliente si stava lavorando (es. si stava
   visualizzando il carrello, aggiungendo prodotti, o discutendo di un cliente specifico),
   usa quel client_id direttamente. NON pianificare search_client e NON chiedere il cliente
   all'utente. L'agente può lavorare su più clienti, ma se il contesto recente punta a uno
   specifico, quello è il cliente corretto.

----------------------------------------------------------------------
📌 STRUTTURA DEI TASK E TOOLS
----------------------------------------------------------------------

- Genera un piano strutturato in task, dove ogni task ha: ID univoco, tool specifico, args corretti, deps verso task da completare prima, status iniziale "pending".
----------------------------------------------------------------------

1️⃣ **search_client** — args: SearchClientArgs(placeholder, query_text)
- Cerca il client_id di un cliente non ancora noto nel database.
- NON usare se il client_id è già in "Lista clienti già risolti".

2️⃣ **search_product** — args: SearchProductArgs(placeholder, query, filters_json, top_k, info_only)

  **Modalità RISOLUZIONE SKU** (info_only=False, default):
  - Cerca lo sku di un prodotto non ancora noto per poi aggiungerlo al carrello.
  - NON usare se lo sku è già in "Lista prodotti già risolti".
  - NON dipende mai da search_client — cliente e prodotto si cercano sempre in parallelo.
  - top_k: 10 per ricerche puntuali.

  **Modalità INFO CATALOGO** (info_only=True):
  - Usa quando l'utente chiede informazioni sui prodotti senza volerli ordinare:
    "che brand avete?", "avete la Heineken?", "quali birre sono disponibili?",
    "quant'è il prezzo della Peroni 33cl?", "mostratemi le acque a catalogo".
  - Questo task è STANDALONE: NON creare task manage_cart collegati.
    Il responder mostrerà direttamente i risultati all'utente.
  - placeholder: usa una stringa descrittiva es. "INFO_HEINEKEN", "INFO_BIRRE", "INFO_BRAND".
  - top_k: scegli in base all'ampiezza della richiesta:
      • 10  — prodotto specifico ("avete la Peroni 33cl?")
      • 20  — tutti i prodotti di un brand ("prodotti Heineken")
      • 50  — query per categoria o sottocategoria ("tutte le birre", "le acque")
      • 100 — query molto ampie ("tutto il catalogo alcolici", "mostrami tutto")
    Nota: top_k è un massimo — se il catalogo contiene meno prodotti corrispondenti,
    ne vengono restituiti semplicemente meno.
  - Per domande su brand disponibili o categorie disponibili, usa stringa vuota come query
    e imposta il filtro appropriato.

  **Filtri disponibili** (filters_json = stringa JSON, stringa vuota = nessun filtro):
  - Per brand:     {{"brand": "Heineken"}}
  - Per categoria: {{"categoria": "Alcolici"}}
  - Combinati:     {{"brand": "Heineken", "categoria": "Alcolici"}}
  - ⚠️ Usa ESATTAMENTE i valori dalle liste seguenti (rispetta maiuscole, apostrofi, spazi):

  Brand ammessi:
  {json.dumps(brand_values, ensure_ascii=False)}

  Categorie ammesse:
  {json.dumps(categoria_values, ensure_ascii=False)}

3️⃣ **manage_cart** — args: CartArgs(action, client_id, sku, quantity)
- action: "add" | "remove" | "view" | "clear"
- Pianifica quando l'utente vuole aggiungere, rimuovere, visualizzare o svuotare il carrello.
  L'utente non usa necessariamente il termine "carrello" — interpreta l'intento.
- client_id: usa ID reale da "Lista clienti già risolti" o placeholder di search_client.
- sku: usa SKU reale da "Lista prodotti già risolti" o placeholder di search_product.
- quantity: per action "add", imposta il numero indicato dall'utente. Se l'utente NON ha
  specificato una quantità numerica esplicita, lascia quantity=null. Il sistema chiederà
  automaticamente. NON inventare quantità di default.
- Per "view" / "clear": usa il client_id del cliente menzionato più di recente in conversazione
  o in "Lista clienti già risolti". Se NON è determinabile, DEVI creare un task search_client
  con query_text vuoto (elenca tutti i clienti dell'agente) e fare in modo che manage_cart
  dipenda da esso. NON usare placeholder inventati senza un search_client corrispondente.
- deps: includi search_client se client_id mancante; includi search_product se sku mancante.
  Se entrambi già noti → deps vuoti.

4️⃣ **manage_orders** — args: OrderArgs(action, client_id, date_from, date_to, status_filter)
- action: "insert_order" | "list_orders"
- **insert_order**: conferma e invia l'ordine. Legge il carrello del cliente, crea l'ordine nel
  DB con un ID progressivo automatico (order_id), svuota il carrello.
  ⚠️ PIANIFICA insert_order SOLO se il messaggio dell'utente è una conferma esplicita dell'ordine:
  parole come "sì", "confermo", "invia", "procedi", "ok manda", "vai". NON pianificare
  insert_order quando l'utente sta fornendo una quantità, un nome di prodotto o un nome di
  cliente — in quei casi l'utente sta completando l'ordine, non confermandolo.
  NON combinare mai add (manage_cart) e insert_order nello stesso piano.
  Richiede client_id reale. Se non determinabile, crea search_client con deps.
- **list_orders**: elenca gli ordini dell'agente. Filtri opzionali:
  - client_id: per vedere ordini di un cliente specifico
  - date_from / date_to: intervallo ISO datetime "YYYY-MM-DD HH:MM:SS"
    Per "ordini di oggi": date_from = "{current_datetime[:10]} 00:00:00"
    Per "ultimi N minuti": calcola sottraendo N minuti da current_datetime
  - status_filter: "RECEIVED" (default), "SHIPPED", etc.
- Non dipende da manage_cart.

5️⃣ **manage_catalog** — args: CatalogArgs(action, categoria)
- Usa questo tool SOLO per domande esplicite sul catalogo: "quali brand hai?", "che categorie ci sono?",
  "quali brand di birra hai?", "dimmi i produttori di vino". NON usarlo per ricerche di prodotti specifici.
- action: "list_brands" | "list_categories" | "list_brands_by_category"
- **list_brands**: restituisce tutti i brand del catalogo. Nessun parametro aggiuntivo. Deps: vuoti.
- **list_categories**: restituisce tutte le categorie. Nessun parametro aggiuntivo. Deps: vuoti.
- **list_brands_by_category**: restituisce i brand che hanno prodotti nella categoria indicata.
  Richiede `categoria` (valore esatto da categoria_values). Deps: vuoti.
- Non ha dipendenze da altri task.

6️⃣ **clarify** — args: ClarifyArgs(intent, detail)
- Usa questo tool quando NON devi eseguire operazioni su catalogo/carrello/ordini.
- intent: "explain_capabilities" | "out_of_scope"
- **explain_capabilities**: l'utente chiede cosa sa fare l'assistente ("cosa puoi fare?",
  "come funzioni?", "a cosa servi?"), oppure mostra incertezza su cosa è possibile chiedere
  ("posso chiederti di...?", "sai anche...?"). Nessun `detail`. Deps: vuoti.
- **out_of_scope**: l'utente chiede qualcosa completamente estraneo al dominio commerciale
  bevande (es. meteo, notizie, codice, ricette, sport). Compila `detail` con una breve
  descrizione di cosa ha chiesto (es. "previsioni meteo"). Deps: vuoti.
- ⚠️ NON usare clarify se la richiesta è solo ambigua o incompleta: in quel caso il responder
  chiede chiarimenti da solo. Usalo SOLO per i due casi sopra.
- clarify è sempre standalone: non combinarlo con altri tool nello stesso piano.

----------------------------------------------------------------------
📌 REGOLE DI PIANIFICAZIONE
----------------------------------------------------------------------

- Crea task separati per ogni cliente o prodotto citato.
- Usa placeholder deterministici solo per dati non noti (es. `ph_cliente_gigio`, `ph_prodotto_ichnusa`).
- Dipendenze: search_client → manage_cart/manage_orders (solo se client_id mancante);
  search_product → manage_cart (solo se sku mancante). search_client e search_product NON si dipendono mai.
- Non creare search_client o search_product per dati già presenti nello stato.

----------------------------------------------------------------------
📌 STATO ATTUALE
----------------------------------------------------------------------

Data e ora corrente (usala per calcolare filtri temporali): {current_datetime}

{"⭐ PRIMO MESSAGGIO DELLA SESSIONE: la chat history è vuota." if is_first_message else ""}
{"Se il messaggio non è una richiesta operativa specifica (es. 'ciao', 'prova', 'salve', 'chi sei', domanda generica), pianifica clarify(intent='explain_capabilities'). Se invece è una richiesta specifica (aggiungere prodotti, vedere catalogo, ecc.), eseguila normalmente." if is_first_message else ""}

Lista clienti già risolti (usa direttamente, SENZA search_client):
{json.dumps(known_clients, indent=2, ensure_ascii=False)}

Lista prodotti già risolti (usa direttamente, SENZA search_product):
{json.dumps(known_products, indent=2, ensure_ascii=False)}

----------------------------------------------------------------------
Genera il piano rispettando la REGOLA PRIORITARIA: verifica sempre se
client_id e sku sono già noti prima di pianificare ricerche.
"""

    try:
        # Passa gli ultimi 4 messaggi di chat (2 turni) per dare contesto al planner
        messages = [SystemMessage(content=system_prompt)]
        if state.chat_history:
            messages.extend(state.chat_history[-4:])
        messages.append(HumanMessage(content=state.question or ""))

        plan_raw = llm.invoke(messages)

        #_rewrite_placeholders(plan_raw)

        # 🔹 Validazione output
        plan_adapter = TypeAdapter(Plan)
        plan = plan_adapter.validate_python(plan_raw)

        print(f"DEBUG PLANNER: tasks_count={len(plan.tasks)}")
        for t in plan.tasks:
            args_repr = {k: v for k, v in t.args.__dict__.items() if v is not None and v != "" and v != []}
            print(f"   ├─ [{t.id}] tool={t.tool} deps={t.deps}")
            print(f"   │   args={args_repr}")

        return {
        "next_tasks": plan.tasks,
        "iteration": getattr(state, "iteration", 0) + 1,
        "is_finished": True}

    except Exception as e:
        print(f"💥 ERRORE PLANNER: {e}")
        return {
    "next_tasks": [],
    "iteration": getattr(state, "iteration", 0) + 1,
    "is_finished": False}

# =============================================================================
# EXECUTOR NODE - VERSIONE PULITA E ALLINEATA AL NUOVO PLANNER
# =============================================================================

def executor_node(state: AgentState):
    print(f"\n{'#'*70}")
    print(f"📥 [INPUT EXECUTOR]")

    agent_code = getattr(state, "agent_code", "AG001")

    # --- Ricostruzione Task Pydantic ---
    task_adapter = TypeAdapter(Task)
    raw_tasks = getattr(state, "next_tasks", [])
    tasks = [
        task_adapter.validate_python(t) if isinstance(t, dict) else t
        for t in raw_tasks
    ]

    print(f" - Numero Task Ricevuti: {len(tasks)}")
    for t in tasks:
        print(f"   └─ Task: id={t.id} tool={t.tool}")

    obs = dict(getattr(state, "observations", {}))
    obs.setdefault("placeholder_map", {})
    placeholder_map = obs["placeholder_map"]

    # Copie mutabili delle mappe note, da restituire nello stato
    updated_known_clients = dict(getattr(state, "known_clients", {}))
    updated_known_products = dict(getattr(state, "known_products", {}))

    # Lookup preventivo: per ogni client_id reale nei task non ancora in known_clients,
    # carica nome/ragione_sociale dal DB così il responder può usarlo senza ambiguità.
    _missing_client_ids = {
        getattr(t.args, "client_id", None)
        for t in tasks
        if getattr(t.args, "client_id", None)
        and not str(getattr(t.args, "client_id", "")).startswith("ph_")
        and getattr(t.args, "client_id", None) not in updated_known_clients
    }
    if _missing_client_ids:
        try:
            _db = _get_db_path()
            _conn = sqlite3.connect(_db)
            _conn.row_factory = sqlite3.Row
            for _cid in _missing_client_ids:
                _row = _conn.execute(
                    "SELECT client_id, ragione_sociale, alias, citta FROM clienti_fts WHERE client_id=?",
                    (_cid,)
                ).fetchone()
                if _row:
                    updated_known_clients[_cid] = dict(_row)
                    print(f"   🔍 Client lookup: {_cid} → {_row['ragione_sociale']}")
            _conn.close()
        except Exception as _e:
            print(f"   ⚠️ Client lookup error: {_e}")

    updated_tasks = tasks.copy()

    # -------------------------------------------------------------------------
    # FUNZIONE CENTRALIZZATA DI REPLACE PLACEHOLDER
    # -------------------------------------------------------------------------
    def _replace_placeholders_in_all_tasks():
        for t in updated_tasks:
            if not hasattr(t, "args") or not hasattr(t.args, "__dict__"):
                continue
            for field, value in list(t.args.__dict__.items()):
                if isinstance(value, str) and value in placeholder_map:
                    setattr(t.args, field, placeholder_map[value])

    # -------------------------------------------------------------------------
    # FUNZIONE PER VERIFICARE SE LE DEPENDENCIES SONO RISOLTE
    # -------------------------------------------------------------------------
    def deps_satisfied(task):
        if not getattr(task, "deps", None):
            return True
        for dep_id in task.deps:
            dep_task = next((t for t in updated_tasks if t.id == dep_id), None)
            if not dep_task or dep_task.status != "success":
                return False
        return True

    # -------------------------------------------------------------------------
    # LOOP PRINCIPALE — singolo pass sui task nell'ordine del piano
    # -------------------------------------------------------------------------
    for task in updated_tasks:
        print(f"\n⚙️ ESECUZIONE: {task.tool} (ID: {task.id})")

        if task.status in ("success", "failed"):
            print(f" ℹ️ Task già processato (status={task.status})")
            continue

        if not deps_satisfied(task):
            print(" ⏳ Dipendenze non ancora soddisfatte, task saltato")
            continue

        try:
                # -----------------------------------------------------------------
                # SEARCH CLIENT
                # -----------------------------------------------------------------
                if task.tool == "search_client":
                    res = search_client_smart(agent_code=agent_code, query_text=task.args.query_text)

                    if not res:
                        task.status = "failed"
                        obs[task.id] = {"error": f"Nessun cliente trovato per '{task.args.query_text}'"}
                        print(f" ❌ Nessun cliente trovato: {task.args.query_text}")
                    else:
                        # Aggiorna known_clients con tutti i risultati trovati
                        for r in res:
                            cid = r.get("client_id")
                            if cid:
                                updated_known_clients[cid] = {
                                    "ragione_sociale": r.get("ragione_sociale"),
                                    "alias": r.get("alias"),
                                    "citta": r.get("citta"),
                                }

                        if len(res) == 1 and getattr(task.args, "placeholder", None):
                            # Risolto univocamente
                            resolved_id = res[0].get("client_id")
                            placeholder_map[task.args.placeholder] = resolved_id
                            print(f" 🔁 Placeholder risolto: {task.args.placeholder} → {resolved_id}")
                            _replace_placeholders_in_all_tasks()
                            task.status = "success"
                            obs[task.id] = {"results": res}
                        else:
                            # Più risultati — serve scelta utente
                            obs[task.id] = {"results": res, "pending_selection": True}
                            task.status = "pending"
                            print(f" ⚠️ Disambiguazione necessaria per '{task.args.query_text}' ({len(res)} risultati)")

                # -----------------------------------------------------------------
                # SEARCH PRODUCT
                # -----------------------------------------------------------------
                elif task.tool == "search_product":
                    # Se query è vuota ma c'è un filtro brand/categoria, usa il valore
                    # del filtro come query per evitare il fallback generico "bevande"
                    search_query = task.args.query
                    if not search_query and task.args.filters_json:
                        try:
                            filter_dict = json.loads(task.args.filters_json)
                            search_query = filter_dict.get("brand") or filter_dict.get("categoria") or ""
                        except Exception:
                            pass

                    raw = search_product_smart.invoke({
                        "query": search_query,
                        "planner_motivation": "",
                        "filters_json": task.args.filters_json,
                        "top_k": task.args.top_k,
                    })
                    # Il tool restituisce {"results": [...], "metadata": {...}}
                    results = raw.get("results", []) if isinstance(raw, dict) else []

                    # Log risultati RAG
                    if results:
                        print(f" 📦 RAG: {len(results)} prodotti trovati:")
                        for r in results[:8]:
                            price_str = f"€{r['prezzo']:.2f}" if r.get("prezzo") else "—"
                            stock_str = f"disp:{r.get('stock')}" if r.get("stock") is not None else ""
                            print(f"    • {r.get('sku')} | {r.get('brand')} {r.get('formato')} | {price_str} {stock_str}".rstrip())
                        if len(results) > 8:
                            print(f"    ... +{len(results)-8} altri")
                    else:
                        print(f" ⚠️ RAG: nessun risultato | query={search_query!r} filter={task.args.filters_json!r}")

                    if not results:
                        task.status = "failed"
                        obs[task.id] = {"error": f"Nessun prodotto trovato per '{search_query}'"}
                        print(f" ❌ Nessun prodotto trovato: {search_query}")
                    else:
                        # Aggiorna known_products con tutti i risultati trovati
                        for r in results:
                            sku = r.get("sku")
                            if sku:
                                updated_known_products[sku] = {
                                    "descrizione": r.get("descrizione"),
                                    "brand": r.get("brand"),
                                    "formato": r.get("formato"),
                                    "prezzo": r.get("prezzo"),
                                    "stock": r.get("stock"),
                                }

                        if getattr(task.args, "info_only", False):
                            # Query informativa sul catalogo — mostra tutti i risultati senza disambiguazione
                            obs[task.id] = {"results": results, "info_only": True}
                            task.status = "success"
                            print(f" ℹ️ Info catalogo: {len(results)} prodotti trovati")
                        elif len(results) == 1 and getattr(task.args, "placeholder", None):
                            # Risolto univocamente
                            resolved_sku = results[0].get("sku")
                            placeholder_map[task.args.placeholder] = resolved_sku
                            print(f" 🔁 Placeholder risolto: {task.args.placeholder} → {resolved_sku}")
                            _replace_placeholders_in_all_tasks()
                            task.status = "success"
                            obs[task.id] = {"results": results}
                        else:
                            # Più risultati — serve scelta utente
                            obs[task.id] = {"results": results, "pending_selection": True}
                            task.status = "pending"
                            print(f" ⚠️ Disambiguazione necessaria per '{task.args.query}' ({len(results)} risultati)")

                # -----------------------------------------------------------------
                # MANAGE CART
                # -----------------------------------------------------------------
                elif task.tool in ("manage_cart", "manage_orders"):
                    # Il planner a volte usa manage_cart anche per azioni ordine:
                    # se l'azione è insert_order o list_orders, ridirigiamo al gestore ordini.
                    if task.args.action in ("insert_order", "list_orders"):
                        task.tool = "manage_orders"  # normalizza il tool name
                        print(f" 📦 Azione ordine (redirect): {task.args.action} | client_id={task.args.client_id}")

                        if task.args.action == "insert_order":
                            if not task.args.client_id or str(task.args.client_id).startswith("ph_"):
                                obs[task.id] = {"error": "client_required", "action": "insert_order"}
                                task.status = "failed"
                            else:
                                res = sql_insert_order(agent_code=agent_code, args=task.args)
                                obs[task.id] = res
                                task.status = "success" if "order_id" in res else "failed"
                                if task.status == "success":
                                    send_order_email(res, updated_known_clients, agent_code=agent_code)

                        elif task.args.action == "list_orders":
                            res = sql_list_orders(agent_code=agent_code, args=task.args)
                            obs[task.id] = res
                            task.status = "success" if not isinstance(res, dict) or "error" not in res else "failed"

                        # Salta il resto del blocco manage_cart
                        continue  # noqa

                    print(
                        f" 🛒 Azione: {task.args.action} | "
                        f"client_id={task.args.client_id} | sku={getattr(task.args, 'sku', None)} | qty={getattr(task.args, 'quantity', None)}"
                    )

                    # Verifica che client_id sia stato risolto (non più un placeholder)
                    if str(task.args.client_id).startswith("ph_"):
                        obs[task.id] = {
                            "error": "client_required",
                            "action": task.args.action,
                        }
                        task.status = "failed"
                        print(" ⚠️ client_id non risolto — il responder chiederà il cliente")

                    # Verifica quantità obbligatoria per add
                    elif task.args.action == "add" and not task.args.quantity:
                        obs[task.id] = {
                            "error": "quantity_required",
                            "sku": task.args.sku,
                            "client_id": task.args.client_id,
                        }
                        task.status = "failed"
                        print(" ⚠️ Quantità mancante — il responder chiederà all'utente")
                    else:
                        # Arricchisci price e description da known_products (se add)
                        if task.args.action == "add" and task.args.sku:
                            product_info = updated_known_products.get(task.args.sku, {})
                            if product_info:
                                if task.args.price is None:
                                    task.args.price = product_info.get("prezzo")
                                if task.args.description is None:
                                    brand = product_info.get("brand") or ""
                                    formato = product_info.get("formato") or ""
                                    task.args.description = (
                                        f"{brand} {formato}".strip()
                                        or product_info.get("descrizione")
                                    )

                        res = sql_manage_cart(agent_code=agent_code, args=task.args)
                        obs[task.id] = res
                        if res and "Errore" not in str(res):
                            print(" ✅ Operazione carrello completata")
                            task.status = "success"
                        else:
                            print(" ❌ Errore operazione carrello")
                            task.status = "failed"

                # -----------------------------------------------------------------
                # MANAGE CATALOG
                # -----------------------------------------------------------------
                elif task.tool == "manage_catalog":
                    action = task.args.action
                    print(f" 📚 Catalog action: {action} | categoria={getattr(task.args, 'categoria', None)}")

                    if action == "list_brands":
                        obs[task.id] = {"brands": brand_values, "catalog_info": True}
                        task.status = "success"
                        print(f"   ✅ {len(brand_values)} brand restituiti dalla cache")

                    elif action == "list_categories":
                        obs[task.id] = {"categories": categoria_values, "catalog_info": True}
                        task.status = "success"
                        print(f"   ✅ {len(categoria_values)} categorie restituite dalla cache")

                    elif action == "list_brands_by_category":
                        cat = getattr(task.args, "categoria", None)
                        if not cat:
                            obs[task.id] = {"error": "categoria obbligatoria per list_brands_by_category"}
                            task.status = "failed"
                        else:
                            escaped_cat = cat.replace("'", "''")
                            odata = f"categoria eq '{escaped_cat}'"
                            facet_result = get_catalog_facets(
                                facet_fields=["brand"],
                                odata_filter=odata
                            )
                            brands_in_cat = facet_result.get("brand", [])
                            obs[task.id] = {
                                "brands": brands_in_cat,
                                "categoria": cat,
                                "catalog_info": True
                            }
                            task.status = "success"
                            print(f"   ✅ {len(brands_in_cat)} brand nella categoria '{cat}'")
                    else:
                        obs[task.id] = {"error": f"Azione catalog non riconosciuta: {action}"}
                        task.status = "failed"

                # -----------------------------------------------------------------
                # CLARIFY
                # -----------------------------------------------------------------
                elif task.tool == "clarify":
                    intent = task.args.intent
                    detail = getattr(task.args, "detail", None)
                    obs[task.id] = {"clarify": True, "intent": intent, "detail": detail}
                    task.status = "success"
                    print(f" 💬 Clarify: intent={intent}" + (f" | detail={detail}" if detail else ""))

                else:
                    print(f" ⚠️ TOOL NON RICONOSCIUTO: {task.tool}")
                    task.status = "failed"

        except Exception as e:
            print(f" 💥 ERRORE TASK: {e}")
            obs[task.id] = {"error": str(e)}
            task.status = "failed"

    # -------------------------------------------------------------------------
    # OUTPUT — trim cache FIFO (mantieni solo gli ultimi N inseriti)
    # -------------------------------------------------------------------------
    if len(updated_known_clients) > 20:
        updated_known_clients = dict(list(updated_known_clients.items())[-20:])
        print(f" ✂️  known_clients trimmato a 20")

    if len(updated_known_products) > 50:
        updated_known_products = dict(list(updated_known_products.items())[-50:])
        print(f" ✂️  known_products trimmato a 50")

    output_to_state = {
        "observations": obs,
        "next_tasks": updated_tasks,
        "known_clients": updated_known_clients,
        "known_products": updated_known_products,
    }

    print(f"\n{'#'*70}")
    print(f"📤 [OUTPUT EXECUTOR]")
    print(f" - Totale chiavi Observations: {len(obs)}")
    print(f" - Status Task: {[(t.id, t.status) for t in updated_tasks]}")
    print(f"{'#'*70}\n")

    return output_to_state



# =============================================================================
# RESPONDER NODE
# =============================================================================

def responder_node(state: AgentState):

    # 1️⃣ Check continuità:
    # Se il planner non ha finito e non esiste un task llm_answer,
    # significa che servono altri giri di tool.
    is_finished_planner = getattr(state, "is_finished", False)
    has_llm_answer = any(
        t.tool == "llm_answer"
        for t in getattr(state, "next_tasks", [])
    )

    if not is_finished_planner and not has_llm_answer:
        return {"is_finished": False}

    # 2️⃣ Inizializza LLM con schema strutturato
    llm = get_model("generic", structured_schema=FinalResponse)

    # 3️⃣ Analisi osservazioni reali
    obs = state.observations
    known_clients = getattr(state, "known_clients", {})
    known_products = getattr(state, "known_products", {})
    cart_tasks = [t for t in getattr(state, "next_tasks", []) if t.tool == "manage_cart"]

    # Verifica INSERT riuscita / rimozione
    was_added = any("Aggiunto" in str(v) for v in obs.values())
    was_removed = any("Rimosso" in str(v) for v in obs.values())

    # Tipo di operazione carrello
    is_cart_view = any(getattr(t.args, "action", None) == "view" for t in cart_tasks)
    is_cart_mutated = was_added or was_removed

    # Rilevamento ordine confermato (insert_order riuscito)
    confirmed_order = next(
        (v for v in obs.values() if isinstance(v, dict) and "order_id" in v),
        None
    )

    # Risolvi nome cliente dal primo task manage_cart (usa sempre ragione_sociale)
    cart_client_name = None
    for t in cart_tasks:
        cid = getattr(t.args, "client_id", None)
        if cid and cid in known_clients:
            info = known_clients[cid]
            cart_client_name = info.get("ragione_sociale") or info.get("alias") or cid
            break
        elif cid:
            cart_client_name = cid
            break

    # Verifica presenza liste (clienti/prodotti multipli da disambiguare)
    found_lists = [
        v for v in obs.values()
        if isinstance(v, dict) and v.get("pending_selection")
    ]
    n_pending = len(found_lists)

    # Pre-formatta il blocco testo per disambiguazione multipla (Opzione B)
    multi_disambiguation_text = ""
    if n_pending >= 2:
        blocks = []
        for task_id, v in obs.items():
            if not (isinstance(v, dict) and v.get("pending_selection")):
                continue
            results = v.get("results", [])[:5]  # max 5 opzioni per prodotto
            # Capisce se è ricerca clienti o prodotti dal contenuto
            if results and "ragione_sociale" in results[0]:
                label = results[0].get("ragione_sociale", task_id)
                lines = [f"*Clienti trovati*:"]
                for i, r in enumerate(results, 1):
                    lines.append(f"{i}. {r.get('ragione_sociale')} ({r.get('alias', '')}) — {r.get('citta', '')}")
            else:
                query_label = task_id
                lines = [f"*{results[0].get('brand', task_id) if results else task_id}* ({len(v.get('results', []))} risultati):"]
                for i, r in enumerate(results, 1):
                    prezzo = f"€{r['prezzo']:.2f}" if r.get("prezzo") else ""
                    stock = f"Disp: {r['stock']}" if r.get("stock") is not None else ""
                    desc = f"{r.get('brand', '')} {r.get('formato', '')}".strip() or r.get("descrizione", "")
                    extra = " | ".join(filter(None, [prezzo, stock]))
                    lines.append(f"{i}. {desc}" + (f" — {extra}" if extra else ""))
            blocks.append("\n".join(lines))
        multi_disambiguation_text = "\n\n".join(blocks)

    # Se operazione riuscita, recupera carrello aggiornato per mostrarlo
    cart_after_mutation = None
    if is_cart_mutated and cart_tasks:
        try:
            cid = getattr(cart_tasks[0].args, "client_id", None)
            agent_code_val = getattr(state, "agent_code", "AG001")
            if cid:
                view_args = CartArgs(action="view", client_id=cid)
                raw_view = sql_manage_cart(agent_code_val, view_args)
                if isinstance(raw_view, list):
                    def _resolve_desc(sku, db_desc):
                        if db_desc:
                            return db_desc
                        info = known_products.get(sku, {})
                        brand = info.get("brand") or ""
                        formato = info.get("formato") or ""
                        return (f"{brand} {formato}".strip()
                                or info.get("descrizione")
                                or sku)
                    cart_after_mutation = [
                        {
                            "sku": r.get("sku"),
                            "descrizione": _resolve_desc(r.get("sku"), r.get("description")),
                            "quantità": r.get("quantity"),
                            "prezzo_unitario": r.get("price"),
                        }
                        for r in raw_view
                    ]
        except Exception:
            pass

    # Regole dinamiche carrello
    cart_rules = ""
    if is_cart_view and cart_client_name:
        cart_rules += f"\n18. Inizia la risposta sul carrello con 'Carrello di {cart_client_name}:' prima di elencare i prodotti."
    if is_cart_mutated:
        nome = f"di {cart_client_name}" if cart_client_name else ""
        if cart_after_mutation:
            cart_summary = "\n".join(
                "• {desc} x{qty}{total}".format(
                    desc=r["descrizione"],
                    qty=r["quantità"],
                    total=(
                        f" — €{r['prezzo_unitario'] * r['quantità']:.2f}"
                        if r.get("prezzo_unitario") and r.get("quantità")
                        else ""
                    ),
                )
                for r in cart_after_mutation
            )
            cart_rules += (
                f"\n19. Dopo aver confermato l'operazione, riporta il seguente riepilogo carrello {nome} "
                f"(già formattato, copialo senza modifiche):\n"
                f"{cart_summary}\n"
                f"Poi chiedi: 'Vuoi confermare e inviare l'ordine?'"
            )
        else:
            cart_rules += f"\n19. Dopo aver confermato l'operazione, chiedi: 'Vuoi confermare e inviare l'ordine?'"

    # 4️⃣ System grounding (vincola il modello ai fatti reali)
    agent_nome = getattr(state, "agent_nome", "") or ""
    is_first_message = len(getattr(state, "chat_history", [])) == 0

    system_msg = f"""
Sei un assistente commerciale per WhatsApp.
Il tuo compito è riferire SOLO ciò che è stato effettivamente eseguito dai tool.
Stai parlando con l'agente: {agent_nome or "l'agente"}.
Usa il suo nome ({agent_nome}) SOLO in questi momenti — mai altrove:
  1. Quando pianificato clarify(explain_capabilities) al primo messaggio (regola 16).
  2. Ordine confermato: includi il nome nella conferma ("Ordine inviato, {agent_nome}. Totale...").
  3. Situazione critica o errore che richiede attenzione: menzionalo una volta per empatia.
In tutti gli altri messaggi rispondi in modo diretto ed efficiente, SENZA usare il nome.

STATO REALE (Fonte di Verità):
- Prodotti/clienti aggiunti al DB: {"SÌ" if was_added else "NO"}
- Risultati Tool (Observations):
{json.dumps(obs, indent=2, default=str)}

Mappa clienti (usa ragione_sociale per riferirsi ai clienti):
{json.dumps({cid: info.get("ragione_sociale", cid) for cid, info in known_clients.items()}, indent=2, ensure_ascii=False)}

REGOLE MANDATORIE:

⚠️ REGOLA FONDAMENTALE: Se 'Prodotti/clienti aggiunti al DB' è SÌ, le operazioni sul carrello
sono ANDATE A BUON FINE. In questo caso NON produrre MAI messaggi come "non ho trovato il cliente",
"per quale cliente?", "non riesco a trovare". Stai confermando operazioni già eseguite, non cercando
dati. Usa la mappa clienti sopra per tradurre client_id in nome leggibile.

1. NON DIRE MAI "Ho aggiunto al carrello" se 'Prodotti aggiunti al DB' è NO.
2. SOLO SE 'Prodotti/clienti aggiunti al DB' è NO e nelle osservazioni c'è {{"error": "client_required"}}:
   chiedi "Per quale cliente vuoi ordinare?"
3. Se c'è ESATTAMENTE UN risultato con 'pending_selection: true' (clienti o prodotti),
   usa use_interactive_list=True per elencare le opzioni.
4. Se ci sono DUE O PIÙ risultati con 'pending_selection: true', NON usare la lista
   interattiva. Manda invece un messaggio di testo con tutte le opzioni già formattate:
---
{multi_disambiguation_text if multi_disambiguation_text else "(nessuna disambiguazione multipla)"}
---
   Concludi con: "Rispondimi specificando il prodotto/formato scelto per ciascuno."
5. Sii sintetico e professionale.
6. Usa il grassetto per i nomi dei prodotti (descrizione, non il codice SKU).
7. Se vedi un errore SQLite nelle osservazioni,
   riferisci che c'è stato un problema tecnico.
8. Non inventare mai dati non presenti nelle Observations.
9. Quando mostri prodotti in lista di selezione (pending_selection=true per prodotti),
   per ogni voce mostra: nome/brand, formato, prezzo unitario e disponibilità (stock).
10. Non mostrare mai codici SKU o codici prodotto all'utente, a meno che non lo chieda
    esplicitamente. Usa sempre la descrizione/nome del prodotto.
11. Quando menzioni un cliente, usa SEMPRE la ragione sociale completa dalla mappa clienti
    qui sopra. Non usare alias, soprannomi o nomi parziali.
12. Se nelle osservazioni trovi {{"error": "quantity_required"}}, NON aggiungere nulla al carrello.
    Chiedi esplicitamente quante unità vuole aggiungere, usando il nome del prodotto (non lo SKU).
13. Se nelle osservazioni trovi {{"error": "client_required"}}, il cliente non era specificato.
    Chiedi esplicitamente: "Per quale cliente vuoi [action] il carrello?"
14. Se nelle osservazioni trovi {{"info_only": true, "results": [...]}}, stai rispondendo a una
    domanda informativa sul catalogo. Mostra i prodotti trovati in un elenco chiaro:
    per ogni voce mostra **nome/brand**, formato, prezzo unitario (€), disponibilità (pz).
    NON proporre di aggiungere al carrello a meno che l'utente non lo abbia richiesto.
    Se results è vuoto, rispondi che non ci sono prodotti corrispondenti.
15. Se nelle osservazioni trovi {{"catalog_info": true, "brands": [...]}}, mostra l'elenco dei brand
    in modo leggibile (es. divisi per lettera iniziale o in lista semplice). Se è presente anche
    "categoria", contestualizza: "Brand disponibili nella categoria X: ...".
    Se trovi {{"catalog_info": true, "categories": [...]}}, mostra le categorie disponibili.
    NON proporre prodotti specifici, NON chiedere se vuole aggiungere al carrello.
16. Se nelle osservazioni trovi {{"clarify": true, "intent": "explain_capabilities"}}:
    {"Inizia con 'Ciao " + agent_nome + "!' poi presenta lo strumento in modo caldo e informale." if is_first_message and agent_nome else "Presenta lo strumento brevemente."}
    Elenca le funzionalità in modo conciso, adatto a WhatsApp:
    - Cercare clienti per nome o ragione sociale
    - Gestire il carrello: aggiungere, rimuovere, visualizzare, svuotare
    - Consultare il catalogo: cercare prodotti, vedere brand e categorie
    - Confermare e inviare ordini
    - Consultare lo storico degli ordini
    Concludi con un invito pratico (es. "Dimmi pure cosa ti serve!").
    Se trovi {{"clarify": true, "intent": "out_of_scope", "detail": "..."}}, rispondi con
    gentilezza che quella richiesta è fuori dal tuo ambito (menziona il `detail` se utile) e
    ricorda brevemente a cosa servi, invitando a chiedere qualcosa sul catalogo o sugli ordini.
17. {"ORDINE CONFERMATO — usa solo queste informazioni per rispondere:" if confirmed_order else ""}
    {f"Ordine #{confirmed_order['order_id']} confermato per {confirmed_order.get('client_id', '')}. Totale: €{confirmed_order.get('total', 0):.2f} ({confirmed_order.get('items_count', 0)} prodotti). Rispondi con un messaggio di conferma chiaro, includi il numero ordine." if confirmed_order else ""}
    {"NON chiedere di nuovo se vuole confermare — l'ordine è già stato creato." if confirmed_order else ""}{cart_rules if not confirmed_order else ""}
"""

    # 5️⃣ Invocazione modello
    res = llm.invoke(
        [SystemMessage(content=system_msg)]
        + state.chat_history[-5:]
        + [HumanMessage(content=state.question or "")]
    )

    # 6️⃣ Debug log interno
    print("\n" + "📱" + "=" * 50)
    print(f"VERIFICA INTERNA:")
    print(f" - Was Added? {was_added}")
    print(f" - Disambiguazioni pendenti? {len(found_lists)}")
    print(f"WHATSAPP OUT: {res.text}")

    if res.use_interactive_list:
        print(f"LISTA ATTIVA: {len(res.items)} elementi inviati.")

    print("=" * 50 + "\n")

    return {
        "final_answer": res,
        "is_finished": True
    }


# =============================================================================
# COMPILAZIONE GRAFO
# =============================================================================

def create_graph():
    workflow = StateGraph(AgentState)

    # -----------------------------
    # NODI
    # -----------------------------
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("responder", responder_node)

    # -----------------------------
    # ENTRY POINT
    # -----------------------------
    workflow.set_entry_point("planner")

    # -----------------------------
    # FLOW PRINCIPALE
    # planner → executor → responder
    # -----------------------------
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "responder")

    # -----------------------------
    # LOGICA DI LOOP CONTROL
    # Se responder dice is_finished=True → END
    # Altrimenti torna al planner
    # -----------------------------
    workflow.add_conditional_edges(
        "responder",
        lambda s: "end" if s.is_finished else "continue",
        {
            "continue": "planner",
            "end": END
        }
    )

    # -----------------------------
    # CHECKPOINTER SQLITE
    # -----------------------------
    conn = sqlite3.connect("checkpoints.db", check_same_thread=False)

    return workflow.compile(
        checkpointer=SqliteSaver(conn)
    )


if __name__ == "__main__":
    app = create_graph()

    input_data = {
        "question": "Fammi un ordine per Mario a Milano, vorrei delle birre",
        "chat_history": [],
        "observations": {},
        "iteration": 0,
        "is_finished": False,
        "agent_code": "AG001",
    }

    for event in app.stream(input_data, {"thread_id": "1"}):
        pass
