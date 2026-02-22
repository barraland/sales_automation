"""
Shared state, models, DB helpers, resolvers e implementazioni dei tool.
Importato da tutti i nodi del grafo e da graph_app.py.
"""

import os
import csv
import json
import re
import sqlite3
import yaml
import smtplib
import logging
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Any, Annotated, Dict, List, Optional, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from typing_extensions import TypedDict

from src.tools.rag_tool import search_product_smart, get_catalog_facets, get_openai_client
from src.tools.rag_tool_anagrafica_clienti import search_client_smart

load_dotenv()
logging.basicConfig(level=logging.WARNING)

# =============================================================================
# PROJECT ROOT
# =============================================================================
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# LLM FACTORY
# =============================================================================
def get_model(role: Literal["planner", "generic"] = "generic"):
    provider = os.getenv(f"{role.upper()}_PROVIDER", "google").lower()
    if provider == "google":
        return ChatGoogleGenerativeAI(
            model=os.getenv(f"GEMINI_MODEL_{role.upper()}"),
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0,
            convert_system_message_to_human=True,
        )
    return ChatOpenAI(
        model=os.getenv(f"OPENAI_MODEL_{role.upper()}"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0,
    )


# =============================================================================
# CATALOG FACETS (caricati una volta all'avvio)
# =============================================================================
facets = get_catalog_facets(facet_fields=["brand", "categoria", "sottocategoria"])
brand_values          = facets.get("brand", [])
categoria_values      = facets.get("categoria", [])
sottocategoria_values = facets.get("sottocategoria", [])


# =============================================================================
# VALORI DISCRETI PER TEXT-TO-SQL (caricati da SQLite all'avvio)
# =============================================================================
def _load_discrete_values() -> str:
    """Valori distinti dei campi discreti, caricati da SQLite all'avvio."""
    db_path = os.path.join(_project_root, "sql_lite", "db", "database_ordini.db")
    conn = sqlite3.connect(db_path)
    fields = {
        "prodotti.brand":          "SELECT DISTINCT brand FROM prodotti WHERE brand IS NOT NULL ORDER BY brand",
        "prodotti.categoria":      "SELECT DISTINCT categoria FROM prodotti WHERE categoria IS NOT NULL ORDER BY categoria",
        "prodotti.sottocategoria": "SELECT DISTINCT sottocategoria FROM prodotti WHERE sottocategoria IS NOT NULL ORDER BY sottocategoria",
        "prodotti.famiglia":       "SELECT DISTINCT famiglia FROM prodotti WHERE famiglia IS NOT NULL ORDER BY famiglia",
        "prodotti.formato":        "SELECT DISTINCT formato FROM prodotti WHERE formato IS NOT NULL ORDER BY formato",
        "prodotti.confezione":     "SELECT DISTINCT confezione FROM prodotti WHERE confezione IS NOT NULL ORDER BY confezione",
        "clienti_fts.citta":       "SELECT DISTINCT citta FROM clienti_fts WHERE citta IS NOT NULL ORDER BY citta",
    }
    lines = ["VALORI AMMESSI per campi discreti (usa SOLO questi nei filtri WHERE):"]
    for label, query in fields.items():
        try:
            rows = conn.execute(query).fetchall()
            values = [str(r[0]) for r in rows if r[0]]
            if values:
                lines.append(f"  {label}: {', '.join(values)}")
        except Exception:
            pass
    conn.close()
    return "\n".join(lines)

_DISCRETE_VALUES = _load_discrete_values()


# =============================================================================
# ANAGRAFICA AGENTI
# =============================================================================
_agenti_csv = os.path.join(_project_root, "data", "agenti.csv")

agent_email_map: Dict[str, str] = {}
agent_name_map:  Dict[str, Dict[str, str]] = {}

if os.path.exists(_agenti_csv):
    with open(_agenti_csv, newline="", encoding="utf-8") as _f:
        for _row in csv.DictReader(_f):
            agent_email_map[_row["codice"]] = _row["email"]
            agent_name_map[_row["codice"]]  = {
                "nome":    _row.get("nome", ""),
                "cognome": _row.get("cognome", ""),
            }
    print(f"✅ Agenti caricati: {list(agent_email_map.keys())}")
else:
    print(f"⚠️ File agenti non trovato: {_agenti_csv}")


# =============================================================================
# DATA MODELS — risposta
# =============================================================================
class WhatsAppListItem(BaseModel):
    id: str
    title: str
    description: Optional[str] = None


class WhatsAppSection(BaseModel):
    title: str
    items: List[WhatsAppListItem] = []


class FinalResponse(BaseModel):
    text: str = ""
    use_interactive_list: bool = False
    list_button_text: str = "Vedi opzioni"
    items: List[WhatsAppListItem] = []
    sections: List[WhatsAppSection] = []
    raw_data: Optional[List[Dict[str, Any]]] = None  # per allegato email overflow


# =============================================================================
# TOOL SCHEMAS — usati da dispatcher per bind_tools
# Il nome della classe → nome del tool che l'LLM chiama.
# =============================================================================
class add_to_cart(BaseModel):
    """Aggiunge un prodotto al carrello di un cliente."""
    client_ref:  str           = Field(description="Nome, alias, ragione sociale o client_id (es. C036) del cliente")
    product_ref: str           = Field(description="Nome, brand, descrizione o SKU del prodotto (es. BIR-ICH-NON-33V)")
    quantity:    Optional[int] = Field(None, description="Quantità richiesta. Lascia null se l'utente non l'ha specificata.")


class remove_from_cart(BaseModel):
    """Rimuove un prodotto dal carrello di un cliente."""
    client_ref:  str = Field(description="Nome, alias o client_id del cliente")
    product_ref: str = Field(description="Nome, brand, descrizione o SKU del prodotto")


class clear_cart(BaseModel):
    """Svuota completamente il carrello di un cliente."""
    client_ref: str = Field(description="Nome, alias o client_id del cliente")


class view_cart(BaseModel):
    """Mostra il carrello corrente di un cliente."""
    client_ref: str = Field(description="Nome, alias o client_id del cliente")


class confirm_order(BaseModel):
    """Conferma e invia l'ordine per un cliente.
    Usare SOLO quando l'utente conferma esplicitamente: 'sì', 'confermo', 'invia', 'procedi', 'manda'."""
    client_ref: str = Field(description="Nome, alias o client_id del cliente")


class list_clients(BaseModel):
    """Mostra la lista dei clienti dell'agente.
    Usare SOLO quando l'utente chiede ESPLICITAMENTE la lista ('mostrami i clienti',
    'lista clienti', 'clienti di Milano').
    NON usare come step preliminare per add_to_cart / remove_from_cart / altri tool:
    quei tool gestiscono autonomamente la risoluzione del cliente via ricerca interna."""
    query:       Optional[str] = Field(None, description="Nome o alias da cercare. None per lista completa.")
    city_filter: Optional[str] = Field(None, description="Città (es. 'Milano', 'Como'). None se non specificata.")


class search_products(BaseModel):
    """Cerca prodotti nel catalogo per info su prezzi, disponibilità, brand.
    Usa quando l'utente chiede informazioni senza voler ordinare."""
    query:   str           = Field(description="Nome prodotto, brand o categoria (es. 'Heineken', 'birre artigianali', 'acque')")
    filters: Optional[str] = Field(None, description='JSON opzionale: {"brand"?: str, "categoria"?: str, "sottocategoria"?: str}')


class list_orders(BaseModel):
    """Mostra gli ordini passati dell'agente, opzionalmente filtrati per cliente."""
    client_ref: Optional[str] = Field(None, description="Nome o client_id del cliente. None per tutti gli ordini.")


class query_database(BaseModel):
    """Interroga il database (prodotti, clienti, ordini) con query strutturate.
    Usare per domande aggregate o cross-tabella:
    'quali brand di succhi?', 'prodotti sotto €2', 'quante birre abbiamo?',
    'clienti di Milano con ordini?', 'totale venduto per categoria',
    'ordini di Bar Mario', 'quante Ichnusa ha ordinato Bar Mario questa settimana'.
    NON usare per ricerca semantica di un singolo prodotto (usa search_products)."""
    question:     str           = Field(description="Domanda in linguaggio naturale sul database")
    client_hint:  Optional[str] = Field(default=None, description="Nome o alias del cliente menzionato nella domanda (es. 'Bar Mario') — il motore risolve client_id via FTS5 e lo inietta nel prompt SQL")
    product_hint: Optional[str] = Field(default=None, description="Nome del prodotto menzionato nella domanda (es. 'Ichnusa non filtrata') — il motore risolve lo SKU via RAG e lo inietta nel prompt SQL")


class free_response(BaseModel):
    """Risposta libera: benvenuto al primo accesso, out-of-scope, spiegazione capacità, casi non coperti da altri tool."""
    text: str = Field(description="Testo della risposta in italiano, conciso (max 3 righe)")


ALL_TOOLS = [
    add_to_cart, remove_from_cart, clear_cart, view_cart, confirm_order,
    list_clients, search_products, query_database, list_orders, free_response,
]

# Tool esposti all'LLM nel dispatcher — list_orders è nascosto (usa query_database)
DISPATCHER_TOOLS = [t for t in ALL_TOOLS if t.__name__ != "list_orders"]

TOOL_NODE_NAMES = [t.__name__ for t in ALL_TOOLS]


# =============================================================================
# AGENT STATE (TypedDict per LangGraph con reducer su tool_results)
# =============================================================================
def _results_reducer(old: Optional[List[dict]], new: Optional[List[dict]]) -> List[dict]:
    """
    Reducer per tool_results:
    - new=None  → reset a lista vuota (dispatcher, nuovo turno)
    - new=lista → accumula in parallelo (fan-out)
    """
    if new is None:
        return []
    return (old or []) + new


class AgentState(TypedDict, total=False):
    agent_code:          str
    agent_nome:          str
    agent_cognome:       str
    known_clients:       Dict[str, Any]
    known_products:      Dict[str, Any]
    chat_history:        List[Any]
    question:            Optional[str]
    final_answer:        Optional[Any]           # FinalResponse object o None
    current_datetime:    Optional[str]
    pending_call:        Optional[Dict[str, Any]]  # operazione in sospeso
    tool_calls:          List[Dict[str, Any]]    # tool calls dal dispatcher LLM
    tool_results:        Annotated[List[Dict[str, Any]], _results_reducer]
    current_tool_call:   Optional[Dict[str, Any]]  # impostato da Send per ogni branch


# =============================================================================
# EMAIL CONFERMA ORDINE
# =============================================================================
def send_order_email(order_result: dict, known_clients: dict, agent_code: str = "") -> bool:
    gmail_from     = os.getenv("GMAIL_FROM", "")
    gmail_password = os.getenv("GMAIL_APP_PASSWORD", "")
    if not gmail_from or not gmail_password:
        print("⚠️ GMAIL_FROM o GMAIL_APP_PASSWORD non configurati — email non inviata")
        return False

    agent_to  = agent_email_map.get(agent_code, gmail_from)
    order_id  = order_result.get("order_id")
    client_id = order_result.get("client_id", "")
    total     = order_result.get("total", 0.0)
    items     = order_result.get("items", [])
    cinfo     = known_clients.get(client_id, {})
    cname     = cinfo.get("ragione_sociale") or cinfo.get("alias") or client_id
    ccity     = cinfo.get("citta") or ""

    def _item_line(r: dict) -> str:
        sku   = r.get("sku", "")
        desc  = r.get("description") or sku
        qty   = r.get("quantity") or 0
        unit  = r.get("price")
        if unit is not None:
            line_tot = unit * qty
            return f"  • [{sku}] {desc} × {qty} — €{unit:.2f}/un = €{line_tot:.2f}"
        return f"  • [{sku}] {desc} × {qty}"

    items_lines = "\n".join(_item_line(r) for r in items)
    client_line = f"{cname} ({client_id})" + (f" — {ccity}" if ccity else "")
    body = (
        f"Nuovo ordine confermato dal sistema Sales Bot.\n\n"
        f"Ordine:  #{order_id}\n"
        f"Cliente: {client_line}\n"
        f"Data:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Stato:   RECEIVED\n\n"
        f"Prodotti:\n{items_lines or '  (nessun dettaglio)'}\n\n"
        f"{'─' * 40}\n"
        f"TOTALE ORDINE: €{total:.2f}\n"
        f"{'─' * 40}\n\n"
        f"Sales Automation Bot\n"
    )
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = f"Ordine #{order_id} confermato — {cname}"
    msg["From"]    = gmail_from
    msg["To"]      = agent_to

    print(f"📧 [EMAIL] Invio ordine #{order_id} a {agent_to} — {len(items)} prodotti, totale €{total:.2f}")
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(gmail_from, gmail_password)
            smtp.sendmail(gmail_from, [agent_to], msg.as_string())
        print(f"✅ [EMAIL] Ordine #{order_id} inviata a {agent_to}")
        return True
    except smtplib.SMTPAuthenticationError as e:
        print(f"❌ [EMAIL] Auth fallita: {e}")
    except smtplib.SMTPException as e:
        print(f"❌ [EMAIL] SMTP error: {e}")
    except Exception as e:
        print(f"❌ [EMAIL] Errore generico: {e}")
    return False


# =============================================================================
# DB HELPERS
# =============================================================================
def _get_db_path() -> str:
    return os.path.join(_project_root, "sql_lite", "db", "database_ordini.db")


def _db_manage_cart(
    agent_code: str,
    action: str,
    client_id: str,
    sku: str = None,
    quantity: int = None,
    description: str = None,
    price: float = None,
):
    """Cart CRUD. Returns str (add/remove/clear) o list[dict] (view)."""
    conn = sqlite3.connect(_get_db_path())
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    try:
        cur.execute("PRAGMA foreign_keys = ON;")
        if action == "add":
            cur.execute(
                """
                INSERT INTO cart_item (agent_id, client_id, sku, description, quantity, price)
                VALUES (?,?,?,?,?,?)
                ON CONFLICT(agent_id, client_id, sku)
                DO UPDATE SET
                    quantity    = quantity + excluded.quantity,
                    description = COALESCE(excluded.description, cart_item.description),
                    price       = COALESCE(excluded.price, cart_item.price)
                """,
                (agent_code, client_id, sku, description, quantity or 1, price),
            )
            conn.commit()
            return f"Aggiunto {quantity or 1}x {sku} per {client_id}."
        elif action == "remove":
            cur.execute(
                "DELETE FROM cart_item WHERE agent_id=? AND client_id=? AND sku=?",
                (agent_code, client_id, sku),
            )
            conn.commit()
            return f"Rimosso {sku} dal carrello di {client_id}."
        elif action == "view":
            cur.execute(
                "SELECT sku, description, quantity, price FROM cart_item WHERE agent_id=? AND client_id=?",
                (agent_code, client_id),
            )
            return [dict(r) for r in cur.fetchall()]
        elif action == "clear":
            cur.execute(
                "DELETE FROM cart_item WHERE agent_id=? AND client_id=?",
                (agent_code, client_id),
            )
            conn.commit()
            return "Carrello svuotato."
    except Exception as e:
        conn.rollback()
        print(f"💥 [DB ERROR cart]: {e}")
        return f"Errore DB: {e}"
    finally:
        conn.close()


def _db_insert_order(agent_code: str, client_id: str) -> dict:
    """Converte il carrello in ordine confermato. Svuota cart_item."""
    conn = sqlite3.connect(_get_db_path())
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    try:
        cur.execute("PRAGMA foreign_keys = ON;")
        cur.execute(
            "SELECT sku, description, quantity, price FROM cart_item WHERE agent_id=? AND client_id=?",
            (agent_code, client_id),
        )
        items = [dict(r) for r in cur.fetchall()]
        if not items:
            return {"error": "Carrello vuoto — nessun ordine creato"}

        total = sum((r["quantity"] or 0) * (r["price"] or 0.0) for r in items)
        cur.execute(
            "INSERT INTO [order] (client_id, agent_id, status, total_amount) VALUES (?,?,'RECEIVED',?)",
            (client_id, agent_code, round(total, 2)),
        )
        order_id = cur.lastrowid
        for r in items:
            cur.execute(
                "INSERT INTO order_item (order_id, sku, description, quantity, price_at_order) VALUES (?,?,?,?,?)",
                (order_id, r["sku"], r["description"], r["quantity"], r["price"]),
            )
        cur.execute(
            "DELETE FROM cart_item WHERE agent_id=? AND client_id=?",
            (agent_code, client_id),
        )
        conn.commit()
        print(f"✅ Ordine #{order_id} creato per {client_id} | €{total:.2f}")
        return {"order_id": order_id, "client_id": client_id,
                "total": round(total, 2), "items": items, "status": "RECEIVED"}
    except Exception as e:
        conn.rollback()
        print(f"💥 [DB ERROR insert_order]: {e}")
        return {"error": str(e)}
    finally:
        conn.close()


def _db_list_orders(agent_code: str, client_id: str = None, limit: int = 20) -> list:
    """Ritorna gli ordini con le rispettive righe."""
    conn = sqlite3.connect(_get_db_path())
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    try:
        q = "SELECT order_id, client_id, status, total_amount, created_at FROM [order] WHERE agent_id=?"
        params = [agent_code]
        if client_id:
            q += " AND client_id=?"
            params.append(client_id)
        q += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        orders = [dict(r) for r in cur.execute(q, params).fetchall()]
        for o in orders:
            rows = cur.execute(
                "SELECT sku, description, quantity, price_at_order FROM order_item WHERE order_id=?",
                (o["order_id"],),
            ).fetchall()
            o["items"] = [dict(r) for r in rows]
        return orders
    except Exception as e:
        print(f"💥 [DB ERROR list_orders]: {e}")
        return []
    finally:
        conn.close()


# =============================================================================
# RESPONSE BUILDERS
# =============================================================================
def _cart_text(cart_items: list, client_name: str = "") -> str:
    nome = f" di {client_name}" if client_name else ""
    if not cart_items:
        return f"Il carrello{nome} è vuoto."
    total = sum((r.get("price") or 0) * (r.get("quantity") or 0) for r in cart_items)
    lines = "\n".join(
        "• {desc} × {qty}{sub}".format(
            desc=r.get("description") or r.get("sku", ""),
            qty=r.get("quantity", ""),
            sub=f" — €{r['price'] * r['quantity']:.2f}" if r.get("price") and r.get("quantity") else "",
        )
        for r in cart_items
    )
    total_str = f"\nTotale: €{total:.2f}" if total else ""
    return f"Carrello{nome}:\n{lines}{total_str}"


def _group_clients_by_city(results: list) -> List[WhatsAppSection]:
    """Raggruppa tutti i clienti per città senza cap — il troncamento avviene in send_whatsapp_message."""
    grouped: Dict[str, list] = {}
    for r in results:
        city = r.get("citta") or "Altro"
        grouped.setdefault(city, []).append(r)
    sections = []
    for city in sorted(grouped.keys()):
        items = [
            WhatsAppListItem(
                id=r["client_id"],
                title=(r.get("alias") or r.get("ragione_sociale") or r["client_id"])[:24],
                description=(r.get("ragione_sociale") or "")[:72],
            )
            for r in grouped[city]
        ]
        if items:
            sections.append(WhatsAppSection(title=city, items=items))
    return sections


def _product_items(results: list) -> List[WhatsAppListItem]:
    items = []
    for r in results[:10]:
        sku   = r.get("sku", "")
        brand = r.get("brand") or ""
        fmt   = r.get("formato") or ""
        desc  = r.get("descrizione") or f"{brand} {fmt}".strip() or sku
        price = r.get("prezzo") or 0
        parts = [p for p in [brand, fmt, f"€{float(price):.2f}" if price else ""] if p]
        items.append(WhatsAppListItem(
            id=sku,
            title=desc[:24],
            description=" | ".join(parts)[:72],
        ))
    return items


def _product_list_text(intro: str, items: List[WhatsAppListItem]) -> str:
    """
    Costruisce il testo della risposta con lista numerata e SKU espliciti.
    Viene salvato nella chat history così il dispatcher al turno successivo
    può ricavare il codice SKU quando l'utente dice 'il secondo' o simili.
    """
    lines = [intro]
    for i, item in enumerate(items, 1):
        sku_part  = f" [{item.id}]" if item.id else ""
        desc_part = f" — {item.description}" if item.description else ""
        lines.append(f"  {i}. {item.title}{sku_part}{desc_part}")
    return "\n".join(lines)


# =============================================================================
# RESOLVER HELPERS
# =============================================================================
def _is_client_id(ref: str) -> bool:
    return bool(re.match(r"^C\d+$", ref.strip()))


def _is_sku(ref: str) -> bool:
    return bool(re.match(r"^[A-Z]{2,}-[A-Z0-9-]+$", ref.strip()))


def _resolve_client(client_ref: str, agent_code: str, known_clients: dict):
    """
    Risolve un riferimento cliente in (client_id, info_dict, candidates).
    - Risolto:     (client_id, info, None)
    - Non trovato: (None, None, [])
    - Ambiguo:     (None, None, [lista risultati])
    """
    ref = client_ref.strip()

    if _is_client_id(ref):
        info = known_clients.get(ref)
        if not info:
            rows = search_client_smart(agent_code, client_id=ref)
            info = dict(rows[0]) if rows else {}
        return ref, info, None

    # Exact match su known_clients
    for cid, info in known_clients.items():
        if (info.get("ragione_sociale") or "").lower() == ref.lower():
            return cid, info, None
        if (info.get("alias") or "").lower() == ref.lower():
            return cid, info, None

    # FTS5 search
    results = [dict(r) for r in search_client_smart(agent_code, query_text=ref)]
    if not results:
        return None, None, []
    if len(results) == 1:
        return results[0]["client_id"], results[0], None
    return None, None, results


def _db_client_past_skus(agent_code: str, client_id: str) -> list:
    """Restituisce lista distinta di SKU già ordinati da questo cliente."""
    conn = sqlite3.connect(_get_db_path())
    try:
        rows = conn.execute(
            """SELECT DISTINCT oi.sku
               FROM order_item oi
               JOIN [order] o ON oi.order_id = o.order_id
               WHERE o.agent_id=? AND o.client_id=?""",
            (agent_code, client_id),
        ).fetchall()
        skus = [r[0] for r in rows]
        print(f"   📦 [HISTORY] {client_id}: {len(skus)} SKU storici — {skus[:5]}")
        return skus
    except Exception as e:
        print(f"   ⚠️ [HISTORY DB ERROR]: {e}")
        return []
    finally:
        conn.close()


def _resolve_product(product_ref: str, known_products: dict, client_id: str = None, history_skus: list = None):
    """
    Risolve un riferimento prodotto in (sku, info_dict, candidates).
    - Risolto:     (sku, info, None)
    - Non trovato: (None, None, [])
    - Ambiguo:     (None, None, [lista risultati])
    """
    ref = product_ref.strip()

    if _is_sku(ref):
        info = known_products.get(ref)
        if not info:
            # SKU valido ma non in cache — recupera dettagli precisi dal catalogo
            rag = search_product_smart.invoke({
                "query": ref,
                "planner_motivation": "sku detail lookup",
                "filters_json": "",
                "top_k": 1,
                "sku_whitelist": [ref],
            })
            results = rag.get("results", [])
            info = results[0] if results else {}
        return ref, info or {}, None

    rag = search_product_smart.invoke({
        "query": ref,
        "planner_motivation": "product resolution",
        "filters_json": "",
        "top_k": 10,
    })
    results = rag.get("results", [])
    if not results:
        return None, None, []
    if len(results) == 1:
        return results[0]["sku"], results[0], None
    # History filter: auto-resolve se esattamente 1 candidato è nello storico ordini
    if history_skus and len(results) > 1:
        hist_set = set(history_skus)
        hist_matches = [r for r in results if r["sku"] in hist_set]
        if len(hist_matches) == 1:
            r = hist_matches[0]
            print(f"   ✅ [HISTORY] Auto-resolved {r['sku']} (unico match storico)")
            return r["sku"], r, None
    return None, None, results


# =============================================================================
# TOOL IMPLEMENTATIONS
# Ogni funzione ritorna:
#   (FinalResponse, needs_input, pending_call, upd_clients, upd_products, cart_client_id)
# cart_client_id != None → il merge_node leggerà il carrello di quel cliente
#   dopo aver completato tutte le mutation parallele.
# =============================================================================

def _ok(resp: FinalResponse, upd_c: dict, upd_p: dict, cart_client_id: Optional[str] = None):
    """Shortcut per risposta finale senza input richiesto."""
    return resp, False, None, upd_c, upd_p, cart_client_id


def impl_add_to_cart(client_ref, product_ref, quantity, agent_code, known_clients, known_products):
    upd_c = dict(known_clients)
    upd_p = dict(known_products)

    client_id, client_info, client_cands = _resolve_client(client_ref, agent_code, upd_c)
    if client_id is None:
        if not client_cands:
            return _ok(FinalResponse(text=f"Nessun cliente trovato per '{client_ref}'."), upd_c, upd_p)
        sections = _group_clients_by_city(client_cands)
        return (
            FinalResponse(text=f"Per quale cliente vuoi aggiungere '{product_ref}'?",
                          use_interactive_list=True, sections=sections),
            True,
            {"name": "add_to_cart", "args": {"client_ref": client_ref, "product_ref": product_ref, "quantity": quantity}},
            upd_c, upd_p, None,
        )
    if client_info:
        upd_c[client_id] = client_info
    client_name = (upd_c.get(client_id) or {}).get("ragione_sociale") or \
                  (upd_c.get(client_id) or {}).get("alias") or client_id

    history_skus = _db_client_past_skus(agent_code, client_id) if client_id else []
    sku, prod_info, prod_cands = _resolve_product(product_ref, upd_p, client_id, history_skus=history_skus)
    if sku is None:
        if not prod_cands:
            return _ok(FinalResponse(text=f"Nessun prodotto trovato per '{product_ref}'."), upd_c, upd_p)
        items = _product_items(prod_cands)
        return (
            FinalResponse(text=_product_list_text(
                              f"Ho trovato {len(prod_cands)} prodotti per '{product_ref}'. Quale intendi?",
                              items),
                          use_interactive_list=True, items=items),
            True,
            {"name": "add_to_cart", "args": {"client_ref": client_id, "product_ref": product_ref, "quantity": quantity}},
            upd_c, upd_p, None,
        )
    if prod_info:
        upd_p[sku] = prod_info
    _pi       = upd_p.get(sku) or {}
    prod_name = _pi.get("descrizione") or f"{_pi.get('brand', '')} {_pi.get('formato', '')}".strip() or sku
    _prezzo   = _pi.get("prezzo")
    prod_price = float(_prezzo) if _prezzo is not None else None  # None = prezzo sconosciuto

    if quantity is None:
        return (
            FinalResponse(text=f"Quante unità di **{prod_name}** vuoi aggiungere per {client_name}?"),
            True,
            {"name": "add_to_cart", "args": {"client_ref": client_id, "product_ref": sku, "quantity": None}},
            upd_c, upd_p, None,
        )

    _db_manage_cart(agent_code, "add", client_id, sku=sku, quantity=quantity,
                    description=prod_name, price=prod_price)
    print(f"🛒 Aggiunto {quantity}x {sku} ({prod_name}) per {client_id}")
    # La view_cart viene fatta dal merge_node dopo tutte le mutation parallele
    return _ok(FinalResponse(text=""), upd_c, upd_p, cart_client_id=client_id)


def impl_remove_from_cart(client_ref, product_ref, agent_code, known_clients, known_products):
    upd_c = dict(known_clients)
    upd_p = dict(known_products)

    client_id, client_info, client_cands = _resolve_client(client_ref, agent_code, upd_c)
    if client_id is None:
        if not client_cands:
            return _ok(FinalResponse(text=f"Nessun cliente trovato per '{client_ref}'."), upd_c, upd_p)
        sections = _group_clients_by_city(client_cands)
        return (
            FinalResponse(text=f"Per quale cliente vuoi rimuovere '{product_ref}'?",
                          use_interactive_list=True, sections=sections),
            True,
            {"name": "remove_from_cart", "args": {"client_ref": client_ref, "product_ref": product_ref}},
            upd_c, upd_p, None,
        )
    if client_info:
        upd_c[client_id] = client_info
    client_name = (upd_c.get(client_id) or {}).get("ragione_sociale") or client_id

    history_skus = _db_client_past_skus(agent_code, client_id) if client_id else []
    sku, prod_info, prod_cands = _resolve_product(product_ref, upd_p, client_id, history_skus=history_skus)
    if sku is None:
        if not prod_cands:
            return _ok(FinalResponse(text=f"Nessun prodotto trovato per '{product_ref}'."), upd_c, upd_p)
        items = _product_items(prod_cands)
        return (
            FinalResponse(text=_product_list_text("Quale prodotto vuoi rimuovere?", items),
                          use_interactive_list=True, items=items),
            True,
            {"name": "remove_from_cart", "args": {"client_ref": client_id, "product_ref": product_ref}},
            upd_c, upd_p, None,
        )

    _db_manage_cart(agent_code, "remove", client_id, sku=sku)
    # La view_cart viene fatta dal merge_node dopo tutte le mutation parallele
    return _ok(FinalResponse(text=""), upd_c, upd_p, cart_client_id=client_id)


def impl_clear_cart(client_ref, agent_code, known_clients, known_products):
    upd_c = dict(known_clients)
    upd_p = dict(known_products)

    client_id, client_info, client_cands = _resolve_client(client_ref, agent_code, upd_c)
    if client_id is None:
        if not client_cands:
            return _ok(FinalResponse(text=f"Nessun cliente trovato per '{client_ref}'."), upd_c, upd_p)
        sections = _group_clients_by_city(client_cands)
        return (
            FinalResponse(text="Per quale cliente vuoi svuotare il carrello?",
                          use_interactive_list=True, sections=sections),
            True,
            {"name": "clear_cart", "args": {"client_ref": client_ref}},
            upd_c, upd_p, None,
        )
    if client_info:
        upd_c[client_id] = client_info
    client_name = (upd_c.get(client_id) or {}).get("ragione_sociale") or client_id

    _db_manage_cart(agent_code, "clear", client_id)
    return _ok(FinalResponse(text=f"Carrello di {client_name} svuotato."), upd_c, upd_p)


def impl_view_cart(client_ref, agent_code, known_clients, known_products):
    upd_c = dict(known_clients)
    upd_p = dict(known_products)

    client_id, client_info, client_cands = _resolve_client(client_ref, agent_code, upd_c)
    if client_id is None:
        if not client_cands:
            return _ok(FinalResponse(text=f"Nessun cliente trovato per '{client_ref}'."), upd_c, upd_p)
        sections = _group_clients_by_city(client_cands)
        return (
            FinalResponse(text="Per quale cliente vuoi vedere il carrello?",
                          use_interactive_list=True, sections=sections),
            True,
            {"name": "view_cart", "args": {"client_ref": client_ref}},
            upd_c, upd_p, None,
        )
    if client_info:
        upd_c[client_id] = client_info
    client_name = (upd_c.get(client_id) or {}).get("ragione_sociale") or client_id

    cart = _db_manage_cart(agent_code, "view", client_id) or []
    cart_text = _cart_text(cart if isinstance(cart, list) else [], client_name)
    suffix = "\n\nVuoi confermare e inviare l'ordine?" if isinstance(cart, list) and cart else ""
    return _ok(FinalResponse(text=f"{cart_text}{suffix}"), upd_c, upd_p)


def impl_confirm_order(client_ref, agent_code, known_clients, known_products):
    upd_c = dict(known_clients)
    upd_p = dict(known_products)

    client_id, client_info, client_cands = _resolve_client(client_ref, agent_code, upd_c)
    if client_id is None:
        if not client_cands:
            return _ok(FinalResponse(text=f"Nessun cliente trovato per '{client_ref}'."), upd_c, upd_p)
        sections = _group_clients_by_city(client_cands)
        return (
            FinalResponse(text="Per quale cliente confermi l'ordine?",
                          use_interactive_list=True, sections=sections),
            True,
            {"name": "confirm_order", "args": {"client_ref": client_ref}},
            upd_c, upd_p, None,
        )
    if client_info:
        upd_c[client_id] = client_info
    client_name = (upd_c.get(client_id) or {}).get("ragione_sociale") or client_id

    result = _db_insert_order(agent_code, client_id)
    if "error" in result:
        return _ok(FinalResponse(text=f"❌ {result['error']}"), upd_c, upd_p)

    order_id  = result["order_id"]
    total     = result["total"]
    items     = result["items"]
    def _wa_line(r: dict) -> str:
        desc = r.get("description") or r.get("sku", "")
        qty  = r.get("quantity") or 0
        unit = r.get("price")
        if unit:
            return f"• {desc} × {qty} — €{unit * qty:.2f}"
        return f"• {desc} × {qty}"
    items_txt = "\n".join(_wa_line(r) for r in items)
    send_order_email(result, upd_c, agent_code)
    agent_email = agent_email_map.get(agent_code, "")
    email_note  = f"\n📧 Mail di conferma mandata a {agent_email}" if agent_email else ""
    return _ok(
        FinalResponse(text=f"✅ Ordine #{order_id} confermato per {client_name}.\n{items_txt}\nTotale: €{total:.2f}{email_note}"),
        upd_c, upd_p,
    )


def impl_list_clients(query, city_filter, agent_code, known_clients, known_products):
    upd_c = dict(known_clients)
    upd_p = dict(known_products)

    results = [dict(r) for r in search_client_smart(
        agent_code, query_text=query or None, city_filter=city_filter or None
    )]
    if not results:
        txt = "Nessun cliente trovato"
        if query:       txt += f" per '{query}'"
        if city_filter: txt += f" a {city_filter}"
        return _ok(FinalResponse(text=txt + "."), upd_c, upd_p)

    # Se trovato un solo cliente, testo piano senza lista interattiva
    if len(results) == 1:
        r = results[0]
        cid   = r.get("client_id", "")
        alias = r.get("alias") or ""
        rs    = r.get("ragione_sociale") or ""
        city  = r.get("citta") or ""
        cname = alias or rs or cid
        extra = f" — {rs}" if rs and rs != cname else ""
        city_s = f" ({city})" if city else ""
        upd_c[cid] = r
        return _ok(FinalResponse(text=f"Cliente: {cname}{extra}{city_s}"), upd_c, upd_p)

    sections = _group_clients_by_city(results)
    total = len(results)
    intro = f"Ecco i tuoi {total} clienti — per quale vuoi procedere?" if not query and not city_filter \
            else f"Ho trovato {total} clienti — per quale vuoi procedere?"
    raw_data = [{"client_id": r.get("client_id", ""), "alias": r.get("alias", ""),
                 "ragione_sociale": r.get("ragione_sociale", ""),
                 "citta": r.get("citta", ""), "indirizzo": r.get("indirizzo", "")}
                for r in results]
    return _ok(FinalResponse(text=intro, use_interactive_list=True, sections=sections,
                             raw_data=raw_data), upd_c, upd_p)


def impl_search_products(query, filters, agent_code, known_clients, known_products):
    upd_c = dict(known_clients)
    upd_p = dict(known_products)

    rag = search_product_smart.invoke({
        "query": query,
        "planner_motivation": "catalog info",
        "filters_json": filters or "",
        "top_k": 10,
    })
    results = rag.get("results", [])
    if not results:
        return _ok(FinalResponse(text=f"Nessun prodotto trovato per '{query}'."), upd_c, upd_p)

    items = _product_items(results)
    return _ok(FinalResponse(
        text=_product_list_text(f"Ho trovato {len(results)} prodotti per '{query}':", items),
        use_interactive_list=True, items=items,
    ), upd_c, upd_p)


def _load_db_schema() -> str:
    """Carica sql_lite/db_schema.yaml e genera la stringa di contesto per il prompt LLM."""
    yaml_path = os.path.join(_project_root, "sql_lite", "db_schema.yaml")
    with open(yaml_path, encoding="utf-8") as _f:
        _schema = yaml.safe_load(_f)
    lines = []
    for tname, tdef in _schema["tables"].items():
        lines.append(f"═══ TABELLA: {tname}  ({tdef.get('note', '')})")
        for col, desc in tdef["columns"].items():
            lines.append(f"  {col:<18} {desc}")
        lines.append("")
    lines.append("═══ JOIN PATHS")
    for j in _schema["joins"]:
        lines.append(f"  {j}")
    lines.append("")
    lines.append("═══ QUERY ESEMPIO")
    for ex in _schema["examples"]:
        lines.append(f"-- {ex['desc']}:")
        lines.append(ex["sql"].rstrip())
        lines.append("")
    return "\n".join(lines)


_DB_SCHEMA = _load_db_schema()   # caricato una volta all'import del modulo


def impl_query_database(question: str, known_clients: dict, known_products: dict,
                        agent_code: str = "",
                        client_hint: str = None, product_hint: str = None):
    upd_c = dict(known_clients)
    upd_p = dict(known_products)
    schema = _DB_SCHEMA.replace("{AGENT}", agent_code)

    # Risoluzione hint: FTS5 per cliente, RAG per prodotto
    hint_ctx = ""
    resolved_cid = None

    if client_hint:
        cid, cinfo, cands = _resolve_client(client_hint, agent_code, upd_c)
        if cid:
            resolved_cid = cid
            upd_c[cid] = cinfo or upd_c.get(cid, {})
            hint_ctx += (
                f"\nCLIENTE RISOLTO (usa questi valori nei filtri SQL):\n"
                f"  client_id:       {cid}\n"
                f"  ragione_sociale: {(cinfo or {}).get('ragione_sociale', '')}\n"
                f"  alias:           {(cinfo or {}).get('alias', '')}\n"
                f"  citta:           {(cinfo or {}).get('citta', '')}\n"
                f"→ Filtra con client_id='{cid}'\n"
            )
            print(f"   👤 [HINT CLIENT]: {client_hint} → {cid}")
        elif cands:
            hint_ctx += f"\nCLIENTI TROVATI per '{client_hint}' (scegli il più pertinente):\n"
            for c in cands[:5]:
                hint_ctx += f"  {c['client_id']}: {c.get('ragione_sociale', '')} alias={c.get('alias', '')} città={c.get('citta', '')}\n"
            print(f"   👤 [HINT CLIENT]: {client_hint} → {len(cands)} candidati")

    if product_hint:
        sku, pinfo, pcands = _resolve_product(product_hint, upd_p, client_id=resolved_cid)
        if sku:
            upd_p[sku] = pinfo or upd_p.get(sku, {})
            hint_ctx += (
                f"\nPRODOTTO RISOLTO (usa questi valori nei filtri SQL):\n"
                f"  sku:         {sku}\n"
                f"  descrizione: {(pinfo or {}).get('descrizione', '')}\n"
                f"  brand:       {(pinfo or {}).get('brand', '')}\n"
                f"  formato:     {(pinfo or {}).get('formato', '')}\n"
                f"→ Filtra con sku='{sku}'\n"
            )
            print(f"   📦 [HINT PRODUCT]: {product_hint} → {sku}")
        elif pcands:
            hint_ctx += f"\nPRODOTTI TROVATI per '{product_hint}' (scegli il più pertinente):\n"
            for p in pcands[:5]:
                hint_ctx += f"  {p['sku']}: {p.get('descrizione', '')} ({p.get('brand', '')} {p.get('formato', '')})\n"
            print(f"   📦 [HINT PRODUCT]: {product_hint} → {len(pcands)} candidati")

    prompt = (
        f"Sei un assistente SQL per un sistema vendite Horeca.\n\n"
        f"Schema:\n{schema}\n\n"
        f"{_DISCRETE_VALUES}\n\n"
        f"Agente corrente: '{agent_code}'\n"
        f"{hint_ctx}"
        f"Domanda: \"{question}\"\n\n"
        f"REGOLE:\n"
        f"- Solo SELECT, nessuna modifica ai dati\n"
        f"- Filtra SEMPRE per agent_id='{agent_code}' su [order], cart_item, clienti_fts\n"
        f"- La tabella prodotti NON ha agent_id — non aggiungere filtro agente\n"
        f"- Usa COLLATE NOCASE per confronti su stringhe\n"
        f"- Per la tabella [order] usa SEMPRE le parentesi quadre\n"
        f"- ORDER BY può referenziare solo colonne presenti nel SELECT o i loro alias\n"
        f"- Evita UNION ALL: usa OR o LIKE '%termine%' per ricerche su più campi dello stesso cliente\n"
        f"- Per cercare un cliente per nome usa LIKE '%nome%' COLLATE NOCASE su alias o ragione_sociale (non UNION)\n"
        f"- Per filtrare su brand, categoria, sottocategoria, famiglia, formato, confezione o città, "
        f"usa ESCLUSIVAMENTE i valori dalla sezione VALORI AMMESSI sopra\n"
        f"- Rispondi SOLO con la query SQL, senza markdown, senza commenti"
    )
    _ERR_MSG = "Non riesco a formulare la query per questa richiesta. Prova a riformulare."
    try:
        resp = get_openai_client().chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        sql = resp.choices[0].message.content.strip().replace("```sql", "").replace("```", "").strip()
    except Exception as e:
        print(f"   ⚠️ [DB QUERY LLM ERROR]: {e}")
        return _ok(FinalResponse(text=_ERR_MSG), upd_c, upd_p)

    print(f"   🗃️ [DB QUERY]: {sql}")
    if not sql.upper().startswith("SELECT"):
        return _ok(FinalResponse(text="Query non consentita."), upd_c, upd_p)

    # Tentativo 1
    conn = sqlite3.connect(_get_db_path())
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(sql).fetchall()
        conn.close()
    except Exception as e1:
        conn.close()
        print(f"   ⚠️ [DB QUERY ERROR #1]: {e1} — retry con LLM...")
        fix_prompt = (
            f"Questa query SQLite ha prodotto un errore:\n\n"
            f"Query:\n{sql}\n\n"
            f"Errore SQLite: {e1}\n\n"
            f"Schema:\n{schema}\n\n"
            f"Scrivi una query corretta che risponde alla stessa domanda.\n"
            f"REGOLE:\n"
            f"- Solo SELECT\n"
            f"- ORDER BY deve referenziare solo colonne nel SELECT o alias espliciti\n"
            f"- Evita UNION ALL: usa OR o LIKE per ricerche su più campi\n"
            f"- Per la tabella [order] usa sempre le parentesi quadre\n"
            f"- Rispondi SOLO con la query SQL, senza markdown, senza commenti"
        )
        try:
            fix_resp = get_openai_client().chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": fix_prompt}],
                temperature=0,
            )
            sql = fix_resp.choices[0].message.content.strip().replace("```sql", "").replace("```", "").strip()
            print(f"   🔄 [DB QUERY RETRY]: {sql}")
            conn2 = sqlite3.connect(_get_db_path())
            conn2.row_factory = sqlite3.Row
            try:
                rows = conn2.execute(sql).fetchall()
                conn2.close()
            except Exception as e2:
                conn2.close()
                print(f"   ❌ [DB QUERY ERROR #2]: {e2}")
                return _ok(FinalResponse(text=_ERR_MSG), upd_c, upd_p)
        except Exception as e_fix:
            print(f"   ❌ [DB QUERY FIX LLM ERROR]: {e_fix}")
            return _ok(FinalResponse(text=_ERR_MSG), upd_c, upd_p)

    if not rows:
        return _ok(FinalResponse(text=f"Nessun risultato per: {question}"), upd_c, upd_p)
    cols = list(rows[0].keys())
    raw = [{c: row[c] for c in cols} for row in rows[:500]]
    if len(cols) == 1:
        vals = [str(r[0]) for r in rows[:20]]
        result_text = ", ".join(vals)
    else:
        lines = [" | ".join(f"{c}: {row[c]}" for c in cols) for row in rows[:15]]
        result_text = "\n".join(lines)
        if len(rows) > 15:
            result_text += f"\n… e altri {len(rows) - 15}"
    return _ok(FinalResponse(text=result_text, raw_data=raw), upd_c, upd_p)


def impl_list_orders(client_ref, agent_code, known_clients, known_products):
    upd_c = dict(known_clients)
    upd_p = dict(known_products)

    client_id   = None
    client_name = None
    if client_ref:
        cid, cinfo, cands = _resolve_client(client_ref, agent_code, upd_c)
        if cid is None:
            if not cands:
                return _ok(FinalResponse(text=f"Nessun cliente trovato per '{client_ref}'."), upd_c, upd_p)
            sections = _group_clients_by_city(cands)
            return (
                FinalResponse(text="Di quale cliente vuoi vedere gli ordini?",
                              use_interactive_list=True, sections=sections),
                True,
                {"name": "list_orders", "args": {"client_ref": client_ref}},
                upd_c, upd_p, None,
            )
        client_id = cid
        if cinfo:
            upd_c[client_id] = cinfo
        client_name = (upd_c.get(client_id) or {}).get("ragione_sociale") or client_id

    orders = _db_list_orders(agent_code, client_id=client_id)
    if not orders:
        txt = "Nessun ordine trovato"
        if client_name: txt += f" per {client_name}"
        return _ok(FinalResponse(text=txt + "."), upd_c, upd_p)

    lines = []
    for o in orders:
        cid = o.get("client_id", "")
        if cid not in upd_c:
            rows = search_client_smart(agent_code, client_id=cid)
            if rows:
                upd_c[cid] = dict(rows[0])
        cinfo = upd_c.get(cid, {})
        cname = cinfo.get("ragione_sociale") or cinfo.get("alias") or cid
        date  = (o.get("created_at") or "")[:10]
        lines.append(
            f"• Ordine #{o['order_id']} — {cname} — €{o.get('total_amount', 0):.2f} — {date} — {o.get('status', '')}"
        )

    header = f"Ordini{' di ' + client_name if client_name else ''}:"
    return _ok(FinalResponse(text=header + "\n" + "\n".join(lines)), upd_c, upd_p)
