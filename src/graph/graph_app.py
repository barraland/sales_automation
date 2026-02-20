"""
Sales Automation — Beverage Agent
Architettura: singolo dispatcher node con LLM tool-calling.

Flusso per turno:
  1. dispatcher_node: LLM legge il messaggio e chiama uno o più tool
  2. Ogni tool implementa la propria logica (risoluzione cliente/prodotto,
     disambiguazione, DB) e restituisce direttamente una FinalResponse
  3. Se il tool ha bisogno di input dall'utente (disambiguazione o quantità)
     salva pending_call nello stato e restituisce la domanda all'utente
  4. Il turno successivo l'LLM vede pending_call nel sistema prompt e riprende
"""

import os
import csv
import json
import re
import sqlite3
import smtplib
import logging
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Any, Dict, List, Optional, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from src.tools.rag_tool import search_product_smart, get_catalog_facets
from src.tools.rag_tool_anagrafica_clienti import search_client_smart

load_dotenv()
logging.basicConfig(level=logging.WARNING)

# =============================================================================
# LLM FACTORY
# =============================================================================
def get_model(role: Literal["planner", "generic"] = "generic", structured_schema: Any = None):
    provider = os.getenv(f"{role.upper()}_PROVIDER", "google").lower()
    if provider == "google":
        llm = ChatGoogleGenerativeAI(
            model=os.getenv(f"GEMINI_MODEL_{role.upper()}"),
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0,
            convert_system_message_to_human=True,
        )
    else:
        llm = ChatOpenAI(
            model=os.getenv(f"OPENAI_MODEL_{role.upper()}"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0,
        )
    if structured_schema:
        return llm.with_structured_output(structured_schema)
    return llm


# =============================================================================
# CATALOG FACETS (caricati una volta all'avvio)
# =============================================================================
facets = get_catalog_facets(facet_fields=["brand", "categoria", "sottocategoria"])
brand_values         = facets.get("brand", [])
categoria_values     = facets.get("categoria", [])
sottocategoria_values = facets.get("sottocategoria", [])


# =============================================================================
# ANAGRAFICA AGENTI
# =============================================================================
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_agenti_csv   = os.path.join(_project_root, "data", "agenti.csv")

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
# DATA MODELS
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


class AgentState(BaseModel):
    agent_code: str
    agent_nome: str = ""
    agent_cognome: str = ""
    known_clients:  Dict[str, Any] = {}
    known_products: Dict[str, Any] = {}
    chat_history:   List[Any] = []
    question:       Optional[str] = None
    final_answer:   Optional[Any] = None
    current_datetime: Optional[str] = None
    pending_call:   Optional[Dict[str, Any]] = None   # operazione in attesa di input utente


# =============================================================================
# TOOL SCHEMAS — input schemas per il tool-calling LLM
# Il nome della classe diventa il nome del tool chiamato dall'LLM.
# =============================================================================
class add_to_cart(BaseModel):
    """Aggiunge un prodotto al carrello di un cliente."""
    client_ref:  str          = Field(description="Nome, alias, ragione sociale o client_id (es. C036) del cliente")
    product_ref: str          = Field(description="Nome, brand, descrizione o SKU del prodotto (es. BIR-ICH-NON-33V)")
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
    """Mostra la lista dei clienti dell'agente, con filtro opzionale per nome o città."""
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


class free_response(BaseModel):
    """Risposta libera: benvenuto al primo accesso, out-of-scope, spiegazione capacità, casi non coperti da altri tool."""
    text: str = Field(description="Testo della risposta in italiano, conciso (max 3 righe)")


ALL_TOOLS = [
    add_to_cart, remove_from_cart, clear_cart, view_cart, confirm_order,
    list_clients, search_products, list_orders, free_response,
]


# =============================================================================
# EMAIL CONFERMA ORDINE
# =============================================================================
def send_order_email(order_result: dict, known_clients: dict, agent_code: str = "") -> bool:
    gmail_from     = os.getenv("GMAIL_FROM", "")
    gmail_password = os.getenv("GMAIL_APP_PASSWORD", "")
    if not gmail_from or not gmail_password:
        print("⚠️ GMAIL_FROM o GMAIL_APP_PASSWORD non configurati — email non inviata")
        return False

    agent_to   = agent_email_map.get(agent_code, gmail_from)
    order_id   = order_result.get("order_id")
    client_id  = order_result.get("client_id", "")
    total      = order_result.get("total", 0.0)
    items      = order_result.get("items", [])
    cinfo      = known_clients.get(client_id, {})
    cname      = cinfo.get("ragione_sociale") or cinfo.get("alias") or client_id
    ccity      = cinfo.get("citta") or ""

    items_lines = "\n".join(
        "  • [{sku}] {desc} x{qty} — €{tot:.2f}".format(
            sku=r.get("sku", ""),
            desc=r.get("description") or r.get("sku", ""),
            qty=r.get("quantity", ""),
            tot=(r.get("price") or 0.0) * (r.get("quantity") or 0),
        )
        for r in items
    )
    client_line = f"{cname} ({client_id})" + (f" — {ccity}" if ccity else "")
    body = (
        f"Nuovo ordine confermato dal sistema Sales Bot.\n\n"
        f"Ordine: #{order_id}\nCliente: {client_line}\n"
        f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nStato: RECEIVED\n\n"
        f"Prodotti:\n{items_lines or '  (nessun dettaglio)'}\n\nTotale: €{total:.2f}\n---\nSales Automation Bot\n"
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
    """Cart CRUD. Returns str (add/remove/clear) or list[dict] (view)."""
    db_path = _get_db_path()
    conn = sqlite3.connect(db_path)
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
    db_path = _get_db_path()
    conn = sqlite3.connect(db_path)
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
    db_path = _get_db_path()
    conn = sqlite3.connect(db_path)
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
        "• {desc} x{qty}{sub}".format(
            desc=r.get("description") or r.get("sku", ""),
            qty=r.get("quantity", ""),
            sub=f" — €{r['price'] * r['quantity']:.2f}" if r.get("price") and r.get("quantity") else "",
        )
        for r in cart_items
    )
    total_str = f"\nTotale: €{total:.2f}" if total else ""
    return f"Carrello{nome}:\n{lines}{total_str}"


def _group_clients_by_city(results: list) -> List[WhatsAppSection]:
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
            for r in grouped[city][:10]
        ]
        if items:
            sections.append(WhatsAppSection(title=city, items=items))
    return sections[:10]


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
    - Risolto:    (client_id, info, None)
    - Non trovato: (None, None, [])
    - Ambiguo:    (None, None, [lista risultati])
    """
    ref = client_ref.strip()

    if _is_client_id(ref):
        info = known_clients.get(ref)
        if not info:
            rows = search_client_smart(agent_code, client_id=ref)
            info = dict(rows[0]) if rows else {}
        return ref, info, None

    # Verifica exact match su known_clients
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


def _resolve_product(product_ref: str, known_products: dict, client_id: str = None):
    """
    Risolve un riferimento prodotto in (sku, info_dict, candidates).
    - Risolto:    (sku, info, None)
    - Non trovato: (None, None, [])
    - Ambiguo:    (None, None, [lista risultati])
    """
    ref = product_ref.strip()

    if _is_sku(ref):
        return ref, known_products.get(ref, {}), None

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
    return None, None, results


# =============================================================================
# TOOL IMPLEMENTATIONS
# Ogni funzione ritorna: (FinalResponse, needs_input, pending_call, upd_clients, upd_products)
# needs_input=True  → l'utente deve rispondere (disambiguazione o quantità mancante)
# pending_call      → dict con {name, args} da salvare nello stato
# =============================================================================

def _ok(resp, upd_c, upd_p):
    """Shortcut per risposta finale senza input richiesto."""
    return resp, False, None, upd_c, upd_p


def impl_add_to_cart(client_ref, product_ref, quantity, agent_code, known_clients, known_products):
    upd_c = dict(known_clients)
    upd_p = dict(known_products)

    # 1. Risolvi cliente
    client_id, client_info, client_cands = _resolve_client(client_ref, agent_code, upd_c)
    if client_id is None:
        if not client_cands:
            return _ok(FinalResponse(text=f"Nessun cliente trovato per '{client_ref}'."), upd_c, upd_p)
        sections = _group_clients_by_city(client_cands)
        n = len(client_cands)
        return (
            FinalResponse(text=f"Ho trovato {n} clienti per '{client_ref}'. Quale intendi?",
                          use_interactive_list=True, sections=sections),
            True,
            {"name": "add_to_cart", "args": {"client_ref": client_ref, "product_ref": product_ref, "quantity": quantity}},
            upd_c, upd_p,
        )
    if client_info:
        upd_c[client_id] = client_info
    client_name = (upd_c.get(client_id) or {}).get("ragione_sociale") or \
                  (upd_c.get(client_id) or {}).get("alias") or client_id

    # 2. Risolvi prodotto
    sku, prod_info, prod_cands = _resolve_product(product_ref, upd_p, client_id)
    if sku is None:
        if not prod_cands:
            return _ok(FinalResponse(text=f"Nessun prodotto trovato per '{product_ref}'."), upd_c, upd_p)
        items = _product_items(prod_cands)
        n = len(prod_cands)
        return (
            FinalResponse(text=f"Ho trovato {n} prodotti per '{product_ref}'. Quale intendi?",
                          use_interactive_list=True, items=items),
            True,
            {"name": "add_to_cart", "args": {"client_ref": client_id, "product_ref": product_ref, "quantity": quantity}},
            upd_c, upd_p,
        )
    if prod_info:
        upd_p[sku] = prod_info
    prod_name  = (upd_p.get(sku) or {}).get("descrizione") or (upd_p.get(sku) or {}).get("brand") or sku
    prod_price = float((upd_p.get(sku) or {}).get("prezzo") or 0) or None

    # 3. Quantità mancante?
    if quantity is None:
        return (
            FinalResponse(text=f"Quante unità di **{prod_name}** vuoi aggiungere per {client_name}?"),
            True,
            {"name": "add_to_cart", "args": {"client_ref": client_id, "product_ref": sku, "quantity": None}},
            upd_c, upd_p,
        )

    # 4. Aggiungi al carrello
    _db_manage_cart(agent_code, "add", client_id, sku=sku, quantity=quantity,
                    description=prod_name, price=prod_price)
    print(f"🛒 Aggiunto {quantity}x {sku} ({prod_name}) per {client_id}")

    # 5. Mostra carrello aggiornato
    cart = _db_manage_cart(agent_code, "view", client_id) or []
    cart_text = _cart_text(cart if isinstance(cart, list) else [], client_name)
    return _ok(FinalResponse(text=f"{cart_text}\n\nVuoi confermare e inviare l'ordine?"), upd_c, upd_p)


def impl_remove_from_cart(client_ref, product_ref, agent_code, known_clients, known_products):
    upd_c = dict(known_clients)
    upd_p = dict(known_products)

    client_id, client_info, client_cands = _resolve_client(client_ref, agent_code, upd_c)
    if client_id is None:
        if not client_cands:
            return _ok(FinalResponse(text=f"Nessun cliente trovato per '{client_ref}'."), upd_c, upd_p)
        sections = _group_clients_by_city(client_cands)
        return (
            FinalResponse(text=f"Quale cliente?", use_interactive_list=True, sections=sections),
            True,
            {"name": "remove_from_cart", "args": {"client_ref": client_ref, "product_ref": product_ref}},
            upd_c, upd_p,
        )
    if client_info:
        upd_c[client_id] = client_info
    client_name = (upd_c.get(client_id) or {}).get("ragione_sociale") or client_id

    sku, prod_info, prod_cands = _resolve_product(product_ref, upd_p, client_id)
    if sku is None:
        if not prod_cands:
            return _ok(FinalResponse(text=f"Nessun prodotto trovato per '{product_ref}'."), upd_c, upd_p)
        items = _product_items(prod_cands)
        return (
            FinalResponse(text=f"Quale prodotto vuoi rimuovere?", use_interactive_list=True, items=items),
            True,
            {"name": "remove_from_cart", "args": {"client_ref": client_id, "product_ref": product_ref}},
            upd_c, upd_p,
        )

    _db_manage_cart(agent_code, "remove", client_id, sku=sku)
    cart = _db_manage_cart(agent_code, "view", client_id) or []
    cart_text = _cart_text(cart if isinstance(cart, list) else [], client_name)
    suffix = "\n\nVuoi confermare e inviare l'ordine?" if isinstance(cart, list) and cart else ""
    return _ok(FinalResponse(text=f"{cart_text}{suffix}"), upd_c, upd_p)


def impl_clear_cart(client_ref, agent_code, known_clients, known_products):
    upd_c = dict(known_clients)
    upd_p = dict(known_products)

    client_id, client_info, client_cands = _resolve_client(client_ref, agent_code, upd_c)
    if client_id is None:
        if not client_cands:
            return _ok(FinalResponse(text=f"Nessun cliente trovato per '{client_ref}'."), upd_c, upd_p)
        sections = _group_clients_by_city(client_cands)
        return (
            FinalResponse(text="Quale cliente?", use_interactive_list=True, sections=sections),
            True,
            {"name": "clear_cart", "args": {"client_ref": client_ref}},
            upd_c, upd_p,
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
            FinalResponse(text="Quale cliente vuoi vedere?", use_interactive_list=True, sections=sections),
            True,
            {"name": "view_cart", "args": {"client_ref": client_ref}},
            upd_c, upd_p,
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
            upd_c, upd_p,
        )
    if client_info:
        upd_c[client_id] = client_info
    client_name = (upd_c.get(client_id) or {}).get("ragione_sociale") or client_id

    result = _db_insert_order(agent_code, client_id)
    if "error" in result:
        return _ok(FinalResponse(text=f"❌ {result['error']}"), upd_c, upd_p)

    order_id = result["order_id"]
    total    = result["total"]
    items    = result["items"]
    items_txt = "\n".join(
        f"• {r.get('description') or r.get('sku')} x{r.get('quantity')}"
        + (f" — €{(r.get('price') or 0) * r.get('quantity',0):.2f}" if r.get("price") else "")
        for r in items
    )
    send_order_email(result, upd_c, agent_code)
    return _ok(
        FinalResponse(text=f"✅ Ordine #{order_id} confermato per {client_name}.\n{items_txt}\nTotale: €{total:.2f}"),
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

    sections = _group_clients_by_city(results)
    intro = "Ecco i tuoi clienti:" if not query and not city_filter else f"Ho trovato {len(results)} clienti:"
    return _ok(FinalResponse(text=intro, use_interactive_list=True, sections=sections), upd_c, upd_p)


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
    intro = f"Ho trovato {len(results)} prodotti per '{query}':"
    return _ok(FinalResponse(text=intro, use_interactive_list=True, items=items), upd_c, upd_p)


def impl_list_orders(client_ref, agent_code, known_clients, known_products):
    upd_c = dict(known_clients)
    upd_p = dict(known_products)

    client_id = None
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
                upd_c, upd_p,
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
        cid   = o.get("client_id", "")
        cinfo = upd_c.get(cid, {})
        cname = cinfo.get("ragione_sociale") or cinfo.get("alias") or cid
        date  = (o.get("created_at") or "")[:10]
        lines.append(f"• Ordine #{o['order_id']} — {cname} — €{o.get('total_amount',0):.2f} — {date} — {o.get('status','')}")

    header = f"Ordini{' di ' + client_name if client_name else ''}:"
    return _ok(FinalResponse(text=header + "\n" + "\n".join(lines)), upd_c, upd_p)


# =============================================================================
# DISPATCHER (routing tool calls → implementations)
# =============================================================================
def _dispatch(name: str, args: dict, agent_code: str, known_clients: dict, known_products: dict):
    if name == "add_to_cart":
        return impl_add_to_cart(
            args.get("client_ref", ""), args.get("product_ref", ""), args.get("quantity"),
            agent_code, known_clients, known_products,
        )
    if name == "remove_from_cart":
        return impl_remove_from_cart(
            args.get("client_ref", ""), args.get("product_ref", ""),
            agent_code, known_clients, known_products,
        )
    if name == "clear_cart":
        return impl_clear_cart(args.get("client_ref", ""), agent_code, known_clients, known_products)
    if name == "view_cart":
        return impl_view_cart(args.get("client_ref", ""), agent_code, known_clients, known_products)
    if name == "confirm_order":
        return impl_confirm_order(args.get("client_ref", ""), agent_code, known_clients, known_products)
    if name == "list_clients":
        return impl_list_clients(args.get("query"), args.get("city_filter"),
                                  agent_code, known_clients, known_products)
    if name == "search_products":
        return impl_search_products(args.get("query", ""), args.get("filters"),
                                     agent_code, known_clients, known_products)
    if name == "list_orders":
        return impl_list_orders(args.get("client_ref"), agent_code, known_clients, known_products)
    if name == "free_response":
        return _ok(FinalResponse(text=args.get("text", "")), known_clients, known_products)
    return _ok(FinalResponse(text=f"Tool '{name}' non riconosciuto."), known_clients, known_products)


# =============================================================================
# DISPATCHER NODE
# =============================================================================
def dispatcher_node(state: AgentState):
    agent_code     = state.agent_code
    known_clients  = dict(state.known_clients)
    known_products = dict(state.known_products)
    question       = state.question or ""

    # Riepilogo clienti già risolti in sessione
    clients_str = "\n".join(
        f"  {cid}: {info.get('ragione_sociale') or info.get('alias') or cid}"
        f" ({info.get('citta', '')})"
        for cid, info in list(known_clients.items())[:20]
    ) or "  (nessuno risolto in questa sessione)"

    # Contesto operazione in sospeso
    pending_ctx = ""
    if state.pending_call:
        pc = state.pending_call
        pending_ctx = (
            f"\n\nOPERAZIONE IN SOSPESO: {pc['name']}\n"
            f"Parametri già noti: {json.dumps(pc['args'], ensure_ascii=False)}\n"
            f"→ Se il messaggio è una risposta/selezione per questa operazione, "
            f"chiama {pc['name']} con i parametri aggiornati.\n"
            f"→ Se l'utente cambia argomento, ignora l'operazione in sospeso."
        )

    is_first = len(state.chat_history) == 0

    system_prompt = (
        f"Sei l'assistente vendite Horeca per {state.agent_nome or state.agent_code}.\n"
        f"Data e ora: {state.current_datetime or ''}\n\n"
        f"CLIENTI RISOLTI IN SESSIONE:\n{clients_str}"
        f"{pending_ctx}\n\n"
        f"ISTRUZIONI:\n"
        f"- Chiama lo strumento appropriato in base al messaggio.\n"
        f"- Puoi chiamare più strumenti in parallelo se l'utente fa richieste multiple "
        f"(es. 'aggiungi per Mario E per Giuseppe').\n"
        f"- Per confirm_order: SOLO su conferma esplicita ('sì', 'confermo', 'invia', 'procedi', 'manda').\n"
        f"- Se il messaggio fornisce una quantità o una selezione per un'operazione in sospeso, "
        f"riprendi quell'operazione con i parametri aggiornati.\n"
        f"- Se l'utente seleziona da una lista precedente, usa client_id o SKU dalla lista.\n"
        f"- Per richieste informative sul catalogo (prezzi, disponibilità, brand) usa search_products.\n"
        f"- Per il primo messaggio usa free_response con un breve benvenuto a "
        f"{state.agent_nome or state.agent_code}.\n"
        f"- Per richieste fuori ambito usa free_response."
    )

    llm = get_model("generic").bind_tools(ALL_TOOLS)
    messages = (
        [SystemMessage(content=system_prompt)]
        + state.chat_history[-10:]
        + [HumanMessage(content=question)]
    )

    print(f"\n💬 [DISPATCHER] {question[:80]}")
    ai_resp = llm.invoke(messages)

    # Fallback: LLM ha risposto con testo senza tool call
    if not getattr(ai_resp, "tool_calls", None):
        text = getattr(ai_resp, "content", "") or "Come posso aiutarti?"
        print(f"   ⚠️ Nessun tool call — risposta testo libero")
        return {"final_answer": FinalResponse(text=str(text)), "pending_call": None}

    print(f"   🔧 Tool calls: {[tc['name'] for tc in ai_resp.tool_calls]}")

    # Esegui tool calls (stop al primo che richiede input)
    all_responses: List[FinalResponse] = []
    new_pending = None
    upd_c = known_clients
    upd_p = known_products

    for tc in ai_resp.tool_calls:
        resp, needs_input, pc, upd_c, upd_p = _dispatch(
            tc["name"], tc["args"], agent_code, upd_c, upd_p
        )
        all_responses.append(resp)
        if needs_input:
            new_pending = pc
            break   # aspetta risposta utente prima di elaborare le eventuali call successive

    # Combina risposte multiple (es. add per Mario + add per Giuseppe)
    if len(all_responses) == 1:
        final = all_responses[0]
    else:
        combined_text = "\n\n".join(r.text for r in all_responses if r.text)
        last_list = next((r for r in reversed(all_responses) if r.sections or r.items), None)
        final = FinalResponse(
            text=combined_text,
            use_interactive_list=bool(last_list),
            items=last_list.items if last_list else [],
            sections=last_list.sections if last_list else [],
        )

    print(f"   📤 Risposta: {final.text[:100]}")
    if final.use_interactive_list:
        n = sum(len(s.items) for s in final.sections) if final.sections else len(final.items)
        print(f"   📋 Lista: {n} elementi")
    print(f"   ⏳ Pending: {new_pending['name'] if new_pending else 'None'}\n")

    return {
        "final_answer":    final,
        "known_clients":   upd_c,
        "known_products":  upd_p,
        "pending_call":    new_pending,
    }


# =============================================================================
# GRAPH
# =============================================================================
def create_graph():
    wf = StateGraph(AgentState)
    wf.add_node("dispatcher", dispatcher_node)
    wf.set_entry_point("dispatcher")
    wf.add_edge("dispatcher", END)
    conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
    return wf.compile(checkpointer=SqliteSaver(conn))


if __name__ == "__main__":
    app = create_graph()
    for event in app.stream(
        {"question": "ciao", "chat_history": [], "agent_code": "AG001"},
        {"thread_id": "test"},
    ):
        pass
