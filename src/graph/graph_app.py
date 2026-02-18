import os
import json
import logging
import sqlite3
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
    quantity: Optional[int] = 1

class SearchProductArgs(BaseModel):
    placeholder: str   
    query: str
    filters_json: str = ""
    top_k: int = 10

# =============================================================================
# TASK E PIANO
# =============================================================================
class Task(BaseModel):
    id: str
    tool: Literal[
        "search_client", 
        "search_product", 
        "manage_cart",
    ]
    args: Union[SearchClientArgs, CartArgs, SearchProductArgs]
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
                INSERT INTO cart_item (agent_id, client_id, sku, quantity)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(agent_id, client_id, sku)
                DO UPDATE SET quantity = quantity + excluded.quantity
            """, (agent_code, args.client_id, args.sku, args.quantity or 1))

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

# =============================================================================
# PLANNER NODE
# =============================================================================
def planner_node(state: AgentState):
    #llm = get_model("planner", structured_schema=Plan)  # rimosso 'method' se get_model non lo supporta
    llm = get_model("planner", structured_schema=Plan, method="function_calling")

    agent_code = getattr(state, "agent_code", "AG001")
    known_clients = getattr(state, "known_clients", {})
    known_products = getattr(state, "known_products", {})
    
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

----------------------------------------------------------------------
📌 STRUTTURA DEI TASK E TOOLS
----------------------------------------------------------------------

- Genera un piano strutturato in task, dove ogni task ha: ID univoco, tool specifico, args corretti, deps verso task da completare prima, status iniziale "pending".
----------------------------------------------------------------------

1️⃣ **search_client** — args: SearchClientArgs(placeholder, query_text)
- Cerca il client_id di un cliente non ancora noto nel database.
- NON usare se il client_id è già in "Lista clienti già risolti".

2️⃣ **search_product** — args: SearchProductArgs(placeholder, query, filters_json, top_k)
- Cerca lo sku di un prodotto non ancora noto (ricerca semantica RAG).
- top_k: usa 10 per ricerche puntuali.
- NON usare se lo sku è già in "Lista prodotti già risolti".
- NON dipende mai da search_client — cliente e prodotto si cercano sempre in parallelo.

3️⃣ **manage_cart** — args: CartArgs(action, client_id, sku, quantity)
- action: "add" | "remove" | "view" | "clear"
- Pianifica quando l'utente vuole ordinare, rimuovere, visualizzare o svuotare il carrello.
  L'utente non usa necessariamente il termine "carrello" — interpreta l'intento.
- client_id: usa ID reale da "Lista clienti già risolti" o placeholder di search_client.
- sku: usa SKU reale da "Lista prodotti già risolti" o placeholder di search_product.
- Per "view" / "clear": usa il client_id del cliente menzionato più di recente in conversazione.
  Solo se non determinabile, crea search_client per chiederlo.
- deps: includi search_client se client_id mancante; includi search_product se sku mancante.
  Se entrambi già noti → deps vuoti.

----------------------------------------------------------------------
📌 REGOLE DI PIANIFICAZIONE
----------------------------------------------------------------------

- Crea task separati per ogni cliente o prodotto citato.
- Usa placeholder deterministici solo per dati non noti (es. `ph_cliente_gigio`, `ph_prodotto_ichnusa`).
- Dipendenze: search_client → manage_cart (solo se client_id mancante); search_product → manage_cart (solo se sku mancante). search_client e search_product NON si dipendono mai.
- Non creare search_client o search_product per dati già presenti nello stato.

----------------------------------------------------------------------
📌 STATO ATTUALE
----------------------------------------------------------------------

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
                    raw = search_product_smart.invoke({
                        "query": task.args.query,
                        "planner_motivation": "",
                        "filters_json": task.args.filters_json,
                        "top_k": task.args.top_k,
                    })
                    # Il tool restituisce {"results": [...], "metadata": {...}}
                    results = raw.get("results", []) if isinstance(raw, dict) else []

                    if not results:
                        task.status = "failed"
                        obs[task.id] = {"error": f"Nessun prodotto trovato per '{task.args.query}'"}
                        print(f" ❌ Nessun prodotto trovato: {task.args.query}")
                    else:
                        # Aggiorna known_products con tutti i risultati trovati
                        for r in results:
                            sku = r.get("sku")
                            if sku:
                                updated_known_products[sku] = {
                                    "descrizione": r.get("descrizione"),
                                    "brand": r.get("brand"),
                                    "formato": r.get("formato"),
                                }

                        if len(results) == 1 and getattr(task.args, "placeholder", None):
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
                elif task.tool == "manage_cart":
                    print(
                        f" 🛒 Azione: {task.args.action} | "
                        f"client_id={task.args.client_id} | sku={task.args.sku}"
                    )
                    res = sql_manage_cart(agent_code=agent_code, args=task.args)
                    obs[task.id] = res
                    if res and "Errore" not in str(res):
                        print(" ✅ Operazione carrello completata")
                        task.status = "success"
                    else:
                        print(" ❌ Errore operazione carrello")
                        task.status = "failed"

                else:
                    print(f" ⚠️ TOOL NON RICONOSCIUTO: {task.tool}")
                    task.status = "failed"

        except Exception as e:
            print(f" 💥 ERRORE TASK: {e}")
            obs[task.id] = {"error": str(e)}
            task.status = "failed"

    # -------------------------------------------------------------------------
    # OUTPUT
    # -------------------------------------------------------------------------
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

    # Verifica INSERT riuscita
    cart_actions = [v for v in obs.values() if "Aggiunto" in str(v)]
    was_added = len(cart_actions) > 0

    # Verifica presenza liste (clienti/prodotti multipli da disambiguare)
    found_lists = [
        v for v in obs.values()
        if isinstance(v, dict) and v.get("pending_selection")
    ]

    # 4️⃣ System grounding (vincola il modello ai fatti reali)
    system_msg = f"""
Sei un assistente commerciale per WhatsApp.
Il tuo compito è riferire SOLO ciò che è stato effettivamente eseguito dai tool.

STATO REALE (Fonte di Verità):
- Prodotti/clienti aggiunti al DB: {"SÌ" if was_added else "NO"}
- Risultati Tool (Observations):
{json.dumps(obs, indent=2, default=str)}

REGOLE MANDATORIE:

1. NON DIRE MAI "Ho aggiunto al carrello" se 'Prodotti aggiunti al DB' è NO.
2. Se l'operazione è fallita perché manca il cliente,
   chiedi: "Per quale cliente vuoi ordinare?"
3. Se ci sono risultati con 'pending_selection: true' per clienti,
   usa use_interactive_list=True per elencarli.
4. Se ci sono risultati con 'pending_selection: true' per prodotti,
   usa use_interactive_list=True per elencarli.
5. Sii sintetico e professionale.
6. Usa il grassetto per i prodotti.
7. Se vedi un errore SQLite nelle osservazioni,
   riferisci che c'è stato un problema tecnico.
8. Non inventare mai dati non presenti nelle Observations.
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
