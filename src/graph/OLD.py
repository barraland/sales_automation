import os
import json
import logging
import sqlite3
from typing import Any, Dict, List, Optional, Literal, Union, TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic import TypeAdapter

# Import Provider
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

# Import client e tool (Assicurati che i path siano corretti)
from src.tools.rag_tool import rag_beverage_search, get_search_client
from src.tools.rag_tool_anagrafica_clienti import search_client_smart

load_dotenv()
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("graph_app")


# =============================================================================
# FACTORY PER I MODELLI
# =============================================================================

def get_model(role: Literal["planner", "generic"], structured_schema: Any = None):
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
        return llm.with_structured_output(structured_schema)

    return llm


# =============================================================================
# SINCRONIZZAZIONE METADATI
# =============================================================================

def _get_distinct_values_azure(field: str) -> List[str]:
    client = get_search_client()
    vals = set()
    try:
        res = client.search(
            search_text="*",
            top=0,
            facets=[f"{field},count:1000"],
            select=['id']
        )
        facets = res.get_facets()
        if facets and field in facets:
            for facet_item in facets[field]:
                val = facet_item.get("value") if isinstance(facet_item, dict) else getattr(facet_item, "value", None)
                if val:
                    vals.add(str(val))
    except Exception as e:
        print(f"❌ Errore facet {field}: {e}")
    return sorted(list(vals))


def _load_catalog_meta():
    print("\n" + "=" * 50 + "\n🔄 SINCRONIZZAZIONE METADATI PRODOTTI\n" + "=" * 50)
    fields = ["brand", "categoria", "sottocategoria", "famiglia", "formato"]
    meta = {f: _get_distinct_values_azure(f) for f in fields}

    for k, v in meta.items():
        print(f"{'✅' if v else '⚠️'} {k.upper()}: {len(v)} valori caricati")

    print("=" * 50 + "\n")
    return meta


CATALOG_META = _load_catalog_meta()


# =============================================================================
# SCHEMI E STATO
# =============================================================================
class ClientArgs(BaseModel):
    query_text: Optional[str] = Field(None, description="Nome, alias o location")
    client_id: Optional[str] = Field(None, description="ID specifico")

class RAGArgs(BaseModel):
    query: str
    planner_motivation: str
    filters_json: str = ""
    top_k: int = Field(default=10)

class LLMArgs(BaseModel):
    question: str

class CartArgs(BaseModel):
    action: Literal["add", "remove", "view", "clear"]
    client_id: str
    sku: Optional[str] = None
    description: Optional[str] = None
    quantity: Optional[int] = 1
    price: Optional[float] = 0.0

from typing import Union
from pydantic import BaseModel, Field
from typing_extensions import Annotated

class SearchClientTask(BaseModel):
    tool: Literal["search_client_smart"]
    args: ClientArgs

class RagTask(BaseModel):
    tool: Literal["rag_beverage_search"]
    args: RAGArgs

class CartTask(BaseModel):
    tool: Literal["manage_cart"]
    args: CartArgs

class LLMAnswerTask(BaseModel):
    tool: Literal["llm_answer"]
    args: LLMArgs

Task = Annotated[
    Union[
        SearchClientTask,
        RagTask,
        CartTask,
        LLMAnswerTask
    ],
    Field(discriminator="tool")
]

class PlanStep(BaseModel):
    thought: str
    next_tasks: List[Task]
    is_finished: bool
    
class WhatsAppListItem(BaseModel):
    id: str
    title: str
    description: Optional[str]

class FinalResponse(BaseModel):
    text: str
    use_interactive_list: bool = False
    list_button_text: str = "Vedi opzioni"
    items: List[WhatsAppListItem] = []

class AgentState(TypedDict):
    question: str
    chat_history: List[BaseMessage]
    observations: Dict[str, Any]
    next_tasks: List[Task]
    iteration: int
    is_finished: bool
    final_answer: Optional[FinalResponse]
    agent_code: str
    selected_client: Optional[Dict]
    last_search_results: Optional[List[Dict]]


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
                DO UPDATE SET quantity = quantity + excluded.quantity
            """, (agent_code, args.client_id, args.sku, args.description, args.quantity, args.price))

            res = f"Aggiunto {args.quantity}x {args.description} al carrello."

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

        conn.commit()
        return res

    except Exception as e:
        print(f"💥 [SQL ERROR]: {e}")
        return f"Errore DB: {e}"

    finally:
        conn.close()


# =============================================================================
# NODI DEL GRAFO
# =============================================================================
# =============================================================================
# PLANNER NODE
# =============================================================================
def planner_node(state: AgentState):
    llm = get_model("planner", structured_schema=PlanStep)
    obs = state.get("observations", {})
    recent_products = state.get("last_search_results", [])
    selected_client = state.get("selected_client")
    agent_code = state.get("agent_code", "AG001")

    system_prompt = f"""
Sei l'Agente Commerciale AI esperto per il settore Horeca.

Il tuo compito è generare UN PIANO COMPLETO E DEFINITIVO per soddisfare la richiesta dell'agente umano {state['agent_code']}.

⚠️ REGOLA FONDAMENTALE:
Devi generare il piano UNA SOLA VOLTA.
Il piano deve essere completo e deve SEMPRE terminare con un task llm_answer.
Non devi ripianificare nei turni successivi.

----------------------------------------------------------------------
📌 STATO ATTUALE (FONTE DI VERITÀ)
----------------------------------------------------------------------

Observations (Risultati Tool già eseguiti):
{json.dumps(obs, indent=2)}

Task già presenti con stato:
{json.dumps([t.dict() for t in state.get("next_tasks", [])], indent=2)}

Cliente attualmente selezionato:
{json.dumps(selected_client) if selected_client else "NESSUNO"}

Cache prodotti recente:
{json.dumps(recent_products) if recent_products else "VUOTA"}

----------------------------------------------------------------------
🎯 OBIETTIVO
----------------------------------------------------------------------
Generare un piano atomico di task che:
1. Risolva eventuali ambiguità (cliente/prodotto)
2. Esegua le operazioni necessarie
3. Termini SEMPRE con llm_answer
4. Non generi loop o retry inutili

----------------------------------------------------------------------
🚫 REGOLE ANTI-LOOP (OBBLIGATORIE)
----------------------------------------------------------------------

1. NON puoi creare un task se ne esiste già uno con lo stesso scopo.
2. NON puoi rieseguire un task con status "failed".
3. Se un task critico è fallito, devi creare SOLO un task llm_answer.
4. NON puoi pianificare più di una rag_beverage_search per lo stesso prodotto.
5. Se nelle Observations esiste:
   - "fallimento_cliente_X"
   - "fallimento_prodotto_X"
   NON è consentito riprovare la stessa ricerca.
6. Se esistono task con status "pending", NON devi crearne di nuovi.
7. Il piano deve essere deterministico: niente retry automatici.

----------------------------------------------------------------------
🛒 REGOLA CRITICA CARRELLO
----------------------------------------------------------------------

Per usare manage_cart(action="add") devono essere noti:
- client_id
- sku

Se manca uno dei due:
❌ NON chiamare tool
❌ NON usare RAG
❌ NON usare search_client
✅ Devi creare solo un task llm_answer per chiedere il dato mancante

----------------------------------------------------------------------
🔍 LOGICA DI AZIONE
----------------------------------------------------------------------

1️⃣ CLIENTE

Se:
- Il cliente è già in selected_client → usalo
- È citato nel messaggio ma non noto → usa search_client_smart
- Non è citato e non è noto → chiedi "Per quale cliente vuoi ordinare?"

Se la ricerca cliente produce più risultati:
→ Non aggiungere al carrello
→ Termina con llm_answer e lista interattiva

Se la ricerca fallisce:
→ Termina con llm_answer comunicando che non è stato trovato

----------------------------------------------------------------------
2️⃣ PRODOTTO

Se:
- SKU già noto (da cache prodotti) → usalo
- Prodotto citato ma SKU non noto → UNA sola rag_beverage_search
- Prodotto non trovato → termina con llm_answer
- Risultati multipli → termina con llm_answer e lista interattiva

----------------------------------------------------------------------
3️⃣ MULTI-ORDINE

Se l’utente cita più clienti o prodotti:
- Genera un task per ciascuna entità mancante
- Puoi pianificare più manage_cart nello stesso piano
- Il piano deve comunque terminare con un solo llm_answer

----------------------------------------------------------------------
📦 STRUTTURA DEL PIANO
----------------------------------------------------------------------

Il piano deve essere una lista di task atomici con:
- id
- tool
- args
- status="pending"
- depends_on (se necessario)

⚠️ L’ULTIMO TASK DEVE ESSERE SEMPRE:
tool="llm_answer"

----------------------------------------------------------------------
🧠 GESTIONE ERRORI
----------------------------------------------------------------------

Se durante la pianificazione capisci che:
- Manca un dato essenziale
- Una ricerca è già fallita
- Non hai informazioni sufficienti

NON tentare di indovinare.
NON fare retry.
Crea solo un task llm_answer chiaro e professionale.

----------------------------------------------------------------------
⚠️ DIVIETI ASSOLUTI
----------------------------------------------------------------------

- Vietato ripianificare
- Vietato retry automatici
- Vietato aggiungere al carrello senza client_id e sku
- Vietato rieseguire task failed
- Vietato concludere senza llm_answer

----------------------------------------------------------------------

Genera ora il piano completo.
"""


    try:
        plan_raw = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=state["question"])
        ])

        # 🔹 Validazione output
        from pydantic import TypeAdapter
        plan_adapter = TypeAdapter(PlanStep)
        plan = plan_adapter.validate_python(plan_raw)

        print(f"DEBUG PLANNER: is_finished={plan.is_finished} | tasks_count={len(plan.next_tasks)}")
        print(f"📋 PIANO: {[{'tool': t.tool} for t in plan.next_tasks]}")

        return {
            "next_tasks": plan.next_tasks,
            "iteration": state.get("iteration", 0) + 1,
            "is_finished": plan.is_finished
        }

    except Exception as e:
        print(f"💥 ERRORE PLANNER: {e}")
        return {
            "next_tasks": [],
            "iteration": state.get("iteration", 0) + 1,
            "is_finished": False
        }

# =============================================================================
# EXECUTOR NODE
# =============================================================================

def executor_node(state: AgentState):
    print(f"\n{'#'*70}")
    print(f"📥 [INPUT EXECUTOR]")

    sel_client = state.get('selected_client')
    client_name = sel_client.get('ragione_sociale', 'NESSUNO') if isinstance(sel_client, dict) else 'NESSUNO'
    agent_code = state.get("agent_code", "AG001")

    # --- Ricostruisci Task Pydantic se arrivano come dict ---
    from pydantic import TypeAdapter
    task_adapter = TypeAdapter(Task)
    tasks = [
        task_adapter.validate_python(t) if isinstance(t, dict) else t
        for t in state.get("next_tasks", [])
    ]

    print(f" - Numero Task Ricevuti: {len(tasks)}")
    try:
        tasks_preview = [t.model_dump() for t in tasks]
        print(f" - Dettaglio Task: {json.dumps(tasks_preview, indent=2)}")
    except Exception as e:
        print(f" ⚠️ Errore nel log dei task: {e}")

    print(f"{'#'*70}\n")

    # --- Stato iniziale ---
    obs = state.get("observations", {})
    new_client = state.get("selected_client")
    new_search_results = state.get("last_search_results", [])
    updated_tasks = []

    # ============================================================
    # LOOP TASK
    # ============================================================
    for task in tasks:
        print(f"⚙️ ESECUZIONE: {task.tool} (ID: {task.id})")

        if getattr(task, "status", "pending") in ["success", "failed"]:
            print(f" ℹ️ Task già processato (status={task.status})")
            updated_tasks.append(task)
            continue

        try:
            # ---------------------
            # 0️⃣ LLM ANSWER
            # ---------------------
            if task.tool == "llm_answer":
                print(" ℹ️ llm_answer marcato come success (gestito dal responder)")
                task.status = "success"
                updated_tasks.append(task)
                continue

            # ---------------------
            # 1️⃣ SEARCH CLIENT
            # ---------------------
            elif task.tool == "search_client_smart":
                q_text = getattr(task.args, 'query_text', None)
                c_id = getattr(task.args, 'client_id', None)

                print(f" 🔍 Ricerca cliente: '{q_text}'")
                res = search_client_smart(agent_code=agent_code, query_text=q_text, client_id=c_id)

                if not res:
                    msg = f"Nessun cliente trovato per '{q_text}'"
                    obs[task.id] = {"error": msg}
                    obs[f"fallimento_cliente_{str(q_text).replace(' ', '_')}"] = msg
                    task.status = "failed"
                    print(f" ❌ {msg}")
                else:
                    res_data = {"results": res}
                    obs[task.id] = res_data
                    if q_text:
                        obs[f"risultato_cliente_{q_text.replace(' ', '_')}"] = res_data
                    if isinstance(res, list) and len(res) == 1:
                        new_client = res[0]
                        print(f" ✅ MATCH CLIENTE: {new_client.get('ragione_sociale')}")
                    else:
                        print(f" ℹ️ Trovati {len(res)} clienti")
                    task.status = "success"

            # ---------------------
            # 2️⃣ RAG BEVERAGE SEARCH
            # ---------------------
            elif task.tool == "rag_beverage_search":
                print(f" 🔍 Ricerca prodotto: '{task.args.query}'")
                res = rag_beverage_search.invoke(task.args.model_dump())

                if not res or not isinstance(res, dict) or not res.get("results"):
                    msg = f"Prodotto '{task.args.query}' non trovato"
                    obs[task.id] = {"error": msg}
                    obs[f"fallimento_prodotto_{task.args.query.replace(' ', '_')}"] = msg
                    task.status = "failed"
                    print(f" ❌ {msg}")
                else:
                    obs[task.id] = res
                    obs[f"risultato_prodotto_{task.args.query.replace(' ', '_')}"] = res

                    products = res.get("results", [])
                    new_search_results = []
                    for p in products:
                        meta = p.get("metadata", {})
                        new_search_results.append({
                            "sku": meta.get("sku") or p.get("id"),
                            "description": meta.get("product_name") or p.get("content", "")[:100],
                            "price": meta.get("prezzo_listino") or meta.get("price") or 0.0
                        })
                    print(f" ✅ PRODOTTI TROVATI: {len(new_search_results)}")
                    task.status = "success"

            # ---------------------
            # 3️⃣ MANAGE CART
            # ---------------------
            elif task.tool == "manage_cart":
                print(f" 🛒 Azione Carrello: {task.args.action}")
                res = sql_manage_cart(agent_code=agent_code, args=task.args)
                obs[task.id] = res

                if res and "Errore" not in str(res):
                    print(" ✅ Operazione carrello completata")
                    task.status = "success"
                else:
                    print(" ❌ Errore operazione carrello")
                    task.status = "failed"

            # ---------------------
            # TOOL NON RICONOSCIUTO
            # ---------------------
            else:
                print(f" ⚠️ TOOL NON RICONOSCIUTO: {task.tool}")
                task.status = "failed"

        except Exception as e:
            print(f" 💥 ERRORE TASK: {e}")
            obs[task.id] = {"error": str(e)}
            task.status = "failed"

        updated_tasks.append(task)

    # ============================================================
    # OUTPUT STATE
    # ============================================================
    output_to_state = {
        "observations": obs,
        "selected_client": new_client,
        "last_search_results": new_search_results,
        "next_tasks": updated_tasks
    }

    print(f"\n{'#'*70}")
    print(f"📤 [OUTPUT EXECUTOR]")
    print(f" - Totale chiavi Observations: {len(obs)}")
    print(f" - Status Task: {[ (t.id, t.status) for t in updated_tasks ]}")
    print(f"{'#'*70}\n")

    return output_to_state


# =============================================================================
# RESPONDER NODE
# =============================================================================

def responder_node(state: AgentState):

    # 1️⃣ Check continuità:
    # Se il planner non ha finito e non esiste un task llm_answer,
    # significa che servono altri giri di tool.
    is_finished_planner = state.get("is_finished", False)
    has_llm_answer = any(
        t.tool == "llm_answer"
        for t in state.get("next_tasks", [])
    )

    if not is_finished_planner and not has_llm_answer:
        return {"is_finished": False}

    # 2️⃣ Inizializza LLM con schema strutturato
    llm = get_model("generic", structured_schema=FinalResponse)

    # 3️⃣ Analisi osservazioni reali
    obs = state.get("observations", {})
    client_info = state.get("selected_client")

    # Verifica INSERT riuscita
    cart_actions = [
        v for v in obs.values()
        if "Aggiunto" in str(v)
    ]
    was_added = len(cart_actions) > 0

    # Verifica presenza liste (clienti/prodotti multipli)
    found_lists = [
        v for v in obs.values()
        if isinstance(v, list)
        or (isinstance(v, dict) and "results" in v)
    ]

    # 4️⃣ System grounding (vincola il modello ai fatti reali)
    system_msg = f"""
Sei un assistente commerciale per WhatsApp.
Il tuo compito è riferire SOLO ciò che è stato effettivamente eseguito dai tool.

STATO REALE (Fonte di Verità):
- Cliente Selezionato: {json.dumps(client_info) if client_info else "NESSUNO"}
- Prodotti aggiunti al DB: {"SÌ" if was_added else "NO"}
- Risultati Tool (Observations):
{json.dumps(obs, indent=2)}

REGOLE MANDATORIE:

1. NON DIRE MAI "Ho aggiunto al carrello" se 'Prodotti aggiunti al DB' è NO.
2. Se l'operazione è fallita perché manca il cliente,
   chiedi: "Per quale cliente vuoi ordinare?"
3. Se il Planner ha cercato un cliente ma ne sono usciti molti,
   usa use_interactive_list=True per elencarli.
4. Se esistono risultati multipli di prodotti,
   usa use_interactive_list=True.
5. Sii sintetico e professionale.
6. Usa il grassetto per i prodotti.
7. Se vedi un errore SQLite nelle osservazioni,
   riferisci che c'è stato un problema tecnico.
8. Non inventare mai dati non presenti nelle Observations.
"""

    # 5️⃣ Invocazione modello
    res = llm.invoke(
        [SystemMessage(content=system_msg)]
        + state.get("chat_history", [])[-5:]  # ultimi 5 messaggi
        + [HumanMessage(content=state["question"])]
    )

    # 6️⃣ Debug log interno
    print("\n" + "📱" + "=" * 50)
    print(f"VERIFICA INTERNA:")
    print(f" - Was Added? {was_added}")
    print(f" - Has Client? {client_info is not None}")
    print(f" - Liste trovate? {len(found_lists)}")
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
        lambda s: "end" if s.get("is_finished") else "continue",
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
        "selected_client": None
    }

    for event in app.stream(input_data, {"thread_id": "1"}):
        pass
