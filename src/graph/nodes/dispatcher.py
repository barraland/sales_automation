"""
Dispatcher node: chiama l'LLM con bind_tools, legge il messaggio dell'utente
e decide quali tool invocare. Restituisce la lista di tool_calls allo stato.
La funzione route_to_tools usa Send per il fan-out parallelo.
"""

import json
import os
from typing import List, Union

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Send

from src.graph.shared import (
    AgentState,
    DISPATCHER_TOOLS,
    FinalResponse,
    get_model,
    brand_values,
    categoria_values,
    sottocategoria_values,
)

_HISTORY_WINDOW = int(os.getenv("CHAT_HISTORY_WINDOW", "10"))


def dispatcher_node(state: AgentState) -> dict:
    agent_code  = state.get("agent_code", "")
    agent_nome  = state.get("agent_nome", "")
    known_clients  = state.get("known_clients") or {}
    question    = state.get("question") or ""
    chat_history = state.get("chat_history") or []

    # Clienti già risolti in sessione
    clients_str = "\n".join(
        f"  {cid}: {info.get('ragione_sociale') or info.get('alias') or cid}"
        f" ({info.get('citta', '')})"
        for cid, info in list(known_clients.items())[:20]
    ) or "  (nessuno risolto in questa sessione)"

    # Contesto operazioni in sospeso (supporta 1 o più pending_call)
    raw_pending   = state.get("pending_call")
    pending_calls = raw_pending if isinstance(raw_pending, list) else ([raw_pending] if raw_pending else [])
    pending_ctx   = ""
    if len(pending_calls) == 1:
        pc = pending_calls[0]
        pending_ctx = (
            f"\n\n⚠️ OPERAZIONE IN SOSPESO: {pc['name']}\n"
            f"Parametri già noti: {json.dumps(pc['args'], ensure_ascii=False)}\n"
            f"REGOLA CRITICA: se il messaggio è una selezione o risposta per questa operazione "
            f"(nome cliente, ragione sociale, numero, quantità, SKU), chiama SEMPRE e SOLO "
            f"{pc['name']} con i parametri aggiornati. "
            f"NON chiamare list_clients, search_products o altri tool.\n"
            f"→ Se l'utente cita un numero (es. 'la 1', 'il 2'), estrai lo SKU o client_id "
            f"corrispondente dall'ultima lista numerata in cronologia e usalo come parametro, "
            f"NON il nome testuale dell'operazione in sospeso.\n"
            f"→ Ignora solo se l'utente cambia esplicitamente argomento "
            f"(es. 'lascia perdere', 'voglio fare altro')."
        )
    elif len(pending_calls) > 1:
        letters = "ABCDEFGHIJ"
        lines = "\n".join(
            f"  {letters[i]}. {pc['name']} — {json.dumps(pc['args'], ensure_ascii=False)}"
            for i, pc in enumerate(pending_calls)
        )
        pending_ctx = (
            f"\n\n⚠️ OPERAZIONI IN SOSPESO ({len(pending_calls)}):\n{lines}\n"
            f"REGOLA CRITICA: risolvi TUTTE le operazioni in sospeso.\n"
            f"Se l'utente cita numeri (es. 'la 1', 'il 3', 'la 1 ed il 4'), quei numeri si "
            f"riferiscono agli elementi dell'ULTIMA LISTA NUMERATA in cronologia (prodotti o clienti), "
            f"NON alle lettere A/B/C delle operazioni qui sopra.\n"
            f"→ Estrai lo SKU o client_id corrispondente dalla lista in chat e usalo direttamente "
            f"come product_ref o client_ref — NON usare il nome testuale dall'operazione in sospeso.\n"
            f"→ Chiama un tool separato per ogni selezione dell'utente.\n"
            f"NON chiamare list_clients, search_products o altri tool aggiuntivi.\n"
            f"→ Ignora solo se l'utente cambia esplicitamente argomento "
            f"(es. 'lascia perdere', 'voglio fare altro')."
        )

    is_first = len(chat_history) == 0

    catalog_ctx = (
        f"\nCATALOGO — FILTRI DISPONIBILI PER search_products:\n"
        f"  Brand:          {', '.join(brand_values)}\n"
        f"  Categorie:      {', '.join(categoria_values)}\n"
        f"  Sottocategorie: {', '.join(sottocategoria_values)}\n"
        f"Usa questi valori esatti in filters_json quando l'utente filtra per brand o categoria.\n"
    )

    system_prompt = (
        f"Sei l'assistente vendite Horeca per {agent_nome or agent_code}.\n"
        f"Data e ora: {state.get('current_datetime') or ''}\n\n"
        f"CLIENTI RISOLTI IN SESSIONE:\n{clients_str}"
        f"{catalog_ctx}"
        f"{pending_ctx}\n\n"
        f"ISTRUZIONI:\n"
        f"- Chiama lo strumento appropriato in base al messaggio.\n"
        f"- Puoi chiamare più strumenti in parallelo se l'utente fa richieste multiple "
        f"(es. 'aggiungi per Mario E per Giuseppe', 'ordina 10 Ichnusa e 5 Martini').\n"
        f"- REGOLA: NON chiamare list_clients come step preliminare per add_to_cart / "
        f"remove_from_cart / clear_cart / view_cart / confirm_order. Quei tool trovano "
        f"il cliente da soli. Usa list_clients SOLO se l'utente chiede esplicitamente "
        f"la lista ('mostrami i clienti', 'clienti di Milano', ecc.).\n"
        f"- Se l'utente nomina un cliente e dei prodotti da aggiungere, chiama add_to_cart "
        f"per ogni prodotto direttamente — anche con typo nel nome, il tool troverà il cliente.\n"
        f"- Per confirm_order: SOLO su conferma esplicita ('sì', 'confermo', 'invia', 'procedi', 'manda').\n"
        f"- Se il messaggio fornisce una quantità o una selezione per un'operazione in sospeso, "
        f"riprendi quell'operazione con i parametri aggiornati.\n"
        f"- Se l'utente fa riferimento a un elemento per numero ('il 3', 'il cliente 22', 'il prodotto 1'), "
        f"cerca nella cronologia della chat la lista numerata corrispondente e usa l'ID reale "
        f"(client_id tipo C022, o SKU tipo BIR-HEI-CLA-33V) come parametro, NON il numero grezzo.\n"
        f"- Se nella cronologia vedi già client_id o SKU espliciti relativi alla richiesta corrente, "
        f"usali direttamente senza chiedere conferma.\n"
        f"- Per ricerca semantica di un prodotto specifico (es. 'trovami una birra leggera', 'Ichnusa non filtrata') usa search_products.\n"
        f"- Per domande su dati del database usa SEMPRE query_database — anche per elenchi, filtri, aggregazioni:\n"
        f"  'quali brand di birra?', 'prodotti sotto €2', 'quante birre abbiamo?',\n"
        f"  'clienti di Milano', 'totale ordini per cliente', 'prodotti disponibili > 100',\n"
        f"  'ordini di Bar Mario', 'quante Ichnusa ha ordinato Bar Mario questa settimana',\n"
        f"  'brand di succhi', 'birre in lattina', 'formati disponibili per Heineken'.\n"
        f"  → Se la domanda menziona un cliente specifico, popola client_hint con il nome del cliente.\n"
        f"  → Se la domanda menziona un prodotto specifico, popola product_hint con il nome del prodotto.\n"
        f"  Esempi: 'ordini di Bar Mario' → client_hint='Bar Mario';\n"
        f"          'quante Ichnusa ha ordinato Bar Mario' → client_hint='Bar Mario', product_hint='Ichnusa'.\n"
        f"- Usa free_response SOLO per: primo saluto, risposte fuori ambito, o quando i dati della "
        f"sezione CATALOGO sopra bastano a rispondere COMPLETAMENTE senza filtri (es. 'elenca tutte le categorie').\n"
        f"  Se c'è un filtro ('brand DI BIRRA', 'clienti DI MILANO') usa SEMPRE query_database.\n"
        f"- Per il primo messaggio usa free_response con un breve benvenuto a "
        f"{agent_nome or agent_code}.\n"
        f"- Per richieste fuori ambito usa free_response."
    )

    llm = get_model("generic").bind_tools(DISPATCHER_TOOLS)
    messages = (
        [SystemMessage(content=system_prompt)]
        + chat_history[-_HISTORY_WINDOW:]
        + [HumanMessage(content=question)]
    )

    print(f"\n💬 [DISPATCHER] {question[:80]}")
    ai_resp = llm.invoke(messages)

    # Fallback: LLM ha risposto con testo libero senza tool call
    if not getattr(ai_resp, "tool_calls", None):
        text = getattr(ai_resp, "content", "") or "Come posso aiutarti?"
        print(f"   ⚠️ Nessun tool call — risposta testo libero")
        return {
            "final_answer": FinalResponse(text=str(text)),
            "pending_call": None,
            "tool_calls": [],
            "tool_results": None,   # reset reducer
        }

    tool_calls = list(ai_resp.tool_calls)   # list of dicts: {name, args, id, type}
    print(f"   🔧 Tool calls: {[tc['name'] for tc in tool_calls]}")

    return {
        "tool_calls": tool_calls,
        "tool_results": None,   # reset reducer prima del fan-out
    }


def route_to_tools(state: AgentState) -> Union[str, List[Send]]:
    """
    Conditional edge: dispatcher → tool nodes (fan-out parallelo via Send).
    Se non ci sono tool_calls va direttamente a merge (final_answer già settato).
    """
    tool_calls = state.get("tool_calls") or []
    if not tool_calls:
        return "merge"
    # Fan-out: ogni tool call diventa un branch parallelo
    return [
        Send(tc["name"], {**state, "current_tool_call": tc})
        for tc in tool_calls
    ]
