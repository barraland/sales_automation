## src.tools.rag_tool_anagrfica_clienti.py
import sqlite3
import os
import re

# Determiniamo il percorso assoluto del DB per evitare "unable to open database file"
# Si assume che il file si trovi in: /sql_lite/db/database_ordini.db rispetto alla radice
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(BASE_DIR, "sql_lite", "db", "database_ordini.db")

def search_client_smart(agent_code: str, query_text: str = None, client_id: str = None, city_filter: str = None):
    """
    Ricerca intelligente dei clienti con supporto FTS5 e gestione errori di sintassi.
    """
    if not agent_code:
        return []

    # Verifica preventiva esistenza file
    if not os.path.exists(DB_PATH):
        print(f"❌ Errore: Database non trovato in {DB_PATH}")
        return []

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row 
    cursor = conn.cursor()

    try:
        # 1. COSTRUZIONE QUERY BASE (Schema aggiornato senza 'destinazioni')
        base_query = """
            SELECT client_id, ragione_sociale, alias, indirizzo, citta 
            FROM clienti_fts 
        """
        
        # 2. GESTIONE FILTRI LOGICI
        if client_id:
            # CASO A: Selezione diretta tramite ID (es. da bottone WhatsApp)
            cursor.execute(f"{base_query} WHERE client_id = ? AND agent_id = ?", (client_id, agent_code))

        elif city_filter and not query_text:
            # CASO B1: Filtro solo per città (es. "clienti di Milano")
            # FTS5 column-specific search: 'citta : Milano*'
            clean_city = re.sub(r'[^\w\s]', ' ', city_filter).strip()
            fts_city = f"citta : {clean_city}*"
            cursor.execute(f"""
                {base_query}
                WHERE clienti_fts MATCH ? AND agent_id = ?
                LIMIT 50
            """, (fts_city, agent_code))

        elif city_filter and query_text:
            # CASO B2: Filtro per città + testo (es. "bar di Milano")
            clean_city = re.sub(r'[^\w\s]', ' ', city_filter).strip()
            clean_query = re.sub(r'[^\w\s]', ' ', query_text).strip()
            words = [f"{w}*" for w in clean_query.split() if w]
            if not words:
                fts_query = f"citta : {clean_city}*"
            else:
                fts_query = " AND ".join(words) + f" AND citta : {clean_city}*"
            cursor.execute(f"""
                {base_query}
                WHERE clienti_fts MATCH ? AND agent_id = ?
                LIMIT 20
            """, (fts_query, agent_code))

        elif not query_text or query_text.strip() == "*":
            # CASO C: Lista generica (fallback)
            cursor.execute(f"{base_query} WHERE agent_id = ? LIMIT 50", (agent_code,))

        else:
            # CASO D: Ricerca Full-Text con pulizia per evitare crash (syntax error near ".")
            # Rimuoviamo tutto ciò che non è alfanumerico o spazio
            clean_query = re.sub(r'[^\w\s]', ' ', query_text).strip()

            # Trasformiamo in query FTS5 valida (es: "Mario Rossi" -> "Mario* AND Rossi*")
            words = [f"{w}*" for w in clean_query.split() if w]

            if not words:
                # Se dopo la pulizia non rimane nulla (solo simboli), restituiamo vuoto
                return []

            fts_query = " AND ".join(words)

            cursor.execute(f"""
                {base_query}
                WHERE clienti_fts MATCH ? AND agent_id = ?
                LIMIT 20
            """, (fts_query, agent_code))

        rows = cursor.fetchall()
        results = []
        
        for row in rows:
            # Creiamo un campo sintetico per l'AI per facilitare la risposta testuale
            full_addr = f"{row['indirizzo']}, {row['citta']}".strip(", ")
            
            results.append({
                "client_id": row["client_id"],
                "ragione_sociale": row["ragione_sociale"],
                "alias": row["alias"],
                "indirizzo": row["indirizzo"],
                "citta": row["citta"],
                "full_address": full_addr
            })

        return results

    except sqlite3.Error as e:
        print(f"❌ Errore SQLite ricerca clienti: {e}")
        return []
    except Exception as e:
        print(f"❌ Errore generico ricerca clienti: {e}")
        return []
    finally:
        conn.close()

# =============================================================================
# MAIN DI TEST AGGIORNATO
# =============================================================================
if __name__ == "__main__":
    if not os.path.exists(DB_PATH):
        print(f"⚠️ DB non trovato in {DB_PATH}")
    else:
        print("🚀 TEST RICERCA CLIENTI (SCHEMA NUOVO)")
        # Test: Cerca i clienti di Milano
        res = search_client_smart("AG001", query_text="Milano")
        for c in res:
            print(f"- {c['ragione_sociale']} ({c['alias']}) -> {c['full_address']}")