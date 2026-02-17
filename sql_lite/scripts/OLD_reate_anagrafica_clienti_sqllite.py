import sqlite3
import json
import os

DB_PATH = "../db/database_ordini.db"

def setup_client_database():
    # Assicuriamoci che la cartella esista
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS clienti_fts")

    # MODIFICA: 'destinazioni' NON deve essere UNINDEXED se vuoi cercarci dentro
    cursor.execute("""
        CREATE VIRTUAL TABLE clienti_fts USING fts5(
            client_id UNINDEXED, 
            ragione_sociale, 
            alias, 
            agent_id UNINDEXED, 
            destinazioni, 
            tokenize='unicode61'
        )
    """)

    # Dati Dummy - Nota: inseriamo gli indirizzi come testo leggibile
    clienti_dummy = [
        ("C001", "Mario Rossi S.r.l.", "Bar Mario", "AG001", 
         "Via Roma 10 Milano, Via Milano 5 Monza"),
        
        ("C002", "Pizzeria da Gigio di Luigi B.", "Gigio", "AG001", 
         "Corso Italia 22 Milano"),
        
        ("C003", "Bevande e Bollicine S.p.A.", "B&B", "AG002", 
         "Zona Industriale Paderno"),
        
        ("C004", "Beck's Corner Pub", "Il Pubbe", "AG001", 
         "Viale Abruzzi 101 Milano"),

        ("C005", "Ristorante Peroni & Figli", "Da Peroni", "AG001", 
         "Piazza Duomo 1 Milano")
    ]

    cursor.executemany("""
        INSERT INTO clienti_fts (client_id, ragione_sociale, alias, agent_id, destinazioni)
        VALUES (?, ?, ?, ?, ?)
    """, clienti_dummy)

    conn.commit()
    conn.close()
    print("✅ Database SQLite FTS5 pronto e SEARCHABLE su indirizzi!")

def search_client(query: str, agent_id: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Trasformiamo "Mario Milano" in "Mario* AND Milano*"
    formatted_query = " AND ".join([f"{word}*" for word in query.split()])

    print(f"\n🔍 RICERCA: '{query}' | AGENTE: {agent_id}")
    
    try:
        cursor.execute("""
            SELECT client_id, ragione_sociale, alias, destinazioni
            FROM clienti_fts 
            WHERE clienti_fts MATCH ? AND agent_id = ?
            ORDER BY rank
            LIMIT 5
        """, (formatted_query, agent_id))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "id": row[0],
                "ragione_sociale": row[1],
                "alias": row[2],
                "destinazioni": row[3]
            })
            print(f"   - [{row[0]}] {row[1]} | 📍 {row[3]}")
        
        if not results: print("⚠️ Nessun match.")
        return results
    except Exception as e:
        print(f"❌ Errore: {e}")
        return []
    finally:
        conn.close()

if __name__ == "__main__":
    setup_client_database()
    
    # ORA QUESTO FUNZIONERÀ!
    search_client("Milano", "AG001")
    search_client("Mario Roma", "AG001")