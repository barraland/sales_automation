import csv
import json as _json
import sqlite3
import os

# 1. Trova la cartella 'scripts' dove si trova questo file
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Sali di un livello per arrivare a 'sql_lite' e poi entra in 'db'
DB_PATH   = os.path.join(os.path.dirname(CURRENT_DIR), "db", "database_ordini.db")
DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(CURRENT_DIR)), "data")

def setup_full_database():
    print(f"🔧 Creazione/Aggiornamento database in: {DB_PATH}")
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON;")

    # 1. CLIENTI (FTS5)
    cursor.execute("DROP TABLE IF EXISTS clienti_fts")
    cursor.execute("""
        CREATE VIRTUAL TABLE clienti_fts USING fts5(
            client_id UNINDEXED, 
            ragione_sociale, 
            alias, 
            agent_id UNINDEXED, 
            indirizzo, 
            citta,
            tokenize='unicode61'
        )
    """)

    # 2. TABELLA ORDER (Testata)
    # created_at è in formato ISO 8601: 'YYYY-MM-DD HH:MM:SS' (UTC)
    # Usata per filtrare ordini per data/ora (es. "ordini di oggi", "ultimi 5 min")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS [order] (
            order_id INTEGER PRIMARY KEY AUTOINCREMENT,
            client_id TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            status TEXT DEFAULT 'RECEIVED',
            total_amount REAL DEFAULT 0.0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            note TEXT
        )
    """)
    # Indice per query efficienti su data/ora
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_order_created_at
        ON [order](created_at)
    """)

    # 3. TABELLA ORDER_ITEM (Righe)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS order_item (
            item_id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id INTEGER NOT NULL,
            sku TEXT NOT NULL,         
            description TEXT,          
            quantity INTEGER NOT NULL,
            price_at_order REAL,       
            FOREIGN KEY(order_id) REFERENCES [order](order_id) ON DELETE CASCADE
        )
    """)

    # 4. TABELLA CART_ITEM (Sandbox)
    # È fondamentale che questa tabella esista per l'azione 'add'
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cart_item (
            cart_item_id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id TEXT NOT NULL,
            client_id TEXT NOT NULL,
            sku TEXT NOT NULL,
            description TEXT,
            quantity INTEGER DEFAULT 0,
            price REAL,
            added_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(agent_id, client_id, sku) 
        )
    """)

    # Inserimento clienti da CSV
    csv_path = os.path.join(os.path.dirname(os.path.dirname(CURRENT_DIR)), "data", "clienti.csv")
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        clienti = [
            (row["client_id"], row["ragione_sociale"], row["alias"],
             row["agent_id"], row["indirizzo"], row["citta"])
            for row in reader
        ]
    cursor.executemany("INSERT INTO clienti_fts VALUES (?, ?, ?, ?, ?, ?)", clienti)
    print(f"   → {len(clienti)} clienti caricati da clienti.csv")

    # 5. TABELLA PRODOTTI (catalogo bevande per query strutturate)
    setup_prodotti(conn)

    # 6. TABELLA SCONTI (sconti cliente×SKU negoziati)
    cursor.execute("DROP TABLE IF EXISTS sconti")
    cursor.execute("""
        CREATE TABLE sconti (
            client_id   TEXT NOT NULL,
            sku         TEXT NOT NULL,
            sconto_pct  REAL NOT NULL CHECK(sconto_pct >= 0 AND sconto_pct <= 100),
            PRIMARY KEY (client_id, sku)
        )
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_sconti_client
        ON sconti(client_id)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_sconti_sku
        ON sconti(sku)
    """)

    conn.commit()
    conn.close()
    print("✅ Database configurato correttamente!")


def setup_prodotti(conn):
    """Crea e popola la tabella prodotti da data/anagrafica_bevande.json."""
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS prodotti")
    cursor.execute("""
        CREATE TABLE prodotti (
            sku TEXT PRIMARY KEY,
            descrizione TEXT,
            brand TEXT,
            categoria TEXT,
            sottocategoria TEXT,
            famiglia TEXT,
            formato TEXT,
            confezione TEXT,
            prezzo_unitario REAL,
            prezzo_confezione TEXT,
            disponibilita REAL,
            note_ricerca TEXT
        )
    """)
    json_path = os.path.join(DATA_ROOT, "anagrafica_bevande.json")
    with open(json_path, encoding="utf-8") as f:
        products = [_json.loads(line) for line in f if line.strip()]
    cursor.executemany(
        """INSERT INTO prodotti
           (sku, descrizione, brand, categoria, sottocategoria, famiglia,
            formato, confezione, prezzo_unitario, prezzo_confezione, disponibilita, note_ricerca)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [(p.get("sku"), p.get("descrizione"), p.get("brand"), p.get("categoria"),
          p.get("sottocategoria"), p.get("famiglia"), p.get("formato"), p.get("confezione"),
          p.get("prezzo_unitario"), p.get("prezzo_confezione"), p.get("disponibilita"),
          p.get("note_ricerca")) for p in products]
    )
    print(f"   → {len(products)} prodotti caricati da anagrafica_bevande.json")


if __name__ == "__main__":
    setup_full_database()