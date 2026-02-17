import sqlite3
import os

# 1. Trova la cartella 'scripts' dove si trova questo file
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Sali di un livello per arrivare a 'sql_lite' e poi entra in 'db'
DB_PATH = os.path.join(os.path.dirname(CURRENT_DIR), "db", "database_ordini.db")

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

    # Inserimento dati dummy
    clienti_dummy = [
        ("C001", "Mario Rossi S.r.l.", "Bar Mario", "AG001", "Via Roma 10", "Milano"),
        ("C002", "Pizzeria da Gigio di Luigi B.", "Gigio", "AG001", "Corso Italia 22", "Milano"),
        ("C003", "Bevande e Bollicine S.p.A.", "B&B", "AG002", "Zona Industriale", "Paderno"),
        ("C004", "Beck's Corner Pub", "Il Pubbe", "AG001", "Viale Abruzzi 101", "Milano"),
        ("C005", "Ristorante Peroni & Figli", "Da Peroni", "AG001", "Piazza Duomo 1", "Milano")
    ]
    cursor.executemany("INSERT INTO clienti_fts VALUES (?, ?, ?, ?, ?, ?)", clienti_dummy)

    conn.commit()
    conn.close()
    print("✅ Database configurato correttamente con tabella cart_item!")

if __name__ == "__main__":
    setup_full_database()