"""
Popola la tabella `sconti` con il 20% delle combinazioni cliente×SKU,
scelto casualmente, con sconto randomico tra 5% e 30% (1 decimale).

Uso:
    python3 sql_lite/scripts/populate_sconti.py
    python3 sql_lite/scripts/populate_sconti.py --seed 42   # riproducibile

La tabella deve già esistere (creata da create_database_ordini._anagrafica_cli.py).
Eseguire questo script DOPO il setup del database.
"""

import argparse
import os
import random
import sqlite3

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(os.path.dirname(CURRENT_DIR), "db", "database_ordini.db")

COVERAGE = 0.20       # frazione di combinazioni da materializzare
SCONTO_MIN = 5.0      # sconto minimo %
SCONTO_MAX = 30.0     # sconto massimo %


def populate_sconti(seed: int = None):
    if seed is not None:
        random.seed(seed)
        print(f"🎲 Seed: {seed}")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Leggi tutti i client_id e SKU presenti
    client_ids = [r[0] for r in cursor.execute("SELECT DISTINCT client_id FROM clienti_fts").fetchall()]
    skus       = [r[0] for r in cursor.execute("SELECT sku FROM prodotti").fetchall()]

    if not client_ids or not skus:
        print("⚠️  Nessun cliente o prodotto trovato — eseguire prima il setup del database.")
        conn.close()
        return

    # Genera tutte le combinazioni e campiona il 20%
    all_combos   = [(c, s) for c in client_ids for s in skus]
    n_sample     = max(1, round(len(all_combos) * COVERAGE))
    sampled      = random.sample(all_combos, n_sample)

    print(f"📊 Combinazioni totali: {len(all_combos):,}  ({len(client_ids)} clienti × {len(skus)} SKU)")
    print(f"   Campione ({COVERAGE*100:.0f}%): {n_sample:,} righe da inserire")
    print(f"   Sconto: {SCONTO_MIN}% – {SCONTO_MAX}%")

    # Pulisce eventuali sconti precedenti e reinserisce
    cursor.execute("DELETE FROM sconti")

    rows = [
        (client_id, sku, round(random.uniform(SCONTO_MIN, SCONTO_MAX), 1))
        for client_id, sku in sampled
    ]
    cursor.executemany(
        "INSERT INTO sconti (client_id, sku, sconto_pct) VALUES (?, ?, ?)",
        rows,
    )
    conn.commit()

    # Statistiche
    stats = cursor.execute("""
        SELECT
            COUNT(*)                    AS n_righe,
            ROUND(AVG(sconto_pct), 2)   AS media_sconto,
            MIN(sconto_pct)             AS min_sconto,
            MAX(sconto_pct)             AS max_sconto
        FROM sconti
    """).fetchone()
    conn.close()

    print(f"✅ Inserite {stats[0]:,} righe — sconto medio: {stats[1]}%"
          f"  (min {stats[2]}%, max {stats[3]}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Popola tabella sconti con dati random")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed per riproducibilità (default: random)")
    args = parser.parse_args()
    populate_sconti(seed=args.seed)
