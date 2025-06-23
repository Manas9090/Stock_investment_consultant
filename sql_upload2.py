import psycopg2
import csv

DB_CONFIG = {
    "dbname": "stock_db",
    "user": "stockuser",
    "password": "123456",
    "host": "localhost",
    "port": "5432"
}

def create_table():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS stock_symbols (
            id SERIAL PRIMARY KEY,
            company_name TEXT NOT NULL,
            symbol TEXT NOT NULL UNIQUE,
            exchange TEXT,
            country TEXT
        )
    """)
    conn.commit()
    cur.close()
    conn.close()
    print("✅ Table created.")

def insert_data_from_csv():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    with open("stock_symbols2.csv", "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            cur.execute("""
                INSERT INTO stock_symbols (company_name, symbol, exchange, country)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (symbol) DO NOTHING;
            """, (row['company_name'], row['symbol'], row['exchange'], row['country']))
    conn.commit()
    cur.close()
    conn.close()
    print("✅ Data inserted.")

if __name__ == "__main__":
    create_table()
    insert_data_from_csv()
