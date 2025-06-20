import psycopg2
import csv

DB_CONFIG = {
    "dbname": "postgres",
    "user": "postgres",
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
            symbol TEXT NOT NULL,
            exchange TEXT,
            country TEXT
        )
    """)
    conn.commit()
    cur.close()
    conn.close()
    print("✅ Table 'stock_symbols' created or already exists.")

def insert_data_from_csv():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    with open("company_tickers.csv", "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            cur.execute("""
                INSERT INTO stock_symbols (company_name, symbol, exchange, country)
                VALUES (%s, %s, %s, %s)
            """, (row['company_name'], row['symbol'], row['exchange'], row['country']))
    conn.commit()
    cur.close()
    conn.close()
    print("✅ Data loaded into 'stock_symbols' table.")  # fixed message

if __name__ == "__main__":
    create_table()
    insert_data_from_csv()
