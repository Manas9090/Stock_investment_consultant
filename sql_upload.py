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
        CREATE TABLE IF NOT EXISTS company_tickers (
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
    print("✅ Table 'company_tickers' created or already exists.")

def insert_data_from_csv():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    with open("company_tickers.csv", "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            cur.execute("""
                INSERT INTO company_tickers (company_name, symbol, exchange, country)
                VALUES (%s, %s, %s, %s)
            """, (row['company_name'], row['symbol'], row['exchange'], row['country']))
    conn.commit()
    cur.close()
    conn.close()
    print("✅ Data loaded into 'company_tickers' table.") 

if __name__ == "__main__":
    create_table()
    insert_data_from_csv()
