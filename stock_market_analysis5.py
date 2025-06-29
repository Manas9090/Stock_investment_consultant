import openai
import requests
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import faiss
import streamlit as st
import datetime
from pymongo import MongoClient

# --- API Keys ---
openai.api_key = st.secrets["api_keys"]["openai"]
twelvedata_api_key = st.secrets["api_keys"]["twelvedata"]
news_api_key = st.secrets["api_keys"]["newsapi"]
alpha_vantage_api_key = st.secrets["api_keys"]["alphavantage"]

# --- Load Embedding Model ---
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# --- MongoDB Config ---
MONGO_URI = st.secrets["database"]["mongo_uri"]
DB_NAME = "my_database"
COLLECTION_NAME = "my_collection"

# Ensure secure connection with tls=True
client = MongoClient(MONGO_URI, tls=True, tlsAllowInvalidCertificates=False)
collection = client[DB_NAME][COLLECTION_NAME]

# --- Fetch Company Symbol & Exchange from MongoDB ---
def fetch_ticker_from_db(company_name):
    doc = collection.find_one({"company_name": {"$regex": f"^{company_name}$", "$options": "i"}})
    if doc:
        return doc.get("symbol"), doc.get("exchange")
    return None

# --- Match Query with Company in MongoDB ---
def match_company_in_db(user_query):
    all_companies = collection.distinct("company_name")
    for name in all_companies:
        if name.lower() in user_query.lower():
            return name
    return None

# --- Fetch Live News ---
def fetch_live_news(company):
    url = f"https://newsapi.org/v2/everything?q={company}&sortBy=publishedAt&language=en&apiKey={news_api_key}"
    response = requests.get(url)
    articles = response.json().get("articles", [])
    return [f"{a['title']}. {a.get('description', '')}" for a in articles if a.get('title')]

# --- Build FAISS Index ---
def build_faiss_index(corpus):
    embeddings = embedding_model.encode(corpus)
    embeddings = normalize(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, corpus

# --- Historical Data from Alpha Vantage ---
def get_alpha_vantage_data(symbol):
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "apikey": alpha_vantage_api_key,
        "outputsize": "full"
    }
    r = requests.get(url, params=params)
    data = r.json()
    if "Time Series (Daily)" not in data:
        raise ValueError(f"Alpha Vantage error: {data}")
    df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index')
    df.rename(columns={
        '1. open': 'open', '2. high': 'high', '3. low': 'low',
        '4. close': 'close', '5. volume': 'volume'
    }, inplace=True)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df.last("6M")[['close']].astype(float)

# --- Historical Data from Twelve Data ---
def get_twelve_data(symbol, exchange):
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=180)
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": "1day",
        "start_date": start_date.date(),
        "end_date": end_date.date(),
        "outputsize": 500,
        "apikey": twelvedata_api_key
    }
    if exchange:
        params["exchange"] = exchange
    response = requests.get(url, params=params)
    data = response.json()
    if "values" not in data:
        raise ValueError(f"Twelve Data error: {data}")
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)
    return df[["close"]].astype(float)

# --- Generate Investment Insight ---
def get_stock_insight(query, corpus, index, symbol, hist_df):
    query_embedding = embedding_model.encode([query])
    query_embedding = normalize(query_embedding)
    D, I = index.search(query_embedding, k=3)
    context = "\n".join([corpus[i] for i in I[0]])

    latest_price = hist_df["close"].iloc[-1]
    past_price = hist_df["close"].iloc[0]
    change_pct = ((latest_price - past_price) / past_price) * 100

    trend_context = (
        f"The current stock price of {symbol} is {latest_price:.2f}. "
        f"It changed from {past_price:.2f} over the past 6 months, a {change_pct:.2f}% move."
    )

    prompt = f"""
    You are a financial advisor. Based on the stock trend and recent news, provide an investment insight:

    Stock Trend:
    {trend_context}

    News:
    {context}

    Question:
    {query}
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "
