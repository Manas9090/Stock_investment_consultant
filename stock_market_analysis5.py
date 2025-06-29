import openai
import requests
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import faiss
import streamlit as st
import datetime
import boto3
from io import StringIO

# --- API Keys ---
openai.api_key = st.secrets["api_keys"]["openai"]
twelvedata_api_key = st.secrets["api_keys"]["twelvedata"]
news_api_key = st.secrets["api_keys"]["newsapi"]
alpha_vantage_api_key = st.secrets["api_keys"]["alphavantage"]

# --- Load Embedding Model ---
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# --- AWS S3 Config ---
AWS_ACCESS_KEY = st.secrets["aws"]["aws_access_key_id"]
AWS_SECRET_KEY = st.secrets["aws"]["aws_secret_access_key"]
AWS_REGION = st.secrets["aws"]["region_name"]
BUCKET_NAME = st.secrets["aws"]["bucket_name"]
SYMBOL_FILE_KEY = "stock_symbols2.csv"

# Initialize S3 Client
s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

# Load Stock Symbols from S3
@st.cache_data
def load_symbols_from_s3():
    response = s3.get_object(Bucket=BUCKET_NAME, Key=SYMBOL_FILE_KEY)
    csv_data = response['Body'].read().decode('utf-8')
    df = pd.read_csv(StringIO(csv_data))
    return df

symbols_df = load_symbols_from_s3()

# --- Fetch Company Symbol & Exchange from DataFrame ---
def fetch_ticker_from_df(company_name):
    match = symbols_df[symbols_df["company_name"].str.lower() == company_name.lower()]
    if not match.empty:
        return match.iloc[0]["symbol"], match.iloc[0]["exchange"]
    return None

# --- Match Company in DataFrame ---
def match_company_in_df(user_query):
    for name in symbols_df["company_name"]:
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

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful financial advisor."},
            {"role": "user", "content": f"""
            You are a financial advisor. Based on the stock trend and recent news, provide an investment insight:

            Stock Trend:
            {trend_context}

            News:
            {context}

            Question:
            {query}
            """}
        ]
    )

    return response["choices"][0]["message"]["content"].strip()

# --- Streamlit UI ---
st.set_page_config(page_title="Stock Investment Consultant", layout="centered")

st.title("\U0001F4C8 Stock Market Investment Consultant")
st.write("Ask about a company's stock trend, news, and get AI-generated investment insights.")

query = st.text_input("Enter your query (mention company name):")

if query:
    company_name = match_company_in_df(query)

    if company_name:
        symbol, exchange = fetch_ticker_from_df(company_name)

        if symbol:
            st.success(f"Company found: {company_name} ({symbol}) on {exchange}")

            st.write("Fetching latest news...")
            news_list = fetch_live_news(company_name)

            if news_list:
                st.subheader("Recent News Articles:")
                for article in news_list[:5]:
                    st.write(f"- {article}")
            else:
                st.info("No recent news found.")

            try:
                use_alpha = False
                if exchange and exchange.lower() in ["bse", "bo", "nse", "ns"]:
                    use_alpha = True

                if use_alpha:
                    hist_df = get_alpha_vantage_data(symbol)
                else:
                    hist_df = get_twelve_data(symbol, exchange)

                st.line_chart(hist_df["close"])

                faiss_index, corpus = build_faiss_index(news_list)
                insight = get_stock_insight(query, corpus, faiss_index, symbol, hist_df)

                st.subheader("\U0001F4A1 Investment Insight:")
                st.write(insight)

            except Exception as e:
                st.error(f"Error fetching stock data: {e}")

        else:
            st.warning("Symbol not found for the matched company.")
    else:
        st.warning("No matching company found in symbol database.")
