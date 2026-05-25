import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="AI Trading Signal App", layout="wide")

st.title("📈 AI Trading Signal App")
st.caption("Professional AI Trading Dashboard")
st.warning("Educational only. Not financial advice.")

ALPHA_KEY = st.secrets.get("ALPHA_VANTAGE_API_KEY", "")
FINNHUB_API_KEY = st.secrets.get("FINNHUB_API_KEY", "")

DEFAULT_TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","NFLX",
    "AMD","SPY","QQQ","PLTR","SMCI","AVGO","CRM","MU","INTC",
    "UBER","SHOP","COIN","SOFI","PYPL","ADBE","SNOW","PANW"
]

@st.cache_data(ttl=86400)
def get_all_tickers():
    tickers = set(DEFAULT_TICKERS)

    sources = [
        "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
    ]

    for url in sources:
        try:
            r = requests.get(url, timeout=30)
            lines = r.text.splitlines()

            for line in lines[1:]:
                parts = line.split("|")

                if len(parts) > 1:
                    symbol = parts[0].strip()

                    if (
                        symbol.isalpha()
                        and len(symbol) <= 5
                    ):
                        tickers.add(symbol)

        except:
            pass

    return sorted(list(tickers))

ALL_TICKERS = get_all_tickers()

def safe_num(x, default=0):
    try:
        if x is None or pd.isna(x):
            return default
        return float(x)
    except:
        return default

@st.cache_data(ttl=3600)
def load_price_data(ticker):

    if not ALPHA_KEY:
        return pd.DataFrame(), "Missing API key"

    try:
        url = "https://www.alphavantage.co/query"

        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": ticker,
            "outputsize": "full",
            "apikey": ALPHA_KEY
        }

        response = requests.get(url, params=params, timeout=30)
        data = response.json()

        if "Time Series (Daily)" not in data:
            return pd.DataFrame(), str(data)

        rows = []

        for date, values in data["Time Series (Daily)"].items():

            rows.append({
                "Date": pd.to_datetime(date),
                "Open": float(values["1. open"]),
                "High": float(values["2. high"]),
                "Low": float(values["3. low"]),
                "Close": float(values["4. close"]),
                "Volume": float(values["5. volume"])
            })

        df = pd.DataFrame(rows).sort_values("Date")

        return df, ""

    except Exception as e:
        return pd.DataFrame(), str(e)

def add_indicators(df):

    df = df.copy()

    df["Return"] = df["Close"].pct_change()

    df["MA9"] = df["Close"].rolling(9).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()

    delta = df["Close"].diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["RSI"] = 100 - (100 / (1 + rs))

    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()

    df["MACD"] = exp1 - exp2
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()

    df = df.ffill().bfill()

    return df

def make_chart(df, ticker):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Close"],
            mode="lines",
            name="Close"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["MA20"],
            mode="lines",
            name="MA20"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["MA50"],
            mode="lines",
            name="MA50"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["MA200"],
            mode="lines",
            name="MA200"
        )
    )

    fig.update_layout(
        title=f"{ticker} Price Chart",
        height=550
    )

    return fig

def get_signal(df, term_type):

    latest = df.iloc[-1]

    score = 0

    if latest["Close"] > latest["MA20"]:
        score += 1

    if latest["Close"] > latest["MA50"]:
        score += 1

    if latest["Close"] > latest["MA200"]:
        score += 1

    if latest["MACD"] > latest["MACD_SIGNAL"]:
        score += 1

    if 45 <= latest["RSI"] <= 70:
        score += 1

    if term_type == "Short-Term":

        if score >= 4:
            return "🔥 Strong Buy"

        elif score >= 3:
            return "✅ Buy"

        elif score >= 2:
            return "⏳ Hold"

        else:
            return "⚠️ Sell"

    else:

        if latest["Close"] > latest["MA200"] and score >= 4:
            return "🚀 Long-Term Buy"

        elif score >= 3:
            return "📈 Long-Term Hold"

        else:
            return "⚠️ Weak Long-Term"

st.sidebar.header("Settings")

term_type = st.sidebar.radio(
    "Trading Style",
    ["Short-Term", "Long-Term"]
)

selected_ticker = st.sidebar.selectbox(
    "Select Stock",
    ALL_TICKERS
)

scan_count = st.sidebar.selectbox(
    "Stocks to Scan",
    [10,25,50,100,250],
    index=1
)

tab1, tab2, tab3 = st.tabs([
    "Single Stock",
    "Scanner",
    "All Tickers"
])

with tab1:

    st.subheader(selected_ticker)

    df, error = load_price_data(selected_ticker)

    if df.empty:

        st.error(error)

    else:

        df = add_indicators(df)

        latest = df.iloc[-1]

        signal = get_signal(df, term_type)

        c1,c2,c3,c4 = st.columns(4)

        c1.metric("Price", f"${latest['Close']:.2f}")
        c2.metric("RSI", f"{latest['RSI']:.2f}")
        c3.metric("Signal", signal)
        c4.metric("Style", term_type)

        st.plotly_chart(
            make_chart(df, selected_ticker),
            use_container_width=True
        )

        st.dataframe(
            df.tail(20),
            use_container_width=True
        )

with tab2:

    st.subheader("AI Scanner")

    rows = []

    scan_list = ALL_TICKERS[:scan_count]

    progress = st.progress(0)

    for i, ticker in enumerate(scan_list):

        df, error = load_price_data(ticker)

        if not df.empty:

            try:

                df = add_indicators(df)

                latest = df.iloc[-1]

                signal = get_signal(df, term_type)

                rows.append({
                    "Ticker": ticker,
                    "Price": round(latest["Close"],2),
                    "RSI": round(latest["RSI"],2),
                    "Signal": signal
                })

            except:
                pass

        progress.progress((i+1)/len(scan_list))

    if rows:

        result_df = pd.DataFrame(rows)

        st.dataframe(
            result_df,
            use_container_width=True
        )

        st.download_button(
            "Download CSV",
            result_df.to_csv(index=False).encode("utf-8"),
            "scanner.csv",
            "text/csv"
        )

with tab3:

    st.subheader("All Available Tickers")

    st.write(f"Total Tickers Loaded: {len(ALL_TICKERS)}")

    st.dataframe(
        pd.DataFrame({"Ticker": ALL_TICKERS}),
        use_container_width=True,
        height=700
    )