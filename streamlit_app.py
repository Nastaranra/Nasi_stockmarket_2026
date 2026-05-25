import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="AI Trading Signal App", layout="wide")

st.title("📈 AI Trading Signal App")
st.caption("Professional AI Trading Dashboard")
st.warning("Educational only. Not financial advice. No signal is guaranteed.")

ALPHA_KEY = st.secrets.get("ALPHA_VANTAGE_API_KEY", "")
FINNHUB_API_KEY = st.secrets.get("FINNHUB_API_KEY", "")

TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "META", "TSLA", "NFLX", "AMD", "SPY",
    "QQQ", "PLTR", "SMCI", "AVGO", "CRM"
]

def safe_num(x, default=0):
    try:
        if x is None or pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default

@st.cache_data(ttl=3600)
def load_price_data(ticker):
    if not ALPHA_KEY:
        return pd.DataFrame(), "Missing Alpha Vantage API key"

    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": ticker,
            "outputsize": "compact",
            "apikey": ALPHA_KEY
        }

        response = requests.get(url, params=params, timeout=30)
        data = response.json()

        if "Note" in data:
            return pd.DataFrame(), "Alpha Vantage API limit reached. Try again later."

        if "Information" in data:
            return pd.DataFrame(), data["Information"]

        if "Error Message" in data:
            return pd.DataFrame(), data["Error Message"]

        time_series = data.get("Time Series (Daily)")

        if not time_series:
            return pd.DataFrame(), f"No price data returned. Response: {data}"

        rows = []

        for date, values in time_series.items():
            rows.append({
                "Date": pd.to_datetime(date),
                "Open": float(values.get("1. open", np.nan)),
                "High": float(values.get("2. high", np.nan)),
                "Low": float(values.get("3. low", np.nan)),
                "Close": float(values.get("4. close", np.nan)),
                "Volume": float(values.get("5. volume", 0))
            })

        df = pd.DataFrame(rows)
        df = df.sort_values("Date")
        df = df.dropna(subset=["Date", "Open", "High", "Low", "Close"])

        if df.empty:
            return pd.DataFrame(), "Empty dataframe after cleaning"

        return df, ""

    except Exception as e:
        return pd.DataFrame(), str(e)

def add_indicators(df):
    df = df.copy()

    df["Return"] = df["Close"].pct_change()
    df["MA9"] = df["Close"].rolling(9, min_periods=3).mean()
    df["MA20"] = df["Close"].rolling(20, min_periods=5).mean()
    df["MA50"] = df["Close"].rolling(50, min_periods=10).mean()
    df["MA200"] = df["Close"].rolling(200, min_periods=20).mean()

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(14, min_periods=5).mean()
    avg_loss = loss.rolling(14, min_periods=5).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["RSI"] = 100 - (100 / (1 + rs))

    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()

    df["MACD"] = exp1 - exp2
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - df["Close"].shift()).abs()
    tr3 = (df["Low"] - df["Close"].shift()).abs()

    df["TR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR"] = df["TR"].rolling(14, min_periods=5).mean()

    df["Support"] = df["Close"].rolling(30, min_periods=5).min()
    df["Resistance"] = df["Close"].rolling(30, min_periods=5).max()
    df["Volatility"] = df["Return"].rolling(20, min_periods=5).std() * np.sqrt(252)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill()

    return df.dropna()

@st.cache_data(ttl=1800)
def load_fundamentals(ticker):
    if not FINNHUB_API_KEY:
        return {}, "No Finnhub API key"

    try:
        url = f"https://finnhub.io/api/v1/stock/metric?symbol={ticker}&metric=all&token={FINNHUB_API_KEY}"
        response = requests.get(url, timeout=20)

        if response.status_code != 200:
            return {}, f"HTTP {response.status_code}"

        metric = response.json().get("metric", {})

        return {
            "Market Cap": metric.get("marketCapitalization"),
            "P/E Ratio": metric.get("peBasicExclExtraTTM"),
            "Forward P/E": metric.get("forwardPE"),
            "Profit Margin": metric.get("netProfitMarginTTM"),
            "Revenue Growth": metric.get("revenueGrowthTTMYoy"),
            "Debt to Equity": metric.get("totalDebt/totalEquityQuarterly"),
            "ROE": metric.get("roeTTM"),
            "Beta": metric.get("beta")
        }, ""

    except Exception as e:
        return {}, str(e)

@st.cache_data(ttl=1800)
def load_news_sentiment(ticker):
    if not FINNHUB_API_KEY:
        return pd.DataFrame(), 0, "No Finnhub API key"

    try:
        today = datetime.today().date()
        start = today - timedelta(days=7)

        url = "https://finnhub.io/api/v1/company-news"
        params = {
            "symbol": ticker,
            "from": str(start),
            "to": str(today),
            "token": FINNHUB_API_KEY
        }

        response = requests.get(url, params=params, timeout=20)

        if response.status_code != 200:
            return pd.DataFrame(), 0, f"HTTP {response.status_code}"

        news = response.json()

        if not news:
            return pd.DataFrame(), 0, "No Recent News"

        positive_words = [
            "beat", "growth", "strong", "upgrade", "surge", "record",
            "profit", "higher", "bullish", "positive", "gain", "rally",
            "outperform", "raises", "increase", "ai", "partnership"
        ]

        negative_words = [
            "miss", "drop", "fall", "lawsuit", "weak", "downgrade",
            "loss", "lower", "bearish", "negative", "decline", "cut",
            "warning", "risk", "investigation", "delay"
        ]

        rows = []
        total_score = 0

        for item in news[:20]:
            headline = str(item.get("headline", ""))
            summary = str(item.get("summary", ""))
            text = (headline + " " + summary).lower()

            pos = sum(1 for w in positive_words if w in text)
            neg = sum(1 for w in negative_words if w in text)

            score = pos - neg
            total_score += score

            sentiment = "Positive" if score > 0 else "Negative" if score < 0 else "Neutral"

            rows.append({
                "Date": datetime.fromtimestamp(item.get("datetime")).strftime("%Y-%m-%d") if item.get("datetime") else None,
                "Headline": headline,
                "Sentiment": sentiment,
                "Source": item.get("source"),
                "URL": item.get("url")
            })

        avg_score = total_score / max(len(rows), 1)

        if avg_score > 0.25:
            label = "Positive News"
        elif avg_score < -0.25:
            label = "Negative News"
        else:
            label = "Neutral News"

        return pd.DataFrame(rows), avg_score, label

    except Exception as e:
        return pd.DataFrame(), 0, str(e)

def technical_score_only(latest):
    score = 0
    reasons = []

    close = safe_num(latest["Close"])
    ma9 = safe_num(latest["MA9"])
    ma20 = safe_num(latest["MA20"])
    ma50 = safe_num(latest["MA50"])
    ma200 = safe_num(latest["MA200"])
    rsi = safe_num(latest["RSI"])
    macd = safe_num(latest["MACD"])
    macd_signal = safe_num(latest["MACD_Signal"])
    vol = safe_num(latest["Volatility"], 1)

    if close > ma9:
        score += 1
        reasons.append("Price is above MA9.")
    if close > ma20:
        score += 1
        reasons.append("Price is above MA20.")
    if close > ma50:
        score += 1
        reasons.append("Price is above MA50.")
    if close > ma200:
        score += 1
        reasons.append("Price is above MA200.")
    if ma9 > ma20:
        score += 1
        reasons.append("Short-term trend is positive.")
    if ma20 > ma50:
        score += 1
        reasons.append("Medium-term trend is positive.")
    if macd > macd_signal:
        score += 1
        reasons.append("MACD is bullish.")

    if 45 <= rsi <= 68:
        score += 1
        reasons.append("RSI is healthy.")
    elif rsi > 72:
        score -= 1
        reasons.append("RSI is overbought.")
    elif rsi < 32:
        score -= 1
        reasons.append("RSI is weak or oversold.")

    if vol > 0.65:
        risk = "High"
        score -= 2
        reasons.append("Volatility is high.")
    elif vol > 0.35:
        risk = "Medium"
        reasons.append("Volatility is moderate.")
    else:
        risk = "Low"
        reasons.append("Volatility is low.")

    return score, risk, reasons

def fundamental_score_only(fundamentals):
    score = 0
    reasons = []

    pe = safe_num(fundamentals.get("P/E Ratio"), None)
    fpe = safe_num(fundamentals.get("Forward P/E"), None)
    margin = safe_num(fundamentals.get("Profit Margin"), None)
    growth = safe_num(fundamentals.get("Revenue Growth"), None)
    debt = safe_num(fundamentals.get("Debt to Equity"), None)
    roe = safe_num(fundamentals.get("ROE"), None)

    if pe is not None and 0 < pe < 45:
        score += 1
        reasons.append("P/E ratio is acceptable.")
    if fpe is not None and 0 < fpe < 45:
        score += 1
        reasons.append("Forward P/E is acceptable.")
    if margin is not None and margin > 8:
        score += 1
        reasons.append("Profit margin is strong.")
    if growth is not None and growth > 5:
        score += 1
        reasons.append("Revenue growth is positive.")
    if debt is not None and debt < 220:
        score += 1
        reasons.append("Debt-to-equity is manageable.")
    if roe is not None and roe > 10:
        score += 1
        reasons.append("ROE is strong.")

    return score, reasons

def estimate_future_price(df, days):
    recent = df.tail(100).copy()

    if len(recent) < 30:
        return pd.DataFrame(), 0, "Not enough data"

    returns = recent["Close"].pct_change().dropna()
    avg_return = returns.mean()
    volatility = returns.std()
    last_price = recent["Close"].iloc[-1]

    base_price = last_price
    bull_price = last_price
    bear_price = last_price

    for _ in range(days):
        base_price *= (1 + avg_return)
        bull_price *= (1 + avg_return + volatility * 0.25)
        bear_price *= (1 + avg_return - volatility * 0.25)

    expected_return = (base_price / last_price) - 1

    if expected_return >= 0.08:
        label = "Strong Positive Estimate"
    elif expected_return >= 0.03:
        label = "Positive Estimate"
    elif expected_return <= -0.08:
        label = "Strong Negative Estimate"
    elif expected_return <= -0.03:
        label = "Negative Estimate"
    else:
        label = "Neutral Estimate"

    out = pd.DataFrame({
        "Forecast Horizon": [f"{days} days"],
        "Current Price": [round(last_price, 2)],
        "Base Estimated Price": [round(base_price, 2)],
        "Bull Case Price": [round(bull_price, 2)],
        "Bear Case Price": [round(bear_price, 2)],
        "Estimated Return": [f"{expected_return:.2%}"],
        "Forecast Label": [label]
    })

    return out, expected_return, label

def final_signal(total_score, risk, news_score, expected_return):
    if risk == "High":
        return "Avoid / High Risk"
    if news_score < -0.25:
        return "Avoid / Negative News"
    if total_score >= 14 and expected_return > 0:
        return "Strong Buy"
    elif total_score >= 11 and expected_return > 0:
        return "Buy Signal"
    elif total_score >= 8:
        return "Buy on Dip"
    elif total_score >= 5:
        return "Hold / Wait"
    else:
        return "Sell / High Caution"

def confidence_score(total_score, risk, expected_return, news_score):
    confidence = 45 + total_score * 2

    if expected_return > 0.05:
        confidence += 5
    elif expected_return < -0.05:
        confidence -= 5

    if risk == "Low":
        confidence += 5
    elif risk == "High":
        confidence -= 15

    if news_score > 0.25:
        confidence += 4
    elif news_score < -0.25:
        confidence -= 8

    return int(max(35, min(90, confidence)))

def trade_plan(latest, signal, confidence, expected_return, horizon_days):
    close = safe_num(latest["Close"])
    atr = safe_num(latest["ATR"], close * 0.02)
    support = safe_num(latest["Support"], close - atr)
    resistance = safe_num(latest["Resistance"], close + atr)

    if signal in ["Strong Buy", "Buy Signal"]:
        buy_low = max(support, close - 0.7 * atr)
        buy_high = min(close + 0.25 * atr, close * 1.015)
        target = max(resistance, close + 1.7 * atr)
        stop_loss = buy_low - 1.05 * atr
        action = "BUY SETUP"
    elif signal == "Buy on Dip":
        buy_low = max(support, close - 1.2 * atr)
        buy_high = close - 0.3 * atr
        target = close + 1.5 * atr
        stop_loss = buy_low - 1.0 * atr
        action = "BUY ON DIP"
    elif "Sell" in signal or "Avoid" in signal:
        buy_low = np.nan
        buy_high = np.nan
        target = min(support, close - 1.4 * atr)
        stop_loss = close + 1.0 * atr
        action = "AVOID / SELL SETUP"
    else:
        buy_low = close - 0.6 * atr
        buy_high = close + 0.2 * atr
        target = close + 1.1 * atr
        stop_loss = close - 1.0 * atr
        action = "WAIT / HOLD SETUP"

    if horizon_days <= 5:
        hold = "1-5 days"
    elif horizon_days <= 14:
        hold = "5-14 days"
    elif horizon_days <= 60:
        hold = "2-8 weeks"
    else:
        hold = "2-6 months"

    return pd.DataFrame({
        "Action": [action],
        "Buy Zone Low": [round(buy_low, 2) if not pd.isna(buy_low) else None],
        "Buy Zone High": [round(buy_high, 2) if not pd.isna(buy_high) else None],
        "Target": [round(target, 2)],
        "Stop Loss": [round(stop_loss, 2)],
        "Expected Hold": [hold],
        "Confidence": [f"{confidence}%"],
        "Estimated Return": [f"{expected_return:.2%}"]
    })

def make_price_chart(df, ticker):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Close"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA9"], mode="lines", name="MA9"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA20"], mode="lines", name="MA20"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA50"], mode="lines", name="MA50"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA200"], mode="lines", name="MA200"))

    fig.update_layout(
        title=f"{ticker} Price Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        height=500
    )

    return fig

def analyze_stock(ticker, horizon_days):
    df, price_error = load_price_data(ticker)

    if df.empty:
        return None, price_error

    df = add_indicators(df)

    if df.empty:
        return None, "Not enough data after indicators"

    fundamentals, fund_error = load_fundamentals(ticker)
    news_df, news_score, news_label = load_news_sentiment(ticker)

    estimate_df, expected_return, forecast_label = estimate_future_price(df, horizon_days)
    latest = df.iloc[-1]

    tech_score, risk, tech_reasons = technical_score_only(latest)
    fund_score, fund_reasons = fundamental_score_only(fundamentals)

    forecast_score = 2 if expected_return >= 0.05 else -2 if expected_return <= -0.05 else 0
    news_component = 2 if news_score > 0.25 else -2 if news_score < -0.25 else 0

    total_score = tech_score + fund_score + forecast_score + news_component
    signal = final_signal(total_score, risk, news_score, expected_return)
    confidence = confidence_score(total_score, risk, expected_return, news_score)
    plan_df = trade_plan(latest, signal, confidence, expected_return, horizon_days)

    result = {
        "df": df,
        "latest": latest,
        "fundamentals": fundamentals,
        "fund_error": fund_error,
        "news_df": news_df,
        "news_score": news_score,
        "news_label": news_label,
        "estimate_df": estimate_df,
        "expected_return": expected_return,
        "forecast_label": forecast_label,
        "tech_score": tech_score,
        "fund_score": fund_score,
        "forecast_score": forecast_score,
        "news_component": news_component,
        "total_score": total_score,
        "risk": risk,
        "signal": signal,
        "confidence": confidence,
        "plan_df": plan_df,
        "reasons": tech_reasons + fund_reasons
    }

    return result, ""

st.sidebar.header("Settings")

selected_ticker = st.sidebar.selectbox("Select one stock", TICKERS)

horizon_days = st.sidebar.selectbox(
    "Forecast horizon",
    [1, 3, 5, 10, 14, 30, 60, 90],
    index=2
)

scan_count = st.sidebar.selectbox(
    "How many stocks to scan?",
    [3, 5, 10, 15],
    index=1
)

if not ALPHA_KEY:
    st.error("ALPHA_VANTAGE_API_KEY is missing. Add it in Streamlit Secrets.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["Single Stock", "Scanner", "Ticker List"])

with tab1:
    st.subheader(f"{selected_ticker} Analysis")

    result, error = analyze_stock(selected_ticker, horizon_days)

    if result is None:
        st.error(f"Could not analyze {selected_ticker}. Reason: {error}")
    else:
        latest = result["latest"]

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric("Current Price", f"${latest['Close']:.2f}")
        c2.metric("Signal", result["signal"])
        c3.metric("Risk", result["risk"])
        c4.metric("Confidence", f"{result['confidence']}%")
        c5.metric("Expected Return", f"{result['expected_return']:.2%}")

        st.markdown("### Trading Plan")
        st.dataframe(result["plan_df"], use_container_width=True)

        st.markdown("### Price Chart")
        st.plotly_chart(make_price_chart(result["df"], selected_ticker), use_container_width=True)

        st.markdown("### Estimated Future Price")
        st.dataframe(result["estimate_df"], use_container_width=True)

        st.markdown("### Score Breakdown")
        score_df = pd.DataFrame({
            "Category": ["Technical", "Fundamental", "Forecast", "News", "Total"],
            "Score": [
                result["tech_score"],
                result["fund_score"],
                result["forecast_score"],
                result["news_component"],
                result["total_score"]
            ]
        })

        st.dataframe(score_df, use_container_width=True)

        st.markdown("### Why This Signal?")
        for r in result["reasons"]:
            st.write(f"- {r}")

        st.markdown("### Fundamentals")
        if result["fundamentals"]:
            st.dataframe(
                pd.DataFrame({
                    "Metric": list(result["fundamentals"].keys()),
                    "Value": list(result["fundamentals"].values())
                }),
                use_container_width=True
            )
        else:
            st.warning(f"Fundamentals unavailable. {result['fund_error']}")

        st.markdown("### Recent News")
        st.write(f"News Sentiment: {result['news_label']}")

        if result["news_df"].empty:
            st.warning("No recent news loaded.")
        else:
            st.dataframe(result["news_df"], use_container_width=True)

with tab2:
    st.subheader("Scanner")

    rows = []
    progress = st.progress(0)

    selected_scan = TICKERS[:scan_count]

    for i, ticker in enumerate(selected_scan):
        result, error = analyze_stock(ticker, horizon_days)

        if result is not None:
            rows.append({
                "Ticker": ticker,
                "Price": round(safe_num(result["latest"]["Close"]), 2),
                "Signal": result["signal"],
                "Risk": result["risk"],
                "Confidence": result["confidence"],
                "Expected Return %": round(result["expected_return"] * 100, 2),
                "Technical Score": result["tech_score"],
                "Fundamental Score": result["fund_score"],
                "News": result["news_label"],
                "Total Score": result["total_score"]
            })

        progress.progress((i + 1) / len(selected_scan))

    if rows:
        scanner_df = pd.DataFrame(rows)

        order = {
            "Strong Buy": 1,
            "Buy Signal": 2,
            "Buy on Dip": 3,
            "Hold / Wait": 4,
            "Avoid / Negative News": 5,
            "Avoid / High Risk": 6,
            "Sell / High Caution": 7
        }

        scanner_df["Sort"] = scanner_df["Signal"].map(order).fillna(9)

        scanner_df = scanner_df.sort_values(
            ["Sort", "Confidence", "Total Score"],
            ascending=[True, False, False]
        ).drop(columns=["Sort"])

        st.dataframe(scanner_df, use_container_width=True)

        st.download_button(
            "Download Scanner Results",
            scanner_df.to_csv(index=False).encode("utf-8"),
            "stock_scanner_results.csv",
            "text/csv"
        )
    else:
        st.warning("No scanner results found.")

with tab3:
    st.subheader("Ticker List")
    st.dataframe(pd.DataFrame({"Ticker": TICKERS}), use_container_width=True)