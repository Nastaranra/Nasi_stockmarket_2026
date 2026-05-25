

Conversations
me
ai_trading_app_no_alpha_limit
 
Attachment:
ai_trading_app_no_alpha_limit.py
5:36 PM
me
ai_trading_app_full_fixed
 
Attachment:
ai_trading_app_full_fixed.py
5:25 PM
88% of 100 GB used
Terms · Privacy · Program Policies
Last account activity: 0 minutes ago
Open in 1 other location · Details
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
from io import StringIO

st.set_page_config(page_title="AI Trading Signal App", layout="wide")

st.title("📈 AI Trading Signal App")
st.caption("Short-Term + Long-Term + Scanner | No Alpha Vantage limit")
st.warning("Educational only. Not financial advice. No signal is guaranteed.")

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "TSLA",
    "NFLX", "AMD", "SPY", "QQQ", "PLTR", "SMCI", "AVGO", "CRM",
    "MU", "INTC", "UBER", "SHOP", "COIN", "SOFI", "PYPL", "ADBE",
    "SNOW", "PANW", "MSTR", "ARM", "BABA", "DIS", "NKE", "COST",
    "WMT", "TGT", "JPM", "BAC", "V", "MA", "UNH", "LLY", "XOM",
    "CVX", "HOOD", "RIVN", "LCID", "NIO", "F", "GM", "BA", "GE"
]


def safe_num(x, default=0):
    try:
        if x is None or pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


@st.cache_data(ttl=86400)
def get_all_tickers():
    tickers = set(DEFAULT_TICKERS)

    urls = [
        "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
    ]

    for url in urls:
        try:
            response = requests.get(url, timeout=30)
            lines = response.text.splitlines()

            for line in lines[1:]:
                parts = line.split("|")

                if len(parts) > 1:
                    symbol = parts[0].strip()

                    if symbol.isalpha() and 1 <= len(symbol) <= 5:
                        tickers.add(symbol)

        except Exception:
            pass

    return sorted(list(tickers))


ALL_TICKERS = get_all_tickers()


@st.cache_data(ttl=1800)
def load_price_data(ticker):
    try:
        symbol = ticker.lower() + ".us"
        url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"

        response = requests.get(url, timeout=30)

        if response.status_code != 200:
            return pd.DataFrame(), f"HTTP Error {response.status_code}"

        text = response.text.strip()

        if not text or "No data" in text or len(text.splitlines()) <= 1:
            return pd.DataFrame(), f"No price data found for {ticker}"

        df = pd.read_csv(StringIO(text))

        if df.empty:
            return pd.DataFrame(), f"No price data found for {ticker}"

        required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]

        for col in required_cols:
            if col not in df.columns:
                return pd.DataFrame(), f"Missing column {col} for {ticker}"

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")

        df = df.dropna(subset=["Date", "Open", "High", "Low", "Close"])

        if len(df) > 800:
            df = df.tail(800)

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
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()

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


def score_stock(latest, term_type):
    score = 0
    reasons = []

    close = safe_num(latest["Close"])
    ma9 = safe_num(latest["MA9"])
    ma20 = safe_num(latest["MA20"])
    ma50 = safe_num(latest["MA50"])
    ma200 = safe_num(latest["MA200"])
    rsi = safe_num(latest["RSI"])
    macd = safe_num(latest["MACD"])
    macd_signal = safe_num(latest["MACD_SIGNAL"])
    volatility = safe_num(latest["Volatility"], 0)

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
        score += 2 if term_type == "Long-Term" else 1
        reasons.append("Price is above MA200.")

    if ma9 > ma20:
        score += 1
        reasons.append("Short-term trend is positive.")

    if ma20 > ma50:
        score += 1
        reasons.append("Medium-term trend is positive.")

    if ma50 > ma200:
        score += 2 if term_type == "Long-Term" else 1
        reasons.append("Long-term trend is positive.")

    if macd > macd_signal:
        score += 1
        reasons.append("MACD is bullish.")

    if 45 <= rsi <= 68:
        score += 1
        reasons.append("RSI is healthy.")
    elif rsi > 75:
        score -= 1
        reasons.append("RSI is overbought.")
    elif rsi < 32:
        score -= 1
        reasons.append("RSI is weak or oversold.")

    if volatility > 0.65:
        risk = "High"
        score -= 2
        reasons.append("Volatility is high.")
    elif volatility > 0.35:
        risk = "Medium"
        reasons.append("Volatility is moderate.")
    else:
        risk = "Low"
        reasons.append("Volatility is low.")

    return score, risk, reasons


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


def final_signal(score, risk, expected_return, term_type):
    if risk == "High" and score < 6:
        return "⚠️ Avoid / High Risk"

    if term_type == "Short-Term":
        if score >= 8 and expected_return > 0:
            return "🔥 Strong Buy"
        elif score >= 6 and expected_return > 0:
            return "✅ Buy Signal"
        elif score >= 4:
            return "📉 Buy on Dip"
        elif score >= 2:
            return "⏳ Hold / Wait"
        else:
            return "🔻 Sell / High Caution"

    else:
        if score >= 9 and expected_return > 0:
            return "🚀 Strong Long-Term Buy"
        elif score >= 7 and expected_return > 0:
            return "✅ Long-Term Buy"
        elif score >= 5:
            return "📉 Long-Term Buy on Dip"
        elif score >= 3:
            return "⏳ Long-Term Hold / Watch"
        else:
            return "⚠️ Avoid Long-Term"


def confidence_score(score, risk, expected_return):
    confidence = 45 + score * 4

    if expected_return > 0.05:
        confidence += 5
    elif expected_return < -0.05:
        confidence -= 5

    if risk == "Low":
        confidence += 5
    elif risk == "High":
        confidence -= 15

    return int(max(35, min(90, confidence)))


def trade_plan(latest, signal, confidence, expected_return, horizon_days):
    close = safe_num(latest["Close"])
    atr = safe_num(latest["ATR"], close * 0.02)
    support = safe_num(latest["Support"], close - atr)
    resistance = safe_num(latest["Resistance"], close + atr)

    if "Strong" in signal or "Buy Signal" in signal or "Long-Term Buy" in signal:
        buy_low = max(support, close - 0.7 * atr)
        buy_high = min(close + 0.25 * atr, close * 1.015)
        target = max(resistance, close + 1.7 * atr)
        stop_loss = buy_low - 1.05 * atr
        action = "BUY SETUP"

    elif "Buy on Dip" in signal:
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


def signal_box(signal):
    st.markdown("### Signal")

    if "Strong" in signal or "Buy Signal" in signal or "Long-Term Buy" in signal:
        st.success(f"**{signal}**")
    elif "Buy on Dip" in signal:
        st.warning(f"**{signal}**")
    elif "Hold" in signal or "Wait" in signal:
        st.info(f"**{signal}**")
    else:
        st.error(f"**{signal}**")


def analyze_stock(ticker, horizon_days, term_type):
    df, price_error = load_price_data(ticker)

    if df.empty:
        return None, price_error

    df = add_indicators(df)

    if df.empty:
        return None, "Not enough data after indicators."

    latest = df.iloc[-1]

    score, risk, reasons = score_stock(latest, term_type)
    estimate_df, expected_return, forecast_label = estimate_future_price(df, horizon_days)

    signal = final_signal(score, risk, expected_return, term_type)
    confidence = confidence_score(score, risk, expected_return)
    plan_df = trade_plan(latest, signal, confidence, expected_return, horizon_days)

    return {
        "df": df,
        "latest": latest,
        "score": score,
        "risk": risk,
        "reasons": reasons,
        "estimate_df": estimate_df,
        "expected_return": expected_return,
        "forecast_label": forecast_label,
        "signal": signal,
        "confidence": confidence,
        "plan_df": plan_df
    }, ""


st.sidebar.header("Settings")

term_type = st.sidebar.radio(
    "Trading Style",
    ["Short-Term", "Long-Term"],
    index=0
)

selected_ticker = st.sidebar.selectbox(
    "Select one stock",
    ALL_TICKERS,
    index=ALL_TICKERS.index("AAPL") if "AAPL" in ALL_TICKERS else 0
)

horizon_days = st.sidebar.selectbox(
    "Forecast horizon",
    [1, 3, 5, 10, 14, 30, 60, 90],
    index=2
)

scan_count = st.sidebar.selectbox(
    "How many stocks to scan?",
    [5, 10, 25, 50, 100, 250],
    index=1
)

tab1, tab2, tab3 = st.tabs(["Single Stock", "Scanner", "Ticker List"])

with tab1:
    st.subheader(f"{selected_ticker} Analysis")

    result, error = analyze_stock(selected_ticker, horizon_days, term_type)

    if result is None:
        st.error(f"Could not analyze {selected_ticker}. Reason: {error}")

    else:
        latest = result["latest"]

        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:
            st.markdown("### Current Price")
            st.success(f"**${latest['Close']:.2f}**")

        with c2:
            st.markdown("### Risk")
            if result["risk"] == "Low":
                st.success(f"**{result['risk']}**")
            elif result["risk"] == "Medium":
                st.warning(f"**{result['risk']}**")
            else:
                st.error(f"**{result['risk']}**")

        with c3:
            st.markdown("### Confidence")
            st.info(f"**{result['confidence']}%**")

        with c4:
            st.markdown("### Expected Return")
            st.info(f"**{result['expected_return']:.2%}**")

        with c5:
            st.markdown("### Style")
            st.info(f"**{term_type}**")

        signal_box(result["signal"])

        st.markdown("### Trading Plan")
        st.dataframe(result["plan_df"], use_container_width=True)

        st.markdown("### Price Chart")
        st.plotly_chart(make_price_chart(result["df"], selected_ticker), use_container_width=True)

        st.markdown("### Estimated Future Price")
        st.dataframe(result["estimate_df"], use_container_width=True)

        st.markdown("### Score Breakdown")
        score_df = pd.DataFrame({
            "Category": ["Technical Score", "Total Score"],
            "Score": [result["score"], result["score"]]
        })
        st.dataframe(score_df, use_container_width=True)

        st.markdown("### Why This Signal?")
        for r in result["reasons"]:
            st.write(f"- {r}")

with tab2:
    st.subheader("Scanner")

    rows = []
    progress = st.progress(0)

    selected_scan = ALL_TICKERS[:scan_count]

    for i, ticker in enumerate(selected_scan):
        result, error = analyze_stock(ticker, horizon_days, term_type)

        if result is not None:
            rows.append({
                "Ticker": ticker,
                "Price": round(safe_num(result["latest"]["Close"]), 2),
                "Signal": result["signal"],
                "Style": term_type,
                "Risk": result["risk"],
                "Confidence": result["confidence"],
                "Expected Return %": round(result["expected_return"] * 100, 2),
                "Score": result["score"]
            })

        progress.progress((i + 1) / len(selected_scan))

    if rows:
        scanner_df = pd.DataFrame(rows)

        order = {
            "🔥 Strong Buy": 1,
            "🚀 Strong Long-Term Buy": 1,
            "✅ Buy Signal": 2,
            "✅ Long-Term Buy": 2,
            "📉 Buy on Dip": 3,
            "📉 Long-Term Buy on Dip": 3,
            "⏳ Hold / Wait": 4,
            "⏳ Long-Term Hold / Watch": 4,
            "⚠️ Avoid / High Risk": 5,
            "🔻 Sell / High Caution": 6,
            "⚠️ Avoid Long-Term": 7
        }

        scanner_df["Sort"] = scanner_df["Signal"].map(order).fillna(9)

        scanner_df = scanner_df.sort_values(
            ["Sort", "Confidence", "Score"],
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
    st.write(f"Total Tickers Loaded: {len(ALL_TICKERS)}")
    st.dataframe(pd.DataFrame({"Ticker": ALL_TICKERS}), use_container_width=True, height=700)