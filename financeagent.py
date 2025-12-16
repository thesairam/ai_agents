from ollama import chat
import yfinance as yf
import pandas as pd
import numpy as np
import json
import time

# -----------------------------
# SIMPLE IN-MEMORY CACHE
# -----------------------------
CACHE = {}
CACHE_TTL = 300  # seconds (5 min)


def cache_get(key):
    if key in CACHE:
        value, timestamp = CACHE[key]
        if time.time() - timestamp < CACHE_TTL:
            return value
    return None


def cache_set(key, value):
    CACHE[key] = (value, time.time())


# -----------------------------
# TECHNICAL INDICATORS
# -----------------------------
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return round(float(rsi.iloc[-1]), 2)


def compute_macd(series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()

    return {
        "macd": round(float(macd.iloc[-1]), 3),
        "signal": round(float(signal.iloc[-1]), 3),
    }


# -----------------------------
# STOCK + INDICATORS + TREND
# -----------------------------
def get_stock_analysis(ticker: str):
    cached = cache_get(ticker)
    if cached:
        return cached

    stock = yf.Ticker(ticker)
    df = stock.history(period="1mo")

    if df.empty:
        return {"error": "No market data found"}

    close = df["Close"]

    # Guard against short history windows
    last_price = round(float(close.iloc[-1]), 2)
    change_5d = None
    if len(close) >= 5:
        change_5d = round(((close.iloc[-1] / close.iloc[-5]) - 1) * 100, 2)
    else:
        change_5d = "insufficient data"

    change_20d = None
    if len(close) >= 20:
        change_20d = round(((close.iloc[-1] / close.iloc[-20]) - 1) * 100, 2)
    else:
        change_20d = "insufficient data"

    analysis = {
        "ticker": ticker,
        "current_price": last_price,
        "5d_change_pct": change_5d,
        "20d_change_pct": change_20d,
        "rsi": compute_rsi(close),
        "macd": compute_macd(close),
    }

    cache_set(ticker, analysis)
    return analysis


# -----------------------------
# NEWS SENTIMENT (LLM-BASED)
# -----------------------------
def get_news_sentiment(ticker: str):
    stock = yf.Ticker(ticker)
    # yfinance may return None or items without 'title'; be defensive
    try:
        raw_news = stock.news or []
    except Exception:
        raw_news = []

    news = raw_news[:5]
    headlines = []
    for n in news:
        if isinstance(n, dict):
            title = n.get("title") or n.get("headline")
            if title:
                headlines.append(title)

    if not headlines:
        return "No recent news found"

    prompt = f"""
Analyze the sentiment of the following stock news headlines.
Classify overall sentiment as Positive, Neutral, or Negative.

Headlines:
{headlines}
"""

    response = chat(
        model="gemma:7b",
        messages=[{"role": "user", "content": prompt}],
    )

    # Ollama chat responses can be dict-like; normalize access safely
    content = None
    if isinstance(response, dict):
        content = (
            response.get("message", {}).get("content")
            or response.get("content")
        )
    else:
        try:
            content = response.message.content
        except Exception:
            content = None

    return (content or "Sentiment analysis unavailable").strip()


# -----------------------------
# COMPANY → TICKER
# -----------------------------
def find_ticker(company_name: str):
    prompt = f"""
What is the official stock ticker for "{company_name}"?
Respond ONLY with the ticker symbol.
"""

    response = chat(
        model="gemma:7b",
        messages=[{"role": "user", "content": prompt}],
    )

    return response.message.content.strip().upper()


# -----------------------------
# AGENT
# -----------------------------
def agentic_stock_ai(user_question: str):

    decision_prompt = f"""
You are an AI planning agent.

Decide if this question needs stock market data.
If yes, extract the company name.

Respond ONLY in JSON:
{{
  "needs_data": true or false,
  "company_name": "Apple"
}}

Question:
{user_question}
"""

    decision_response = chat(
        model="gemma:7b",
        messages=[{"role": "user", "content": decision_prompt}],
    )

    try:
        decision = json.loads(decision_response.message.content)
    except:
        return "Error: Invalid planner output"

    if not decision["needs_data"]:
        return chat(
            model="gemma:7b",
            messages=[{"role": "user", "content": user_question}],
        ).message.content

    company = decision["company_name"]
    ticker = find_ticker(company)

    stock_analysis = get_stock_analysis(ticker)
    sentiment = get_news_sentiment(ticker)

    final_prompt = f"""
User question:
{user_question}

Company: {company}
Ticker: {ticker}

Market analysis:
{stock_analysis}

News sentiment:
{sentiment}

Explain:
- Short-term momentum
- Technical indicators meaning
- Trend over time
- News impact

Be neutral and cautious.
State that this is not financial advice.
"""

    response = chat(
        model="gemma:7b",
        messages=[{"role": "user", "content": final_prompt}],
    )

    return response.message.content


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    while True:
        q = input("\nAsk a stock question (or exit): ")
        if q.lower() == "exit":
            break

        print("\nAnswer:\n", agentic_stock_ai(q))
