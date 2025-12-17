from ollama import chat
import yfinance as yf
import pandas as pd
import numpy as np
import json
import time
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

# -----------------------------
# SIMPLE IN-MEMORY CACHE
# -----------------------------
CACHE = {}
CACHE_TTL = 300  # seconds (5 min)
model_name = "gemma:2b"
MODEL_TIMEOUT = 12  # seconds

# Common name→ticker hints to avoid depending solely on LLM
COMMON_TICKERS = {
    "tesla": "TSLA",
    "nvidia": "NVDA",
    "apple": "AAPL",
    "microsoft": "MSFT",
    "amazon": "AMZN",
    "alphabet": "GOOGL",
    "google": "GOOGL",
    "meta": "META",
    "facebook": "META",
    "netflix": "NFLX",
    "amd": "AMD",
    "advanced micro devices": "AMD",
    "intel": "INTC",
    "tsmc": "TSM",
    "taiwan semiconductor": "TSM",
}


def _normalize_llm_content(response):
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
    return content


def safe_chat(prompt: str, model: str | None = None, timeout: int = MODEL_TIMEOUT) -> str | None:
    model = model or model_name
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(lambda: chat(model=model, messages=[{"role": "user", "content": prompt}]))
        try:
            resp = fut.result(timeout=timeout)
        except FuturesTimeout:
            return None
        except Exception:
            return None
    return _normalize_llm_content(resp)

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
    if len(series) < period + 1:
        return "insufficient data"
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    value = rsi.iloc[-1]
    try:
        return round(float(value), 2)
    except Exception:
        return "insufficient data"


def compute_macd(series):
    if len(series) < 26:
        return {"macd": "insufficient data", "signal": "insufficient data"}
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
    # Try multiple periods in case shorter ones are empty for some tickers
    df = pd.DataFrame()
    for period in ["1mo", "3mo", "6mo"]:
        try:
            df = stock.history(period=period, interval="1d")
        except Exception:
            df = pd.DataFrame()
        if not df.empty:
            break

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

    # Additional simple indicators
    sma20 = round(float(close.rolling(20).mean().iloc[-1]), 2) if len(close) >= 20 else "insufficient data"
    sma50 = round(float(close.rolling(50).mean().iloc[-1]), 2) if len(close) >= 50 else "insufficient data"
    # 20-day annualized historical volatility
    if len(close) >= 21:
        returns = close.pct_change().dropna()
        vol20 = float(returns.iloc[-20:].std() * np.sqrt(252) * 100)
        vol20 = round(vol20, 2)
    else:
        vol20 = "insufficient data"

    analysis = {
        "ticker": ticker,
        "current_price": last_price,
        "5d_change_pct": change_5d,
        "20d_change_pct": change_20d,
        "sma20": sma20,
        "sma50": sma50,
        "volatility20_annualized_pct": vol20,
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
Analyze the sentiment of these stock news headlines.
Return one of: Positive, Neutral, Negative. Add a one-sentence rationale.

Headlines:
{headlines}
"""

    content = safe_chat(prompt, model=model_name, timeout=MODEL_TIMEOUT)

    if not content:
        # Simple rule-based fallback
        pos_words = ["beat", "beats", "upgrade", "surge", "rally", "record", "strong", "rise", "soar"]
        neg_words = ["miss", "downgrade", "plunge", "fall", "weak", "lawsuit", "recall", "drop", "slump"]
        score = 0
        text = " ".join(headlines).lower()
        score += sum(w in text for w in pos_words)
        score -= sum(w in text for w in neg_words)
        label = "Neutral"
        if score >= 2:
            label = "Positive"
        elif score <= -2:
            label = "Negative"
        return f"{label} (rule-based fallback)"

    return content.strip()


# -----------------------------
# COMPANY → TICKER
# -----------------------------
def find_ticker(company_name: str):
    # First try local hints
    key = company_name.strip().lower()
    for name, sym in COMMON_TICKERS.items():
        if name in key:
            return sym

    prompt = f"""
What is the primary US stock ticker for "{company_name}"?
Respond ONLY with the ticker symbol like TSLA.
"""
    content = safe_chat(prompt, model=model_name, timeout=MODEL_TIMEOUT)
    if not content:
        return company_name.upper()[:5]
    return content.strip().upper().split()[0]


def is_valid_ticker(symbol: str) -> bool:
    try:
        h = yf.Ticker(symbol).history(period="5d")
        return not h.empty
    except Exception:
        return False


def resolve_ticker_from_question(question: str) -> tuple[str | None, str | None]:
    # Try explicit uppercase tokens (1-5 chars) that validate as tickers
    candidates = set(re.findall(r"\b[A-Z]{1,5}\b", question))
    for c in sorted(candidates, key=len, reverse=True):
        if is_valid_ticker(c):
            return c, None

    # Try common names mapping
    lower_q = question.lower()
    for name, sym in COMMON_TICKERS.items():
        if name in lower_q and is_valid_ticker(sym):
            return sym, name.title()

    # Extract a probable company word sequence (e.g., "Nvidia", "Tesla") and ask LLM
    content = safe_chat(
        f"Extract the company name (one or two words) from: {question}. Return only the name.",
        model=model_name,
        timeout=MODEL_TIMEOUT,
    )
    if content:
        sym = find_ticker(content.strip())
        if is_valid_ticker(sym):
            return sym, content.strip()

    return None, None


# -----------------------------
# AGENT
# -----------------------------
def agentic_stock_ai(user_question: str):
    # Heuristic ticker resolution first
    ticker, company = resolve_ticker_from_question(user_question)

    if not ticker:
        # If not a data question, just answer conversationally
        content = safe_chat(
            f"Does this need stock data? Respond true/false only: {user_question}",
            model=model_name,
            timeout=6,
        )
        if content and content.strip().lower().startswith("f"):
            generic = safe_chat(user_question, model=model_name, timeout=MODEL_TIMEOUT)
            return generic or "I can help with stock questions if you provide a company or ticker."

        # Last attempt: ask planner for company and map to ticker
        plan = safe_chat(
            f"Extract the company name from this question (just the name): {user_question}",
            model=model_name,
            timeout=6,
        )
        if plan:
            sym = find_ticker(plan)
            if is_valid_ticker(sym):
                ticker, company = sym, plan

    if not ticker:
        return "I couldn't determine the ticker. Please mention the company name or ticker (e.g., NVDA)."

    stock_analysis = get_stock_analysis(ticker)
    if "error" in stock_analysis:
        return f"Could not fetch market data for {ticker}: {stock_analysis['error']}"

    sentiment = get_news_sentiment(ticker)

    final_prompt = f"""
User question:
{user_question}

Company: {company or 'Unknown'}
Ticker: {ticker}

Market analysis:
{stock_analysis}

News sentiment:
{sentiment}

Explain succinctly:
- Short-term momentum
- What RSI, MACD, SMAs imply
- Trend over time
- How news may impact

Be neutral and cautious. This is not financial advice.
"""

    content = safe_chat(final_prompt, model=model_name, timeout=MODEL_TIMEOUT)
    if content:
        return content

    # Fallback templated answer
    lines = [
        f"Ticker {ticker}: price {stock_analysis['current_price']}",
        f"5d change: {stock_analysis['5d_change_pct']}",
        f"20d change: {stock_analysis['20d_change_pct']}",
        f"SMA20/SMA50: {stock_analysis['sma20']} / {stock_analysis['sma50']}",
        f"RSI: {stock_analysis['rsi']}, MACD: {stock_analysis['macd']}",
        f"News sentiment: {sentiment}",
        "Note: This is not financial advice.",
    ]
    return "\n".join(lines)


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    while True:
        q = input("\nAsk a stock question (or exit): ")
        if q.lower() == "exit":
            break

        print("\nAnswer:\n", agentic_stock_ai(q))
