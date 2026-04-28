import anthropic
import requests
import feedparser
import yfinance as yf
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from flask import Flask, jsonify, request as flask_request
import os
import json as json_lib

try:
    from dotenv import dotenv_values
    _env = dotenv_values(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
    os.environ.update({k: v for k, v in _env.items() if v})
except:
    pass

FRED_API_KEY      = os.environ.get("FRED_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
FINNHUB_API_KEY   = os.environ.get("FINNHUB_API_KEY", "")

_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
app = Flask(__name__)

# In-memory cache
_cache: dict = {"html": None, "analysis": None, "macro": None, "headlines": None, "markets": None, "ts": 0.0}
_building = False
_build_lock = threading.Lock()

def _cache_ttl():
    """Expire at 4:30 PM ET (market close) or midnight ET — whichever is sooner from now."""
    et = ZoneInfo("America/New_York")
    now = datetime.now(et)
    market_close = now.replace(hour=16, minute=30, second=0, microsecond=0)
    midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    if now < market_close:
        return max((market_close - now).total_seconds(), 300)
    return max((midnight - now).total_seconds(), 300)

IMPACT_RULES = [
    {"keywords": ["oil", "crude", "opec", "petroleum", "energy prices"],
     "impacts": {"Energy": "positive", "Airlines": "negative", "Consumer Discretionary": "negative", "Materials": "positive"},
     "assets": {"XLE": "positive", "OIL": "positive"}},
    {"keywords": ["fed", "federal reserve", "interest rate", "rate hike", "rate cut", "fomc", "powell", "treasury yield"],
     "impacts": {"Financials": "positive", "Real Estate": "negative", "Utilities": "negative", "Technology": "negative"},
     "assets": {"TLT": "negative", "XLF": "positive", "XLRE": "negative"}},
    {"keywords": ["china", "tariff", "trade war", "import tax", "sanctions"],
     "impacts": {"Technology": "negative", "Industrials": "negative", "Materials": "mixed", "Consumer Discretionary": "negative"},
     "assets": {"AAPL": "negative", "NVDA": "negative", "GLD": "positive"}},
    {"keywords": ["war", "conflict", "geopolitical", "invasion", "military", "nato"],
     "impacts": {"Defense": "positive", "Energy": "positive", "Technology": "negative", "Consumer Discretionary": "negative"},
     "assets": {"GLD": "positive", "XLE": "positive", "TLT": "positive"}},
    {"keywords": ["inflation", "cpi", "pce", "price index", "prices rose"],
     "impacts": {"Consumer Staples": "negative", "Real Estate": "negative", "Energy": "positive", "Commodities": "positive"},
     "assets": {"TLT": "negative", "GLD": "positive", "TIP": "positive"}},
]

# ── FRED ──────────────────────────────────────────────────────────────────────
def get_fred(series_id, label):
    url = (f"https://api.stlouisfed.org/fred/series/observations"
           f"?series_id={series_id}&api_key={FRED_API_KEY}"
           f"&sort_order=desc&limit=1&file_type=json")
    try:
        r = requests.get(url, timeout=10).json()
        val = r["observations"][0]["value"]
        return {"label": label, "value": val}
    except:
        return {"label": label, "value": "N/A"}

def get_fred_history(series_id, limit=24):
    url = (f"https://api.stlouisfed.org/fred/series/observations"
           f"?series_id={series_id}&api_key={FRED_API_KEY}"
           f"&sort_order=desc&limit={limit}&file_type=json")
    try:
        r = requests.get(url, timeout=10).json()
        obs = r["observations"]
        obs.reverse()
        return [(o["date"], float(o["value"])) for o in obs if o["value"] != "."]
    except:
        return []

def fetch_macro():
    series = [
        ("FEDFUNDS",     "Fed Funds Rate (%)"),
        ("CPIAUCSL",     "CPI Index"),
        ("CPILFESL",     "Core CPI"),
        ("UNRATE",       "Unemployment (%)"),
        ("GDP",          "GDP (B USD)"),
        ("RSAFS",        "Retail Sales (M USD)"),
        ("HOUST",        "Housing Starts (K)"),
        ("DCOILWTICO",   "WTI Crude ($/bbl)"),
        ("T10Y2Y",       "10Y-2Y Spread"),
        ("T10YIE",       "10Y Breakeven Inflation"),
        ("M2SL",         "M2 Money Supply (B)"),
        ("UMCSENT",      "Consumer Sentiment"),
        ("PAYEMS",       "Nonfarm Payrolls (K)"),
        ("INDPRO",       "Industrial Production"),
        ("MORTGAGE30US", "30Y Mortgage Rate (%)"),
    ]
    with ThreadPoolExecutor(max_workers=15) as ex:
        futures = {ex.submit(get_fred, sid, lbl): i for i, (sid, lbl) in enumerate(series)}
        results = [None] * len(series)
        for f in as_completed(futures):
            results[futures[f]] = f.result()
    return results

def fetch_chart_data():
    print("  Fetching chart history...")
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_fed = ex.submit(get_fred_history, "FEDFUNDS", 24)
        f_cpi = ex.submit(get_fred_history, "CPIAUCSL", 24)
        return {"fed": f_fed.result(), "cpi": f_cpi.result()}

# ── NEWS ──────────────────────────────────────────────────────────────────────
def fetch_news():
    feeds = [
        ("NYT Economy",    "https://rss.nytimes.com/services/xml/rss/nyt/Economy.xml"),
        ("NYT Business",   "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml"),
        ("NYT Markets",    "https://rss.nytimes.com/services/xml/rss/nyt/YourMoney.xml"),
        ("WSJ Markets",    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml"),
        ("WSJ Business",   "https://feeds.a.dj.com/rss/WSJcomUSBusiness.xml"),
        ("AP Business",    "https://feeds.apnews.com/rss/apf-business"),
        ("MarketWatch",    "https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines"),
        ("CNBC",           "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
        ("Seeking Alpha",  "https://seekingalpha.com/feed.xml"),
        ("Investopedia",   "https://www.investopedia.com/feedbuilder/feed/getfeed?feedName=rss_headline"),
    ]
    _headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        )
    }

    def _fetch_feed(source, url):
        try:
            resp = requests.get(url, timeout=8, headers=_headers)
            feed = feedparser.parse(resp.content)
            items = []
            for entry in feed.entries[:5]:
                title = getattr(entry, "title", None) or entry.get("title", "")
                link  = getattr(entry, "link",  None) or entry.get("link",  "#")
                if title:
                    items.append({"source": source, "title": title, "link": link})
            return items
        except:
            return []

    headlines = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = [ex.submit(_fetch_feed, src, url) for src, url in feeds]
        for f in as_completed(futures):
            headlines.extend(f.result())
    return headlines[:36]

# ── MARKETS ───────────────────────────────────────────────────────────────────
def _fetch_one_ticker(name, ticker):
    for _ in range(2):
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            if len(hist) >= 2:
                prev = hist["Close"].iloc[-2]
                curr = hist["Close"].iloc[-1]
                chg  = ((curr - prev) / prev) * 100
                return name, {
                    "name": name, "price": f"{curr:,.2f}",
                    "chg_str": f"{'▲' if chg>=0 else '▼'} {abs(chg):.2f}%",
                    "chg_num": f"{'▲' if chg>=0 else '▼'}{abs(chg):.2f}%",
                    "up": chg >= 0, "chg_val": round(chg, 2)
                }
        except:
            pass
    return name, {"name": name, "price": "N/A", "chg_str": "—", "chg_num": "—", "up": True, "chg_val": 0}

def fetch_markets():
    categories = {
        "US Equities":   {"S&P 500":"^GSPC","Nasdaq 100":"^NDX","Dow Jones":"^DJI","Russell 2000":"^RUT","VIX":"^VIX"},
        "Sector ETFs":   {"Technology":"XLK","Financials":"XLF","Energy":"XLE","Healthcare":"XLV","Industrials":"XLI","Consumer Disc":"XLY","Utilities":"XLU","Real Estate":"XLRE"},
        "Fixed Income":  {"10Y Treasury":"^TNX","2Y Treasury":"^IRX","TLT (20Y)":"TLT","HYG (HY Bond)":"HYG","LQD (IG Bond)":"LQD"},
        "Commodities":   {"Crude Oil":"CL=F","Natural Gas":"NG=F","Gold":"GC=F","Silver":"SI=F","Copper":"HG=F","Wheat":"ZW=F"},
        "Currencies":    {"USD Index":"DX-Y.NYB","EUR/USD":"EURUSD=X","GBP/USD":"GBPUSD=X","USD/JPY":"JPY=X","USD/CNY":"CNY=X","USD/INR":"INR=X"},
        "Crypto":        {"Bitcoin":"BTC-USD","Ethereum":"ETH-USD","Solana":"SOL-USD"},
    }
    # Flatten all tickers, fetch all in parallel, then re-group
    all_tasks = []  # (cat, name, ticker)
    for cat, tickers in categories.items():
        for name, ticker in tickers.items():
            all_tasks.append((cat, name, ticker))

    cat_order = {n: i for i, n in enumerate(categories[list(categories.keys())[0]])}
    results = {cat: {} for cat in categories}

    with ThreadPoolExecutor(max_workers=30) as ex:
        futures = {ex.submit(_fetch_one_ticker, name, ticker): (cat, name)
                   for cat, name, ticker in all_tasks}
        for f in as_completed(futures):
            cat, _ = futures[f]
            name, data = f.result()
            results[cat][name] = data

    # Restore original order per category
    ordered = {}
    for cat, tickers in categories.items():
        ordered[cat] = [results[cat].get(name, {"name":name,"price":"N/A","chg_str":"—","chg_num":"—","up":True,"chg_val":0})
                        for name in tickers]
    return ordered

def fetch_sparklines():
    print("  Fetching sparkline data...")
    tickers = {"S&P 500":"^GSPC","10Y Treasury":"^TNX","Crude Oil":"CL=F","Gold":"GC=F"}

    def _fetch(name, ticker):
        try:
            hist = yf.Ticker(ticker).history(period="30d")
            if len(hist) > 0:
                return name, {
                    "prices": [round(float(v), 2) for v in hist["Close"].tolist()],
                    "dates":  [str(d.date()) for d in hist.index.tolist()]
                }
        except:
            pass
        return name, {"prices": [], "dates": []}

    result = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        for name, data in ex.map(lambda kv: _fetch(*kv), tickers.items()):
            result[name] = data
    return result

# ── AGENT RUNNER ──────────────────────────────────────────────────────────────
def run_agent(client, system_prompt, user_content, max_tokens=2000):
    msg = client.messages.create(
        model="claude-opus-4-6", max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role":"user","content":user_content}]
    )
    return msg.content[0].text

# ── MULTI-AGENT ───────────────────────────────────────────────────────────────
def run_all_agents(macro, headlines, markets):
    macro_text     = "\n".join([f"{m['label']}: {m['value']}" for m in macro])
    headlines_text = "\n".join([f"[{h['source']}] {h['title']}" for h in headlines])
    markets_text   = ""
    for cat, items in markets.items():
        markets_text += f"\n{cat}:\n" + "\n".join([f"  {i['name']}: {i['price']} ({i['chg_str']})" for i in items])
    base = f"MACRO:\n{macro_text}\n\nMARKETS:\n{markets_text}\n\nNEWS:\n{headlines_text}"
    headlines_for_bias = "\n".join([f"{i+1}. [{h['source']}] {h['title']}" for i,h in enumerate(headlines)])

    print("  Single comprehensive analysis call...")
    prompt = f"""{base}

You are a team of 8 specialist analysts. Return ONLY a single valid JSON object with ALL of these keys. No markdown, no preamble.

{{
  "macro_regime": {{
    "regime": "Regime label",
    "regime_color": "red",
    "regime_summary": "2-3 sentences",
    "top_drivers": [
      {{"title": "Driver name", "what_changed": "1 sentence", "market_impact": "1 sentence", "winners": "assets/sectors", "losers": "assets/sectors"}}
    ],
    "cross_asset": [
      {{"signal": "Equities", "reading": "Bearish", "color": "red", "detail": "1 sentence"}},
      {{"signal": "Yields", "reading": "Rising", "color": "red", "detail": "1 sentence"}},
      {{"signal": "Oil", "reading": "Bullish", "color": "green", "detail": "1 sentence"}},
      {{"signal": "Gold", "reading": "Defensive", "color": "amber", "detail": "1 sentence"}},
      {{"signal": "USD", "reading": "Weak", "color": "red", "detail": "1 sentence"}},
      {{"signal": "VIX", "reading": "Elevated", "color": "red", "detail": "1 sentence"}}
    ]
  }},
  "impact_cards": [
    {{
      "title": "Short event title",
      "headline": "The actual headline driving this",
      "why_it_matters": "2 sentences max",
      "sectors": [{{"name": "Energy", "direction": "bullish"}}],
      "assets": [{{"name": "Crude Oil", "direction": "bullish"}}],
      "overall": "bearish",
      "horizon": "near-term",
      "score": "8/10"
    }}
  ],
  "geo_risk": [
    {{
      "title": "Short risk title",
      "region": "Region",
      "trigger": "1-2 sentences",
      "market_exposure": "2 sentences",
      "sectors_impacted": "comma separated with direction",
      "asset_impact": "comma separated with direction",
      "risk_level": "HIGH",
      "what_to_watch": "1-2 sentences"
    }}
  ],
  "cb_watch": [
    {{
      "bank": "Federal Reserve",
      "stance": "Hawkish Hold",
      "stance_color": "red",
      "latest_signals": "2 sentences",
      "market_interpretation": "2 sentences",
      "yields_impact": "1 sentence",
      "equities_impact": "1 sentence",
      "usd_impact": "1 sentence",
      "key_risk": "1 sentence"
    }}
  ],
  "sector_map": [
    {{
      "sector": "Technology",
      "sentiment": "Cautiously Bullish",
      "sentiment_color": "green",
      "tailwinds": ["bullet 1", "bullet 2", "bullet 3"],
      "headwinds": ["bullet 1", "bullet 2"],
      "macro_sensitivity": "Rate-sensitive / Growth-dependent",
      "key_catalyst": "1 sentence",
      "key_names": "NVDA, MSFT, AAPL"
    }}
  ],
  "sentiment": {{
    "bullish_themes": [{{"theme": "Theme", "detail": "2 sentences", "assets": "assets"}}],
    "bearish_themes": [{{"theme": "Theme", "detail": "2 sentences", "assets": "assets"}}],
    "emerging_debates": [{{"debate": "Debate", "bull_case": "1 sentence", "bear_case": "1 sentence"}}],
    "rotation_signals": [{{"from": "asset", "to": "asset", "signal": "1 sentence"}}]
  }},
  "strategist": {{
    "headline": "One punchy sentence summarizing today",
    "paragraphs": ["paragraph 1 (3-4 sentences)", "paragraph 2 (3-4 sentences)", "paragraph 3 (3-4 sentences)"],
    "biggest_risk": "1 sentence",
    "most_sensitive_sector": "sector and why",
    "watchlist": [
      {{"event": "Event name", "why_now": "1 sentence", "bullish": "1 sentence", "bearish": "1 sentence", "assets": "key assets"}}
    ]
  }},
  "news_bias": [{{"bias": "center"}}]
}}

Requirements: 5 top_drivers, 6 cross_asset (Equities Yields Oil Gold USD VIX), 6 impact_cards, 5 geo_risk, 5 cb_watch (Fed ECB BoJ BoE PBOC), 8 sector_map (Technology Financials Energy Healthcare Industrials Consumer-Disc Real-Estate Utilities), 3 bullish_themes, 3 bearish_themes, 2 emerging_debates, 4 rotation_signals, 3 brief paragraphs, 5 watchlist items, {len(headlines)} news_bias objects. Colors: red green amber only."""

    msg = _client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=16000,
        messages=[{"role": "user", "content": prompt}],
    )
    text = msg.content[0].text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    combined = json_lib.loads(text)

    return {
        "macro_regime": json_lib.dumps(combined.get("macro_regime", {})),
        "impact_cards": json_lib.dumps(combined.get("impact_cards", [])),
        "geo_risk":     json_lib.dumps(combined.get("geo_risk", [])),
        "cb_watch":     json_lib.dumps(combined.get("cb_watch", [])),
        "sector_map":   json_lib.dumps(combined.get("sector_map", [])),
        "sentiment":    json_lib.dumps(combined.get("sentiment", {})),
        "strategist":   json_lib.dumps(combined.get("strategist", {})),
        "news_bias":    json_lib.dumps(combined.get("news_bias", [])),
    }


def get_or_build_html():
    global _cache, _building
    if _cache["html"] and _cache["ts"] > 0 and time.time() < _cache["ts"]:
        return _cache["html"]
    with _build_lock:
        # Double-check after acquiring lock
        if _cache["html"] and _cache["ts"] > 0 and time.time() < _cache["ts"]:
            return _cache["html"]
        _building = True
        try:
            print("[ 1/4 ] Fetching all data in parallel...")
            with ThreadPoolExecutor(max_workers=5) as ex:
                f_macro      = ex.submit(fetch_macro)
                f_news       = ex.submit(fetch_news)
                f_markets    = ex.submit(fetch_markets)
                f_chart      = ex.submit(fetch_chart_data)
                f_sparklines = ex.submit(fetch_sparklines)
                macro      = f_macro.result()
                headlines  = f_news.result()
                markets    = f_markets.result()
                chart_data = f_chart.result()
                sparklines = f_sparklines.result()
            print("[ 2/4 ] Data fetched. Running AI analysis...")
            try:
                analysis = run_all_agents(macro, headlines, markets)
            except Exception as e:
                print(f"  Analysis error: {e}")
                analysis = {k: "{}" if k not in ("impact_cards","geo_risk","cb_watch","sector_map","news_bias") else "[]"
                            for k in ("macro_regime","impact_cards","geo_risk","cb_watch","sector_map","sentiment","strategist","news_bias")}
            print("[ 3/4 ] Building HTML...")
            html = build_html(macro, headlines, markets, analysis, chart_data, sparklines)
            _cache["html"] = html
            _cache["macro"] = macro
            _cache["headlines"] = headlines
            _cache["ts"] = time.time() + _cache_ttl()  # expire at market close / midnight ET
            ttl_mins = int(_cache_ttl() / 60)
            print(f"[ 4/4 ] Done. Cache expires in ~{ttl_mins} min (market close / midnight ET).")
        finally:
            _building = False
    return _cache["html"]


def _start_background_build():
    """Fire-and-forget: warm the cache in a background thread so the first HTTP request returns instantly."""
    def _run():
        try:
            get_or_build_html()
        except Exception as e:
            print(f"Background build failed: {e}")
    t = threading.Thread(target=_run, daemon=True)
    t.start()


_LOADING_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Macro Dashboard — Loading…</title>
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  body{background:#0a0a0f;color:#e2e8f0;font-family:'Inter',sans-serif;
       display:flex;flex-direction:column;align-items:center;justify-content:center;
       min-height:100vh;text-align:center;gap:24px}
  .logo{font-size:2rem;font-weight:700;letter-spacing:-.5px}
  .logo span{color:#3b82f6}
  .spinner{width:48px;height:48px;border:3px solid #1e293b;border-top-color:#3b82f6;
           border-radius:50%;animation:spin 1s linear infinite}
  @keyframes spin{to{transform:rotate(360deg)}}
  .msg{color:#94a3b8;font-size:.95rem;max-width:360px;line-height:1.6}
  .steps{display:flex;flex-direction:column;gap:8px;margin-top:8px}
  .step{font-size:.8rem;color:#64748b;display:flex;align-items:center;gap:8px}
  .dot{width:6px;height:6px;border-radius:50%;background:#3b82f6;
       animation:pulse 1.5s ease-in-out infinite}
  @keyframes pulse{0%,100%{opacity:.3}50%{opacity:1}}
</style>
</head>
<body>
<div class="logo">Macro<span>Dashboard</span></div>
<div class="spinner"></div>
<div class="msg">
  Fetching live market data &amp; running AI analysis…<br>
  <strong style="color:#e2e8f0">This takes about 60–90 seconds on first load.</strong>
</div>
<div class="steps">
  <div class="step"><div class="dot"></div>Pulling FRED economic indicators</div>
  <div class="step"><div class="dot" style="animation-delay:.3s"></div>Fetching market prices &amp; news</div>
  <div class="step"><div class="dot" style="animation-delay:.6s"></div>Running Claude AI macro analysis</div>
</div>
<script>
  (function poll(){
    fetch('/api/status').then(r=>r.json()).then(d=>{
      if(d.ready){ window.location.reload(); }
      else { setTimeout(poll, 4000); }
    }).catch(()=>setTimeout(poll,5000));
  })();
</script>
</body>
</html>"""


def run_agent(client, system_prompt, user_content, max_tokens=2000):
    pass  # no longer used — run_all_agents now makes a single consolidated call

# ── JSON SAFE PARSE ───────────────────────────────────────────────────────────
def safe_json(text, fallback):
    try:
        text = text.strip()
        start = text.find("[") if "[" in text else text.find("{")
        if "[" in text and "{" in text:
            start = min(text.find("["), text.find("{"))
            if text.find("[") < text.find("{"):
                end = text.rfind("]") + 1
            else:
                end = text.rfind("}") + 1
        elif "[" in text:
            start = text.find("["); end = text.rfind("]") + 1
        else:
            start = text.find("{"); end = text.rfind("}") + 1
        return json_lib.loads(text[start:end])
    except:
        return fallback

# ── COLOR HELPERS ─────────────────────────────────────────────────────────────
def color_cls(c):
    return {"green":"clr-green","red":"clr-red","amber":"clr-amber","blue":"clr-blue"}.get(str(c).lower(),"clr-muted")

def dir_color(d):
    d = str(d).lower()
    if any(w in d for w in ["bullish","positive","up"]): return "clr-green"
    if any(w in d for w in ["bearish","negative","down"]): return "clr-red"
    return "clr-amber"

def risk_cls(r):
    r = str(r).upper()
    if "HIGH" in r: return "badge-red"
    if "MEDIUM" in r or "MODERATE" in r: return "badge-amber"
    return "badge-green"

def sentiment_cls(s):
    s = str(s).lower()
    if any(w in s for w in ["bullish","positive","constructive"]): return "badge-green"
    if any(w in s for w in ["bearish","negative","cautious"]): return "badge-red"
    return "badge-amber"

# ── BUILD HTML ────────────────────────────────────────────────────────────────
def build_html(macro, headlines, markets, analysis, chart_data, sparklines):
    now = datetime.now().strftime("%A, %B %d, %Y — %I:%M %p")

    # Parse all JSON responses
    regime   = safe_json(analysis["macro_regime"],  {})
    impacts  = safe_json(analysis["impact_cards"],  [])
    geo      = safe_json(analysis["geo_risk"],      [])
    cb       = safe_json(analysis["cb_watch"],      [])
    sectors  = safe_json(analysis["sector_map"],    [])
    sent     = safe_json(analysis["sentiment"],     {})
    strat    = safe_json(analysis["strategist"],    {})
    biases   = safe_json(analysis["news_bias"],     [])

    # Ticker
    ticker_html = ""
    for cat, items in markets.items():
        for m in items:
            c = "#22c55e" if m["up"] else "#ef4444"
            ticker_html += f'<span class="ti"><span class="tn">{m["name"]}</span><span class="tp">{m["price"]}</span><span style="color:{c}">{m["chg_num"]}</span></span>'

    def gv(lbl): return next((m["value"] for m in macro if lbl in m["label"]), "N/A")
    fed=gv("Fed Funds"); unem=gv("Unemployment"); cpi=gv("CPI Index")
    sprd=gv("10Y-2Y"); sent_val=gv("Consumer Sentiment"); wti=gv("WTI")

    sparkline_js  = "const sparklineData = " + json_lib.dumps(sparklines) + ";"
    geo_risk_js   = "const GEO_RISK_DATA = " + json_lib.dumps(geo) + ";"
    fed_labels   = json_lib.dumps([d for d,v in chart_data["fed"]])
    fed_values   = json_lib.dumps([v for d,v in chart_data["fed"]])
    cpi_labels   = json_lib.dumps([d for d,v in chart_data["cpi"]])
    cpi_values   = json_lib.dumps([v for d,v in chart_data["cpi"]])

    sector_items      = markets.get("Sector ETFs", [])
    sector_names      = json_lib.dumps([m["name"] for m in sector_items])
    sector_values     = json_lib.dumps([round(m["chg_val"], 2) for m in sector_items])
    sector_colors     = json_lib.dumps(["#22c55e" if m["up"] else "#ef4444" for m in sector_items])

    # ── OVERVIEW ─────────────────────────────────────────────────────────────
    regime_color = color_cls(regime.get("regime_color","amber"))
    regime_label = regime.get("regime","—")
    regime_summary = regime.get("regime_summary","—")

    cross_asset_html = ""
    for ca in regime.get("cross_asset",[]):
        cc = color_cls(ca.get("color","amber"))
        cross_asset_html += f'''<div class="ca-row">
          <span class="ca-signal">{ca.get("signal","—")}</span>
          <span class="ca-reading {cc}">{ca.get("reading","—")}</span>
          <span class="ca-detail">{ca.get("detail","—")}</span>
        </div>'''

    drivers_html = ""
    for d in regime.get("top_drivers",[]):
        drivers_html += f'''<div class="driver-card">
          <div class="driver-title">{d.get("title","—")}</div>
          <div class="driver-row"><span class="driver-lbl">CHANGED</span><span class="driver-val">{d.get("what_changed","—")}</span></div>
          <div class="driver-row"><span class="driver-lbl">IMPACT</span><span class="driver-val">{d.get("market_impact","—")}</span></div>
          <div class="driver-row"><span class="driver-lbl">WINNERS</span><span class="driver-val clr-green">{d.get("winners","—")}</span></div>
          <div class="driver-row"><span class="driver-lbl">LOSERS</span><span class="driver-val clr-red">{d.get("losers","—")}</span></div>
        </div>'''

    # ── MORNING BRIEF ─────────────────────────────────────────────────────────
    brief_headline = strat.get("headline","—")
    brief_paras    = "".join([f'<p class="brief-para">{p}</p>' for p in strat.get("paragraphs",[])])
    brief_risk     = strat.get("biggest_risk","—")
    brief_sector   = strat.get("most_sensitive_sector","—")
    watchlist_html = ""
    for w in strat.get("watchlist",[]):
        watchlist_html += f'''<tr>
          <td class="wl-event">{w.get("event","—")}</td>
          <td>{w.get("why_now","—")}</td>
          <td class="clr-green">{w.get("bullish","—")}</td>
          <td class="clr-red">{w.get("bearish","—")}</td>
          <td class="clr-amber">{w.get("assets","—")}</td>
        </tr>'''

    # ── MARKET IMPACT ─────────────────────────────────────────────────────────
    impact_html = ""
    for card in impacts:
        ov  = card.get("overall","neutral")
        ovc = dir_color(ov)
        sectors_html = "".join([f'<span class="tag {dir_color(s.get("direction","neutral"))}">{s.get("name","")}: {s.get("direction","").title()}</span>' for s in card.get("sectors",[])])
        assets_html  = "".join([f'<span class="tag {dir_color(a.get("direction","neutral"))}">{a.get("name","")}: {a.get("direction","").title()}</span>' for a in card.get("assets",[])])
        impact_html += f'''<div class="impact-card">
          <div class="impact-hdr">
            <span class="impact-title">{card.get("title","—")}</span>
            <div style="display:flex;gap:6px;align-items:center;">
              <span class="badge {dir_color(ov).replace('clr-','badge-')}">{ov.upper()}</span>
              <span class="badge badge-muted">{card.get("horizon","—").upper()}</span>
              <span class="badge badge-amber">SCORE {card.get("score","—")}</span>
            </div>
          </div>
          <div class="impact-body">
            <div class="impact-headline">"{card.get("headline","—")}"</div>
            <p class="impact-why">{card.get("why_it_matters","—")}</p>
            <div class="tag-row"><span class="tag-lbl">SECTORS</span>{sectors_html}</div>
            <div class="tag-row"><span class="tag-lbl">ASSETS</span>{assets_html}</div>
          </div>
        </div>'''

    # ── GEO RISK ─────────────────────────────────────────────────────────────
    geo_rows = ""
    for i, g in enumerate(geo):
        rl  = g.get("risk_level","MEDIUM")
        geo_rows += f'''<tr class="geo-tr" onclick="toggleGeo({i})">
          <td><span class="geo-title">{g.get("title","—")}</span></td>
          <td class="geo-region">{g.get("region","—")}</td>
          <td><span class="badge {risk_cls(rl)}">{rl}</span></td>
          <td class="geo-sectors">{g.get("sectors_impacted","—")}</td>
          <td class="geo-expand">▼</td>
        </tr>
        <tr id="geo-detail-{i}" class="geo-detail-row" style="display:none;">
          <td colspan="5">
            <div class="geo-detail">
              <div class="geo-detail-grid">
                <div><span class="dl">TRIGGER</span><p>{g.get("trigger","—")}</p></div>
                <div><span class="dl">MARKET EXPOSURE</span><p>{g.get("market_exposure","—")}</p></div>
                <div><span class="dl">ASSET IMPACT</span><p>{g.get("asset_impact","—")}</p></div>
                <div><span class="dl">WHAT TO WATCH</span><p>{g.get("what_to_watch","—")}</p></div>
              </div>
            </div>
          </td>
        </tr>'''

    # Conflict Map tab cards (reuses geo data)
    geo_cards_html = ""
    for i, g in enumerate(geo):
        rl = g.get("risk_level","MEDIUM")
        rc = risk_cls(rl)
        geo_cards_html += f'''<div class="card al-r" style="cursor:pointer;" onclick="showConflictDetail({i})">
          <div class="card-hdr"><span class="card-title">{g.get("title","—")}</span><span class="badge {rc}">{rl}</span></div>
          <div class="card-body" style="font-size:.83rem;">
            <div style="color:var(--sub);font-size:.72rem;margin-bottom:4px;">REGION</div>
            <div style="margin-bottom:8px;font-weight:600;">{g.get("region","—")}</div>
            <div style="color:var(--sub);font-size:.72rem;margin-bottom:4px;">TRIGGER</div>
            <div style="margin-bottom:8px;line-height:1.5;">{g.get("trigger","—")}</div>
            <div style="color:var(--sub);font-size:.72rem;margin-bottom:4px;">SECTORS IMPACTED</div>
            <div style="color:var(--amber);">{g.get("sectors_impacted","—")}</div>
          </div>
        </div>'''

    # ── CENTRAL BANKS ─────────────────────────────────────────────────────────
    cb_html = ""
    for bank in cb:
        sc = color_cls(bank.get("stance_color","amber"))
        cb_html += f'''<div class="cb-card">
          <div class="cb-hdr">
            <span class="cb-name">{bank.get("bank","—")}</span>
            <span class="cb-stance {sc}">{bank.get("stance","—")}</span>
          </div>
          <div class="cb-body">
            <div class="cb-section"><span class="cb-lbl">LATEST SIGNALS</span><p>{bank.get("latest_signals","—")}</p></div>
            <div class="cb-section"><span class="cb-lbl">MARKET INTERPRETATION</span><p>{bank.get("market_interpretation","—")}</p></div>
            <div class="cb-impacts">
              <div><span class="cb-lbl">YIELDS</span><p>{bank.get("yields_impact","—")}</p></div>
              <div><span class="cb-lbl">EQUITIES</span><p>{bank.get("equities_impact","—")}</p></div>
              <div><span class="cb-lbl">USD</span><p>{bank.get("usd_impact","—")}</p></div>
            </div>
            <div class="cb-risk"><span class="cb-lbl">KEY RISK</span> {bank.get("key_risk","—")}</div>
          </div>
        </div>'''

    # ── SECTORS ──────────────────────────────────────────────────────────────
    sector_cards_html = ""
    for sec in sectors:
        sc     = sentiment_cls(sec.get("sentiment",""))
        tails  = "".join([f'<li>{t}</li>' for t in sec.get("tailwinds",[])])
        heads  = "".join([f'<li>{h}</li>' for h in sec.get("headwinds",[])])
        sector_cards_html += f'''<div class="sec-card">
          <div class="sec-hdr">
            <span class="sec-name">{sec.get("sector","—")}</span>
            <span class="badge {sc}">{sec.get("sentiment","—")}</span>
          </div>
          <div class="sec-body">
            <div class="sec-col">
              <span class="sec-lbl clr-green">▲ TAILWINDS</span>
              <ul class="sec-list">{tails}</ul>
            </div>
            <div class="sec-col">
              <span class="sec-lbl clr-red">▼ HEADWINDS</span>
              <ul class="sec-list">{heads}</ul>
            </div>
          </div>
          <div class="sec-footer">
            <div><span class="sec-lbl">MACRO SENSITIVITY</span> {sec.get("macro_sensitivity","—")}</div>
            <div><span class="sec-lbl">KEY CATALYST</span> {sec.get("key_catalyst","—")}</div>
            <div><span class="sec-lbl">WATCH</span> <span class="clr-amber">{sec.get("key_names","—")}</span></div>
          </div>
        </div>'''

    # ── SENTIMENT ─────────────────────────────────────────────────────────────
    bullish_html = "".join([f'''<div class="sent-item">
      <div class="sent-theme clr-green">{t.get("theme","—")}</div>
      <p class="sent-detail">{t.get("detail","—")}</p>
      <span class="sent-assets">{t.get("assets","—")}</span>
    </div>''' for t in sent.get("bullish_themes",[])])

    bearish_html = "".join([f'''<div class="sent-item">
      <div class="sent-theme clr-red">{t.get("theme","—")}</div>
      <p class="sent-detail">{t.get("detail","—")}</p>
      <span class="sent-assets">{t.get("assets","—")}</span>
    </div>''' for t in sent.get("bearish_themes",[])])

    debates_html = "".join([f'''<div class="debate-item">
      <div class="debate-title">{d.get("debate","—")}</div>
      <div class="debate-sides">
        <div><span class="clr-green">BULL:</span> {d.get("bull_case","—")}</div>
        <div><span class="clr-red">BEAR:</span> {d.get("bear_case","—")}</div>
      </div>
    </div>''' for d in sent.get("emerging_debates",[])])

    rotation_html = "".join([f'''<tr>
      <td class="rot-from">{r.get("from","—")}</td>
      <td class="rot-arrow">→</td>
      <td class="rot-to clr-green">{r.get("to","—")}</td>
      <td class="rot-signal">{r.get("signal","—")}</td>
    </tr>''' for r in sent.get("rotation_signals",[])])

    # ── NEWS IMPACT HEADLINES ─────────────────────────────────────────────────
    ni_headlines_html = ""
    for h in headlines[:24]:
        safe_title = h["title"].replace("'", "\\'").replace('"', '&quot;')
        ni_headlines_html += f'''<div class="ni-row">
          <div style="flex:1;">
            <div style="font-size:10px;color:var(--sub);margin-bottom:3px;">{h["source"]}</div>
            <div style="font-size:13px;color:var(--text);">{h["title"]}</div>
          </div>
          <button class="ni-analyze" onclick="showNIResult(this,'{safe_title}')">Analyze</button>
          <div class="ni-result"></div>
        </div>'''

    # ── NEWS FEED ─────────────────────────────────────────────────────────────
    news_html = ""
    if not headlines:
        news_html = '<div style="padding:24px;text-align:center;color:var(--sub);font-size:13px;">No headlines loaded — RSS feeds may be temporarily unavailable. Re-run the script to retry.</div>'
    for i, h in enumerate(headlines):
        bias = biases[i]["bias"] if i < len(biases) and isinstance(biases[i], dict) else "center"
        bias_color = {"left":"#3b82f6","center":"#6b7280","right":"#ef4444"}.get(bias,"#6b7280")
        bias_label = {"left":"LEFT","center":"CENTER","right":"RIGHT"}.get(bias,"CENTER")
        news_html += f'''<div class="news-row">
          <div class="news-bias-dot" style="background:{bias_color};" title="{bias_label}"></div>
          <div class="news-content">
            <div class="news-meta"><span class="news-src">{h["source"]}</span><span class="news-bias-lbl" style="color:{bias_color};">{bias_label}</span></div>
            <a href="{h["link"]}" target="_blank" class="news-title">{h["title"]}</a>
          </div>
        </div>'''

    # ── MARKETS ──────────────────────────────────────────────────────────────
    def mkt_table(cat_name, items):
        rows = "".join([f'<tr><td>{m["name"]}</td><td class="mkt-price">{m["price"]}</td><td class="{"clr-green" if m["up"] else "clr-red"} mkt-chg">{m["chg_num"]}</td></tr>' for m in items])
        return f'<div class="mkt-block"><div class="mkt-block-title">{cat_name}</div><table class="mkt-tbl"><thead><tr><th>Name</th><th>Price</th><th>Chg%</th></tr></thead><tbody>{rows}</tbody></table></div>'

    markets_html = '<div class="mkt-grid">' + "".join([mkt_table(cat, items) for cat, items in markets.items()]) + '</div>'

    # ── INDICATORS ───────────────────────────────────────────────────────────
    indicators_rows = "".join([f'<tr><td class="ind-lbl">{m["label"]}</td><td class="ind-val">{m["value"]}</td></tr>' for m in macro])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Weekly Macroeconomic Monitoring &amp; Market Conditions Brief</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@600;700&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{{
  --bg:#0a0a0a;--s1:#111111;--s2:#1a1a1a;--s3:#222222;
  --border:#2a2a2a;--border2:#333333;
  --text:#f0f0f0;--sub:#a0a0a0;--dim:#555;
  --green:#22c55e;--red:#ef4444;--amber:#f59e0b;--blue:#3b82f6;--purple:#8b5cf6;--cyan:#06b6d4;
  --sans:'Inter',sans-serif;--head:'Space Grotesk',sans-serif;
}}
*{{box-sizing:border-box;margin:0;padding:0;}}
html{{scroll-behavior:smooth;}}
body{{font-family:var(--sans);background:var(--bg);color:var(--text);font-size:13px;line-height:1.5;}}

/* COLORS */
.clr-green{{color:var(--green);}}.clr-red{{color:var(--red);}}.clr-amber{{color:var(--amber);}}.clr-blue{{color:var(--blue);}}.clr-muted{{color:var(--sub);}}

/* BADGES */
.badge{{font-size:10px;font-weight:600;padding:2px 8px;border-radius:3px;letter-spacing:0.06em;text-transform:uppercase;white-space:nowrap;}}
.badge-green{{background:rgba(34,197,94,0.15);color:var(--green);border:1px solid rgba(34,197,94,0.3);}}
.badge-red{{background:rgba(239,68,68,0.15);color:var(--red);border:1px solid rgba(239,68,68,0.3);}}
.badge-amber{{background:rgba(245,158,11,0.15);color:var(--amber);border:1px solid rgba(245,158,11,0.3);}}
.badge-blue{{background:rgba(59,130,246,0.15);color:var(--blue);border:1px solid rgba(59,130,246,0.3);}}
.badge-muted{{background:rgba(160,160,160,0.1);color:var(--sub);border:1px solid rgba(160,160,160,0.2);}}

/* HEADER */
.hdr{{position:sticky;top:0;z-index:300;background:rgba(10,10,10,0.97);backdrop-filter:blur(20px);border-bottom:1px solid var(--border);height:52px;display:flex;align-items:center;justify-content:space-between;padding:0 24px;}}
.hdr-logo{{font-family:var(--head);font-size:11px;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:var(--text);}}
.hdr-right{{display:flex;align-items:center;gap:12px;}}
.live-pill{{display:flex;align-items:center;gap:5px;background:rgba(34,197,94,0.08);border:1px solid rgba(34,197,94,0.25);border-radius:20px;padding:3px 10px;font-size:9px;font-weight:600;color:var(--green);letter-spacing:0.1em;}}
.live-dot{{width:5px;height:5px;border-radius:50%;background:var(--green);animation:blink 2s infinite;}}
@keyframes blink{{0%,100%{{opacity:1;}}50%{{opacity:0.2;}}}}
.hdr-time{{font-size:10px;color:var(--sub);}}

/* TICKER */
.ticker-bar{{overflow:hidden;background:var(--s1);border-bottom:1px solid var(--border);padding:8px 0;}}
.ticker-inner{{display:flex;gap:0;animation:scroll 90s linear infinite;white-space:nowrap;}}
.ticker-inner:hover{{animation-play-state:paused;}}
@keyframes scroll{{from{{transform:translateX(0);}}to{{transform:translateX(-50%);}}}}
.ti{{display:inline-flex;align-items:center;gap:7px;padding:0 16px;border-right:1px solid var(--border);font-size:11px;}}
.tn{{color:var(--sub);font-weight:500;}}.tp{{color:var(--text);font-weight:600;}}

/* NAV */
.nav{{background:var(--s1);border-bottom:1px solid var(--border);display:flex;overflow-x:auto;padding:0 24px;}}
.nav::-webkit-scrollbar{{height:0;}}
.ntab{{padding:13px 18px;font-size:10px;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;color:var(--sub);cursor:pointer;border-bottom:2px solid transparent;white-space:nowrap;transition:all 0.18s;user-select:none;font-family:var(--head);}}
.ntab:hover{{color:var(--text);}}.ntab.active{{color:var(--blue);border-bottom-color:var(--blue);}}

/* LAYOUT */
.wrap{{max-width:1500px;margin:0 auto;padding:24px;}}
.tab{{display:none;}}.tab.active{{display:block;}}

/* SECTION HEADERS */
.sec-hdr-row{{display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;margin-top:28px;}}
.sec-hdr-row:first-child{{margin-top:0;}}
.sec-hdr{{font-family:var(--head);font-size:15px;font-weight:700;color:var(--text);}}
.sec-sub{{font-size:11px;color:var(--sub);margin-top:2px;}}

/* CARDS */
.card{{background:var(--s1);border:1px solid var(--border);border-radius:8px;overflow:hidden;}}
.card-hdr{{padding:12px 16px;border-bottom:1px solid var(--border);background:var(--s2);display:flex;align-items:center;justify-content:space-between;}}
.card-title{{font-family:var(--head);font-size:11px;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;color:var(--sub);}}
.card-body{{padding:16px;}}

/* STAT ROW */
.stat-row{{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin-bottom:24px;}}
.stat{{background:var(--s1);border:1px solid var(--border);border-radius:8px;padding:14px 16px;}}
.stat-lbl{{font-size:9px;color:var(--sub);letter-spacing:0.1em;text-transform:uppercase;margin-bottom:6px;font-weight:500;}}
.stat-val{{font-size:22px;font-weight:600;color:var(--text);font-family:var(--head);}}
.stat-sub2{{font-size:10px;color:var(--dim);margin-top:3px;}}

/* SPARKLINES */
.spark-row{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:24px;}}
.spark-card{{background:var(--s1);border:1px solid var(--border);border-radius:8px;padding:14px;}}
.spark-name{{font-size:10px;color:var(--sub);letter-spacing:0.06em;text-transform:uppercase;margin-bottom:8px;font-weight:500;}}

/* REGIME SECTION */
.regime-grid{{display:grid;grid-template-columns:280px 1fr 1fr;gap:16px;margin-bottom:24px;}}
.regime-badge-wrap{{display:flex;flex-direction:column;justify-content:center;align-items:center;background:var(--s1);border:1px solid var(--border);border-radius:8px;padding:24px 16px;text-align:center;gap:10px;}}
.regime-label{{font-family:var(--head);font-size:13px;font-weight:700;}}
.regime-summary{{font-size:12px;color:var(--sub);line-height:1.6;margin-top:6px;}}
.ca-list{{background:var(--s1);border:1px solid var(--border);border-radius:8px;overflow:hidden;}}
.ca-row{{display:grid;grid-template-columns:90px 120px 1fr;gap:12px;align-items:start;padding:10px 16px;border-bottom:1px solid var(--border);}}
.ca-row:last-child{{border-bottom:none;}}
.ca-signal{{font-size:11px;color:var(--sub);font-weight:500;}}
.ca-reading{{font-size:11px;font-weight:600;}}
.ca-detail{{font-size:11px;color:var(--sub);}}
.drivers-list{{background:var(--s1);border:1px solid var(--border);border-radius:8px;overflow:hidden;}}
.driver-card{{padding:12px 16px;border-bottom:1px solid var(--border);}}
.driver-card:last-child{{border-bottom:none;}}
.driver-title{{font-family:var(--head);font-size:12px;font-weight:700;color:var(--text);margin-bottom:8px;}}
.driver-row{{display:flex;gap:8px;margin-bottom:4px;align-items:baseline;}}
.driver-lbl{{font-size:9px;font-weight:600;color:var(--dim);letter-spacing:0.08em;text-transform:uppercase;min-width:55px;flex-shrink:0;}}
.driver-val{{font-size:12px;color:var(--sub);line-height:1.5;}}

/* BRIEF */
.brief-headline{{font-family:var(--head);font-size:18px;font-weight:700;color:var(--text);line-height:1.4;margin-bottom:20px;padding-bottom:16px;border-bottom:1px solid var(--border);}}
.brief-para{{font-size:13px;color:#c0c0c0;line-height:1.8;margin-bottom:14px;}}
.brief-callout{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:20px;}}
.brief-callout-item{{background:var(--s2);border:1px solid var(--border);border-radius:6px;padding:12px 14px;}}
.brief-callout-lbl{{font-size:9px;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:var(--sub);margin-bottom:6px;}}
.brief-callout-val{{font-size:12px;color:var(--text);line-height:1.5;}}
.watchlist-tbl{{width:100%;border-collapse:collapse;margin-top:0;}}
.watchlist-tbl th{{font-size:9px;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;color:var(--dim);padding:8px 12px;border-bottom:1px solid var(--border);text-align:left;background:var(--s2);}}
.watchlist-tbl td{{font-size:12px;color:var(--sub);padding:10px 12px;border-bottom:1px solid var(--border);vertical-align:top;line-height:1.5;}}
.watchlist-tbl tr:last-child td{{border-bottom:none;}}
.watchlist-tbl tr:hover td{{background:var(--s2);}}
.wl-event{{color:var(--text);font-weight:500;}}

/* IMPACT CARDS */
.impact-grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px;}}
.impact-card{{background:var(--s1);border:1px solid var(--border);border-radius:8px;overflow:hidden;}}
.impact-hdr{{padding:12px 16px;background:var(--s2);border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap;}}
.impact-title{{font-family:var(--head);font-size:12px;font-weight:700;color:var(--text);}}
.impact-body{{padding:14px 16px;}}
.impact-headline{{font-size:12px;color:var(--sub);font-style:italic;margin-bottom:10px;line-height:1.5;border-left:3px solid var(--border2);padding-left:10px;}}
.impact-why{{font-size:12px;color:#b0b0b0;line-height:1.6;margin-bottom:12px;}}
.tag-row{{display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-bottom:6px;}}
.tag-lbl{{font-size:9px;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;color:var(--dim);min-width:50px;}}
.tag{{font-size:10px;padding:2px 8px;border-radius:3px;font-weight:500;}}
.tag.clr-green{{background:rgba(34,197,94,0.1);color:var(--green);border:1px solid rgba(34,197,94,0.2);}}
.tag.clr-red{{background:rgba(239,68,68,0.1);color:var(--red);border:1px solid rgba(239,68,68,0.2);}}
.tag.clr-amber{{background:rgba(245,158,11,0.1);color:var(--amber);border:1px solid rgba(245,158,11,0.2);}}

/* GEO RISK */
.geo-tbl{{width:100%;border-collapse:collapse;}}
.geo-tbl th{{font-size:9px;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;color:var(--dim);padding:10px 14px;border-bottom:1px solid var(--border);text-align:left;background:var(--s2);}}
.geo-tr{{cursor:pointer;transition:background 0.15s;}}
.geo-tr:hover td{{background:var(--s2);}}
.geo-tr td{{padding:12px 14px;border-bottom:1px solid var(--border);vertical-align:middle;font-size:12px;color:var(--sub);}}
.geo-title{{color:var(--text);font-weight:600;font-size:12px;}}
.geo-region{{color:var(--sub);font-size:11px;}}
.geo-sectors{{font-size:11px;max-width:280px;}}
.geo-expand{{color:var(--dim);font-size:10px;text-align:right;}}
.geo-detail-row td{{padding:0;border-bottom:1px solid var(--border);}}
.geo-detail{{padding:14px 16px;background:var(--s2);}}
.geo-detail-grid{{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:16px;}}
.dl{{font-size:9px;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;color:var(--dim);display:block;margin-bottom:4px;}}
.geo-detail p{{font-size:12px;color:var(--sub);line-height:1.6;}}

/* CENTRAL BANKS */
.cb-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:14px;}}
.cb-card{{background:var(--s1);border:1px solid var(--border);border-radius:8px;overflow:hidden;}}
.cb-hdr{{padding:14px 16px;background:var(--s2);border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;}}
.cb-name{{font-family:var(--head);font-size:13px;font-weight:700;color:var(--text);}}
.cb-stance{{font-size:11px;font-weight:700;padding:3px 10px;border-radius:4px;}}
.cb-stance.clr-green{{background:rgba(34,197,94,0.12);color:var(--green);border:1px solid rgba(34,197,94,0.25);}}
.cb-stance.clr-red{{background:rgba(239,68,68,0.12);color:var(--red);border:1px solid rgba(239,68,68,0.25);}}
.cb-stance.clr-amber{{background:rgba(245,158,11,0.12);color:var(--amber);border:1px solid rgba(245,158,11,0.25);}}
.cb-body{{padding:0;}}
.cb-section{{padding:10px 16px;border-bottom:1px solid var(--border);}}
.cb-section p{{font-size:12px;color:var(--sub);line-height:1.6;margin-top:4px;}}
.cb-impacts{{display:grid;grid-template-columns:1fr 1fr 1fr;border-bottom:1px solid var(--border);}}
.cb-impacts>div{{padding:10px 16px;border-right:1px solid var(--border);}}
.cb-impacts>div:last-child{{border-right:none;}}
.cb-impacts p{{font-size:11px;color:var(--sub);line-height:1.5;margin-top:4px;}}
.cb-risk{{padding:10px 16px;font-size:12px;color:var(--red);}}
.cb-lbl{{font-size:9px;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;color:var(--dim);}}

/* SECTORS */
.sec-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:14px;}}
.sec-card{{background:var(--s1);border:1px solid var(--border);border-radius:8px;overflow:hidden;}}
.sec-hdr{{padding:12px 16px;background:var(--s2);border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;}}
.sec-name{{font-family:var(--head);font-size:13px;font-weight:700;color:var(--text);}}
.sec-body{{display:grid;grid-template-columns:1fr 1fr;gap:0;border-bottom:1px solid var(--border);}}
.sec-col{{padding:12px 14px;}}
.sec-col:first-child{{border-right:1px solid var(--border);}}
.sec-lbl{{font-size:9px;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;display:block;margin-bottom:6px;}}
.sec-list{{list-style:none;padding:0;}}
.sec-list li{{font-size:11px;color:var(--sub);padding:3px 0;line-height:1.5;padding-left:10px;position:relative;}}
.sec-list li::before{{content:"•";position:absolute;left:0;color:var(--dim);}}
.sec-footer{{padding:10px 14px;display:flex;flex-direction:column;gap:5px;font-size:11px;color:var(--sub);}}

/* SENTIMENT */
.sent-grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:20px;}}
.sent-col-title{{font-family:var(--head);font-size:13px;font-weight:700;margin-bottom:12px;}}
.sent-item{{background:var(--s1);border:1px solid var(--border);border-radius:6px;padding:12px 14px;margin-bottom:10px;}}
.sent-theme{{font-size:12px;font-weight:600;margin-bottom:5px;}}
.sent-detail{{font-size:12px;color:var(--sub);line-height:1.6;margin-bottom:6px;}}
.sent-assets{{font-size:10px;color:var(--dim);}}
.debate-item{{background:var(--s1);border:1px solid var(--border);border-radius:6px;padding:12px 14px;margin-bottom:10px;}}
.debate-title{{font-size:12px;font-weight:600;color:var(--text);margin-bottom:8px;}}
.debate-sides{{display:flex;flex-direction:column;gap:5px;font-size:12px;color:var(--sub);}}
.rot-tbl{{width:100%;border-collapse:collapse;}}
.rot-tbl tr:hover td{{background:var(--s2);}}
.rot-tbl td{{padding:10px 12px;border-bottom:1px solid var(--border);font-size:12px;vertical-align:top;}}
.rot-tbl tr:last-child td{{border-bottom:none;}}
.rot-from{{color:var(--sub);width:160px;}}
.rot-arrow{{color:var(--dim);width:30px;text-align:center;}}
.rot-to{{font-weight:600;width:160px;}}
.rot-signal{{color:var(--sub);}}

/* NEWS */
.news-list{{display:flex;flex-direction:column;gap:0;}}
.news-row{{display:flex;align-items:flex-start;gap:12px;padding:12px 0;border-bottom:1px solid var(--border);}}
.news-row:last-child{{border-bottom:none;}}
.news-bias-dot{{width:8px;height:8px;border-radius:50%;flex-shrink:0;margin-top:4px;}}
.news-content{{flex:1;}}
.news-meta{{display:flex;align-items:center;gap:8px;margin-bottom:4px;}}
.news-src{{font-size:10px;font-weight:600;letter-spacing:0.06em;text-transform:uppercase;color:var(--sub);}}
.news-bias-lbl{{font-size:9px;font-weight:600;letter-spacing:0.06em;}}
.news-title{{font-size:13px;color:var(--text);text-decoration:none;line-height:1.5;display:block;}}
.news-title:hover{{color:var(--blue);}}

/* MARKETS */
.mkt-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:12px;}}
.mkt-block{{background:var(--s1);border:1px solid var(--border);border-radius:8px;overflow:hidden;}}
.mkt-block-title{{font-family:var(--head);font-size:10px;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:var(--sub);padding:10px 14px;border-bottom:1px solid var(--border);background:var(--s2);}}
.mkt-tbl{{width:100%;border-collapse:collapse;}}
.mkt-tbl th{{font-size:9px;color:var(--dim);font-weight:600;letter-spacing:0.06em;text-transform:uppercase;padding:7px 12px;text-align:left;border-bottom:1px solid var(--border);}}
.mkt-tbl td{{padding:9px 12px;border-bottom:1px solid var(--border);font-size:12px;}}
.mkt-tbl tr:last-child td{{border-bottom:none;}}
.mkt-tbl tr:hover td{{background:var(--s2);}}
.mkt-price{{font-weight:600;font-family:var(--head);}}
.mkt-chg{{font-weight:600;text-align:right;}}

/* INDICATORS */
.ind-tbl{{width:100%;border-collapse:collapse;}}
.ind-tbl tr:hover td{{background:var(--s2);}}
.ind-tbl td{{padding:10px 16px;border-bottom:1px solid var(--border);font-size:13px;}}
.ind-tbl tr:last-child td{{border-bottom:none;}}
.ind-lbl{{color:var(--sub);}}
.ind-val{{text-align:right;font-weight:600;font-family:var(--head);color:var(--text);}}

/* FOOTER */
.footer{{text-align:center;padding:20px 24px;border-top:1px solid var(--border);margin-top:32px;font-size:11px;color:var(--dim);letter-spacing:0.05em;}}
.footer a{{color:var(--dim);text-decoration:none;}}.footer a:hover{{color:var(--sub);}}

/* SCROLL */
.scroll-box{{max-height:520px;overflow-y:auto;scrollbar-width:thin;scrollbar-color:var(--border) transparent;}}
.scroll-box::-webkit-scrollbar{{width:4px;}}.scroll-box::-webkit-scrollbar-thumb{{background:var(--border);}}

/* LEFT ACCENTS */
.al-g{{border-left:3px solid var(--green);}}.al-r{{border-left:3px solid var(--red);}}.al-b{{border-left:3px solid var(--blue);}}.al-a{{border-left:3px solid var(--amber);}}

/* GIES BADGE */
.gies-badge{{display:flex;align-items:center;gap:8px;flex-shrink:0;}}
.gies-i{{display:flex;align-items:center;justify-content:center;width:28px;height:28px;border-radius:4px;background:#E84A27;color:#fff;font-family:Georgia,serif;font-size:18px;font-style:italic;font-weight:700;line-height:1;}}
.gies-text{{display:flex;flex-direction:column;gap:1px;}}
.gies-top{{font-size:9px;font-weight:700;letter-spacing:0.1em;color:#fff;text-transform:uppercase;}}
.gies-bot{{font-size:8px;font-weight:500;letter-spacing:0.06em;color:var(--sub);text-transform:uppercase;}}
.hdr-divider{{width:1px;height:28px;background:var(--border2);}}

/* PORTFOLIO LAB */
.port-form-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:20px;}}
.port-textarea{{width:100%;background:#0d0d0d;color:var(--text);border:1px solid var(--border);border-radius:6px;padding:10px 12px;font:inherit;font-size:12px;min-height:130px;resize:vertical;}}
.port-label{{font-size:10px;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;color:var(--sub);margin-bottom:6px;display:block;}}
.port-btn{{background:linear-gradient(90deg,#1d74f5,#38a3ff);color:#fff;border:none;border-radius:6px;padding:10px 18px;font-weight:700;cursor:pointer;font-size:12px;}}
.port-btn:hover{{opacity:.9;}}
.port-btn.sec{{background:#1a1a1a;border:1px solid var(--border);color:var(--sub);}}
.stress-tbl{{width:100%;border-collapse:collapse;}}
.stress-tbl th{{font-size:9px;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;color:var(--dim);padding:8px 12px;border-bottom:1px solid var(--border);text-align:left;background:var(--s2);}}
.stress-tbl td{{padding:10px 12px;border-bottom:1px solid var(--border);font-size:12px;}}
.stress-tbl tr:last-child td{{border-bottom:none;}}
.stress-pos{{color:var(--green);font-weight:700;}}
.stress-neg{{color:var(--red);font-weight:700;}}
.port-loading{{text-align:center;padding:32px;color:var(--sub);font-size:12px;display:none;}}

/* AI CHAT PANEL */
#ai-chat-btn{{position:fixed;bottom:24px;right:24px;z-index:9000;background:linear-gradient(135deg,#1d74f5,#38a3ff);color:#fff;border:none;border-radius:999px;padding:12px 20px;font-size:13px;font-weight:700;cursor:pointer;box-shadow:0 4px 20px rgba(29,116,245,.4);display:flex;align-items:center;gap:8px;transition:.2s;}}
#ai-chat-btn:hover{{transform:translateY(-2px);box-shadow:0 8px 28px rgba(29,116,245,.5);}}
#ai-chat-panel{{position:fixed;top:0;right:-440px;width:420px;height:100vh;z-index:9001;background:#0d0d0d;border-left:1px solid var(--border);display:flex;flex-direction:column;transition:right .3s ease;box-shadow:-8px 0 32px rgba(0,0,0,.5);}}
#ai-chat-panel.open{{right:0;}}
#ai-chat-hdr{{padding:16px 18px;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;align-items:center;background:linear-gradient(90deg,rgba(29,116,245,.12),transparent);}}
#ai-chat-hdr h3{{margin:0;font-size:15px;font-weight:700;}}
#ai-chat-close{{background:none;border:none;color:var(--sub);font-size:20px;cursor:pointer;}}
#ai-chat-close:hover{{color:var(--text);}}
#ai-chat-msgs{{flex:1;overflow-y:auto;padding:14px;display:flex;flex-direction:column;gap:10px;}}
.ai-msg{{max-width:88%;padding:10px 13px;border-radius:12px;font-size:12px;line-height:1.6;}}
.ai-msg.user{{align-self:flex-end;background:rgba(29,116,245,.22);border:1px solid rgba(29,116,245,.3);}}
.ai-msg.ai{{align-self:flex-start;background:var(--s2);border:1px solid var(--border);color:#c0c8d0;}}
.ai-msg.err{{align-self:flex-start;background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.2);color:var(--red);}}
.ai-quick{{padding:10px 14px;display:flex;flex-wrap:wrap;gap:6px;border-top:1px solid var(--border);}}
.ai-qbtn{{font-size:11px;padding:5px 10px;border-radius:6px;cursor:pointer;background:rgba(59,130,246,.08);border:1px solid rgba(59,130,246,.2);color:var(--blue);transition:.15s;}}
.ai-qbtn:hover{{background:rgba(59,130,246,.18);}}
#ai-chat-loading{{padding:6px 14px;font-size:11px;color:var(--sub);display:none;}}
#ai-chat-input-row{{padding:12px 14px;border-top:1px solid var(--border);display:flex;gap:8px;}}
#ai-chat-input{{flex:1;background:#111;border:1px solid var(--border);border-radius:8px;padding:9px 12px;color:var(--text);font:inherit;font-size:12px;resize:none;}}
#ai-chat-send{{background:linear-gradient(90deg,#1d74f5,#38a3ff);color:#fff;border:none;border-radius:8px;padding:9px 14px;cursor:pointer;font-weight:700;font-size:12px;}}
#ai-chat-send:disabled{{opacity:.5;cursor:not-allowed;}}

/* NEWS IMPACT */
.ni-row{{display:flex;align-items:flex-start;gap:10px;padding:12px 0;border-bottom:1px solid var(--border);}}
.ni-row:last-child{{border-bottom:none;}}
.ni-analyze{{font-size:10px;padding:2px 8px;border-radius:4px;cursor:pointer;background:rgba(59,130,246,.08);border:1px solid rgba(59,130,246,.2);color:var(--blue);flex-shrink:0;margin-top:2px;}}
.ni-analyze:hover{{background:rgba(59,130,246,.18);}}
.ni-result{{margin-top:8px;padding:10px 12px;background:var(--s2);border-radius:6px;font-size:11px;display:none;border-left:3px solid var(--blue);}}
.ni-badge{{display:inline-block;padding:2px 8px;border-radius:3px;font-size:10px;font-weight:600;margin-bottom:6px;}}
.ni-badge.bullish{{background:rgba(34,197,94,.12);color:var(--green);}}
.ni-badge.bearish{{background:rgba(239,68,68,.12);color:var(--red);}}
.ni-badge.neutral,.ni-badge.mixed{{background:rgba(245,158,11,.12);color:var(--amber);}}
.ni-sec-tag{{display:inline-block;padding:2px 6px;border-radius:3px;font-size:10px;font-weight:600;margin:2px;}}
.ni-sec-tag.pos{{background:rgba(34,197,94,.1);color:var(--green);}}
.ni-sec-tag.neg{{background:rgba(239,68,68,.1);color:var(--red);}}
.custom-ni-row{{display:flex;gap:8px;margin-bottom:14px;}}
.custom-ni-input{{flex:1;background:#0d0d0d;color:var(--text);border:1px solid var(--border);border-radius:6px;padding:9px 12px;font:inherit;font-size:12px;}}
/* WATCHLIST */
.wl-row{{cursor:pointer;transition:background .15s;}}
.wl-row:hover{{background:var(--s2)!important;}}
.wl-row td{{padding:11px 16px;border-bottom:1px solid var(--bdr);font-size:.875rem;}}
.wl-price{{text-align:right;font-weight:600;font-variant-numeric:tabular-nums;}}
.wl-chg{{text-align:right;font-weight:600;font-variant-numeric:tabular-nums;}}
.wl-chart-btn{{text-align:center;}}
.wl-chart-btn button{{background:rgba(59,130,246,.1);border:1px solid rgba(59,130,246,.25);color:var(--blue);padding:3px 10px;border-radius:4px;cursor:pointer;font-size:.75rem;}}
.wl-chart-btn button:hover{{background:rgba(59,130,246,.22);}}
.wl-del{{text-align:center;}}
.wl-del button{{background:none;border:none;color:var(--sub);cursor:pointer;font-size:.85rem;padding:4px 8px;border-radius:4px;}}
.wl-del button:hover{{color:var(--red);background:rgba(239,68,68,.08);}}
/* CHART MODAL */
.cp-btn{{background:var(--s1);border:1px solid var(--bdr);color:var(--sub);padding:5px 14px;border-radius:5px;cursor:pointer;font-size:.8rem;transition:all .15s;}}
.cp-btn:hover,.cp-btn.active{{background:#3b82f6;border-color:#3b82f6;color:#fff;}}
/* LEAFLET overrides */
.leaflet-container{{background:#0d1117!important;}}
.leaflet-tile{{filter:brightness(.7) saturate(.5);}}
.conflict-popup .leaflet-popup-content-wrapper{{background:var(--s1);border:1px solid var(--bdr);color:var(--fg);border-radius:8px;box-shadow:0 8px 32px rgba(0,0,0,.6);}}
.conflict-popup .leaflet-popup-tip{{background:var(--s1);}}
</style>
</head>
<body>

<div class="hdr">
  <div style="display:flex;align-items:center;gap:16px;">
    <div class="gies-badge">
      <span class="gies-i">i</span>
      <div class="gies-text">
        <span class="gies-top">GIES COLLEGE OF BUSINESS</span>
        <span class="gies-bot">UNIVERSITY OF ILLINOIS</span>
      </div>
    </div>
    <div class="hdr-divider"></div>
    <div class="hdr-logo">Weekly Macroeconomic Monitoring &amp; Market Conditions Brief</div>
  </div>
  <div class="hdr-right">
    <div class="live-pill"><div class="live-dot"></div>LIVE DATA</div>
    <div class="hdr-time" id="hdr-clock"></div>
  </div>
</div>

<div class="ticker-bar">
  <div class="ticker-inner">{ticker_html}{ticker_html}</div>
</div>

<div class="nav">
  <div class="ntab active" onclick="show('overview',this)">Overview</div>
  <div class="ntab" onclick="show('brief',this)">Morning Brief</div>
  <div class="ntab" onclick="show('regime',this)">Macro Regime</div>
  <div class="ntab" onclick="show('impact',this)">Market Impact</div>
  <div class="ntab" onclick="show('geo',this)">Geo Risk</div>
  <div class="ntab" onclick="show('cb',this)">Central Banks</div>
  <div class="ntab" onclick="show('sectors',this)">Sectors</div>
  <div class="ntab" onclick="show('sentiment',this)">Sentiment</div>
  <div class="ntab" onclick="show('markets',this)">Markets</div>
  <div class="ntab" onclick="show('indicators',this)">Indicators</div>
  <div class="ntab" onclick="show('news',this)">News Feed</div>
  <div class="ntab" onclick="show('newsimpact',this)">News Impact</div>
  <div class="ntab" onclick="show('conflictmap',this)">&#x1F5FA; Conflict Map</div>
  <div class="ntab" onclick="show('watchlist',this)">&#x1F4C8; Watchlist</div>
  <div class="ntab" onclick="show('portfolio',this)">&#x1F9EA; Portfolio Lab</div>
</div>

<!-- ═══ OVERVIEW ═══ -->
<div id="tab-overview" class="tab active"><div class="wrap">
  <div class="stat-row">
    <div class="stat al-g"><div class="stat-lbl">Fed Funds Rate</div><div class="stat-val">{fed}%</div><div class="stat-sub2">Current target</div></div>
    <div class="stat al-b"><div class="stat-lbl">Unemployment</div><div class="stat-val">{unem}%</div><div class="stat-sub2">Latest reading</div></div>
    <div class="stat al-a"><div class="stat-lbl">CPI Index</div><div class="stat-val">{cpi}</div><div class="stat-sub2">Latest reading</div></div>
    <div class="stat al-g"><div class="stat-lbl">Yield Curve</div><div class="stat-val">{sprd}</div><div class="stat-sub2">10Y minus 2Y</div></div>
    <div class="stat al-r"><div class="stat-lbl">Consumer Sentiment</div><div class="stat-val">{sent_val}</div><div class="stat-sub2">U of Michigan</div></div>
    <div class="stat al-a"><div class="stat-lbl">WTI Crude</div><div class="stat-val">${wti}</div><div class="stat-sub2">Per barrel</div></div>
  </div>

  <div class="spark-row">
    <div class="spark-card al-g"><div class="spark-name">S&amp;P 500 — 30 Day Price (USD)</div><canvas id="sp-spark" height="55"></canvas></div>
    <div class="spark-card al-b"><div class="spark-name">10Y Treasury Yield (%) — 30 Day</div><canvas id="tnx-spark" height="55"></canvas></div>
    <div class="spark-card al-a"><div class="spark-name">Crude Oil (USD/bbl) — 30 Day</div><canvas id="oil-spark" height="55"></canvas></div>
    <div class="spark-card" style="border-left:3px solid var(--purple);background:var(--s1);border-radius:8px;padding:14px;"><div class="spark-name">Gold (USD/oz) — 30 Day</div><canvas id="gold-spark" height="55"></canvas></div>
  </div>

  <div class="regime-grid">
    <div class="regime-badge-wrap">
      <div class="stat-lbl" style="text-align:center;">CURRENT MARKET REGIME</div>
      <div class="regime-label {regime_color}" style="font-size:14px;">{regime_label}</div>
      <p class="regime-summary">{regime_summary}</p>
    </div>
    <div class="ca-list">
      <div class="card-hdr"><span class="card-title">Cross-Asset Signals</span></div>
      {cross_asset_html}
    </div>
    <div class="drivers-list">
      <div class="card-hdr"><span class="card-title">Top Macro Drivers</span><span class="badge badge-blue">AI</span></div>
      <div class="scroll-box">{drivers_html}</div>
    </div>
  </div>

  <div style="display:grid;grid-template-columns:1fr 340px;gap:16px;">
    <div class="card al-b">
      <div class="card-hdr"><span class="card-title">FRED Macro Indicators</span><span class="badge badge-blue">FEDERAL RESERVE</span></div>
      <div class="card-body" style="padding:0;">
        <table class="ind-tbl"><tbody>{indicators_rows}</tbody></table>
      </div>
    </div>
    <div class="card al-g">
      <div class="card-hdr"><span class="card-title">Live Headlines</span><span class="badge badge-green">LIVE</span></div>
      <div class="scroll-box" style="max-height:400px;padding:0 16px;">{news_html}</div>
    </div>
  </div>
</div></div>

<!-- ═══ MORNING BRIEF ═══ -->
<div id="tab-brief" class="tab"><div class="wrap">
  <div class="brief-headline">{brief_headline}</div>
  {brief_paras}
  <div class="brief-callout">
    <div class="brief-callout-item al-r">
      <div class="brief-callout-lbl">⚠ Biggest Risk Today</div>
      <div class="brief-callout-val clr-red">{brief_risk}</div>
    </div>
    <div class="brief-callout-item al-a">
      <div class="brief-callout-lbl">📍 Most Sensitive Sector</div>
      <div class="brief-callout-val clr-amber">{brief_sector}</div>
    </div>
  </div>
  <div class="sec-hdr-row"><div><div class="sec-hdr">Macro Watchlist</div><div class="sec-sub">Upcoming catalysts with bull/bear scenarios</div></div></div>
  <div class="card">
    <table class="watchlist-tbl">
      <thead><tr><th>Event</th><th>Why It Matters</th><th>Bull Scenario</th><th>Bear Scenario</th><th>Key Assets</th></tr></thead>
      <tbody>{watchlist_html}</tbody>
    </table>
  </div>
</div></div>

<!-- ═══ MACRO REGIME ═══ -->
<div id="tab-regime" class="tab"><div class="wrap">
  <div class="regime-grid" style="grid-template-columns:280px 1fr 1fr;">
    <div class="regime-badge-wrap">
      <div class="stat-lbl" style="text-align:center;">CURRENT REGIME</div>
      <div class="regime-label {regime_color}" style="font-size:16px;">{regime_label}</div>
      <p class="regime-summary">{regime_summary}</p>
    </div>
    <div class="ca-list">
      <div class="card-hdr"><span class="card-title">Cross-Asset Signal Summary</span></div>
      {cross_asset_html}
    </div>
    <div class="drivers-list">
      <div class="card-hdr"><span class="card-title">Top 5 Macro Drivers</span></div>
      <div class="scroll-box">{drivers_html}</div>
    </div>
  </div>
</div></div>

<!-- ═══ MARKET IMPACT ═══ -->
<div id="tab-impact" class="tab"><div class="wrap">
  <div class="impact-grid">{impact_html}</div>
</div></div>

<!-- ═══ GEO RISK ═══ -->
<div id="tab-geo" class="tab"><div class="wrap">
  <div class="card">
    <div class="card-hdr"><span class="card-title">Global Market Risk Monitor</span><span class="badge badge-red">AI</span></div>
    <table class="geo-tbl">
      <thead><tr><th>Risk Event</th><th>Region</th><th>Risk Level</th><th>Sectors Impacted</th><th></th></tr></thead>
      <tbody>{geo_rows}</tbody>
    </table>
  </div>
</div></div>

<!-- ═══ CENTRAL BANKS ═══ -->
<div id="tab-cb" class="tab"><div class="wrap">
  <div class="cb-grid">{cb_html}</div>
</div></div>

<!-- ═══ SECTORS ═══ -->
<div id="tab-sectors" class="tab"><div class="wrap">
  <div class="card" style="margin-bottom:20px;">
    <div class="card-hdr"><span class="card-title">Sector ETF Performance Today (% Change)</span><span class="badge badge-blue">LIVE</span></div>
    <div class="card-body"><canvas id="sector-chart" height="70"></canvas></div>
  </div>
  <div class="sec-grid">{sector_cards_html}</div>
</div></div>

<!-- ═══ SENTIMENT ═══ -->
<div id="tab-sentiment" class="tab"><div class="wrap">
  <div class="sent-grid">
    <div>
      <div class="sent-col-title clr-green">▲ Bullish Themes</div>
      {bullish_html}
    </div>
    <div>
      <div class="sent-col-title clr-red">▼ Bearish Themes</div>
      {bearish_html}
    </div>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">
    <div>
      <div class="sec-hdr-row"><div class="sec-hdr">Emerging Debates</div></div>
      {debates_html}
    </div>
    <div>
      <div class="sec-hdr-row"><div class="sec-hdr">Capital Rotation Signals</div></div>
      <div class="card">
        <table class="rot-tbl">
          <thead><tr><th style="font-size:9px;color:var(--dim);padding:8px 12px;border-bottom:1px solid var(--border);text-align:left;background:var(--s2);">FROM</th><th></th><th style="font-size:9px;color:var(--dim);padding:8px 12px;border-bottom:1px solid var(--border);text-align:left;background:var(--s2);">TO</th><th style="font-size:9px;color:var(--dim);padding:8px 12px;border-bottom:1px solid var(--border);text-align:left;background:var(--s2);">SIGNAL</th></tr></thead>
          <tbody>{rotation_html}</tbody>
        </table>
      </div>
    </div>
  </div>
</div></div>

<!-- ═══ MARKETS ═══ -->
<div id="tab-markets" class="tab"><div class="wrap">{markets_html}</div></div>

<!-- ═══ INDICATORS ═══ -->
<div id="tab-indicators" class="tab"><div class="wrap">
  <div class="chart-grid" style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:20px;">
    <div class="card al-g">
      <div class="card-hdr"><span class="card-title">Fed Funds Rate (%) — 24 Month History</span><span class="badge badge-green">FRED</span></div>
      <div class="card-body"><canvas id="fed-chart" height="110"></canvas></div>
    </div>
    <div class="card al-a">
      <div class="card-hdr"><span class="card-title">CPI Index — 24 Month History</span><span class="badge badge-amber">FRED</span></div>
      <div class="card-body"><canvas id="cpi-chart" height="110"></canvas></div>
    </div>
  </div>
  <div class="card">
    <div class="card-hdr"><span class="card-title">All FRED Macro Indicators</span><span class="badge badge-blue">FEDERAL RESERVE</span></div>
    <table class="ind-tbl"><tbody>{indicators_rows}</tbody></table>
  </div>
</div></div>

<!-- ═══ NEWS FEED ═══ -->
<div id="tab-news" class="tab"><div class="wrap">
  <div style="display:flex;align-items:center;gap:16px;margin-bottom:16px;">
    <span style="font-size:11px;color:var(--sub);">Bias indicators:</span>
    <span style="display:flex;align-items:center;gap:5px;font-size:11px;color:#3b82f6;"><span style="width:8px;height:8px;border-radius:50%;background:#3b82f6;display:inline-block;"></span> Left-leaning</span>
    <span style="display:flex;align-items:center;gap:5px;font-size:11px;color:#6b7280;"><span style="width:8px;height:8px;border-radius:50%;background:#6b7280;display:inline-block;"></span> Center</span>
    <span style="display:flex;align-items:center;gap:5px;font-size:11px;color:#ef4444;"><span style="width:8px;height:8px;border-radius:50%;background:#ef4444;display:inline-block;"></span> Right-leaning</span>
  </div>
  <div class="card">
    <div class="news-list" style="padding:0 16px;">{news_html}</div>
  </div>
</div></div>

<div class="footer">
  Developed by Jake Joseph &nbsp;|&nbsp;
  <a href="mailto:jakemjoseph@gmail.com">jakemjoseph@gmail.com</a> &nbsp;|&nbsp;
  <a href="https://www.linkedin.com/in/jakemarleyjoseph/" target="_blank">linkedin.com/in/jakemarleyjoseph</a>
</div>

<!-- ═══ NEWS IMPACT ═══ -->
<div id="tab-newsimpact" class="tab"><div class="wrap">
  <div class="sec-hdr-row"><div><div class="sec-hdr">News Impact Analyzer</div><div class="sec-sub">Analyze how any headline affects sectors and assets</div></div></div>
  <div class="card" style="margin-bottom:18px;">
    <div class="card-hdr"><span class="card-title">Custom Headline</span></div>
    <div class="card-body">
      <div class="custom-ni-row">
        <input class="custom-ni-input" id="custom-ni-input" placeholder="Enter any headline to analyze its market impact..." />
        <button class="port-btn" onclick="analyzeCustomHeadline()">Analyze</button>
      </div>
      <div id="custom-ni-result" class="ni-result"></div>
    </div>
  </div>
  <div class="card">
    <div class="card-hdr"><span class="card-title">Today's Headlines — Click to Analyze Impact</span></div>
    <div class="card-body" id="ni-headlines-list">
      {ni_headlines_html}
    </div>
  </div>
</div></div>

<!-- ═══ CONFLICT MAP ═══ -->
<div id="tab-conflictmap" class="tab"><div class="wrap">
  <div class="sec-hdr-row">
    <div><div class="sec-hdr">Global Conflict &amp; Geopolitical Risk Map</div>
    <div class="sec-sub">AI-identified hotspots and their market implications — click any marker for details</div></div>
  </div>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <div id="conflict-map" style="height:480px;border-radius:12px;overflow:hidden;border:1px solid var(--bdr);"></div>
  <div id="conflict-detail" style="display:none;margin-top:16px;" class="card al-r">
    <div class="card-hdr"><span class="card-title" id="cd-title"></span><span class="badge" id="cd-risk-badge"></span></div>
    <div class="card-body">
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;font-size:.85rem;">
        <div><div style="color:var(--sub);font-size:.75rem;margin-bottom:4px;">TRIGGER</div><div id="cd-trigger"></div></div>
        <div><div style="color:var(--sub);font-size:.75rem;margin-bottom:4px;">SECTORS IMPACTED</div><div id="cd-sectors"></div></div>
        <div><div style="color:var(--sub);font-size:.75rem;margin-bottom:4px;">ASSET IMPACT</div><div id="cd-assets"></div></div>
      </div>
      <div style="margin-top:12px;padding-top:12px;border-top:1px solid var(--bdr);">
        <div style="color:var(--sub);font-size:.75rem;margin-bottom:4px;">MARKET EXPOSURE</div>
        <div id="cd-exposure" style="font-size:.85rem;line-height:1.6;"></div>
      </div>
      <div style="margin-top:12px;padding-top:12px;border-top:1px solid var(--bdr);">
        <div style="color:var(--sub);font-size:.75rem;margin-bottom:4px;">WHAT TO WATCH</div>
        <div id="cd-watch" style="font-size:.85rem;line-height:1.6;color:var(--amber);"></div>
      </div>
    </div>
  </div>
  <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:14px;margin-top:20px;">
    {geo_cards_html}
  </div>
</div></div>

<!-- ═══ WATCHLIST ═══ -->
<div id="tab-watchlist" class="tab"><div class="wrap">
  <div class="sec-hdr-row" style="flex-wrap:wrap;gap:12px;">
    <div><div class="sec-hdr">My Watchlist</div>
    <div class="sec-sub">Saved tickers — prices update on page load. Click any row to view chart.</div></div>
    <div style="display:flex;gap:8px;align-items:center;">
      <input id="wl-input" type="text" placeholder="Add ticker (e.g. AAPL)" style="background:var(--s1);border:1px solid var(--bdr);color:var(--fg);padding:8px 12px;border-radius:6px;font-size:.85rem;width:180px;text-transform:uppercase;" onkeydown="if(event.key==='Enter')wlAdd()"/>
      <button onclick="wlAdd()" style="background:#3b82f6;color:#fff;border:none;padding:8px 16px;border-radius:6px;cursor:pointer;font-size:.85rem;">+ Add</button>
      <button onclick="wlRefresh()" style="background:var(--s1);color:var(--fg);border:1px solid var(--bdr);padding:8px 14px;border-radius:6px;cursor:pointer;font-size:.85rem;">↻ Refresh</button>
    </div>
  </div>
  <div id="wl-error" style="display:none;color:var(--red);font-size:.85rem;margin-bottom:8px;"></div>
  <div class="card" style="padding:0;overflow:hidden;">
    <table class="ind-tbl" id="wl-table">
      <thead><tr style="background:var(--s2);">
        <th style="padding:10px 16px;text-align:left;color:var(--sub);font-size:.75rem;font-weight:600;">TICKER</th>
        <th style="padding:10px 16px;text-align:right;color:var(--sub);font-size:.75rem;font-weight:600;">PRICE</th>
        <th style="padding:10px 16px;text-align:right;color:var(--sub);font-size:.75rem;font-weight:600;">CHANGE</th>
        <th style="padding:10px 16px;text-align:center;color:var(--sub);font-size:.75rem;font-weight:600;">CHART</th>
        <th style="padding:10px 16px;text-align:center;color:var(--sub);font-size:.75rem;font-weight:600;"></th>
      </tr></thead>
      <tbody id="wl-body"></tbody>
    </table>
  </div>
  <div id="wl-empty" style="text-align:center;padding:48px;color:var(--sub);display:none;">
    No tickers saved yet. Add one above (e.g. AAPL, SPY, NVDA, BTC-USD).
  </div>
</div></div>

<!-- ═══ STOCK CHART MODAL ═══ -->
<div id="chart-modal" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,.75);z-index:9999;align-items:center;justify-content:center;">
  <div style="background:var(--bg);border:1px solid var(--bdr);border-radius:14px;width:min(860px,96vw);padding:24px;position:relative;">
    <button onclick="closeChartModal()" style="position:absolute;top:14px;right:14px;background:var(--s1);border:1px solid var(--bdr);color:var(--fg);width:30px;height:30px;border-radius:50%;cursor:pointer;font-size:1rem;line-height:1;">✕</button>
    <div style="display:flex;align-items:baseline;gap:12px;margin-bottom:4px;">
      <span id="cm-ticker" style="font-size:1.3rem;font-weight:700;"></span>
      <span id="cm-name" style="font-size:.85rem;color:var(--sub);"></span>
    </div>
    <div style="display:flex;align-items:baseline;gap:10px;margin-bottom:16px;">
      <span id="cm-price" style="font-size:1.6rem;font-weight:700;"></span>
      <span id="cm-chg" style="font-size:.95rem;font-weight:600;"></span>
    </div>
    <div style="display:flex;gap:6px;margin-bottom:14px;flex-wrap:wrap;">
      <button class="cp-btn active" onclick="loadChart(null,'1w',this)">1W</button>
      <button class="cp-btn" onclick="loadChart(null,'1m',this)">1M</button>
      <button class="cp-btn" onclick="loadChart(null,'3m',this)">3M</button>
      <button class="cp-btn" onclick="loadChart(null,'6m',this)">6M</button>
      <button class="cp-btn" onclick="loadChart(null,'1y',this)">1Y</button>
      <button class="cp-btn" onclick="loadChart(null,'5y',this)">5Y</button>
      <button class="cp-btn" onclick="loadChart(null,'max',this)">MAX</button>
    </div>
    <div id="cm-loading" style="text-align:center;padding:40px;color:var(--sub);">Loading…</div>
    <canvas id="cm-canvas" style="display:none;max-height:320px;"></canvas>
  </div>
</div>

<!-- ═══ PORTFOLIO LAB ═══ -->
<div id="tab-portfolio" class="tab"><div class="wrap">
  <div class="sec-hdr-row">
    <div><div class="sec-hdr">Portfolio Lab</div><div class="sec-sub">Enter your holdings to stress test against historical crises and get AI analysis</div></div>
  </div>

  <div class="card" style="margin-bottom:18px;">
    <div class="card-hdr"><span class="card-title">Portfolio Holdings (ticker, weight%)</span></div>
    <div class="card-body">
      <div class="port-form-grid">
        <div>
          <label class="port-label">Portfolio A</label>
          <textarea class="port-textarea" id="port-a" placeholder="VOO,40&#10;QQQ,20&#10;MSFT,15&#10;AAPL,15&#10;GLD,10">VOO,40
QQQ,20
MSFT,15
AAPL,15
GLD,10</textarea>
        </div>
        <div>
          <label class="port-label">Portfolio B</label>
          <textarea class="port-textarea" id="port-b" placeholder="VTI,50&#10;SCHD,25&#10;GLD,15&#10;TLT,10">VTI,50
SCHD,25
GLD,15
TLT,10</textarea>
        </div>
        <div>
          <label class="port-label">Portfolio C</label>
          <textarea class="port-textarea" id="port-c" placeholder="NVDA,25&#10;MSFT,25&#10;META,20&#10;AMZN,20&#10;GOOGL,10">NVDA,25
MSFT,25
META,20
AMZN,20
GOOGL,10</textarea>
        </div>
      </div>
      <div style="display:flex;gap:10px;">
        <button class="port-btn" onclick="runStressTest()">&#x26A1; Run Stress Test</button>
        <button class="port-btn sec" onclick="askAboutPortfolio()">&#x1F4AC; Ask AI About My Portfolio</button>
      </div>
      <div class="port-loading" id="port-loading">&#x23F3; Fetching historical data...</div>
    </div>
  </div>

  <div id="stress-results" style="display:none;">
    <div class="sec-hdr-row" style="margin-top:0;"><div class="sec-hdr">Stress Test Results</div></div>
    <div class="card">
      <div class="card-body" style="padding:0;">
        <table class="stress-tbl">
          <thead><tr><th>Period</th><th>Portfolio A</th><th>Portfolio B</th><th>Portfolio C</th><th>SPY</th></tr></thead>
          <tbody id="stress-tbody"></tbody>
        </table>
      </div>
    </div>
  </div>
</div></div>

<!-- AI CHAT PANEL -->
<button id="ai-chat-btn" onclick="toggleAIChat()">&#x1F916; Ask AI Analyst</button>
<div id="ai-chat-panel">
  <div id="ai-chat-hdr">
    <h3>&#x1F916; AI Analyst</h3>
    <button id="ai-chat-close" onclick="toggleAIChat()">&#x2715;</button>
  </div>
  <div id="ai-chat-msgs">
    <div class="ai-msg ai">Hello! I'm your AI Analyst. Ask me about the macro regime, your portfolio, sector outlook, or anything market-related.</div>
  </div>
  <div class="ai-quick">
    <button class="ai-qbtn" onclick="aiQuick('What is the current macro regime and what does it mean for equities?')">Current regime?</button>
    <button class="ai-qbtn" onclick="aiQuick('Which sectors should I overweight right now?')">Best sectors?</button>
    <button class="ai-qbtn" onclick="aiQuick('What are the biggest risks to markets this week?')">Key risks?</button>
    <button class="ai-qbtn" onclick="aiQuick('How should I position given the Fed stance?')">Fed impact?</button>
    <button class="ai-qbtn" onclick="aiQuick('Summarize the morning brief for me.')">Brief summary?</button>
  </div>
  <div id="ai-chat-loading">&#x23F3; AI is thinking...</div>
  <div id="ai-chat-input-row">
    <textarea id="ai-chat-input" rows="2" placeholder="Ask about macro, your portfolio, or market outlook..."></textarea>
    <button id="ai-chat-send" onclick="sendAIChat()">Send</button>
  </div>
</div>

<script>
{sparkline_js}
{geo_risk_js}

const sparkOpts = (color, fill) => ({{
  responsive: true,
  plugins: {{ legend: {{ display: false }}, tooltip: {{ callbacks: {{ label: ctx => ctx.parsed.y.toLocaleString() }} }} }},
  scales: {{
    x: {{ display: false }},
    y: {{ display: true, position: 'right',
         ticks: {{ color: '#555', font: {{ size: 9 }}, maxTicksLimit: 4, callback: v => v >= 1000 ? (v/1000).toFixed(1)+'K' : v.toFixed(2) }},
         grid: {{ color: 'rgba(42,42,42,0.8)' }} }}
  }},
  elements: {{ point: {{ radius: 0 }}, line: {{ tension: 0.3, borderWidth: 2 }} }}
}});

function makeSparkline(id, key, color, fill) {{
  const d = sparklineData[key];
  if (!d || !d.prices.length) return;
  new Chart(document.getElementById(id), {{
    type: 'line',
    data: {{ labels: d.dates, datasets: [{{ data: d.prices, borderColor: color, backgroundColor: fill, fill: true }}] }},
    options: sparkOpts(color, fill)
  }});
}}

const barOpts = () => ({{
  responsive: true,
  plugins: {{ legend: {{ display: false }} }},
  scales: {{
    x: {{ ticks: {{ color: '#777', font: {{ size: 10 }} }}, grid: {{ color: '#1a1a1a' }} }},
    y: {{ ticks: {{ color: '#777', font: {{ size: 10 }}, callback: v => parseFloat(v.toFixed(2)) + '%' }}, grid: {{ color: '#1a1a1a' }} }}
  }}
}});

const lineOpts = (unit) => ({{
  responsive: true,
  plugins: {{ legend: {{ display: false }} }},
  scales: {{
    x: {{ ticks: {{ color: '#777', font: {{ size: 9 }}, maxTicksLimit: 6 }}, grid: {{ color: '#1a1a1a' }} }},
    y: {{ ticks: {{ color: '#777', font: {{ size: 9 }}, callback: v => v + unit }}, grid: {{ color: '#1a1a1a' }} }}
  }}
}});

window.addEventListener('load', function() {{
  makeSparkline('sp-spark',   'S&P 500',      '#22c55e', 'rgba(34,197,94,0.08)');
  makeSparkline('tnx-spark',  '10Y Treasury', '#3b82f6', 'rgba(59,130,246,0.08)');
  makeSparkline('oil-spark',  'Crude Oil',    '#f59e0b', 'rgba(245,158,11,0.08)');
  makeSparkline('gold-spark', 'Gold',         '#8b5cf6', 'rgba(139,92,246,0.08)');

  const sc = document.getElementById('sector-chart');
  if (sc) new Chart(sc, {{ type:'bar', data:{{ labels:{sector_names}, datasets:[{{ data:{sector_values}, backgroundColor:{sector_colors}, borderRadius:4 }}] }}, options:barOpts() }});

  const fc = document.getElementById('fed-chart');
  if (fc) new Chart(fc, {{ type:'line', data:{{ labels:{fed_labels}, datasets:[{{ data:{fed_values}, borderColor:'#22c55e', backgroundColor:'rgba(34,197,94,0.06)', fill:true, tension:0.3, borderWidth:2, pointRadius:0 }}] }}, options:lineOpts('%') }});

  const ci = document.getElementById('cpi-chart');
  if (ci) new Chart(ci, {{ type:'line', data:{{ labels:{cpi_labels}, datasets:[{{ data:{cpi_values}, borderColor:'#f59e0b', backgroundColor:'rgba(245,158,11,0.06)', fill:true, tension:0.3, borderWidth:2, pointRadius:0 }}] }}, options:lineOpts('') }});
}});

function show(id,el){{
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.ntab').forEach(t=>t.classList.remove('active'));
  document.getElementById('tab-'+id).classList.add('active');
  el.classList.add('active');
  if (id === 'conflictmap') setTimeout(initConflictMap, 150);
  if (id === 'watchlist')   {{ wlRender(); wlRefresh(); }}
}}

function toggleGeo(i){{
  const row = document.getElementById('geo-detail-'+i);
  const isOpen = row.style.display !== 'none';
  document.querySelectorAll('[id^="geo-detail-"]').forEach(r => r.style.display = 'none');
  document.querySelectorAll('.geo-expand').forEach(e => e.textContent = '▼');
  if (!isOpen) {{
    row.style.display = 'table-row';
    document.querySelectorAll('.geo-tr')[i].querySelector('.geo-expand').textContent = '▲';
  }}
}}

// ── AI CHAT ────────────────────────────────────────────────────────────────
function toggleAIChat() {{
  document.getElementById('ai-chat-panel').classList.toggle('open');
}}

function appendAIMsg(role, text) {{
  const msgs = document.getElementById('ai-chat-msgs');
  const div = document.createElement('div');
  div.className = 'ai-msg ' + role;
  div.textContent = text;
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
}}

async function sendAIChat() {{
  const input = document.getElementById('ai-chat-input');
  const send = document.getElementById('ai-chat-send');
  const loading = document.getElementById('ai-chat-loading');
  const message = input.value.trim();
  if (!message) return;
  appendAIMsg('user', message);
  input.value = '';
  send.disabled = true;
  loading.style.display = 'block';
  try {{
    const resp = await fetch('/chat', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{ message, context: {{ regime: document.querySelector('.regime-label')?.textContent || '', source: 'macro-dashboard' }} }}),
    }});
    const data = await resp.json();
    if (data.error) appendAIMsg('err', 'Error: ' + data.error);
    else appendAIMsg('ai', data.response);
  }} catch(e) {{
    appendAIMsg('err', 'Request failed: ' + e.message);
  }} finally {{
    send.disabled = false;
    loading.style.display = 'none';
  }}
}}

function aiQuick(prompt) {{
  document.getElementById('ai-chat-input').value = prompt;
  sendAIChat();
}}

document.getElementById('ai-chat-input')?.addEventListener('keydown', e => {{
  if (e.key === 'Enter' && !e.shiftKey) {{ e.preventDefault(); sendAIChat(); }}
}});

// ── STRESS TEST ─────────────────────────────────────────────────────────────
function parsePortText(text) {{
  const weights = {{}};
  let total = 0;
  text.trim().split('\\n').forEach(line => {{
    const parts = line.split(',');
    if (parts.length === 2) {{
      const t = parts[0].trim().toUpperCase();
      const w = parseFloat(parts[1].trim());
      if (t && !isNaN(w)) {{ weights[t] = w / 100; total += w; }}
    }}
  }});
  if (total > 0 && Math.abs(total - 100) > 1) {{
    Object.keys(weights).forEach(k => weights[k] /= (total / 100));
  }}
  return weights;
}}

async function runStressTest() {{
  const loading = document.getElementById('port-loading');
  const results = document.getElementById('stress-results');
  loading.style.display = 'block';
  results.style.display = 'none';

  const portfolios = {{
    'Portfolio A': parsePortText(document.getElementById('port-a').value),
    'Portfolio B': parsePortText(document.getElementById('port-b').value),
    'Portfolio C': parsePortText(document.getElementById('port-c').value),
  }};

  try {{
    const responses = await Promise.all(
      Object.entries(portfolios).map(([name, weights]) =>
        fetch('/api/stress-test', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{ weights }}),
        }}).then(r => r.json()).then(d => ({{ name, periods: d.periods || [] }}))
      )
    );

    const periods = responses[0]?.periods?.map(p => p.name) || [];
    const tbody = document.getElementById('stress-tbody');
    tbody.innerHTML = '';
    periods.forEach((period, i) => {{
      const tr = document.createElement('tr');
      let html = `<td style="font-weight:600;color:var(--text)">${{period}}</td>`;
      responses.forEach(port => {{
        const p = port.periods[i] || {{}};
        const v = p.portfolio_return;
        if (v === null || v === undefined) {{
          html += `<td style="color:var(--dim)">N/A</td>`;
        }} else {{
          const cls = v >= 0 ? 'stress-pos' : 'stress-neg';
          html += `<td class="${{cls}}">${{v >= 0 ? '+' : ''}}${{v.toFixed(1)}}%</td>`;
        }}
      }});
      const spy = responses[0]?.periods[i]?.spy_return;
      if (spy !== null && spy !== undefined) {{
        const cls = spy >= 0 ? 'stress-pos' : 'stress-neg';
        html += `<td class="${{cls}}">${{spy >= 0 ? '+' : ''}}${{spy.toFixed(1)}}%</td>`;
      }} else html += `<td style="color:var(--dim)">N/A</td>`;
      tr.innerHTML = html;
      tbody.appendChild(tr);
    }});
    results.style.display = 'block';
  }} catch(e) {{
    alert('Stress test failed: ' + e.message);
  }} finally {{
    loading.style.display = 'none';
  }}
}}

function askAboutPortfolio() {{
  const portA = document.getElementById('port-a').value;
  document.getElementById('ai-chat-panel').classList.add('open');
  document.getElementById('ai-chat-input').value = `Given the current macro regime, analyze my portfolio: ${{portA}}. What are my biggest risks and opportunities?`;
  sendAIChat();
}}

// ── NEWS IMPACT ─────────────────────────────────────────────────────────────
const IMPACT_RULES = [
  {{ keywords: ['oil','crude','opec','petroleum'], sectors: {{'Energy':'pos','Airlines':'neg','Consumer Discretionary':'neg'}} }},
  {{ keywords: ['fed','federal reserve','interest rate','rate hike','rate cut','fomc','powell'], sectors: {{'Financials':'pos','Real Estate':'neg','Utilities':'neg','Technology':'neg'}} }},
  {{ keywords: ['china','tariff','trade war','sanctions'], sectors: {{'Technology':'neg','Industrials':'neg','Consumer Discretionary':'neg'}} }},
  {{ keywords: ['war','conflict','geopolitical','invasion','military'], sectors: {{'Defense':'pos','Energy':'pos','Technology':'neg'}} }},
  {{ keywords: ['inflation','cpi','pce','prices rose'], sectors: {{'Real Estate':'neg','Energy':'pos','Consumer Staples':'neg'}} }},
];

function analyzeHeadline(headline) {{
  const lower = headline.toLowerCase();
  const matched = IMPACT_RULES.filter(r => r.keywords.some(k => lower.includes(k)));
  if (!matched.length) return {{ overall: 'neutral', sectors: {{}} }};
  const sectors = {{}};
  matched.forEach(r => Object.assign(sectors, r.sectors));
  const pos = Object.values(sectors).filter(v => v === 'pos').length;
  const neg = Object.values(sectors).filter(v => v === 'neg').length;
  const overall = pos > neg ? 'bullish' : neg > pos ? 'bearish' : 'mixed';
  return {{ overall, sectors }};
}}

function showNIResult(el, headline) {{
  const result = el.nextElementSibling;
  const analysis = analyzeHeadline(headline);
  const tags = Object.entries(analysis.sectors).map(([s,d]) =>
    `<span class="ni-sec-tag ${{d}}">${{s}}: ${{d === 'pos' ? '▲ Positive' : '▼ Negative'}}</span>`
  ).join('');
  result.innerHTML = `<div class="ni-badge ${{analysis.overall}}">${{analysis.overall.toUpperCase()}}</div><br/>${{tags || '<span style="color:var(--sub)">No specific sector impact identified</span>'}}`;
  result.style.display = result.style.display === 'block' ? 'none' : 'block';
}}

function analyzeCustomHeadline() {{
  const input = document.getElementById('custom-ni-input');
  const result = document.getElementById('custom-ni-result');
  const headline = input.value.trim();
  if (!headline) return;
  const analysis = analyzeHeadline(headline);
  const tags = Object.entries(analysis.sectors).map(([s,d]) =>
    `<span class="ni-sec-tag ${{d}}">${{s}}: ${{d === 'pos' ? '▲ Positive' : '▼ Negative'}}</span>`
  ).join('');
  result.innerHTML = `<div class="ni-badge ${{analysis.overall}}">${{analysis.overall.toUpperCase()}}</div><br/>${{tags || '<span style="color:var(--sub)">No specific sector impact identified</span>'}}`;
  result.style.display = 'block';
}}

// ── Local clock (updates every second) ───────────────────────────────────────
(function() {{
  function updateClock() {{
    const now = new Date();
    const opts = {{ weekday:'long', year:'numeric', month:'long', day:'numeric', hour:'2-digit', minute:'2-digit', hour12:true }};
    const el = document.getElementById('hdr-clock');
    if (el) el.textContent = now.toLocaleString('en-US', opts);
  }}
  updateClock();
  setInterval(updateClock, 1000);
}})();

// ── CONFLICT MAP ─────────────────────────────────────────────────────────────
const REGION_COORDS = {{
  "Middle East": [29, 45], "Ukraine": [49, 32], "Russia": [55, 37],
  "Taiwan": [23.5, 121], "China": [35, 105], "North Korea": [40, 127],
  "South China Sea": [12, 115], "Iran": [32, 53], "Israel": [31.5, 35],
  "Gaza": [31.4, 34.3], "Syria": [35, 38], "Yemen": [15.5, 48],
  "Africa": [5, 20], "Sahel": [15, 5], "Sudan": [15, 30],
  "India-Pakistan": [28, 70], "Venezuela": [8, -66], "Haiti": [19, -72],
  "Kosovo": [42.6, 20.9], "Armenia-Azerbaijan": [40, 47],
  "Ethiopia": [9, 40], "Myanmar": [17, 96], "Libya": [27, 17],
  "Eastern Europe": [50, 30], "Europe": [50, 15], "Asia": [34, 100],
  "Latin America": [-10, -60], "North Africa": [27, 15],
  "Southeast Asia": [10, 110], "Central Asia": [42, 63],
}};
function getCoords(region) {{
  if (!region) return [20, 0];
  for (const [key, coords] of Object.entries(REGION_COORDS)) {{
    if (region.toLowerCase().includes(key.toLowerCase()) ||
        key.toLowerCase().includes(region.toLowerCase())) return coords;
  }}
  return [20, 0];
}}
let _conflictMap = null;
function initConflictMap() {{
  if (_conflictMap) return;
  const el = document.getElementById('conflict-map');
  if (!el) return;
  _conflictMap = L.map('conflict-map', {{
    center:[20,15], zoom:2, zoomControl:true,
    worldCopyJump: false,
    maxBounds: [[-85,-180],[85,180]],
    maxBoundsViscosity: 1.0,
    minZoom: 2
  }});
  L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
    attribution:'© OpenStreetMap © CartoDB', subdomains:'abcd', maxZoom:10, noWrap:true
  }}).addTo(_conflictMap);
  (GEO_RISK_DATA || []).forEach((g, i) => {{
    const coords = getCoords(g.region);
    const rl = (g.risk_level||'MEDIUM').toUpperCase();
    const color = rl.includes('HIGH') ? '#ef4444' : rl.includes('MED') ? '#f59e0b' : '#22c55e';
    const marker = L.circleMarker(coords, {{
      radius: rl.includes('HIGH') ? 14 : rl.includes('MED') ? 10 : 8,
      fillColor: color, color: color, weight: 2,
      opacity: .9, fillOpacity: .35,
    }}).addTo(_conflictMap);
    marker.bindPopup(`<div style="font-family:Inter,sans-serif;min-width:200px;">
      <strong style="font-size:.9rem;">${{g.title||'—'}}</strong><br/>
      <span style="color:#94a3b8;font-size:.75rem;">${{g.region||'—'}}</span><br/>
      <span style="display:inline-block;margin-top:4px;padding:2px 8px;border-radius:3px;font-size:.7rem;font-weight:700;
        background:${{rl.includes('HIGH')?'rgba(239,68,68,.2)':rl.includes('MED')?'rgba(245,158,11,.2)':'rgba(34,197,94,.2)'}};
        color:${{color}};">${{rl}}</span><br/>
      <p style="margin:8px 0 0;font-size:.78rem;color:#cbd5e1;line-height:1.5;">${{(g.trigger||'').substring(0,120)}}…</p>
    </div>`, {{ className:'conflict-popup' }});
    marker.on('click', () => showConflictDetail(i));
  }});
  setTimeout(() => _conflictMap.invalidateSize(), 200);
}}
function showConflictDetail(i) {{
  const g = (GEO_RISK_DATA||[])[i];
  if (!g) return;
  document.getElementById('cd-title').textContent = g.title||'—';
  const rl = (g.risk_level||'MEDIUM').toUpperCase();
  const badge = document.getElementById('cd-risk-badge');
  badge.textContent = rl;
  badge.className = 'badge ' + (rl.includes('HIGH')?'badge-red':rl.includes('MED')?'badge-amber':'badge-green');
  document.getElementById('cd-trigger').textContent   = g.trigger||'—';
  document.getElementById('cd-sectors').textContent   = g.sectors_impacted||'—';
  document.getElementById('cd-assets').textContent    = g.asset_impact||'—';
  document.getElementById('cd-exposure').textContent  = g.market_exposure||'—';
  document.getElementById('cd-watch').textContent     = g.what_to_watch||'—';
  document.getElementById('conflict-detail').style.display = 'block';
  document.getElementById('conflict-detail').scrollIntoView({{behavior:'smooth',block:'nearest'}});
}}
// Map + watchlist init is handled inside show() above

// ── WATCHLIST ─────────────────────────────────────────────────────────────────
const WL_KEY = 'macro_watchlist_v1';
function wlLoad() {{ try {{ return JSON.parse(localStorage.getItem(WL_KEY)||'[]'); }} catch{{return [];}} }}
function wlSave(list) {{ localStorage.setItem(WL_KEY, JSON.stringify(list)); }}
function wlAdd() {{
  const input = document.getElementById('wl-input');
  const ticker = (input.value||'').trim().toUpperCase().replace(/[^A-Z0-9.\-^=]/g,'');
  if (!ticker) return;
  const list = wlLoad();
  if (list.includes(ticker)) {{ input.value=''; return; }}
  list.push(ticker);
  wlSave(list);
  input.value = '';
  wlRender();
  wlRefresh();
}}
function wlRemove(ticker) {{
  wlSave(wlLoad().filter(t => t !== ticker));
  wlRender();
}}
function wlRender() {{
  const list = wlLoad();
  const body = document.getElementById('wl-body');
  const empty = document.getElementById('wl-empty');
  if (!body) return;
  if (!list.length) {{ body.innerHTML=''; empty.style.display='block'; return; }}
  empty.style.display = 'none';
  body.innerHTML = list.map(t => `
    <tr class="wl-row" id="wl-row-${{t}}">
      <td style="font-weight:700;font-size:.95rem;letter-spacing:.5px;">${{t}}</td>
      <td class="wl-price" id="wl-price-${{t}}">—</td>
      <td class="wl-chg" id="wl-chg-${{t}}">—</td>
      <td class="wl-chart-btn"><button onclick="openChartModal('${{t}}')">📈 Chart</button></td>
      <td class="wl-del"><button onclick="wlRemove('${{t}}')" title="Remove">✕</button></td>
    </tr>`).join('');
}}
async function wlRefresh() {{
  const list = wlLoad();
  if (!list.length) return;
  const errEl = document.getElementById('wl-error');
  try {{
    const resp = await fetch('/api/quotes', {{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{tickers:list}})}});
    const data = await resp.json();
    list.forEach(t => {{
      const q = data[t];
      const priceEl = document.getElementById('wl-price-'+t);
      const chgEl   = document.getElementById('wl-chg-'+t);
      if (!priceEl) return;
      if (q) {{
        priceEl.textContent = '$' + q.price.toLocaleString('en-US', {{minimumFractionDigits:2,maximumFractionDigits:2}});
        const liveTag = q.source==='finnhub' ? '<span style="font-size:.6rem;padding:1px 5px;background:rgba(34,197,94,.15);color:var(--green);border-radius:3px;margin-left:6px;vertical-align:middle;">LIVE</span>' : '<span style="font-size:.6rem;padding:1px 5px;background:rgba(245,158,11,.1);color:var(--amber);border-radius:3px;margin-left:6px;vertical-align:middle;">15m</span>';
        chgEl.innerHTML   = (q.chg>=0?'▲ +':'▼ ') + Math.abs(q.chg).toFixed(2) + '%' + liveTag;
        chgEl.style.color   = q.up ? 'var(--green)' : 'var(--red)';
      }} else {{
        priceEl.textContent = 'N/A'; chgEl.textContent = '—';
      }}
    }});
    if (errEl) errEl.style.display='none';
  }} catch(e) {{
    if (errEl) {{ errEl.textContent='Could not fetch quotes: '+e.message; errEl.style.display='block'; }}
  }}
}}

// ── STOCK CHART MODAL ────────────────────────────────────────────────────────
let _chartInstance = null;
let _chartTicker   = null;
function openChartModal(ticker) {{
  _chartTicker = ticker;
  document.getElementById('chart-modal').style.display = 'flex';
  document.getElementById('cm-ticker').textContent = ticker;
  document.getElementById('cm-name').textContent   = '';
  document.getElementById('cm-price').textContent  = '';
  document.getElementById('cm-chg').textContent    = '';
  document.querySelectorAll('.cp-btn').forEach((b,i) => b.classList.toggle('active', i===2)); // default 3M
  loadChart(ticker, '3m', null);
}}
function closeChartModal() {{
  document.getElementById('chart-modal').style.display = 'none';
  if (_chartInstance) {{ _chartInstance.destroy(); _chartInstance = null; }}
}}
async function loadChart(ticker, period, btn) {{
  const t = ticker || _chartTicker;
  if (!t) return;
  _chartTicker = t;
  if (btn) {{ document.querySelectorAll('.cp-btn').forEach(b=>b.classList.remove('active')); btn.classList.add('active'); }}
  const loading = document.getElementById('cm-loading');
  const canvas  = document.getElementById('cm-canvas');
  loading.style.display = 'block'; canvas.style.display = 'none';
  if (_chartInstance) {{ _chartInstance.destroy(); _chartInstance = null; }}
  try {{
    const resp = await fetch(`/api/chart?ticker=${{encodeURIComponent(t)}}&period=${{period}}`);
    const d = await resp.json();
    if (d.error) {{ loading.textContent = 'Error: ' + d.error; return; }}
    document.getElementById('cm-ticker').textContent = d.ticker;
    document.getElementById('cm-name').textContent   = d.name;
    const chgColor = d.chg >= 0 ? 'var(--green)' : 'var(--red)';
    document.getElementById('cm-price').textContent  = '$' + (d.current||0).toLocaleString('en-US',{{minimumFractionDigits:2}});
    document.getElementById('cm-chg').textContent    = (d.chg>=0?'▲ +':'▼ ')+Math.abs(d.chg).toFixed(2)+'%';
    document.getElementById('cm-chg').style.color    = chgColor;
    loading.style.display = 'none'; canvas.style.display = 'block';
    const lineColor = d.chg >= 0 ? '#22c55e' : '#ef4444';
    const ctx = canvas.getContext('2d');
    _chartInstance = new Chart(ctx, {{
      type: 'line',
      data: {{
        labels: d.dates,
        datasets: [{{ label: d.ticker, data: d.closes,
          borderColor: lineColor, backgroundColor: lineColor+'18',
          borderWidth: 2, pointRadius: 0, fill: true, tension: .3 }}]
      }},
      options: {{
        responsive:true, maintainAspectRatio:true,
        plugins:{{ legend:{{display:false}}, tooltip:{{
          callbacks:{{ label: ctx => '$' + ctx.parsed.y.toLocaleString('en-US',{{minimumFractionDigits:2}}) }}
        }} }},
        scales:{{
          x:{{ ticks:{{color:'#64748b',maxTicksLimit:8,maxRotation:0}}, grid:{{color:'rgba(255,255,255,.04)'}} }},
          y:{{ ticks:{{color:'#64748b',callback:v=>'$'+v.toLocaleString()}}, grid:{{color:'rgba(255,255,255,.06)'}} }}
        }}
      }}
    }});
  }} catch(e) {{
    loading.textContent = 'Failed to load chart: ' + e.message;
  }}
}}
// Close modal on backdrop click
document.getElementById('chart-modal').addEventListener('click', function(e) {{
  if (e.target === this) closeChartModal();
}});

// ── Local clock (updates every second) ───────────────────────────────────────
(function() {{
  function updateClock() {{
    const now = new Date();
    const opts = {{ weekday:'long', year:'numeric', month:'long', day:'numeric', hour:'2-digit', minute:'2-digit', hour12:true }};
    const el = document.getElementById('hdr-clock');
    if (el) el.textContent = now.toLocaleString('en-US', opts);
  }}
  updateClock();
  setInterval(updateClock, 1000);
}})();
</script>
</body>
</html>"""

# ── FLASK ROUTES ──────────────────────────────────────────────────────────────

@app.route("/api/status")
def api_status():
    ready = bool(_cache["html"] and _cache["ts"] > 0 and time.time() < _cache["ts"])
    return jsonify({"ready": ready, "building": _building})


@app.route("/")
def index():
    if not _cache["html"]:
        return _LOADING_HTML
    return _cache["html"]


@app.route("/api/refresh", methods=["POST"])
def refresh():
    global _cache
    _cache["ts"] = 0.0
    html = get_or_build_html()
    return jsonify({"ok": True})


@app.route("/chat", methods=["POST"])
def chat():
    data = flask_request.get_json(force=True)
    message = data.get("message", "").strip()
    context = data.get("context", {})
    if not message:
        return jsonify({"error": "No message", "response": None})
    ctx_str = json_lib.dumps(context, indent=2) if context else ""
    system = (
        "You are a macro and portfolio analyst. Answer concisely based on real market conditions. "
        "Be clear, practical, and note uncertainty. This is for education, not financial advice.\n\n"
        + (f"Current macro context:\n{ctx_str}" if ctx_str else "")
    )
    try:
        msg = _client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            system=system,
            messages=[{"role": "user", "content": message}],
        )
        return jsonify({"response": msg.content[0].text.strip(), "error": None})
    except Exception as e:
        return jsonify({"error": str(e), "response": None})


@app.route("/api/stress-test", methods=["POST"])
def stress_test():
    data = flask_request.get_json(force=True)
    weights = data.get("weights", {})
    if not weights:
        return jsonify({"error": "No weights"}), 400

    PERIODS = [
        {"name": "GFC 2008-09",    "start": "2008-09-12", "end": "2009-03-09"},
        {"name": "COVID Crash",    "start": "2020-02-19", "end": "2020-03-23"},
        {"name": "2022 Bear",      "start": "2022-01-03", "end": "2022-10-13"},
        {"name": "2023 Bull Run",  "start": "2023-01-01", "end": "2023-12-29"},
        {"name": "Rate Hike Cycle","start": "2022-03-15", "end": "2023-07-26"},
    ]
    tickers = list(set([t for t in weights if t != "CASH"] + ["SPY"]))
    try:
        all_data = yf.download(tickers, start="2008-01-01", auto_adjust=True, progress=False)["Close"]
        if isinstance(all_data, __import__("pandas").Series):
            all_data = all_data.to_frame(name=tickers[0])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    results = []
    for p in PERIODS:
        try:
            pdata = all_data.loc[p["start"]:p["end"]].dropna(how="all")
            if len(pdata) < 2:
                results.append({"name": p["name"], "portfolio_return": None, "spy_return": None})
                continue
            port_ret = 0.0
            for ticker, w in weights.items():
                if ticker == "CASH":
                    continue
                if ticker in pdata.columns:
                    s = pdata[ticker].dropna()
                    if len(s) >= 2:
                        port_ret += float((s.iloc[-1] / s.iloc[0] - 1) * 100) * w
            spy_ret = None
            if "SPY" in pdata.columns:
                s = pdata["SPY"].dropna()
                if len(s) >= 2:
                    spy_ret = float((s.iloc[-1] / s.iloc[0] - 1) * 100)
            results.append({"name": p["name"], "portfolio_return": round(port_ret, 2), "spy_return": round(spy_ret, 2) if spy_ret is not None else None})
        except Exception:
            results.append({"name": p["name"], "portfolio_return": None, "spy_return": None})

    return jsonify({"periods": results})


@app.route("/api/chart")
def api_chart():
    ticker = flask_request.args.get("ticker", "").upper().strip()
    period = flask_request.args.get("period", "3mo")
    period_map = {"1w":"5d","1m":"1mo","3m":"3mo","6m":"6mo","1y":"1y","5y":"5y","max":"max"}
    yf_period = period_map.get(period, "3mo")
    if not ticker:
        return jsonify({"error": "No ticker"}), 400
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=yf_period)
        if len(hist) == 0:
            return jsonify({"error": "No data found"}), 404
        name = ticker
        try:
            info = t.info
            name = info.get("longName") or info.get("shortName") or ticker
        except:
            pass
        curr = round(float(hist["Close"].iloc[-1]), 2)
        prev = round(float(hist["Close"].iloc[-2]), 2) if len(hist) >= 2 else curr
        chg = round(((curr - prev) / prev) * 100, 2) if prev else 0
        return jsonify({
            "ticker": ticker, "name": name,
            "current": curr, "chg": chg,
            "dates":  [str(d.date()) for d in hist.index],
            "closes": [round(float(v), 2) for v in hist["Close"]],
            "volumes":[int(v) for v in hist["Volume"]],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/quotes", methods=["POST"])
def api_quotes():
    tickers = flask_request.get_json(force=True).get("tickers", [])[:60]

    def finnhub_quote(ticker):
        """Real-time quote via Finnhub (works for stocks/ETFs/forex)."""
        try:
            url = f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={FINNHUB_API_KEY}"
            r = requests.get(url, timeout=5).json()
            curr = float(r.get("c", 0))
            prev = float(r.get("pc", 0))
            if curr and prev:
                chg = round(((curr - prev) / prev) * 100, 2)
                return {"price": round(curr, 2), "chg": chg, "up": chg >= 0, "source": "finnhub"}
        except:
            pass
        return None

    def yf_quote(ticker):
        """Fallback: yFinance (15-min delayed, handles crypto)."""
        try:
            hist = yf.Ticker(ticker).history(period="2d")
            if len(hist) >= 2:
                prev = float(hist["Close"].iloc[-2])
                curr = float(hist["Close"].iloc[-1])
                chg  = round(((curr - prev) / prev) * 100, 2)
                return {"price": round(curr, 2), "chg": chg, "up": chg >= 0, "source": "yfinance"}
            elif len(hist) == 1:
                curr = float(hist["Close"].iloc[-1])
                return {"price": round(curr, 2), "chg": 0.0, "up": True, "source": "yfinance"}
        except:
            pass
        return None

    def fetch_q(ticker):
        # Crypto tickers (contain - like BTC-USD) go straight to yFinance
        # Finnhub handles them differently and free tier is limited
        is_crypto = "-USD" in ticker or "-USDT" in ticker
        if FINNHUB_API_KEY and not is_crypto:
            data = finnhub_quote(ticker)
            if data and data["price"] > 0:
                return ticker, data
        return ticker, yf_quote(ticker)

    results = {}
    with ThreadPoolExecutor(max_workers=20) as ex:
        for ticker, data in ex.map(fetch_q, tickers):
            results[ticker] = data
    return jsonify(results)


# ── STARTUP: warm cache in background so first HTTP request is instant ─────────
_start_background_build()

if __name__ == "__main__":
    print("Starting Macro Dashboard server...")
    port = int(os.environ.get("PORT", 5002))
    app.run(debug=False, host="0.0.0.0", port=port)
