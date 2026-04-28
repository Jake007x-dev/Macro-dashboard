from __future__ import annotations

import json
import math
import os
import time
import re
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import anthropic
import numpy as np
import openai as _openai_module
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv, dotenv_values
from flask import Flask, jsonify, render_template_string, request

try:
    _env = dotenv_values(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
    os.environ.update({k: v for k, v in _env.items() if v})
except Exception:
    load_dotenv()

app = Flask(__name__)
_anthropic_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
FMP_API_KEY = os.environ.get("FMP_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

TRADING_DAYS = 252
RISK_FREE_RATE = 0.04
DEFAULT_START = "2022-01-01"
LINKEDIN_URL = "https://www.linkedin.com/in/jakemarleyjoseph/"

DEFAULT_PORTFOLIOS = {
    "Portfolio A": "VOO,40%\nQQQ,20%\nMSFT,15%\nAAPL,15%\nGLD,10%",
    "Portfolio B": "VTI,$4000\nSCHD,$2500\nVXUS,$2000\nGLD,$1000\nTLT,$500",
    "Portfolio C": "NVDA,20%\nMSFT,20%\nMETA,15%\nAMZN,15%\nGOOGL,15%\nCash,15%",
}
DEFAULT_CUSTOM_BENCHMARK = "XLF,30%\nXLK,30%\nXLE,20%\nXLV,20%"

MACRO_TICKERS = {
    "S&P 500": "^GSPC", "Nasdaq 100": "^NDX", "10Y Treasury": "^TNX",
    "WTI Crude": "CL=F", "Gold": "GC=F", "VIX": "^VIX",
    "USD Index": "DX-Y.NYB", "Bitcoin": "BTC-USD",
}
GLOBAL_TICKERS = {
    "S&P 500": "^GSPC", "Nasdaq": "^IXIC", "Dow Jones": "^DJI",
    "FTSE 100": "^FTSE", "DAX": "^GDAXI", "Nikkei 225": "^N225",
    "Hang Seng": "^HSI", "Euro Stoxx 50": "^STOXX50E",
    "ASX 200": "^AXJO", "Brazil": "^BVSP", "India Nifty": "^NSEI",
}
SECTOR_ETFS = {
    "Technology": "XLK", "Financials": "XLF", "Energy": "XLE",
    "Healthcare": "XLV", "Industrials": "XLI", "Consumer Disc.": "XLY",
    "Consumer Staples": "XLP", "Utilities": "XLU", "Real Estate": "XLRE",
    "Materials": "XLB", "Comm. Services": "XLC",
}
BENCHMARKS = {
    "SPY": {"SPY": 1.0}, "QQQ": {"QQQ": 1.0},
    "VTI": {"VTI": 1.0}, "60/40": {"SPY": 0.6, "AGG": 0.4},
}
ETF_SECTOR_MAP = {
    "XLK": "Technology", "XLF": "Financials", "XLE": "Energy",
    "XLV": "Healthcare", "XLI": "Industrials", "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples", "XLU": "Utilities", "XLRE": "Real Estate",
    "XLB": "Materials", "XLC": "Communication Services",
    "VOO": "Broad Market", "VTI": "Broad Market", "QQQ": "Technology / Growth",
    "SCHD": "Dividend / Value", "VXUS": "International Equity",
    "GLD": "Gold", "TLT": "Long Duration Bonds", "AGG": "Core Bonds",
}
ASSET_CLASS_MAP = {
    "EQUITY": "Equity", "ETF": "ETF", "MUTUALFUND": "Fund",
    "INDEX": "Index", "CRYPTOCURRENCY": "Crypto", "FUTURE": "Commodity",
}
HIGHER_IS_BETTER = {
    "Cumulative Return": True, "Annualized Return": True, "Volatility": False,
    "Sharpe": True, "Max Drawdown": True, "Beta vs SPY": None,
    "Alpha vs SPY": True, "Correlation vs SPY": None,
}


@dataclass
class Holding:
    ticker: str
    value_input: float
    mode: str


def clean_ticker(t: str) -> str:
    return t.strip().upper()


def parse_portfolio_text(text: str) -> List[Holding]:
    holdings: List[Holding] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Invalid line: '{line}'. Use TICKER,VALUE like AAPL,20% or VOO,$4000")
        ticker = clean_ticker(parts[0])
        vt = parts[1].replace(" ", "")
        if ticker == "CASH":
            if vt.endswith("%"):
                holdings.append(Holding("CASH", float(vt[:-1]), "percent"))
            elif vt.startswith("$"):
                holdings.append(Holding("CASH", float(vt[1:].replace(",", "")), "dollar"))
            else:
                holdings.append(Holding("CASH", float(vt.replace(",", "")), "dollar"))
            continue
        if vt.endswith("%"):
            holdings.append(Holding(ticker, float(vt[:-1]), "percent"))
        elif vt.startswith("$"):
            holdings.append(Holding(ticker, float(vt[1:].replace(",", "")), "dollar"))
        else:
            n = float(vt.replace(",", ""))
            holdings.append(Holding(ticker, n, "percent" if n <= 100 else "dollar"))
    if not holdings:
        raise ValueError("Portfolio is empty.")
    return holdings


def normalize_holdings(holdings: List[Holding]) -> Dict[str, float]:
    if all(h.mode == "percent" for h in holdings):
        total = sum(h.value_input for h in holdings)
        if total <= 0:
            raise ValueError("Portfolio percent total must be positive.")
        return {h.ticker: h.value_input / total for h in holdings}
    if all(h.mode in {"dollar", "cash"} for h in holdings):
        total = sum(h.value_input for h in holdings)
        if total <= 0:
            raise ValueError("Portfolio dollar total must be positive.")
        return {h.ticker: h.value_input / total for h in holdings}
    raise ValueError("Use either all percentages or all dollar amounts within one portfolio.")


def get_ticker_metadata(tickers: List[str]) -> Dict[str, dict]:
    meta: Dict[str, dict] = {}
    for ticker in tickers:
        if ticker == "CASH":
            meta[ticker] = {"shortName": "Cash", "sector": "Cash", "assetClass": "Cash", "quoteType": "CASH"}
            continue
        try:
            info = yf.Ticker(ticker).info
        except Exception:
            info = {}
        qt = str(info.get("quoteType", "")).upper()
        ac = ASSET_CLASS_MAP.get(qt, "Other")
        sector = info.get("sector") or ETF_SECTOR_MAP.get(ticker, ac)
        meta[ticker] = {
            "shortName": info.get("shortName", ticker),
            "sector": sector,
            "assetClass": ac if ticker not in ETF_SECTOR_MAP else "ETF",
            "quoteType": qt or "UNKNOWN",
        }
    return meta


def download_price_data(tickers: List[str], start: str = DEFAULT_START) -> pd.DataFrame:
    live = [t for t in tickers if t != "CASH"]
    if not live:
        return pd.DataFrame(index=pd.date_range(start=start, end=datetime.today(), freq="B"))
    data = yf.download(live, start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame(name=live[0])
    return data.ffill().dropna(how="all")


def build_portfolio_returns(weights: Dict[str, float], returns: pd.DataFrame) -> pd.Series:
    parts = []
    for ticker, w in weights.items():
        if ticker == "CASH":
            parts.append(pd.Series(0.0, index=returns.index) * w)
        elif ticker in returns.columns:
            parts.append(returns[ticker].fillna(0.0) * w)
    if not parts:
        return pd.Series(dtype=float)
    return pd.concat(parts, axis=1).sum(axis=1).dropna()


def price_to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame()
    return prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")


def cumulative_curve(returns: pd.Series) -> pd.Series:
    return (1 + returns.fillna(0)).cumprod()


def max_drawdown(returns: pd.Series) -> float:
    curve = cumulative_curve(returns)
    return float((curve / curve.cummax() - 1).min())


def annualized_return(returns: pd.Series) -> float:
    if not len(returns):
        return 0.0
    total = cumulative_curve(returns).iloc[-1]
    years = len(returns) / TRADING_DAYS
    return float(total ** (1 / years) - 1) if years > 0 else 0.0


def annualized_volatility(returns: pd.Series) -> float:
    return float(returns.std() * math.sqrt(TRADING_DAYS)) if len(returns) else 0.0


def sharpe_ratio(returns: pd.Series, rf: float = RISK_FREE_RATE) -> float:
    if not len(returns):
        return 0.0
    vol = returns.std()
    if vol == 0 or np.isnan(vol):
        return 0.0
    return float(((returns.mean() - rf / TRADING_DAYS) / vol) * math.sqrt(TRADING_DAYS))


def beta_alpha(p: pd.Series, b: pd.Series) -> Tuple[float, float, float]:
    aligned = pd.concat([p, b], axis=1).dropna()
    if aligned.empty or aligned.iloc[:, 1].var() == 0:
        return 0.0, 0.0, 0.0
    pv, bv = aligned.iloc[:, 0], aligned.iloc[:, 1]
    beta = float(np.cov(pv, bv)[0, 1] / np.var(bv))
    alpha = float((pv.mean() - beta * bv.mean()) * TRADING_DAYS)
    corr = float(pv.corr(bv)) if len(aligned) > 1 else 0.0
    return beta, alpha, corr


def top_holdings(weights: Dict[str, float], meta: Dict[str, dict], n: int = 5) -> List[dict]:
    return [
        {"ticker": t, "name": meta.get(t, {}).get("shortName", t), "weight": w}
        for t, w in sorted(weights.items(), key=lambda x: x[1], reverse=True)[:n]
    ]


def exposure_breakdown(weights: Dict[str, float], meta: Dict[str, dict], key: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for t, w in weights.items():
        k = meta.get(t, {}).get(key, "Other")
        out[k] = out.get(k, 0.0) + w
    return dict(sorted(out.items(), key=lambda x: x[1], reverse=True))


def concentration_risk(weights: Dict[str, float]) -> dict:
    vals = np.array(list(weights.values()))
    hhi = float(np.sum(vals ** 2))
    top1 = max(weights.values()) if weights else 0.0
    top3 = sum(sorted(weights.values(), reverse=True)[:3])
    label = "High" if hhi > 0.18 or top1 > 0.25 else "Moderate" if hhi > 0.10 or top1 > 0.18 else "Low"
    return {"hhi": hhi, "top1": top1, "top3": top3, "label": label}


def monte_carlo_projection(sv, ar, av, years=3, paths=500, bull=0.04, bear=-0.05) -> dict:
    days = TRADING_DAYS * years
    dt = 1 / TRADING_DAYS
    def sim(adj):
        mu = ar + adj
        drift = (mu - 0.5 * av ** 2) * dt
        r = np.random.normal(size=(paths, days))
        return sv * np.exp(drift + av * math.sqrt(dt) * r).cumprod(axis=1)
    def summ(arr):
        f = arr[:, -1]
        return {
            "median": float(np.median(f)), "p10": float(np.percentile(f, 10)),
            "p25": float(np.percentile(f, 25)), "p75": float(np.percentile(f, 75)),
            "p90": float(np.percentile(f, 90)), "prob_loss": float(np.mean(f < sv)),
            "sample_path": [float(x) for x in arr[0, ::21][:36]],
        }
    return {"base": summ(sim(0)), "bull": summ(sim(bull)), "bear": summ(sim(bear)),
            "years": years, "start_value": sv}


def build_benchmark_returns(returns: pd.DataFrame, custom: Optional[Dict[str, float]]) -> Dict[str, pd.Series]:
    out = {name: build_portfolio_returns(w, returns) for name, w in BENCHMARKS.items()}
    if custom:
        out["Custom"] = build_portfolio_returns(custom, returns)
    return out


def macro_sensitivity(weights: Dict[str, float], meta: Dict[str, dict]) -> Dict[str, str]:
    se = exposure_breakdown(weights, meta, "sector")
    def sc(*keys):
        return sum(v for k, v in se.items() if any(x.lower() in k.lower() for x in keys))
    growth = sc("Technology", "Communication", "Consumer Discretionary", "Technology / Growth")
    fin = sc("Financial")
    energy = sc("Energy")
    health = sc("Healthcare")
    util = sc("Utilities")
    re_ = sc("Real Estate")
    bonds = sc("Bond", "Duration")
    gold = sc("Gold")
    intl = sc("International")
    def lbl(v, pos, neg):
        if v > 0.35: return pos
        if v > 0.15: return f"Moderately {pos.lower()}"
        return neg
    return {
        "Higher Rates": lbl(growth + re_ + bonds, "Vulnerable", "Limited sensitivity"),
        "Lower Rates": lbl(growth + re_ + bonds, "Likely beneficiary", "Limited upside"),
        "Rising Inflation": lbl(energy + fin + gold, "Likely resilient", "Potential headwind"),
        "Recession": lbl(util + health + bonds, "Defensive cushion", "Cyclical risk"),
        "Oil Shock": lbl(energy, "Net beneficiary", "Likely drag"),
        "Tech Selloff": lbl(growth, "High sensitivity", "Contained sensitivity"),
        "Strong Dollar": lbl(intl, "Potential headwind", "Limited direct sensitivity"),
    }


def factor_tilt(weights: Dict[str, float], meta: Dict[str, dict]) -> Dict[str, float]:
    se = exposure_breakdown(weights, meta, "sector")
    return {
        "Growth / Tech": round(sum(v for k, v in se.items() if any(x in k for x in ["Technology", "Technology / Growth", "Communication"])), 4),
        "Defensive": round(sum(v for k, v in se.items() if any(x in k for x in ["Healthcare", "Utilities", "Consumer Staples"])), 4),
        "Cyclicals": round(sum(v for k, v in se.items() if any(x in k for x in ["Industrials", "Financial", "Consumer Discretionary", "Energy", "Materials"])), 4),
        "Real Assets": round(sum(v for k, v in se.items() if any(x in k for x in ["Energy", "Gold", "Real Estate"])), 4),
        "International": round(sum(v for k, v in se.items() if "International" in k), 4),
    }


def sector_overlap(wa: Dict[str, float], wb: Dict[str, float], meta: Dict[str, dict]) -> float:
    a = exposure_breakdown(wa, meta, "sector")
    b = exposure_breakdown(wb, meta, "sector")
    return float(sum(min(a.get(k, 0.0), b.get(k, 0.0)) for k in set(a) | set(b)))


def parse_custom_benchmark(text: str) -> Optional[Dict[str, float]]:
    if not text.strip():
        return None
    return normalize_holdings(parse_portfolio_text(text))


def rolling_correlation(a: pd.Series, b: pd.Series, window: int = 63) -> pd.Series:
    aligned = pd.concat([a, b], axis=1).dropna()
    if aligned.empty:
        return pd.Series(dtype=float)
    return aligned.iloc[:, 0].rolling(window).corr(aligned.iloc[:, 1]).dropna()


def fmt_pct(x: float) -> str: return f"{x * 100:.1f}%"
def fmt_money(x: float) -> str: return f"${x:,.0f}"


# ── Enrichment functions ──────────────────────────────────────────────────────

def get_macro_cards() -> List[dict]:
    cards = []
    for label, ticker in MACRO_TICKERS.items():
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            if len(hist) >= 2:
                prev, curr = float(hist["Close"].iloc[-2]), float(hist["Close"].iloc[-1])
                chg = ((curr - prev) / prev) * 100 if prev else 0.0
                cards.append({"label": label, "value": curr, "change": chg})
            else:
                cards.append({"label": label, "value": None, "change": 0.0})
        except Exception:
            cards.append({"label": label, "value": None, "change": 0.0})
    return cards


def detect_regime(macro_cards_raw: List[dict]) -> Tuple[str, str]:
    vals = {c["label"]: c.get("value") for c in macro_cards_raw}
    chgs = {c["label"]: c.get("change", 0) for c in macro_cards_raw}
    vix = vals.get("VIX")
    tnx = vals.get("10Y Treasury")
    spy_chg = chgs.get("S&P 500", 0)
    oil_chg = chgs.get("WTI Crude", 0)
    if vix and vix > 30:
        return "High Volatility / Risk Off", f"VIX at {vix:.1f} — elevated stress. Defensive positioning preferred."
    if vix and tnx and vix > 20 and tnx > 4.5:
        return "Stagflation Watch", f"Elevated vol (VIX {vix:.1f}) + high 10Y ({tnx:.2f}%) compressing equity multiples."
    if tnx and tnx > 4.75:
        return "Rate Pressure", f"10Y at {tnx:.2f}% — duration and growth assets under pressure."
    if oil_chg and oil_chg > 3:
        return "Energy / Inflation Surge", f"WTI up {oil_chg:.1f}% — energy benefits; watch consumer margins."
    if vix and vix < 15 and spy_chg and spy_chg > 0:
        return "Risk On / Bull Trend", "Low volatility + positive SPY momentum. Growth assets in favor."
    if vix and vix < 18:
        return "Neutral / Cautiously Optimistic", "VIX contained. Markets stable with selective risk appetite."
    return "Neutral", "Mixed signals across macro indicators. Monitor VIX and rates for direction."


def compute_fear_greed(macro_cards_raw: List[dict]) -> dict:
    vix = next((c.get("value") for c in macro_cards_raw if c["label"] == "VIX"), None)
    spy_chg = next((c.get("change", 0) for c in macro_cards_raw if c["label"] == "S&P 500"), 0)
    score = 50
    if vix is not None:
        score = max(0, min(100, int(110 - vix * 3)))
    if spy_chg:
        mom = max(0, min(100, int(50 + spy_chg * 8)))
        score = int(score * 0.6 + mom * 0.4)
    if score >= 75:
        label, color = "Extreme Greed", "#3bd671"
    elif score >= 55:
        label, color = "Greed", "#86efac"
    elif score >= 45:
        label, color = "Neutral", "#ffb44d"
    elif score >= 25:
        label, color = "Fear", "#fb923c"
    else:
        label, color = "Extreme Fear", "#ff5f6d"
    t = score / 100
    arc = math.pi * 90
    angle = math.pi * (1 - t)
    return {
        "score": score, "label": label, "color": color,
        "dash": round(t * arc, 1), "total_arc": round(arc, 1),
        "nx": round(100 + 85 * math.cos(angle), 1),
        "ny": round(100 - 85 * math.sin(angle), 1),
    }


def get_global_heatmap() -> List[dict]:
    result = []
    for label, ticker in GLOBAL_TICKERS.items():
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            if len(hist) >= 2:
                prev, curr = float(hist["Close"].iloc[-2]), float(hist["Close"].iloc[-1])
                chg = round(((curr - prev) / prev) * 100, 2) if prev else 0.0
            else:
                chg = 0.0
        except Exception:
            chg = 0.0
        result.append({"label": label, "change": chg})
    return result


def get_sector_rotation() -> List[dict]:
    tickers = list(SECTOR_ETFS.values())
    result = []
    try:
        data = yf.download(tickers, period="3mo", auto_adjust=True, progress=False)["Close"]
        if isinstance(data, pd.Series):
            data = data.to_frame(name=tickers[0])
        for sector, ticker in SECTOR_ETFS.items():
            if ticker not in data.columns:
                continue
            s = data[ticker].dropna()
            if len(s) < 5:
                continue
            w1 = round(float((s.iloc[-1] / s.iloc[-6] - 1) * 100), 2) if len(s) >= 6 else 0.0
            m1 = round(float((s.iloc[-1] / s.iloc[-22] - 1) * 100), 2) if len(s) >= 22 else 0.0
            m3 = round(float((s.iloc[-1] / s.iloc[0] - 1) * 100), 2)
            result.append({"sector": sector, "w1": w1, "m1": m1, "m3": m3})
    except Exception:
        pass
    return sorted(result, key=lambda x: x["m1"], reverse=True)


def get_news_feed(n: int = 8) -> List[dict]:
    items = []
    try:
        url = "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC&region=US&lang=en-US"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=6) as resp:
            root = ET.fromstring(resp.read())
            for item in root.findall(".//item")[:n]:
                title = item.findtext("title", "").strip()
                link = item.findtext("link", "").strip()
                pub = item.findtext("pubDate", "").strip()[:22]
                if title:
                    items.append({"title": title, "link": link, "pub": pub})
    except Exception:
        pass
    return items


def portfolio_summary_blurb(weights: Dict[str, float], meta: Dict[str, dict],
                             ann_vol: float, sharpe: float, conc: dict) -> str:
    se = exposure_breakdown(weights, meta, "sector")
    top = max(se, key=se.get) if se else "Mixed"
    style = ("Growth-tilted" if any(x in top for x in ["Technology", "Growth", "Communication"])
             else "Defensively positioned" if any(x in top for x in ["Healthcare", "Utilities", "Staples"])
             else "Income-focused" if any(x in top for x in ["Dividend", "Bond", "Value"])
             else "Broadly diversified")
    vol_d = "high vol" if ann_vol > 0.20 else "moderate vol" if ann_vol > 0.12 else "low vol"
    eff = "strong risk-adj. returns" if sharpe > 1.0 else "avg efficiency" if sharpe > 0.5 else "below-avg efficiency"
    return f"{style} • {conc['label'].lower()} concentration • {vol_d} • {eff} (Sharpe {sharpe:.2f})"


def risk_score(ann_ret: float, ann_vol: float, sharpe: float, mdd: float) -> int:
    s = min(40, max(0, sharpe * 20)) + min(30, max(0, ann_ret * 100)) + min(30, max(0, 30 + mdd * 100))
    return max(1, min(10, round(s / 10)))


def color_comparison(rows: List[dict]) -> List[dict]:
    result = []
    for row in rows:
        cells = row["cells"]
        hib = HIGHER_IS_BETTER.get(row["metric"])
        nums = []
        for v in cells:
            try:
                nums.append(float(v.replace("%", "").replace(",", "")))
            except Exception:
                nums.append(None)
        valid = [n for n in nums if n is not None]
        best = max(valid) if valid else None
        worst = min(valid) if valid else None
        colored = []
        for v, n in zip(cells, nums):
            if n is None or hib is None or best == worst:
                cls = ""
            elif hib:
                cls = "cell-best" if n == best else ("cell-worst" if n == worst else "")
            else:
                cls = "cell-best" if n == worst else ("cell-worst" if n == best else "")
            colored.append({"value": v, "cls": cls})
        result.append({"metric": row["metric"], "colored": colored})
    return result


def get_ai_commentary(name: str, weights: Dict[str, float], metrics: dict,
                       sector_exposure: dict, macro_sens: dict, regime: str) -> str:
    top = ", ".join(f"{t} ({v})" for t, v in list(weights.items())[:5])
    sectors = ", ".join(f"{k}: {v}" for k, v in list(sector_exposure.items())[:3])
    macro = ", ".join(f"{k}: {v}" for k, v in list(macro_sens.items())[:3])
    prompt = f"""You are a professional portfolio analyst. Write exactly 3 sentences for this portfolio dashboard.

Portfolio: {name} | Holdings: {top}
Return: {metrics.get('annual_return')} | Vol: {metrics.get('volatility')} | Sharpe: {metrics.get('sharpe')} | Max DD: {metrics.get('max_drawdown')} | Beta: {metrics.get('beta')}
Sectors: {sectors} | Macro: {macro} | Regime: {regime}

Sentence 1: character/style. Sentence 2: key risk or opportunity given regime. Sentence 3: one specific actionable observation. Direct and professional."""
    try:
        msg = _anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()
    except Exception:
        return ""


def make_series_json(series_map: Dict[str, pd.Series], pct: bool = True) -> str:
    out = {}
    for name, s in series_map.items():
        if s is None or not len(s):
            continue
        vals = s.dropna()
        out[name] = {
            "labels": [d.strftime("%Y-%m-%d") for d in vals.index],
            "values": [round(float(v * 100), 2) if pct else round(float(v), 2) for v in vals.values],
        }
    return json.dumps(out)


IMPACT_RULES = [
    {"keywords": ["oil", "crude", "opec", "middle east", "iran", "hormuz"],
     "impacts": {"Energy": "positive", "Airlines": "negative", "Consumer Discretionary": "negative", "Inflation": "negative"},
     "assets": {"XLE": "positive", "USO": "positive", "AAL": "negative"}},
    {"keywords": ["fed", "rates", "yields", "inflation", "cpi", "hawkish", "dovish"],
     "impacts": {"Technology": "negative", "Utilities": "negative", "Financials": "mixed", "Real Estate": "negative"},
     "assets": {"TLT": "negative", "QQQ": "negative", "XLF": "mixed"}},
    {"keywords": ["china", "tariff", "trade", "semiconductor", "chip"],
     "impacts": {"Technology": "negative", "Industrials": "negative", "Consumer Discretionary": "mixed"},
     "assets": {"NVDA": "negative", "TSM": "negative", "XLK": "negative"}},
    {"keywords": ["war", "conflict", "sanction", "ukraine", "russia", "defense"],
     "impacts": {"Defense": "positive", "Energy": "positive", "Global Equities": "negative"},
     "assets": {"LMT": "positive", "GC=F": "positive", "EEM": "negative"}},
    {"keywords": ["jobs", "unemployment", "payroll", "wage", "labor"],
     "impacts": {"Rates": "mixed", "Consumer Discretionary": "mixed", "Financials": "positive"},
     "assets": {"SPY": "mixed", "XLY": "positive", "TLT": "negative"}},
]


def get_geo_hotspots() -> List[dict]:
    return [
        {"id": "taiwan", "title": "Taiwan Strait Tension", "region": "Asia-Pacific",
         "lat": 23.5, "lng": 121.0, "risk_score": 78, "risk_level": "HIGH",
         "category": "conflict", "color": "#ef4444",
         "summary": "Ongoing cross-strait military activity. Key semiconductor supply chain risk.",
         "market_impact": "Semiconductor supply chains, Taiwan ETF (EWT), global tech equities",
         "affected_assets": ["TSM", "NVDA", "AMAT", "EWT"],
         "affected_sectors": ["Technology", "Semiconductors"],
         "portfolio_keywords": ["TSM", "NVDA", "QCOM", "AMAT", "INTC", "XLK", "QQQ", "SOXX"]},
        {"id": "hormuz", "title": "Strait of Hormuz / Iran", "region": "Middle East",
         "lat": 26.6, "lng": 56.3, "risk_score": 72, "risk_level": "HIGH",
         "category": "oil_chokepoint", "color": "#ef4444",
         "summary": "Iran tensions and Hormuz blockade risk. ~20% of global oil transit.",
         "market_impact": "WTI crude, energy equities, airlines, consumer spending",
         "affected_assets": ["XLE", "USO", "CL=F", "AAL", "DAL"],
         "affected_sectors": ["Energy", "Airlines", "Consumer Discretionary"],
         "portfolio_keywords": ["XLE", "CVX", "XOM", "OXY", "COP", "USO"]},
        {"id": "red_sea", "title": "Red Sea / Suez Shipping", "region": "Middle East",
         "lat": 15.0, "lng": 42.0, "risk_score": 65, "risk_level": "ELEVATED",
         "category": "shipping", "color": "#f97316",
         "summary": "Houthi attacks rerouting global shipping. 12% of global trade affected.",
         "market_impact": "Shipping costs, supply chain inflation, consumer goods",
         "affected_assets": ["ZIM", "MAERSK", "GC=F"],
         "affected_sectors": ["Industrials", "Consumer Staples", "Materials"],
         "portfolio_keywords": ["XLI", "XLP", "XLB"]},
        {"id": "ukraine", "title": "Ukraine / Russia War", "region": "Eastern Europe",
         "lat": 49.0, "lng": 31.0, "risk_score": 68, "risk_level": "HIGH",
         "category": "conflict", "color": "#ef4444",
         "summary": "Ongoing conflict. European energy security and grain supply disruption.",
         "market_impact": "European equities, energy, wheat/grain commodities, defense",
         "affected_assets": ["LMT", "RTX", "NOC", "NG=F", "ZW=F", "EZU"],
         "affected_sectors": ["Defense", "Energy", "Materials", "Agriculture"],
         "portfolio_keywords": ["LMT", "RTX", "NOC", "GD", "XLI", "VXUS", "EFA", "EZU"]},
        {"id": "china_us_trade", "title": "China / US Trade War", "region": "Global",
         "lat": 35.0, "lng": 105.0, "risk_score": 80, "risk_level": "HIGH",
         "category": "tariffs", "color": "#ef4444",
         "summary": "Escalating tariffs and decoupling. Broad impact on global supply chains.",
         "market_impact": "Global trade, tech supply chains, consumer goods inflation",
         "affected_assets": ["NVDA", "AAPL", "QQQ", "EEM", "FXI"],
         "affected_sectors": ["Technology", "Consumer Discretionary", "Industrials"],
         "portfolio_keywords": ["AAPL", "NVDA", "MSFT", "AMZN", "QQQ", "XLK", "FXI", "VXUS"]},
        {"id": "south_china_sea", "title": "South China Sea", "region": "Asia-Pacific",
         "lat": 12.0, "lng": 114.0, "risk_score": 55, "risk_level": "ELEVATED",
         "category": "conflict", "color": "#f97316",
         "summary": "Territorial disputes. Key global shipping lane (~$3T annual trade).",
         "market_impact": "Asian equities, shipping, oil transit",
         "affected_assets": ["EWJ", "EWY", "EWT", "EWH"],
         "affected_sectors": ["Industrials", "Technology"],
         "portfolio_keywords": ["EWJ", "EWY", "EWT", "VXUS", "EFA"]},
        {"id": "europe_energy", "title": "European Energy Crisis", "region": "Europe",
         "lat": 51.0, "lng": 10.0, "risk_score": 48, "risk_level": "WATCH",
         "category": "commodity_shock", "color": "#eab308",
         "summary": "Energy dependency and grid stress. Structural vulnerability post-Ukraine.",
         "market_impact": "European industrial output, EUR/USD, natural gas prices",
         "affected_assets": ["NG=F", "EZU", "EURUSD=X"],
         "affected_sectors": ["Utilities", "Industrials", "Energy"],
         "portfolio_keywords": ["NG=F", "EZU", "EFA", "VXUS", "XLU"]},
        {"id": "latam_commodity", "title": "LatAm Commodity / Currency", "region": "Latin America",
         "lat": -15.0, "lng": -60.0, "risk_score": 42, "risk_level": "WATCH",
         "category": "currency_pressure", "color": "#eab308",
         "summary": "Brazil/Argentina currency volatility and commodity export disruption.",
         "market_impact": "EWZ, copper, agricultural commodities, EM bonds",
         "affected_assets": ["EWZ", "HG=F", "EEM"],
         "affected_sectors": ["Materials", "Agriculture"],
         "portfolio_keywords": ["EWZ", "EEM", "XLB", "HG=F"]},
        {"id": "india_pakistan", "title": "India / Pakistan Tension", "region": "South Asia",
         "lat": 28.0, "lng": 70.0, "risk_score": 52, "risk_level": "ELEVATED",
         "category": "conflict", "color": "#f97316",
         "summary": "Military escalation risk. Limited direct global market impact but EM sentiment.",
         "market_impact": "India ETF (INDA), broader EM sentiment, risk-off flows",
         "affected_assets": ["INDA", "EEM", "GC=F"],
         "affected_sectors": ["Emerging Markets"],
         "portfolio_keywords": ["INDA", "EEM", "VXUS", "EFA"]},
        {"id": "korea", "title": "Korean Peninsula Risk", "region": "Asia-Pacific",
         "lat": 37.5, "lng": 127.5, "risk_score": 45, "risk_level": "WATCH",
         "category": "conflict", "color": "#eab308",
         "summary": "North Korean missile tests and nuclear posturing. Periodic risk-off trigger.",
         "market_impact": "Korean won, EWY, Asian equities, semiconductor supply",
         "affected_assets": ["EWY", "EWJ", "GC=F", "^VIX"],
         "affected_sectors": ["Technology", "Semiconductors"],
         "portfolio_keywords": ["EWY", "EWJ", "VXUS", "EFA", "NVDA", "TSM"]},
        {"id": "middle_east", "title": "Middle East Conflict", "region": "Middle East",
         "lat": 32.0, "lng": 35.0, "risk_score": 70, "risk_level": "HIGH",
         "category": "conflict", "color": "#ef4444",
         "summary": "Israel-Hamas conflict and broader regional escalation risk.",
         "market_impact": "Oil risk premium, gold safe haven, defense, EM risk-off",
         "affected_assets": ["GC=F", "XLE", "LMT", "RTX", "CL=F"],
         "affected_sectors": ["Energy", "Defense", "Gold"],
         "portfolio_keywords": ["GLD", "GC=F", "XLE", "LMT", "RTX", "NOC"]},
    ]


def classify_portfolio_dna(weights, meta, beta, ann_vol, factor_tilts, macro_sens):
    growth = factor_tilts.get("Growth / Tech", 0)
    defensive = factor_tilts.get("Defensive", 0)
    real_assets = factor_tilts.get("Real Assets", 0)
    intl = factor_tilts.get("International", 0)
    cash = weights.get("CASH", 0)
    bonds = sum(w for t, w in weights.items() if meta.get(t, {}).get("sector", "") in ["Long Duration Bonds", "Core Bonds"])

    scores = {
        "Growth Exposure": min(100, int(growth * 200)),
        "Defensive Tilt": min(100, int((defensive + cash + bonds) * 150)),
        "Inflation Sensitivity": min(100, int(real_assets * 200 + 20)),
        "Rate Sensitivity": min(100, int((growth * 0.5 + bonds * 0.8) * 150)),
        "Risk-Off Resilience": min(100, int((defensive + cash + bonds + real_assets * 0.5) * 120)),
        "Concentration": min(100, int(max(weights.values()) * 300)) if weights else 0,
    }

    if growth > 0.5:
        label = "High Beta Tech"
    elif growth > 0.3 and defensive < 0.15:
        label = "Growth Heavy"
    elif defensive > 0.3 or (cash + bonds) > 0.3:
        label = "Defensive Income"
    elif real_assets > 0.2:
        label = "Inflation Hedge"
    elif intl > 0.2:
        label = "Global Diversified"
    elif abs(beta - 1.0) < 0.15 and ann_vol < 0.18:
        label = "Balanced Core"
    else:
        label = "Tactically Mixed"

    drivers = {}
    top_holdings_list = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:3]
    drivers["Growth Exposure"] = f"Driven by {', '.join(t for t, _ in top_holdings_list[:2])}"
    drivers["Defensive Tilt"] = f"{'Cash + bonds anchor' if cash + bonds > 0.15 else 'Sector tilt'}"
    drivers["Inflation Sensitivity"] = f"{'Real assets present' if real_assets > 0.1 else 'Limited real asset exposure'}"
    drivers["Rate Sensitivity"] = f"{'Duration/growth heavy' if scores['Rate Sensitivity'] > 60 else 'Moderate rate exposure'}"

    return {"label": label, "scores": scores, "drivers": drivers}


# ── Macro AI analysis (cached) ────────────────────────────────────────────────

_macro_cache: dict = {"data": None, "ts": 0.0}
_MACRO_TTL = 7200  # 2 hours


def run_macro_analysis() -> dict:
    global _macro_cache
    if _macro_cache["data"] and time.time() - _macro_cache["ts"] < _MACRO_TTL:
        return _macro_cache["data"]

    macro_cards_raw = get_macro_cards()
    news_items = get_news_feed()

    macro_text = "\n".join([f"{m['label']}: {m['value']}" for m in macro_cards_raw])
    news_text = "\n".join([f"[{n['source']}] {n['title']}" for n in news_items[:20]])

    prompt = f"""You are a chief macro strategist. Analyze current market conditions and return ONLY valid JSON — no markdown, no preamble.

MARKET DATA:
{macro_text}

NEWS HEADLINES:
{news_text}

Return this exact JSON structure:
{{
  "morning_brief": {{
    "headline": "One punchy sentence summarizing today",
    "paragraphs": ["paragraph 1 (3-4 sentences)", "paragraph 2 (3-4 sentences)", "paragraph 3 (3-4 sentences)"],
    "biggest_risk": "1 sentence",
    "most_sensitive_sector": "sector name and why in 1 sentence",
    "watchlist": [
      {{"event": "Event name", "why_now": "1 sentence", "bullish": "1 sentence", "bearish": "1 sentence", "assets": "key assets"}}
    ]
  }},
  "macro_regime": {{
    "regime": "Regime label e.g. Risk-Off / Inflationary / Growth-Negative",
    "regime_color": "red",
    "regime_summary": "2-3 sentences describing overall macro regime",
    "top_drivers": [
      {{"title": "Driver name", "what_changed": "1 sentence", "market_impact": "1 sentence", "winners": "assets/sectors", "losers": "assets/sectors"}}
    ],
    "cross_asset": [
      {{"signal": "Equities", "reading": "Bearish", "color": "red", "detail": "1 sentence"}}
    ]
  }},
  "market_impact": [
    {{
      "title": "Short event title",
      "why_it_matters": "2 sentences max",
      "sectors": [{{"name": "Energy", "direction": "bullish"}}],
      "overall": "bearish",
      "horizon": "near-term",
      "score": "8/10"
    }}
  ],
  "central_banks": [
    {{
      "bank": "Federal Reserve",
      "stance": "Hawkish Hold",
      "stance_color": "red",
      "latest_signals": "2 sentences",
      "market_interpretation": "2 sentences",
      "key_risk": "1 sentence"
    }}
  ],
  "sectors": [
    {{
      "sector": "Technology",
      "sentiment": "Cautiously Bullish",
      "sentiment_color": "green",
      "tailwinds": ["bullet point 1", "bullet point 2", "bullet point 3"],
      "headwinds": ["bullet point 1", "bullet point 2"],
      "key_catalyst": "1 sentence on most important catalyst"
    }}
  ],
  "sentiment": {{
    "bullish_themes": [{{"theme": "Theme title", "detail": "2 sentences", "assets": "affected assets"}}],
    "bearish_themes": [{{"theme": "Theme title", "detail": "2 sentences", "assets": "affected assets"}}],
    "rotation_signals": [{{"from": "asset/sector", "to": "asset/sector", "signal": "1 sentence explanation"}}]
  }}
}}

Requirements: 5 top_drivers, 6 cross_asset signals (Equities, Yields, Oil, Gold, USD, VIX), 6 market_impact cards, 5 central_banks (Fed ECB BoJ BoE PBOC in that order), 8 sectors (Technology Financials Energy Healthcare Industrials Consumer-Disc Real-Estate Utilities), 3 bullish_themes, 3 bearish_themes, 4 rotation_signals, 5 watchlist items. Colors must be exactly: red green or amber."""

    try:
        msg = _anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=6000,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        data = json.loads(text)
        _macro_cache["data"] = data
        _macro_cache["ts"] = time.time()
        return data
    except Exception as exc:
        return {"error": str(exc)}


# ── Flask routes ──────────────────────────────────────────────────────────────

@app.route("/api/search")
def search_tickers():
    q = request.args.get("q", "").strip()
    if len(q) < 2 or not FMP_API_KEY:
        return jsonify([])
    try:
        url = f"https://financialmodelingprep.com/api/v3/search?query={urllib.parse.quote(q)}&limit=10&apikey={FMP_API_KEY}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        return jsonify([{"symbol": d.get("symbol", ""), "name": d.get("name", "")} for d in data[:10]])
    except Exception:
        return jsonify([])


@app.route("/api/stress-test", methods=["POST"])
def stress_test():
    data = request.get_json(force=True)
    weights = data.get("weights", {})
    if not weights:
        return jsonify({"error": "No weights provided"}), 400

    STRESS_PERIODS = [
        {"name": "GFC 2008-09",       "start": "2008-09-12", "end": "2009-03-09"},
        {"name": "COVID Crash",        "start": "2020-02-19", "end": "2020-03-23"},
        {"name": "2022 Bear",          "start": "2022-01-03", "end": "2022-10-13"},
        {"name": "2023 Bull",          "start": "2023-01-01", "end": "2023-12-29"},
        {"name": "Rate Hike Cycle",    "start": "2022-03-15", "end": "2023-07-26"},
    ]

    tickers = [t for t in weights if t != "CASH"] + ["SPY"]
    tickers = list(set(tickers))

    try:
        all_data = yf.download(tickers, start="2008-01-01", auto_adjust=True, progress=False)["Close"]
        if isinstance(all_data, pd.Series):
            all_data = all_data.to_frame(name=tickers[0])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    results = []
    for period in STRESS_PERIODS:
        try:
            pdata = all_data.loc[period["start"]:period["end"]].dropna(how="all")
            if len(pdata) < 2:
                results.append({"name": period["name"], "portfolio_return": None, "spy_return": None, "outperformed": None})
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

            outperformed = (port_ret > spy_ret) if spy_ret is not None else None
            results.append({
                "name": period["name"],
                "portfolio_return": round(port_ret, 2),
                "spy_return": round(spy_ret, 2) if spy_ret is not None else None,
                "outperformed": outperformed,
            })
        except Exception:
            results.append({"name": period["name"], "portfolio_return": None, "spy_return": None, "outperformed": None})

    return jsonify({"periods": results})


@app.route("/chat", methods=["POST"])
def chat():
    if not OPENAI_API_KEY:
        return jsonify({"error": "AI chat unavailable — OPENAI_API_KEY not configured", "response": None})
    data = request.get_json(force=True)
    message = data.get("message", "").strip()
    context = data.get("context", {})
    if not message:
        return jsonify({"error": "No message provided", "response": None})

    ctx_str = json.dumps(context, indent=2) if context else "No macro context available."
    system_prompt = (
        "You are a macro and portfolio analyst assistant. Answer based on the provided market data. "
        "Be clear, practical, state uncertainty. Frame output as analysis and education, not financial advice. "
        "Include 'Bias/Limitation:' at end of complex interpretations.\n\n"
        f"Current market context:\n{ctx_str}"
    )

    try:
        client = _openai_module.OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message},
            ],
            max_tokens=600,
        )
        return jsonify({"response": resp.choices[0].message.content.strip(), "error": None})
    except Exception as e:
        return jsonify({"error": str(e), "response": None})


@app.route("/api/news-impact", methods=["POST"])
def news_impact():
    data = request.get_json(force=True)
    headline = data.get("headline", "").lower()
    portfolio_weights = data.get("weights", {})

    matched_rules = []
    sector_impacts: Dict[str, str] = {}
    asset_impacts: Dict[str, str] = {}

    for rule in IMPACT_RULES:
        if any(kw in headline for kw in rule["keywords"]):
            matched_rules.append({"keywords": rule["keywords"], "matched": [kw for kw in rule["keywords"] if kw in headline]})
            for sector, direction in rule["impacts"].items():
                if sector not in sector_impacts:
                    sector_impacts[sector] = direction
            for asset, direction in rule["assets"].items():
                if asset not in asset_impacts:
                    asset_impacts[asset] = direction

    portfolio_relevance = []
    for ticker in portfolio_weights:
        if ticker in asset_impacts:
            portfolio_relevance.append({
                "ticker": ticker,
                "weight": portfolio_weights[ticker],
                "direction": asset_impacts[ticker],
                "reason": f"Directly mentioned in matched rule for {ticker}",
            })

    pos = sum(1 for d in sector_impacts.values() if d == "positive")
    neg = sum(1 for d in sector_impacts.values() if d == "negative")
    if pos > neg:
        overall = "bullish"
    elif neg > pos:
        overall = "bearish"
    elif matched_rules:
        overall = "mixed"
    else:
        overall = "neutral"

    return jsonify({
        "matched_rules": matched_rules,
        "sector_impacts": sector_impacts,
        "portfolio_relevance": portfolio_relevance,
        "overall_sentiment": overall,
    })


@app.route("/api/geo-risk")
def geo_risk():
    return jsonify(get_geo_hotspots())


@app.route("/api/macro-analysis")
def macro_analysis_route():
    return jsonify(run_macro_analysis())


@app.route("/api/refresh-macro", methods=["POST"])
def refresh_macro():
    global _macro_cache
    _macro_cache["ts"] = 0.0
    data = run_macro_analysis()
    return jsonify({"ok": True, "cached_at": _macro_cache["ts"], "data": data})


TEMPLATE = r'''
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>Macro Dashboard • Portfolio Lab</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
:root {
  --bg:#070b11; --panel:#0e1520; --panel2:#131d2b; --panel3:#182536;
  --border:#203146; --text:#edf2f7; --muted:#98a8ba;
  --green:#3bd671; --red:#ff5f6d; --blue:#59a8ff; --amber:#ffb44d; --teal:#1dd1c1;
  --glow:0 0 0 1px rgba(89,168,255,.15),0 12px 28px rgba(0,0,0,.28);
}
*{box-sizing:border-box;}
html{scroll-behavior:smooth;}
body{
  margin:0;
  background:radial-gradient(circle at top right,rgba(89,168,255,.10),transparent 25%),
             radial-gradient(circle at top left,rgba(29,209,193,.08),transparent 25%),var(--bg);
  color:var(--text);
  font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
}
body.light-mode{
  --bg:#f0f4f8;--panel:#fff;--panel2:#e8edf5;--panel3:#dce5f0;
  --border:#c5d0e0;--text:#1a2a3a;--muted:#5a7090;
  --glow:0 0 0 1px rgba(0,0,0,.08),0 8px 20px rgba(0,0,0,.1);
}
body.light-mode .sidebar{background:linear-gradient(180deg,#e8edf5,#dce5f0);}
body.light-mode .card{background:linear-gradient(180deg,#fff,#f5f8fc);}
body.light-mode textarea,body.light-mode input,body.light-mode select{background:#e8edf5;color:#1a2a3a;}
body.light-mode .topbar{background:rgba(240,244,248,.97);}
.app{display:grid;grid-template-columns:260px 1fr;min-height:100vh;}
.sidebar{
  position:sticky;top:0;height:100vh;overflow-y:auto;
  background:linear-gradient(180deg,#09101a,#0b131e);
  border-right:1px solid var(--border);
  padding:20px 16px;
  display:flex;flex-direction:column;gap:0;
}
.brand{font-size:20px;font-weight:800;margin-bottom:4px;}
.brand-sub{color:var(--muted);font-size:12px;margin-bottom:16px;}
.regime-box{
  border:1px solid var(--border);background:rgba(89,168,255,.08);
  border-radius:14px;padding:12px 14px;margin-bottom:14px;box-shadow:var(--glow);
}
.regime-box small{display:block;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;font-size:10px;margin-bottom:5px;}
.regime-box strong{font-size:15px;line-height:1.3;}
.regime-box .rd{color:var(--muted);font-size:11px;margin-top:5px;line-height:1.4;}
.fg-box{border:1px solid var(--border);border-radius:14px;padding:12px 14px;margin-bottom:14px;text-align:center;}
.fg-box small{display:block;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;font-size:10px;margin-bottom:8px;}
.nav a{display:block;color:var(--muted);text-decoration:none;padding:9px 12px;border-radius:10px;margin-bottom:4px;transition:.15s;font-size:13px;}
.nav a:hover{background:var(--panel2);color:var(--text);}
.save-section{margin-top:auto;padding-top:14px;border-top:1px solid var(--border);}
.save-title{font-size:10px;font-weight:800;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);margin-bottom:8px;}
.saved-btn{width:100%;text-align:left;background:var(--panel2);border:1px solid var(--border);border-radius:8px;padding:7px 10px;color:var(--text);cursor:pointer;font-size:12px;margin-bottom:5px;}
.saved-btn:hover{background:var(--panel3);}
.content{padding:0 22px 40px;}
.topbar{
  position:sticky;top:0;z-index:50;
  background:rgba(7,11,17,.95);backdrop-filter:blur(14px);
  border-bottom:1px solid var(--border);margin:0 -22px 18px;
}
.topbar-inner{padding:14px 22px;display:flex;justify-content:space-between;align-items:center;gap:14px;}
.title-wrap h1{margin:0;font-size:28px;font-weight:800;}
.title-wrap .sub{color:var(--muted);margin-top:4px;font-size:13px;}
.topbar-right{display:flex;align-items:center;gap:10px;}
.theme-btn{background:var(--panel2);border:1px solid var(--border);border-radius:10px;padding:8px 12px;cursor:pointer;color:var(--text);font-size:16px;}
.ticker-wrap{overflow:hidden;border-top:1px solid var(--border);background:#09111b;}
.ticker-track{display:flex;width:max-content;animation:scroll 35s linear infinite;}
.ticker-item{display:inline-flex;gap:10px;align-items:center;padding:10px 18px;border-right:1px solid var(--border);font-size:13px;}
.ticker-item .nm{color:var(--muted);}
.ticker-item .vl{font-weight:700;}
.ticker-item .up{color:var(--green);}
.ticker-item .dn{color:var(--red);}
@keyframes scroll{from{transform:translateX(0);}to{transform:translateX(-50%);}}
.section-sep{display:flex;align-items:center;gap:14px;margin:32px 0 20px;}
.section-sep::before,.section-sep::after{content:'';flex:1;height:1px;background:var(--border);}
.section-sep-label{font-size:10px;font-weight:800;text-transform:uppercase;letter-spacing:.15em;color:var(--muted);white-space:nowrap;}
.grid-6{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin-bottom:18px;}
.grid-2{display:grid;grid-template-columns:1.1fr .9fr;gap:18px;margin-bottom:18px;}
.grid-3{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;margin-bottom:18px;}
.grid-4{display:grid;grid-template-columns:repeat(4,1fr);gap:18px;margin-bottom:18px;}
.grid-global{display:grid;grid-template-columns:repeat(auto-fit,minmax(100px,1fr));gap:10px;margin-bottom:18px;}
.card{
  background:linear-gradient(180deg,rgba(19,29,43,.94),rgba(14,21,32,.98));
  border:1px solid var(--border);border-radius:18px;box-shadow:var(--glow);overflow:hidden;
}
.card-header{
  padding:13px 18px;border-bottom:1px solid var(--border);
  background:linear-gradient(90deg,rgba(255,255,255,.02),rgba(89,168,255,.03));
  font-weight:800;display:flex;justify-content:space-between;gap:12px;align-items:center;font-size:14px;
}
.card-body{padding:16px 18px;}
.stat .lbl{color:var(--muted);text-transform:uppercase;letter-spacing:.09em;font-size:11px;margin-bottom:8px;}
.stat .val{font-size:36px;font-weight:800;margin-bottom:6px;}
.pill{padding:5px 9px;border-radius:999px;font-size:11px;font-weight:700;border:1px solid var(--border);}
.pill.up{background:rgba(59,214,113,.08);color:var(--green);}
.pill.dn{background:rgba(255,95,109,.08);color:var(--red);}
.pill.warn{background:rgba(255,180,77,.08);color:var(--amber);}
.pill.info{background:rgba(89,168,255,.08);color:var(--blue);}
.spark{height:52px!important;width:100%!important;}
.big-chart{height:300px!important;width:100%!important;}
.mid-chart{height:240px!important;width:100%!important;}
textarea,input,select{
  width:100%;background:#0b1320;color:var(--text);
  border:1px solid var(--border);border-radius:12px;padding:11px 13px;font:inherit;
}
textarea{min-height:140px;resize:vertical;}
.form-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:12px;}
.portfolio-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;}
.btn{
  background:linear-gradient(90deg,#1d74f5,#38a3ff);color:#fff;
  border:none;border-radius:12px;padding:11px 16px;font-weight:800;cursor:pointer;font-size:13px;
}
.btn.secondary{background:#0d1622;border:1px solid var(--border);color:var(--text);}
.btn.danger{background:rgba(255,95,109,.12);border:1px solid rgba(255,95,109,.25);color:var(--red);}
.muted{color:var(--muted);}
table{width:100%;border-collapse:collapse;}
th,td{padding:10px 10px;border-bottom:1px solid var(--border);text-align:left;font-size:13px;}
th{color:var(--muted);font-weight:700;font-size:11px;text-transform:uppercase;letter-spacing:.06em;}
tr:last-child td{border-bottom:none;}
.heatmap-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:9px;}
.heat{border-radius:12px;padding:11px;border:1px solid var(--border);background:rgba(89,168,255,.05);}
.heat .k{color:var(--muted);font-size:11px;margin-bottom:5px;}
.heat .v{font-size:18px;font-weight:800;}
.global-cell{border-radius:10px;padding:10px 12px;text-align:center;border:1px solid var(--border);}
.global-cell.pos{background:rgba(59,214,113,.10);}
.global-cell.neg{background:rgba(255,95,109,.10);}
.global-cell .gm{font-size:10px;color:var(--muted);margin-bottom:4px;}
.global-cell .gc{font-size:15px;font-weight:800;}
.global-cell .gc.pos{color:var(--green);}
.global-cell .gc.neg{color:var(--red);}
.cell-best{color:var(--green)!important;font-weight:700;}
.cell-worst{color:var(--red)!important;}
.compare-table td:first-child,.compare-table th:first-child{position:sticky;left:0;background:#101928;}
.scenario-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;}
.scenario-card{border:1px solid var(--border);border-radius:14px;padding:14px;background:rgba(255,255,255,.02);}
.blurb{
  margin-top:12px;padding:10px 13px;
  background:rgba(89,168,255,.05);border:1px solid rgba(89,168,255,.12);border-radius:10px;
  font-size:12px;color:var(--muted);line-height:1.5;
}
.risk-badge{
  display:inline-flex;align-items:center;gap:5px;
  padding:3px 9px;border-radius:999px;font-size:11px;font-weight:800;border:1px solid var(--border);
}
.risk-badge.high{background:rgba(255,95,109,.1);color:var(--red);}
.risk-badge.mid{background:rgba(255,180,77,.1);color:var(--amber);}
.risk-badge.low{background:rgba(59,214,113,.1);color:var(--green);}
.ai-box{
  margin-top:12px;padding:12px 14px;
  background:rgba(89,168,255,.06);border:1px solid rgba(89,168,255,.15);border-radius:12px;
}
.ai-box .ai-label{font-size:10px;font-weight:800;text-transform:uppercase;letter-spacing:.1em;color:var(--blue);margin-bottom:6px;}
.ai-box .ai-text{font-size:13px;line-height:1.6;color:#c8d8e8;}
.search-wrap{position:relative;margin-bottom:8px;}
.search-input{
  width:100%;background:#0b1320;color:var(--text);
  border:1px solid var(--border);border-radius:10px;padding:9px 12px;font:inherit;font-size:13px;
}
.search-dd{
  position:absolute;top:100%;left:0;right:0;z-index:300;
  background:#0e1520;border:1px solid var(--border);border-radius:10px;
  max-height:200px;overflow-y:auto;display:none;box-shadow:0 8px 24px rgba(0,0,0,.4);
}
.search-item{padding:10px 12px;cursor:pointer;font-size:13px;border-bottom:1px solid var(--border);}
.search-item:last-child{border-bottom:none;}
.search-item:hover{background:var(--panel2);}
.search-item .sym{font-weight:700;color:var(--blue);}
.search-item .nm{color:var(--muted);font-size:12px;margin-left:8px;}
.news-card{
  display:block;padding:13px 16px;border:1px solid var(--border);border-radius:12px;
  background:rgba(255,255,255,.02);text-decoration:none;color:var(--text);
  transition:.15s;margin-bottom:10px;
}
.news-card:hover{background:rgba(89,168,255,.06);border-color:rgba(89,168,255,.3);}
.news-title{font-size:14px;font-weight:600;margin-bottom:4px;line-height:1.4;}
.news-pub{font-size:11px;color:var(--muted);}
.corr-table td,.corr-table th{padding:8px;text-align:center;font-size:12px;border:none;}
.sector-rot td{font-size:12px;}
.footer{
  text-align:center;padding:28px 0 12px;
  border-top:1px solid var(--border);margin-top:20px;
}
.footer a{color:var(--blue);text-decoration:none;font-weight:700;}
.footer a:hover{text-decoration:underline;}
.small{font-size:12px;}
.good{color:var(--green);}
.bad{color:var(--red);}

/* Stress Test */
.stress-table td,.stress-table th{padding:9px 10px;font-size:13px;}
.stress-ret.pos{color:var(--green);font-weight:700;}
.stress-ret.neg{color:var(--red);font-weight:700;}
.stress-vs.out{color:var(--green);}
.stress-vs.under{color:var(--red);}

/* Portfolio DNA */
.dna-label-badge{
  display:inline-block;padding:7px 18px;border-radius:999px;
  background:linear-gradient(90deg,rgba(89,168,255,.15),rgba(29,209,193,.12));
  border:1px solid rgba(89,168,255,.3);color:var(--blue);
  font-size:15px;font-weight:800;letter-spacing:.04em;margin-bottom:16px;
}
.dna-dim{margin-bottom:12px;}
.dna-dim-label{display:flex;justify-content:space-between;font-size:12px;margin-bottom:4px;}
.dna-bar-bg{height:8px;border-radius:999px;background:rgba(255,255,255,.08);overflow:hidden;}
.dna-bar-fill{height:100%;border-radius:999px;background:linear-gradient(90deg,#59a8ff,#1dd1c1);}
.dna-driver{font-size:11px;color:var(--muted);margin-top:3px;}

/* Rebalance Sandbox */
.sandbox-card{border:1px dashed var(--border);border-radius:18px;padding:28px;text-align:center;background:rgba(255,255,255,.01);}
.sandbox-icon{font-size:36px;margin-bottom:10px;}
.sandbox-title{font-size:15px;font-weight:800;margin-bottom:6px;}
.sandbox-sub{font-size:13px;color:var(--muted);}

/* News Impact Modal */
.impact-panel{
  margin-top:10px;padding:14px;border-radius:12px;
  background:rgba(89,168,255,.05);border:1px solid rgba(89,168,255,.15);
  display:none;
}
.impact-badge{
  display:inline-block;padding:4px 12px;border-radius:999px;font-size:12px;font-weight:700;margin-bottom:10px;
}
.impact-badge.bullish{background:rgba(59,214,113,.12);color:var(--green);border:1px solid rgba(59,214,113,.3);}
.impact-badge.bearish{background:rgba(255,95,109,.12);color:var(--red);border:1px solid rgba(255,95,109,.3);}
.impact-badge.mixed{background:rgba(255,180,77,.12);color:var(--amber);border:1px solid rgba(255,180,77,.3);}
.impact-badge.neutral{background:rgba(255,255,255,.06);color:var(--muted);border:1px solid var(--border);}
.impact-sector-tag{
  display:inline-block;padding:3px 8px;border-radius:6px;font-size:11px;font-weight:700;margin:2px;
}
.impact-sector-tag.pos{background:rgba(59,214,113,.1);color:var(--green);}
.impact-sector-tag.neg{background:rgba(255,95,109,.1);color:var(--red);}
.impact-sector-tag.mixed{background:rgba(255,180,77,.1);color:var(--amber);}
.news-card-wrap{margin-bottom:10px;}
.analyze-btn{
  font-size:11px;padding:3px 9px;border-radius:7px;cursor:pointer;
  background:rgba(89,168,255,.1);border:1px solid rgba(89,168,255,.2);color:var(--blue);
  margin-left:8px;vertical-align:middle;
}
.analyze-btn:hover{background:rgba(89,168,255,.2);}

/* Geo Risk Map */
#geo-map{height:500px;border-radius:14px;overflow:hidden;border:1px solid var(--border);}
.geo-filter-bar{display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap;}
.geo-filter-btn{
  padding:6px 14px;border-radius:999px;font-size:12px;font-weight:700;cursor:pointer;
  background:var(--panel2);border:1px solid var(--border);color:var(--muted);transition:.15s;
}
.geo-filter-btn.active,.geo-filter-btn:hover{background:rgba(89,168,255,.12);color:var(--blue);border-color:rgba(89,168,255,.3);}
#geo-detail{
  margin-top:16px;padding:18px;border-radius:14px;
  background:rgba(89,168,255,.04);border:1px solid rgba(89,168,255,.12);
  display:none;
}
.geo-risk-badge{
  display:inline-block;padding:4px 12px;border-radius:999px;font-size:12px;font-weight:700;margin-left:8px;
}
.geo-risk-badge.HIGH{background:rgba(239,68,68,.12);color:#ef4444;border:1px solid rgba(239,68,68,.3);}
.geo-risk-badge.ELEVATED{background:rgba(249,115,22,.12);color:#f97316;border:1px solid rgba(249,115,22,.3);}
.geo-risk-badge.WATCH{background:rgba(234,179,8,.12);color:#eab308;border:1px solid rgba(234,179,8,.3);}
.geo-risk-badge.LOW{background:rgba(59,214,113,.12);color:var(--green);border:1px solid rgba(59,214,113,.3);}
.asset-tag{
  display:inline-block;padding:3px 8px;border-radius:6px;font-size:11px;font-weight:700;margin:2px;
  background:rgba(89,168,255,.1);color:var(--blue);border:1px solid rgba(89,168,255,.2);
}
.asset-tag.port-match{background:rgba(59,214,113,.12);color:var(--green);border-color:rgba(59,214,113,.3);}

/* AI Chat Panel */
#chat-float-btn{
  position:fixed;bottom:28px;right:28px;z-index:9000;
  background:linear-gradient(135deg,#1d74f5,#38a3ff);color:#fff;
  border:none;border-radius:999px;padding:13px 20px;
  font-size:14px;font-weight:800;cursor:pointer;box-shadow:0 4px 24px rgba(29,116,245,.4);
  transition:.2s;display:flex;align-items:center;gap:8px;
}
#chat-float-btn:hover{transform:translateY(-2px);box-shadow:0 8px 32px rgba(29,116,245,.5);}
#chat-panel{
  position:fixed;top:0;right:-420px;width:400px;height:100vh;z-index:9001;
  background:#0b1320;border-left:1px solid var(--border);
  display:flex;flex-direction:column;transition:right .3s ease;
  box-shadow:-8px 0 32px rgba(0,0,0,.4);
}
#chat-panel.open{right:0;}
#chat-header{
  padding:16px 18px;border-bottom:1px solid var(--border);
  display:flex;justify-content:space-between;align-items:center;
  background:linear-gradient(90deg,rgba(29,116,245,.1),transparent);
}
#chat-header h3{margin:0;font-size:16px;font-weight:800;}
#chat-close{background:none;border:none;color:var(--muted);font-size:22px;cursor:pointer;padding:4px;}
#chat-close:hover{color:var(--text);}
#chat-messages{flex:1;overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:10px;}
.chat-msg{max-width:85%;padding:10px 13px;border-radius:14px;font-size:13px;line-height:1.5;}
.chat-msg.user{align-self:flex-end;background:rgba(29,116,245,.25);border:1px solid rgba(29,116,245,.3);color:var(--text);}
.chat-msg.ai{align-self:flex-start;background:rgba(255,255,255,.05);border:1px solid var(--border);color:#c8d8e8;}
.chat-msg.error{align-self:flex-start;background:rgba(255,95,109,.08);border:1px solid rgba(255,95,109,.2);color:var(--red);}
.chat-quick-btns{padding:10px 16px;display:flex;flex-wrap:wrap;gap:6px;border-top:1px solid var(--border);}
.quick-btn{
  font-size:11px;padding:5px 10px;border-radius:8px;cursor:pointer;
  background:rgba(89,168,255,.08);border:1px solid rgba(89,168,255,.15);color:var(--blue);
  transition:.15s;
}
.quick-btn:hover{background:rgba(89,168,255,.18);}
#chat-input-row{padding:12px 16px;border-top:1px solid var(--border);display:flex;gap:8px;}
#chat-input{flex:1;background:#0e1928;border:1px solid var(--border);border-radius:10px;padding:10px 12px;color:var(--text);font:inherit;font-size:13px;resize:none;}
#chat-send{background:linear-gradient(90deg,#1d74f5,#38a3ff);color:#fff;border:none;border-radius:10px;padding:10px 14px;cursor:pointer;font-weight:700;font-size:13px;}
#chat-send:disabled{opacity:.5;cursor:not-allowed;}
#chat-loading{padding:8px 16px;font-size:12px;color:var(--muted);display:none;}

/* Macro AI sections */
.macro-loading{text-align:center;padding:48px;color:var(--muted);}
.macro-loading .spin{display:inline-block;width:32px;height:32px;border:3px solid var(--border);border-top-color:var(--blue);border-radius:50%;animation:spin .8s linear infinite;margin-bottom:12px;}
@keyframes spin{to{transform:rotate(360deg);}}
.regime-hero{
  border-radius:18px;padding:22px 24px;margin-bottom:18px;
  border:1px solid var(--border);
  background:linear-gradient(135deg,rgba(19,29,43,.97),rgba(14,21,32,.99));
}
.regime-hero .r-label{font-size:10px;font-weight:800;text-transform:uppercase;letter-spacing:.12em;color:var(--muted);margin-bottom:8px;}
.regime-hero .r-name{font-size:24px;font-weight:800;margin-bottom:8px;}
.regime-hero .r-sum{font-size:13px;color:#c8d8e8;line-height:1.6;max-width:700px;}
.regime-hero.red{border-color:rgba(255,95,109,.25);background:linear-gradient(135deg,rgba(255,95,109,.06),rgba(14,21,32,.99));}
.regime-hero.green{border-color:rgba(59,214,113,.25);background:linear-gradient(135deg,rgba(59,214,113,.06),rgba(14,21,32,.99));}
.regime-hero.amber{border-color:rgba(255,180,77,.25);background:linear-gradient(135deg,rgba(255,180,77,.06),rgba(14,21,32,.99));}
.cross-asset-grid{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin-bottom:18px;}
.ca-cell{border:1px solid var(--border);border-radius:12px;padding:12px;text-align:center;background:rgba(255,255,255,.02);}
.ca-cell .ca-sig{font-size:11px;color:var(--muted);margin-bottom:6px;}
.ca-cell .ca-read{font-size:14px;font-weight:800;margin-bottom:4px;}
.ca-cell .ca-det{font-size:10px;color:var(--muted);line-height:1.3;}
.ca-cell.red .ca-read{color:var(--red);}
.ca-cell.green .ca-read{color:var(--green);}
.ca-cell.amber .ca-read{color:var(--amber);}
.driver-item{border:1px solid var(--border);border-radius:12px;padding:14px;background:rgba(255,255,255,.02);margin-bottom:10px;}
.driver-item .d-title{font-weight:800;font-size:14px;margin-bottom:6px;}
.driver-item .d-change{font-size:12px;color:var(--muted);margin-bottom:6px;}
.driver-item .d-impact{font-size:12px;margin-bottom:8px;}
.driver-wl{display:flex;gap:8px;flex-wrap:wrap;font-size:11px;}
.driver-wl span{padding:2px 7px;border-radius:5px;}
.driver-wl .win{background:rgba(59,214,113,.1);color:var(--green);}
.driver-wl .lose{background:rgba(255,95,109,.1);color:var(--red);}
.impact-card{border:1px solid var(--border);border-radius:14px;padding:16px;background:rgba(255,255,255,.02);}
.impact-card .ic-title{font-weight:800;font-size:14px;margin-bottom:6px;}
.impact-card .ic-why{font-size:12px;color:var(--muted);line-height:1.5;margin-bottom:10px;}
.impact-card .ic-meta{display:flex;gap:8px;align-items:center;flex-wrap:wrap;font-size:11px;margin-bottom:8px;}
.ic-overall{padding:3px 9px;border-radius:999px;font-weight:700;}
.ic-overall.bullish{background:rgba(59,214,113,.12);color:var(--green);border:1px solid rgba(59,214,113,.3);}
.ic-overall.bearish{background:rgba(255,95,109,.12);color:var(--red);border:1px solid rgba(255,95,109,.3);}
.ic-overall.neutral{background:rgba(255,255,255,.06);color:var(--muted);border:1px solid var(--border);}
.ic-tags{display:flex;gap:5px;flex-wrap:wrap;}
.ic-tag{padding:2px 7px;border-radius:5px;font-size:10px;font-weight:700;}
.ic-tag.bullish{background:rgba(59,214,113,.1);color:var(--green);}
.ic-tag.bearish{background:rgba(255,95,109,.1);color:var(--red);}
.cb-row{display:grid;grid-template-columns:160px 130px 1fr;gap:12px;align-items:start;padding:14px;border:1px solid var(--border);border-radius:12px;background:rgba(255,255,255,.02);margin-bottom:10px;}
.cb-row .cb-bank{font-weight:800;font-size:14px;}
.cb-stance{padding:4px 10px;border-radius:999px;font-size:12px;font-weight:700;text-align:center;}
.cb-stance.red{background:rgba(255,95,109,.1);color:var(--red);border:1px solid rgba(255,95,109,.3);}
.cb-stance.green{background:rgba(59,214,113,.1);color:var(--green);border:1px solid rgba(59,214,113,.3);}
.cb-stance.amber{background:rgba(255,180,77,.1);color:var(--amber);border:1px solid rgba(255,180,77,.3);}
.cb-detail{font-size:12px;color:#c8d8e8;line-height:1.5;}
.cb-risk{font-size:11px;color:var(--amber);margin-top:4px;}
.sector-ai-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:14px;}
.sector-ai-card{border:1px solid var(--border);border-radius:14px;padding:16px;background:rgba(255,255,255,.02);}
.sector-ai-card .s-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;}
.sector-ai-card .s-name{font-weight:800;font-size:14px;}
.s-sent{padding:3px 9px;border-radius:999px;font-size:11px;font-weight:700;}
.s-sent.green{background:rgba(59,214,113,.12);color:var(--green);border:1px solid rgba(59,214,113,.3);}
.s-sent.red{background:rgba(255,95,109,.12);color:var(--red);border:1px solid rgba(255,95,109,.3);}
.s-sent.amber{background:rgba(255,180,77,.12);color:var(--amber);border:1px solid rgba(255,180,77,.3);}
.bullet-list{margin:0;padding:0 0 0 14px;font-size:12px;color:var(--muted);line-height:1.6;}
.bullet-list li{margin-bottom:2px;}
.s-catalyst{font-size:11px;color:var(--blue);margin-top:8px;padding:6px 10px;background:rgba(89,168,255,.06);border-radius:8px;}
.brief-hero{border:1px solid rgba(89,168,255,.2);border-radius:18px;padding:22px 24px;margin-bottom:18px;background:linear-gradient(135deg,rgba(89,168,255,.06),rgba(14,21,32,.99));}
.brief-hero .b-headline{font-size:18px;font-weight:800;margin-bottom:14px;line-height:1.4;}
.brief-hero .b-para{font-size:13px;color:#c8d8e8;line-height:1.7;margin-bottom:10px;}
.brief-meta{display:flex;gap:14px;flex-wrap:wrap;margin-top:14px;padding-top:14px;border-top:1px solid var(--border);}
.brief-meta-item{font-size:12px;}
.brief-meta-item .bm-lbl{color:var(--muted);text-transform:uppercase;font-size:10px;letter-spacing:.08em;margin-bottom:4px;}
.watchlist-table td,.watchlist-table th{font-size:12px;padding:9px 10px;}
.theme-col{display:flex;flex-direction:column;gap:10px;}
.theme-item{border:1px solid var(--border);border-radius:12px;padding:14px;}
.theme-item .t-title{font-weight:800;font-size:13px;margin-bottom:6px;}
.theme-item .t-detail{font-size:12px;color:var(--muted);line-height:1.5;margin-bottom:6px;}
.theme-item .t-assets{font-size:11px;color:var(--blue);}
.theme-bull .t-title{color:var(--green);}
.theme-bear .t-title{color:var(--red);}
.rotation-item{display:flex;align-items:center;gap:10px;padding:10px 14px;border:1px solid var(--border);border-radius:10px;margin-bottom:8px;font-size:12px;}
.rotation-item .r-from{color:var(--red);font-weight:700;}
.rotation-item .r-arrow{color:var(--muted);}
.rotation-item .r-to{color:var(--green);font-weight:700;}
.rotation-item .r-sig{color:var(--muted);flex:1;text-align:right;}
.refresh-bar{display:flex;align-items:center;gap:12px;margin-bottom:16px;}
.refresh-status{font-size:12px;color:var(--muted);}

@media print{
  .sidebar,.topbar,.btn,.search-wrap,.save-section,.ticker-wrap,
  #chat-float-btn,#chat-panel{display:none!important;}
  .app{grid-template-columns:1fr;}
  .content{padding:0;}
  .card{break-inside:avoid;}
}
@media(max-width:1300px){
  .app{grid-template-columns:1fr;}
  .sidebar{position:relative;height:auto;}
  .grid-6{grid-template-columns:repeat(3,1fr);}
  .grid-2,.grid-3,.grid-4,.portfolio-grid,.scenario-grid{grid-template-columns:1fr;}
  .form-grid{grid-template-columns:repeat(2,1fr);}
  #chat-panel{width:100%;right:-105%;}
}
</style>
</head>
<body>
<div class="app">
<aside class="sidebar">
  <div class="brand">Macro Terminal</div>
  <div class="brand-sub">Portfolio Lab • Built by Jake Joseph</div>
  <div class="regime-box">
    <small>Market Regime</small>
    <strong>{{ regime_label }}</strong>
    <div class="rd">{{ regime_detail }}</div>
  </div>
  <div class="fg-box">
    <small>Fear &amp; Greed</small>
    <svg viewBox="0 0 200 110" width="160" height="88" style="display:block;margin:0 auto 6px;">
      <defs>
        <linearGradient id="arcGrad" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" style="stop-color:#ff5f6d"/>
          <stop offset="50%" style="stop-color:#ffb44d"/>
          <stop offset="100%" style="stop-color:#3bd671"/>
        </linearGradient>
      </defs>
      <path d="M 10 100 A 90 90 0 0 1 190 100" fill="none" stroke="#203146" stroke-width="14" stroke-linecap="round"/>
      <path d="M 10 100 A 90 90 0 0 1 190 100" fill="none" stroke="url(#arcGrad)" stroke-width="14" stroke-linecap="round"
            stroke-dasharray="{{ fear_greed.dash }} {{ fear_greed.total_arc }}"/>
      <line x1="100" y1="100" x2="{{ fear_greed.nx }}" y2="{{ fear_greed.ny }}" stroke="white" stroke-width="2.5" stroke-linecap="round"/>
      <circle cx="100" cy="100" r="5" fill="white"/>
    </svg>
    <div style="font-size:18px;font-weight:800;color:{{ fear_greed.color }};">{{ fear_greed.score }}</div>
    <div style="font-size:12px;color:{{ fear_greed.color }};">{{ fear_greed.label }}</div>
  </div>
  <nav class="nav">
    <a href="#overview">&#x1F4CA; Overview</a>
    <a href="#global">&#x1F30D; Global Markets</a>
    <a href="#portfolios">&#x1F9EA; Portfolio Lab</a>
    <a href="#performance">&#x1F4C8; Performance</a>
    <a href="#comparison">&#x1F50D; Comparison</a>
    <a href="#sectors">&#x1F3DB; Sectors</a>
    <a href="#scenarios">&#x1F3AF; Scenarios</a>
    <a href="#news">&#x1F4F0; News Feed</a>
    <a href="#news-impact">&#x1F4CA; News Impact</a>
    <a href="#geo-risk">&#x1F5FA; Geo Risk Radar</a>
    <div style="height:1px;background:var(--border);margin:8px 0 6px;"></div>
    <div style="font-size:9px;font-weight:800;text-transform:uppercase;letter-spacing:.12em;color:var(--muted);padding:0 12px;margin-bottom:4px;">AI Macro Intel</div>
    <a href="#morning-brief">&#x1F305; Morning Brief</a>
    <a href="#macro-regime">&#x1F9E0; Macro Regime</a>
    <a href="#market-impact">&#x26A1; Market Impact</a>
    <a href="#central-banks">&#x1F3E6; Central Banks</a>
    <a href="#ai-sectors">&#x1F4C9; Sector Outlook</a>
    <a href="#ai-sentiment">&#x1F4AC; Sentiment</a>
  </nav>
  <div class="save-section">
    <div class="save-title">Saved Portfolios</div>
    <div id="saved-list"></div>
    <input id="save-name" placeholder="Name this set..." style="margin-bottom:8px;font-size:12px;padding:8px 10px;"/>
    <button onclick="savePortfolio()" class="btn" style="width:100%;padding:8px;font-size:12px;">Save Current</button>
  </div>
</aside>

<main class="content">
  <div class="topbar">
    <div class="topbar-inner">
      <div class="title-wrap">
        <h1>Macro Dashboard &bull; Portfolio Lab</h1>
        <div class="sub">Live macro tape &bull; portfolio construction &bull; benchmark analytics &bull; AI commentary</div>
      </div>
      <div class="topbar-right">
        <span class="muted small">{{ updated_at }}</span>
        <button class="theme-btn" id="theme-toggle" title="Toggle light/dark mode">&#9790;</button>
        <button class="btn secondary" onclick="sharePortfolio()" style="font-size:12px;padding:8px 12px;">&#128279; Share</button>
        <button class="btn secondary" onclick="window.print()" style="font-size:12px;padding:8px 12px;">&#128438; Print</button>
      </div>
    </div>
    <div class="ticker-wrap">
      <div class="ticker-track">
        {% for item in ticker_items %}
        <div class="ticker-item">
          <span class="nm">{{ item.label }}</span>
          <span class="vl">{{ item.display }}</span>
          <span class="{{ 'up' if item.change >= 0 else 'dn' }}">{{ '%+.2f%%'|format(item.change) }}</span>
        </div>
        {% endfor %}
        {% for item in ticker_items %}
        <div class="ticker-item">
          <span class="nm">{{ item.label }}</span>
          <span class="vl">{{ item.display }}</span>
          <span class="{{ 'up' if item.change >= 0 else 'dn' }}">{{ '%+.2f%%'|format(item.change) }}</span>
        </div>
        {% endfor %}
      </div>
    </div>
  </div>

  <!-- OVERVIEW -->
  <div class="section-sep"><span class="section-sep-label">Live Market Overview</span></div>
  <section id="overview">
    <div class="grid-6">
      {% for card in macro_cards %}
      <div class="card stat">
        <div class="card-body">
          <div class="lbl">{{ card.label }}</div>
          <div class="val" style="font-size:26px;">{{ card.display }}</div>
          <span class="pill {{ 'up' if card.change >= 0 else 'dn' }}">{{ '%+.2f%%'|format(card.change) }}</span>
          <canvas id="spark_{{ loop.index0 }}" class="spark" style="margin-top:8px;"></canvas>
        </div>
      </div>
      {% endfor %}
    </div>
  </section>

  <!-- GLOBAL MARKETS -->
  <div class="section-sep"><span class="section-sep-label">Global Markets</span></div>
  <section id="global">
    <div class="grid-global">
      {% for g in global_heatmap %}
      <div class="global-cell {{ 'pos' if g.change >= 0 else 'neg' }}">
        <div class="gm">{{ g.label }}</div>
        <div class="gc {{ 'pos' if g.change >= 0 else 'neg' }}">{{ '%+.2f%%'|format(g.change) }}</div>
      </div>
      {% endfor %}
    </div>
  </section>

  <!-- PORTFOLIO LAB -->
  <div class="section-sep"><span class="section-sep-label">Portfolio Lab</span></div>
  <section id="portfolios" class="card" style="margin-bottom:20px;">
    <div class="card-header">Portfolio Lab Input <span class="pill info">Percent or Dollar Based</span></div>
    <div class="card-body">
      <form method="post" id="main-form">
        <div class="form-grid">
          <div>
            <label class="muted small">Start Date</label>
            <input name="start_date" value="{{ form.start_date }}"/>
          </div>
          <div>
            <label class="muted small">Risk Free Rate</label>
            <input name="risk_free_rate" value="{{ form.risk_free_rate }}"/>
          </div>
          <div>
            <label class="muted small">Projection Value ($)</label>
            <input name="projection_value" value="{{ form.projection_value }}"/>
          </div>
          <div>
            <label class="muted small">Custom Benchmark Label</label>
            <input name="custom_benchmark_name" value="{{ form.custom_benchmark_name }}"/>
          </div>
        </div>
        <div class="portfolio-grid">
          {% for label, fname in [('Portfolio A','portfolio_a'),('Portfolio B','portfolio_b'),('Portfolio C','portfolio_c')] %}
          <div>
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
              <label class="muted small">{{ label }}</label>
              <button type="button" class="btn danger" onclick="clearPortfolio('{{ fname }}')" style="padding:3px 9px;font-size:11px;">Clear</button>
            </div>
            <div class="search-wrap">
              <input class="search-input" id="search_{{ loop.index0 }}" placeholder="Search ticker to add (e.g. Apple)..." autocomplete="off"/>
              <div class="search-dd" id="search_{{ loop.index0 }}_dd"></div>
            </div>
            <textarea name="{{ fname }}">{{ form[fname] }}</textarea>
          </div>
          {% endfor %}
        </div>
        <div style="margin-top:12px;">
          <label class="muted small">Custom Benchmark Holdings</label>
          <textarea name="custom_benchmark" style="min-height:80px;">{{ form.custom_benchmark }}</textarea>
        </div>
        <div style="display:flex;gap:10px;margin-top:14px;flex-wrap:wrap;">
          <button class="btn" type="submit">&#9654; Run Portfolio Lab <span class="muted" style="font-weight:400;font-size:11px;margin-left:6px;">[R]</span></button>
          <button class="btn secondary" type="button" onclick="window.print()">&#128438; Export PDF <span class="muted" style="font-weight:400;font-size:11px;margin-left:6px;">[P]</span></button>
        </div>
      </form>
    </div>
  </section>

  {% if error %}
  <div class="card" style="margin-bottom:18px;border-color:#5a2226;">
    <div class="card-header">Error</div>
    <div class="card-body bad">{{ error }}</div>
  </div>
  {% endif %}

  {% if portfolios %}
  <!-- PORTFOLIO CARDS -->
  <div class="grid-3">
    {% for name, p in portfolios.items() %}
    <div class="card">
      <div class="card-header">
        {{ name }}
        <div style="display:flex;gap:6px;align-items:center;">
          <span class="risk-badge {{ 'high' if p.risk_score <= 4 else ('low' if p.risk_score >= 7 else 'mid') }}">Score {{ p.risk_score }}/10</span>
          <span class="pill warn">{{ p.concentration.label }} conc.</span>
        </div>
      </div>
      <div class="card-body">
        <div class="blurb">{{ p.blurb }}</div>
        <div class="heatmap-grid" style="margin:12px 0;">
          <div class="heat"><div class="k">Annual Return</div><div class="v">{{ p.metrics.annual_return }}</div></div>
          <div class="heat"><div class="k">Volatility</div><div class="v">{{ p.metrics.volatility }}</div></div>
          <div class="heat"><div class="k">Sharpe</div><div class="v">{{ p.metrics.sharpe }}</div></div>
          <div class="heat"><div class="k">Max Drawdown</div><div class="v bad">{{ p.metrics.max_drawdown }}</div></div>
        </div>
        <div class="muted small" style="margin-bottom:6px;">Top holdings</div>
        <table>
          <tbody>
            {% for h in p.top_holdings %}
            <tr><td style="font-weight:700;">{{ h.ticker }}</td><td class="muted">{{ h.name }}</td><td>{{ h.weight }}</td></tr>
            {% endfor %}
          </tbody>
        </table>
        {% if p.ai_commentary %}
        <div class="ai-box">
          <div class="ai-label">&#x2728; AI Commentary</div>
          <div class="ai-text">{{ p.ai_commentary }}</div>
        </div>
        {% endif %}
        {% if p.dna %}
        <div style="margin-top:12px;padding:14px;border:1px solid rgba(89,168,255,.15);border-radius:14px;background:rgba(89,168,255,.03);">
          <div class="muted small" style="text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px;">&#x1F9EC; Portfolio DNA</div>
          <div class="dna-label-badge">{{ p.dna.label }}</div>
          {% for dim, score in p.dna.scores.items() %}
          <div class="dna-dim">
            <div class="dna-dim-label">
              <span>{{ dim }}</span>
              <span style="font-weight:700;">{{ score }}</span>
            </div>
            <div class="dna-bar-bg"><div class="dna-bar-fill" style="width:{{ score }}%;"></div></div>
            {% if p.dna.drivers.get(dim) %}
            <div class="dna-driver">{{ p.dna.drivers[dim] }}</div>
            {% endif %}
          </div>
          {% endfor %}
        </div>
        {% endif %}
      </div>
    </div>
    {% endfor %}
  </div>

  <!-- STRESS TEST -->
  <div class="card" style="margin-bottom:20px;">
    <div class="card-header">Portfolio Stress Test <span class="muted small">Historical period analysis vs SPY</span></div>
    <div class="card-body">
      <div style="margin-bottom:12px;">
        <button class="btn" id="run-stress-btn" onclick="runStressTest()">&#x26A1; Run Stress Test (Portfolio A)</button>
        <span id="stress-loading" style="display:none;margin-left:12px;color:var(--muted);font-size:13px;">Loading...</span>
      </div>
      <div id="stress-results" style="display:none;">
        <table class="stress-table">
          <thead><tr><th>Period</th><th>Portfolio Return</th><th>SPY Return</th><th>vs SPY</th></tr></thead>
          <tbody id="stress-tbody"></tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- REBALANCE SANDBOX -->
  <div class="card" id="rebalance-sandbox" style="margin-bottom:20px;">
    <div class="card-body sandbox-card">
      <div class="sandbox-icon">&#x1F6A7;</div>
      <div class="sandbox-title">Rebalance Sandbox</div>
      <div class="sandbox-sub">Coming soon — drag sliders to rebalance and see live metric changes</div>
    </div>
  </div>

  <!-- PERFORMANCE -->
  <div class="section-sep"><span class="section-sep-label">Performance</span></div>
  <section id="performance">
    <div class="grid-2">
      <div class="card">
        <div class="card-header">Cumulative Return vs Benchmarks</div>
        <div class="card-body"><canvas id="perfChart" class="big-chart"></canvas></div>
      </div>
      <div class="card">
        <div class="card-header">Drawdown Analysis</div>
        <div class="card-body"><canvas id="ddChart" class="big-chart"></canvas></div>
      </div>
    </div>
  </section>

  <!-- ALLOCATION & SECTORS -->
  <div class="section-sep"><span class="section-sep-label">Allocation &amp; Sectors</span></div>
  <section id="sectors">
    <div class="grid-2">
      <div class="card">
        <div class="card-header">Allocation Breakdown &bull; Portfolio A</div>
        <div class="card-body"><canvas id="allocA" class="mid-chart"></canvas></div>
      </div>
      <div class="card">
        <div class="card-header">Sector Exposure &bull; Portfolio A</div>
        <div class="card-body">
          <div class="heatmap-grid">
            {% for k, v in portfolios['Portfolio A'].sector_exposure.items() %}
            <div class="heat"><div class="k">{{ k }}</div><div class="v">{{ v }}</div></div>
            {% endfor %}
          </div>
        </div>
      </div>
    </div>

    <!-- SECTOR ROTATION -->
    {% if sector_rotation %}
    <div class="card" style="margin-bottom:18px;">
      <div class="card-header">Sector Rotation Tracker</div>
      <div class="card-body" style="overflow-x:auto;">
        <table class="sector-rot">
          <thead>
            <tr><th>Sector</th><th>1 Week</th><th>1 Month</th><th>3 Month</th></tr>
          </thead>
          <tbody>
            {% for row in sector_rotation %}
            <tr>
              <td style="font-weight:700;">{{ row.sector }}</td>
              <td class="{{ 'good' if row.w1 >= 0 else 'bad' }}">{{ '%+.2f%%'|format(row.w1) }}</td>
              <td class="{{ 'good' if row.m1 >= 0 else 'bad' }}">{{ '%+.2f%%'|format(row.m1) }}</td>
              <td class="{{ 'good' if row.m3 >= 0 else 'bad' }}">{{ '%+.2f%%'|format(row.m3) }}</td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
    </div>
    {% endif %}
  </section>

  <!-- COMPARISON -->
  <div class="section-sep"><span class="section-sep-label">Benchmark Comparison</span></div>
  <section id="comparison">
    <div class="card" style="margin-bottom:18px;">
      <div class="card-header">Portfolio vs Benchmark — Full Comparison <span class="muted small">Green = best &bull; Red = worst</span></div>
      <div class="card-body" style="overflow-x:auto;">
        <table class="compare-table">
          <thead>
            <tr>
              <th>Metric</th>
              {% for col in comparison_headers %}<th>{{ col }}</th>{% endfor %}
            </tr>
          </thead>
          <tbody>
            {% for row in colored_rows %}
            <tr>
              <td style="font-weight:700;">{{ row.metric }}</td>
              {% for cell in row.colored %}<td class="{{ cell.cls }}">{{ cell.value }}</td>{% endfor %}
            </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
    </div>

    <!-- CORRELATION MATRIX -->
    <div class="grid-2">
      <div class="card">
        <div class="card-header">Correlation Matrix</div>
        <div class="card-body" id="corrMatrixWrap"></div>
      </div>
      <div class="card">
        <div class="card-header">Rolling Correlation vs Benchmarks &bull; Portfolio A</div>
        <div class="card-body"><canvas id="rollingCorrChart" class="mid-chart"></canvas></div>
      </div>
    </div>
  </section>

  <!-- SCENARIOS -->
  <div class="section-sep"><span class="section-sep-label">Scenario Analysis</span></div>
  <section id="scenarios">
    <div class="grid-2">
      <div class="card">
        <div class="card-header">Monte Carlo Projection &bull; Portfolio A (3 Years)</div>
        <div class="card-body">
          <div class="scenario-grid" style="margin-bottom:16px;">
            {% for sname, block in projections.items() %}
            <div class="scenario-card">
              <div class="pill {{ 'up' if sname == 'Bull' else ('dn' if sname == 'Bear' else 'info') }}" style="margin-bottom:10px;">{{ sname }}</div>
              <div><strong>Median:</strong> {{ block.median }}</div>
              <div><strong>10th pct:</strong> {{ block.p10 }}</div>
              <div><strong>90th pct:</strong> {{ block.p90 }}</div>
              <div class="{{ 'bad' if block.prob_loss_raw > 0.3 else 'good' }}"><strong>Prob. Loss:</strong> {{ block.prob_loss }}</div>
            </div>
            {% endfor %}
          </div>
          <canvas id="projChart" class="mid-chart"></canvas>
        </div>
      </div>
      <div class="card">
        <div class="card-header">Macro Sensitivity &bull; Portfolio A</div>
        <div class="card-body">
          <table>
            <tbody>
              {% for k, v in macro_sens.items() %}
              <tr>
                <td style="font-weight:600;">{{ k }}</td>
                <td class="{{ 'bad' if 'Vulnerable' in v or 'risk' in v.lower() or 'drag' in v.lower() else ('good' if 'beneficiary' in v.lower() or 'cushion' in v.lower() else '') }}">{{ v }}</td>
              </tr>
              {% endfor %}
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- FACTOR + OVERLAP -->
    <div class="grid-3">
      <div class="card">
        <div class="card-header">Factor Tilts &bull; Portfolio A</div>
        <div class="card-body">
          <table>
            <tbody>
              {% for k, v in portfolios['Portfolio A'].factor_tilt.items() %}
              <tr><td>{{ k }}</td><td style="font-weight:700;">{{ v }}</td></tr>
              {% endfor %}
            </tbody>
          </table>
        </div>
      </div>
      <div class="card">
        <div class="card-header">Sector Overlap</div>
        <div class="card-body">
          <table>
            <tbody>
              {% for row in overlap_rows %}
              <tr><td>{{ row.name }}</td><td style="font-weight:700;">{{ row.value }}</td></tr>
              {% endfor %}
            </tbody>
          </table>
        </div>
      </div>
      <div class="card">
        <div class="card-header">Beta &amp; Alpha Summary</div>
        <div class="card-body">
          <table>
            <thead><tr><th>Portfolio</th><th>Beta</th><th>Alpha</th></tr></thead>
            <tbody>
              {% for name, p in portfolios.items() %}
              <tr>
                <td style="font-weight:700;">{{ name }}</td>
                <td>{{ p.metrics.beta }}</td>
                <td class="{{ 'good' if p.metrics.alpha_raw > 0 else 'bad' }}">{{ p.metrics.alpha }}</td>
              </tr>
              {% endfor %}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </section>

  <!-- NEWS FEED -->
  <div class="section-sep"><span class="section-sep-label">Financial News Feed</span></div>
  <section id="news">
    <div class="card">
      <div class="card-header">Live Headlines</div>
      <div class="card-body">
        {% if news_items %}
          {% for item in news_items %}
          <div class="news-card-wrap">
            <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:10px;padding:13px 16px;border:1px solid var(--border);border-radius:12px;background:rgba(255,255,255,.02);">
              <div style="flex:1;">
                <a href="{{ item.link }}" target="_blank" rel="noopener noreferrer" style="text-decoration:none;color:var(--text);">
                  <div class="news-title">{{ item.title }}</div>
                  <div class="news-pub">{{ item.pub }}</div>
                </a>
              </div>
              <button class="analyze-btn" onclick="analyzeHeadline(this, {{ item.title|tojson }})">Analyze Impact</button>
            </div>
            <div class="impact-panel" id="impact-{{ loop.index0 }}">
              <div class="muted small">Loading...</div>
            </div>
          </div>
          {% endfor %}
        {% else %}
          <div class="muted">News feed unavailable right now.</div>
        {% endif %}
      </div>
    </div>
  </section>

  <!-- NEWS IMPACT -->
  <div class="section-sep"><span class="section-sep-label">News Impact Analyzer</span></div>
  <section id="news-impact">
    <div class="card">
      <div class="card-header">Analyze Custom Headline</div>
      <div class="card-body">
        <div style="display:flex;gap:10px;margin-bottom:14px;">
          <input id="custom-headline-input" placeholder="Enter a headline to analyze impact (e.g. Fed raises rates 50bps)..." style="flex:1;"/>
          <button class="btn" onclick="analyzeCustomHeadline()" style="white-space:nowrap;">Analyze</button>
        </div>
        <div id="custom-impact-result" style="display:none;">
          <div id="custom-impact-badge" style="margin-bottom:10px;"></div>
          <div class="grid-2" style="gap:14px;">
            <div>
              <div class="muted small" style="margin-bottom:6px;text-transform:uppercase;letter-spacing:.08em;">Sector Impacts</div>
              <div id="custom-impact-sectors"></div>
            </div>
            <div>
              <div class="muted small" style="margin-bottom:6px;text-transform:uppercase;letter-spacing:.08em;">Portfolio Relevance</div>
              <div id="custom-impact-portfolio"></div>
            </div>
          </div>
        </div>
        <div id="custom-impact-neutral" style="display:none;color:var(--muted);font-size:13px;">No matching impact rules found for this headline.</div>
      </div>
    </div>
  </section>

  <!-- GEO RISK RADAR -->
  <div class="section-sep"><span class="section-sep-label">Global Risk Radar</span></div>
  <section id="geo-risk">
    <div class="card">
      <div class="card-header">Geopolitical Risk Hotspots <span class="muted small">Click a marker for details</span></div>
      <div class="card-body">
        <div class="geo-filter-bar">
          <button class="geo-filter-btn active" onclick="filterGeo('all', this)">All</button>
          <button class="geo-filter-btn" onclick="filterGeo('conflict', this)">Conflict</button>
          <button class="geo-filter-btn" onclick="filterGeo('tariffs', this)">Tariffs</button>
          <button class="geo-filter-btn" onclick="filterGeo('oil', this)">Oil / Shipping</button>
          <button class="geo-filter-btn" onclick="filterGeo('commodity', this)">Commodity</button>
        </div>
        <div id="geo-map"></div>
        <div id="geo-detail">
          <div id="geo-detail-content"></div>
        </div>
      </div>
    </div>
  </section>

  {% endif %}

  <!-- ── AI MACRO INTEL ────────────────────────────────────────────────── -->
  <div class="section-sep"><span class="section-sep-label">AI Macro Intel</span></div>

  <div class="refresh-bar">
    <button class="btn secondary" id="refresh-macro-btn" onclick="refreshMacro()" style="font-size:12px;padding:8px 14px;">&#x1F504; Refresh Analysis</button>
    <span class="refresh-status" id="macro-refresh-status">Loading macro analysis…</span>
  </div>

  <!-- Morning Brief -->
  <section id="morning-brief">
    <div class="section-sep"><span class="section-sep-label">&#x1F305; Morning Brief</span></div>
    <div id="morning-brief-content"><div class="macro-loading"><div class="spin"></div><br/>Running strategist brief…</div></div>
  </section>

  <!-- Macro Regime -->
  <section id="macro-regime">
    <div class="section-sep"><span class="section-sep-label">&#x1F9E0; Macro Regime & Cross-Asset Signals</span></div>
    <div id="macro-regime-content"><div class="macro-loading"><div class="spin"></div><br/>Analyzing macro regime…</div></div>
  </section>

  <!-- Market Impact -->
  <section id="market-impact">
    <div class="section-sep"><span class="section-sep-label">&#x26A1; Market Impact Cards</span></div>
    <div id="market-impact-content"><div class="macro-loading"><div class="spin"></div><br/>Processing market impact…</div></div>
  </section>

  <!-- Central Banks -->
  <section id="central-banks">
    <div class="section-sep"><span class="section-sep-label">&#x1F3E6; Central Banks Watch</span></div>
    <div id="central-banks-content"><div class="macro-loading"><div class="spin"></div><br/>Evaluating central bank signals…</div></div>
  </section>

  <!-- AI Sectors -->
  <section id="ai-sectors">
    <div class="section-sep"><span class="section-sep-label">&#x1F4C9; Sector Outlook (AI)</span></div>
    <div id="ai-sectors-content"><div class="macro-loading"><div class="spin"></div><br/>Building sector analysis…</div></div>
  </section>

  <!-- Sentiment -->
  <section id="ai-sentiment">
    <div class="section-sep"><span class="section-sep-label">&#x1F4AC; Sentiment & Rotation Signals</span></div>
    <div id="ai-sentiment-content"><div class="macro-loading"><div class="spin"></div><br/>Synthesizing sentiment…</div></div>
  </section>

  <div class="footer">
    <span class="muted">Macro Dashboard &bull; Portfolio Lab &bull; Built by </span>
    <a href="{{ linkedin_url }}" target="_blank" rel="noopener noreferrer">Jake Joseph</a>
    <span class="muted"> &bull; {{ updated_at }}</span>
  </div>
</main>
</div>

<!-- AI Chat Panel -->
<button id="chat-float-btn" onclick="toggleChat()">&#x1F4AC; Ask AI</button>
<div id="chat-panel">
  <div id="chat-header">
    <h3>&#x1F916; AI Analyst</h3>
    <button id="chat-close" onclick="toggleChat()">&#x2715;</button>
  </div>
  <div id="chat-messages"></div>
  <div class="chat-quick-btns">
    <button class="quick-btn" onclick="sendQuick('How is my portfolio doing?')">Portfolio health?</button>
    <button class="quick-btn" onclick="sendQuick('What are my biggest risks?')">Biggest risks?</button>
    <button class="quick-btn" onclick="sendQuick('What does the current macro regime mean for my portfolio?')">Macro impact?</button>
    <button class="quick-btn" onclick="sendQuick('Which sectors should I be overweight given the current macro?')">Sector strategy?</button>
    <button class="quick-btn" onclick="sendQuick('How does the Fed stance affect my holdings?')">Fed impact?</button>
    <button class="quick-btn" onclick="sendQuick('Am I too concentrated?')">Concentration?</button>
  </div>
  <div id="chat-loading">&#x23F3; AI is thinking...</div>
  <div id="chat-input-row">
    <textarea id="chat-input" rows="2" placeholder="Ask about your portfolio or macro..."></textarea>
    <button id="chat-send" onclick="sendChat()">Send</button>
  </div>
</div>

<script>
// ── Data from server ──────────────────────────────────────────────────────────
const perfSeries     = {{ perf_series|safe }};
const ddSeries       = {{ drawdown_series|safe }};
const allocA         = {{ allocation_a|safe }};
const projData       = {{ projection_series|safe }};
const rollingCorr    = {{ rolling_corr|safe }};
const sparkData      = {{ spark_data|safe }};
const corrMatrix     = {{ corr_matrix_json|safe }};

const PORTFOLIO_NAMES = ['Portfolio A','Portfolio B','Portfolio C'];
const PALETTE = ['#59a8ff','#3bd671','#ff5f6d','#ffb44d','#1dd1c1','#c084fc','#f97316','#38bdf8'];
function color(i){ return PALETTE[i % PALETTE.length]; }

// ── Chart helpers ─────────────────────────────────────────────────────────────
const baseScaleOpts = {
  x:{ ticks:{color:'#98a8ba',maxTicksLimit:10,maxRotation:0}, grid:{color:'#203146',drawBorder:false} },
  y:{ ticks:{color:'#98a8ba'}, grid:{color:'#203146',drawBorder:false} }
};
const baseTooltip = {
  backgroundColor:'rgba(9,16,26,.95)',borderColor:'#203146',borderWidth:1,
  titleColor:'#98a8ba',bodyColor:'#edf2f7',padding:12,mode:'index',intersect:false
};

function makeLine(id, seriesObj, suffix='%') {
  const el = document.getElementById(id);
  if (!el || !Object.keys(seriesObj).length) return;
  const labels = Object.values(seriesObj)[0]?.labels || [];
  const datasets = Object.entries(seriesObj).map(([name, s], i) => {
    const isPort = PORTFOLIO_NAMES.includes(name);
    const c = color(i);
    return {
      label: name, data: s.values, borderColor: c,
      backgroundColor: isPort ? c+'18' : 'transparent',
      borderWidth: isPort ? 2.5 : 1.5,
      pointRadius: 0, tension: 0.25,
      fill: false, order: isPort ? 0 : 1,
    };
  });
  new Chart(el, {
    type:'line', data:{labels,datasets},
    options:{
      responsive:true, animation:{duration:600},
      interaction:{mode:'index',intersect:false},
      plugins:{
        legend:{labels:{color:'#edf2f7',padding:14,boxWidth:20,font:{size:12}}},
        tooltip:{...baseTooltip, callbacks:{
          label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(2)}${suffix}`
        }}
      },
      scales:{
        x:{...baseScaleOpts.x},
        y:{...baseScaleOpts.y, ticks:{...baseScaleOpts.y.ticks, callback: v => v+suffix}}
      }
    }
  });
}

function makeDonut(id, obj) {
  const el = document.getElementById(id);
  if (!el) return;
  const labels = Object.keys(obj);
  const values = Object.values(obj).map(v => parseFloat(v));
  new Chart(el, {
    type:'doughnut',
    data:{labels, datasets:[{data:values, backgroundColor:labels.map((_,i)=>color(i)), borderWidth:2, borderColor:'#0e1520'}]},
    options:{
      responsive:true,
      plugins:{
        legend:{labels:{color:'#edf2f7',padding:12,font:{size:12}}},
        tooltip:{...baseTooltip, callbacks:{label: ctx=>`${ctx.label}: ${ctx.parsed.toFixed(1)}%`}}
      }
    }
  });
}

function makeProjection(id, obj) {
  const el = document.getElementById(id);
  if (!el || !obj.labels?.length) return;
  const datasets = [
    {label:'Base',data:obj.base,borderColor:'#59a8ff',backgroundColor:'#59a8ff18',fill:true,pointRadius:0,tension:0.25,borderWidth:2},
    {label:'Bull',data:obj.bull,borderColor:'#3bd671',pointRadius:0,tension:0.25,borderWidth:2},
    {label:'Bear',data:obj.bear,borderColor:'#ff5f6d',pointRadius:0,tension:0.25,borderWidth:2},
  ];
  new Chart(el, {
    type:'line', data:{labels:obj.labels,datasets},
    options:{
      responsive:true,
      interaction:{mode:'index',intersect:false},
      plugins:{
        legend:{labels:{color:'#edf2f7',padding:14,font:{size:12}}},
        tooltip:{...baseTooltip, callbacks:{label:ctx=>` ${ctx.dataset.label}: $${ctx.parsed.y.toLocaleString()}`}}
      },
      scales:{
        x:{...baseScaleOpts.x},
        y:{...baseScaleOpts.y, ticks:{...baseScaleOpts.y.ticks, callback:v=>'$'+v.toLocaleString()}}
      }
    }
  });
}

function makeSpark(id, s, c) {
  const el = document.getElementById(id);
  if (!el || !s?.values?.length) return;
  new Chart(el, {
    type:'line',
    data:{labels:s.labels, datasets:[{data:s.values,borderColor:c,backgroundColor:c+'20',fill:true,pointRadius:0,tension:.3,borderWidth:2}]},
    options:{responsive:true,plugins:{legend:{display:false}},scales:{x:{display:false},y:{display:false}}}
  });
}

function buildCorrMatrix(wrap, data) {
  if (!wrap || !data.labels?.length) return;
  const n = data.labels.length;
  let html = '<table class="corr-table" style="width:100%"><thead><tr><th></th>';
  data.labels.forEach(l => { html += `<th style="color:var(--muted)">${l}</th>`; });
  html += '</tr></thead><tbody>';
  data.matrix.forEach((row, i) => {
    html += `<tr><td style="font-weight:700;color:var(--muted)">${data.labels[i]}</td>`;
    row.forEach(val => {
      const v = parseFloat(val);
      const intensity = Math.abs(v);
      let bg = 'transparent';
      if (v > 0.7) bg = `rgba(59,214,113,${intensity * 0.4})`;
      else if (v > 0.3) bg = `rgba(59,214,113,${intensity * 0.2})`;
      else if (v < -0.3) bg = `rgba(255,95,109,${intensity * 0.3})`;
      html += `<td style="background:${bg};border-radius:6px;font-weight:${v===1?'800':'400'}">${v.toFixed(2)}</td>`;
    });
    html += '</tr>';
  });
  html += '</tbody></table>';
  wrap.innerHTML = html;
}

// ── Render charts ─────────────────────────────────────────────────────────────
makeLine('perfChart', perfSeries, '%');
makeLine('ddChart', ddSeries, '%');
makeDonut('allocA', allocA);
makeProjection('projChart', projData);
makeLine('rollingCorrChart', rollingCorr, '');
buildCorrMatrix(document.getElementById('corrMatrixWrap'), corrMatrix);

Object.entries(sparkData).forEach(([id, s], i) => {
  makeSpark(`spark_${i}`, s, PALETTE[i % PALETTE.length]);
});

// ── Ticker search ─────────────────────────────────────────────────────────────
let searchTimer = null;
function setupSearch(inputId, targetName) {
  const input = document.getElementById(inputId);
  const dd = document.getElementById(inputId + '_dd');
  if (!input || !dd) return;
  input.addEventListener('input', () => {
    clearTimeout(searchTimer);
    const q = input.value.trim();
    if (q.length < 2) { dd.style.display = 'none'; return; }
    searchTimer = setTimeout(() => {
      fetch(`/api/search?q=${encodeURIComponent(q)}`)
        .then(r => r.json())
        .then(data => {
          if (!data.length) { dd.style.display = 'none'; return; }
          dd.innerHTML = data.map(d =>
            `<div class="search-item" onclick="addTicker('${targetName}','${d.symbol}')">
              <span class="sym">${d.symbol}</span><span class="nm">${d.name}</span>
            </div>`
          ).join('');
          dd.style.display = 'block';
        })
        .catch(() => { dd.style.display = 'none'; });
    }, 280);
  });
  document.addEventListener('click', e => {
    if (!input.contains(e.target) && !dd.contains(e.target)) dd.style.display = 'none';
  });
}

function addTicker(fieldName, symbol) {
  const ta = document.querySelector(`[name=${fieldName}]`);
  if (!ta) return;
  const cur = ta.value.trim();
  ta.value = cur ? cur + `\n${symbol},10%` : `${symbol},10%`;
  document.querySelectorAll('.search-dd').forEach(d => d.style.display = 'none');
  document.querySelectorAll('.search-input').forEach(i => i.value = '');
}

function clearPortfolio(name) {
  const ta = document.querySelector(`[name=${name}]`);
  if (ta) ta.value = '';
}

setupSearch('search_0', 'portfolio_a');
setupSearch('search_1', 'portfolio_b');
setupSearch('search_2', 'portfolio_c');

// ── Dark / Light mode ─────────────────────────────────────────────────────────
const themeBtn = document.getElementById('theme-toggle');
if (localStorage.getItem('theme') === 'light') {
  document.body.classList.add('light-mode');
  if (themeBtn) themeBtn.textContent = '\u2600\uFE0F';
}
themeBtn?.addEventListener('click', () => {
  document.body.classList.toggle('light-mode');
  const isLight = document.body.classList.contains('light-mode');
  localStorage.setItem('theme', isLight ? 'light' : 'dark');
  themeBtn.textContent = isLight ? '\u2600\uFE0F' : '\u263A';
});

// ── Save / Load portfolios ────────────────────────────────────────────────────
function getFormData() {
  return {
    a: document.querySelector('[name=portfolio_a]')?.value || '',
    b: document.querySelector('[name=portfolio_b]')?.value || '',
    c: document.querySelector('[name=portfolio_c]')?.value || '',
    start: document.querySelector('[name=start_date]')?.value || '',
    rf: document.querySelector('[name=risk_free_rate]')?.value || '',
    pv: document.querySelector('[name=projection_value]')?.value || '',
    bench: document.querySelector('[name=custom_benchmark]')?.value || '',
  };
}
function savePortfolio() {
  const name = document.getElementById('save-name')?.value.trim();
  if (!name) return alert('Please enter a name for this portfolio set.');
  const saved = JSON.parse(localStorage.getItem('portfolioSets') || '{}');
  saved[name] = getFormData();
  localStorage.setItem('portfolioSets', JSON.stringify(saved));
  document.getElementById('save-name').value = '';
  renderSavedList();
}
function loadPortfolio(name) {
  const saved = JSON.parse(localStorage.getItem('portfolioSets') || '{}');
  const d = saved[name];
  if (!d) return;
  if (d.a !== undefined) document.querySelector('[name=portfolio_a]').value = d.a;
  if (d.b !== undefined) document.querySelector('[name=portfolio_b]').value = d.b;
  if (d.c !== undefined) document.querySelector('[name=portfolio_c]').value = d.c;
  if (d.start) document.querySelector('[name=start_date]').value = d.start;
  if (d.rf) document.querySelector('[name=risk_free_rate]').value = d.rf;
  if (d.pv) document.querySelector('[name=projection_value]').value = d.pv;
  if (d.bench) document.querySelector('[name=custom_benchmark]').value = d.bench;
}
function deleteSaved(name) {
  const saved = JSON.parse(localStorage.getItem('portfolioSets') || '{}');
  delete saved[name];
  localStorage.setItem('portfolioSets', JSON.stringify(saved));
  renderSavedList();
}
function renderSavedList() {
  const el = document.getElementById('saved-list');
  if (!el) return;
  const saved = JSON.parse(localStorage.getItem('portfolioSets') || '{}');
  const keys = Object.keys(saved);
  if (!keys.length) {
    el.innerHTML = '<div class="muted small" style="padding:4px 0 8px;">No saved sets.</div>';
    return;
  }
  el.innerHTML = keys.map(k => `
    <div style="display:flex;align-items:center;gap:5px;margin-bottom:5px;">
      <button onclick="loadPortfolio('${k}')" class="saved-btn" style="flex:1;">${k}</button>
      <button onclick="deleteSaved('${k}')" class="btn danger" style="padding:4px 7px;font-size:11px;">&#x2715;</button>
    </div>`).join('');
}

// ── Shareable URL ─────────────────────────────────────────────────────────────
function sharePortfolio() {
  try {
    const encoded = btoa(unescape(encodeURIComponent(JSON.stringify(getFormData()))));
    const url = location.origin + location.pathname + '#share=' + encoded;
    navigator.clipboard.writeText(url).then(() => alert('Shareable link copied to clipboard!'));
  } catch(e) { alert('Could not generate share link.'); }
}
function checkShareHash() {
  const hash = location.hash;
  if (!hash.startsWith('#share=')) return;
  try {
    const d = JSON.parse(decodeURIComponent(escape(atob(hash.slice(7)))));
    if (d.a) document.querySelector('[name=portfolio_a]').value = d.a;
    if (d.b) document.querySelector('[name=portfolio_b]').value = d.b;
    if (d.c) document.querySelector('[name=portfolio_c]').value = d.c;
    if (d.start) document.querySelector('[name=start_date]').value = d.start;
    if (d.rf) document.querySelector('[name=risk_free_rate]').value = d.rf;
    if (d.pv) document.querySelector('[name=projection_value]').value = d.pv;
    if (d.bench) document.querySelector('[name=custom_benchmark]').value = d.bench;
  } catch(e) {}
}

// ── Keyboard shortcuts ────────────────────────────────────────────────────────
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') return;
  if (e.key === 'r' || e.key === 'R') document.getElementById('main-form')?.submit();
  if (e.key === 'p' || e.key === 'P') window.print();
});

// ── Init ──────────────────────────────────────────────────────────────────────
window.addEventListener('load', () => { checkShareHash(); renderSavedList(); initGeoMap(); });

// ── Helpers ───────────────────────────────────────────────────────────────────
function getPortfolioAWeights() {
  const ta = document.querySelector('[name=portfolio_a]');
  if (!ta) return {};
  const weights = {};
  const lines = ta.value.trim().split('\n');
  const entries = [];
  for (const line of lines) {
    const parts = line.split(',');
    if (parts.length !== 2) continue;
    const ticker = parts[0].trim().toUpperCase();
    const val = parts[1].trim();
    let w = 0;
    if (val.endsWith('%')) w = parseFloat(val) / 100;
    else if (val.startsWith('$')) w = parseFloat(val.slice(1).replace(/,/g, ''));
    else w = parseFloat(val.replace(/,/g, ''));
    if (!isNaN(w)) entries.push({ticker, w});
  }
  if (!entries.length) return {};
  const total = entries.reduce((s, e) => s + e.w, 0);
  for (const e of entries) weights[e.ticker] = e.w / total;
  return weights;
}

// ── Stress Test ───────────────────────────────────────────────────────────────
async function runStressTest() {
  const btn = document.getElementById('run-stress-btn');
  const loading = document.getElementById('stress-loading');
  const results = document.getElementById('stress-results');
  const tbody = document.getElementById('stress-tbody');

  const weights = getPortfolioAWeights();
  if (!Object.keys(weights).length) {
    alert('Run Portfolio Lab first, then click Stress Test.');
    return;
  }

  btn.disabled = true;
  loading.style.display = 'inline';
  results.style.display = 'none';

  try {
    const resp = await fetch('/api/stress-test', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({weights}),
    });
    const data = await resp.json();
    if (data.error) { alert('Stress test error: ' + data.error); return; }

    tbody.innerHTML = data.periods.map(p => {
      const pRet = p.portfolio_return;
      const sRet = p.spy_return;
      const pCls = pRet === null ? '' : (pRet >= 0 ? 'pos' : 'neg');
      const sCls = sRet === null ? '' : (sRet >= 0 ? 'pos' : 'neg');
      const vs = p.outperformed === null ? '—' : (p.outperformed ? '&#x2191; Outperformed' : '&#x2193; Underperformed');
      const vsCls = p.outperformed === null ? '' : (p.outperformed ? 'out' : 'under');
      return `<tr>
        <td style="font-weight:700;">${p.name}</td>
        <td class="stress-ret ${pCls}">${pRet === null ? '—' : (pRet >= 0 ? '+' : '') + pRet.toFixed(1) + '%'}</td>
        <td class="stress-ret ${sCls}">${sRet === null ? '—' : (sRet >= 0 ? '+' : '') + sRet.toFixed(1) + '%'}</td>
        <td class="stress-vs ${vsCls}">${vs}</td>
      </tr>`;
    }).join('');
    results.style.display = 'block';
  } catch(e) {
    alert('Stress test failed: ' + e.message);
  } finally {
    btn.disabled = false;
    loading.style.display = 'none';
  }
}

// ── News Impact ───────────────────────────────────────────────────────────────
function renderImpactResult(containerEl, data) {
  const sentiment = data.overall_sentiment || 'neutral';
  const badgeHtml = `<span class="impact-badge ${sentiment}">${sentiment.toUpperCase()}</span>`;

  let sectorsHtml = '';
  for (const [sector, dir] of Object.entries(data.sector_impacts || {})) {
    sectorsHtml += `<span class="impact-sector-tag ${dir === 'positive' ? 'pos' : dir === 'negative' ? 'neg' : 'mixed'}">${sector}: ${dir}</span>`;
  }
  if (!sectorsHtml) sectorsHtml = '<span class="muted small">No sector impacts matched.</span>';

  let portHtml = '';
  for (const item of (data.portfolio_relevance || [])) {
    portHtml += `<div style="font-size:12px;margin-bottom:4px;">
      <strong>${item.ticker}</strong> (${(item.weight*100).toFixed(1)}%) —
      <span class="${item.direction === 'positive' ? 'good' : item.direction === 'negative' ? 'bad' : ''}">${item.direction}</span>
      <span class="muted"> ${item.reason}</span>
    </div>`;
  }
  if (!portHtml) portHtml = '<span class="muted small">No portfolio holdings directly matched.</span>';

  containerEl.innerHTML = `
    ${badgeHtml}
    <div style="margin-bottom:10px;">
      <div class="muted small" style="margin-bottom:5px;text-transform:uppercase;letter-spacing:.08em;">Sector Impacts</div>
      ${sectorsHtml}
    </div>
    <div>
      <div class="muted small" style="margin-bottom:5px;text-transform:uppercase;letter-spacing:.08em;">Portfolio Relevance</div>
      ${portHtml}
    </div>
  `;
}

async function analyzeHeadline(btn, headline) {
  const wrap = btn.closest('.news-card-wrap');
  const panel = wrap ? wrap.querySelector('.impact-panel') : null;
  if (!panel) return;

  const isOpen = panel.style.display === 'block';
  if (isOpen) { panel.style.display = 'none'; return; }

  panel.style.display = 'block';
  panel.innerHTML = '<div class="muted small">Analyzing...</div>';

  const weights = getPortfolioAWeights();
  try {
    const resp = await fetch('/api/news-impact', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({headline, weights}),
    });
    const data = await resp.json();
    renderImpactResult(panel, data);
  } catch(e) {
    panel.innerHTML = `<span class="bad">Error: ${e.message}</span>`;
  }
}

async function analyzeCustomHeadline() {
  const input = document.getElementById('custom-headline-input');
  const headline = input?.value.trim();
  if (!headline) { alert('Enter a headline first.'); return; }

  const resultDiv = document.getElementById('custom-impact-result');
  const neutralDiv = document.getElementById('custom-impact-neutral');
  resultDiv.style.display = 'none';
  neutralDiv.style.display = 'none';

  const weights = getPortfolioAWeights();
  try {
    const resp = await fetch('/api/news-impact', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({headline: headline.toLowerCase(), weights}),
    });
    const data = await resp.json();

    if (!Object.keys(data.sector_impacts || {}).length) {
      neutralDiv.style.display = 'block';
      return;
    }

    document.getElementById('custom-impact-badge').innerHTML =
      `<span class="impact-badge ${data.overall_sentiment}">${data.overall_sentiment.toUpperCase()}</span>`;

    let sectorsHtml = '';
    for (const [sector, dir] of Object.entries(data.sector_impacts || {})) {
      sectorsHtml += `<span class="impact-sector-tag ${dir === 'positive' ? 'pos' : dir === 'negative' ? 'neg' : 'mixed'}">${sector}: ${dir}</span> `;
    }
    document.getElementById('custom-impact-sectors').innerHTML = sectorsHtml;

    let portHtml = '';
    for (const item of (data.portfolio_relevance || [])) {
      portHtml += `<div style="font-size:12px;margin-bottom:4px;">
        <strong>${item.ticker}</strong> —
        <span class="${item.direction === 'positive' ? 'good' : item.direction === 'negative' ? 'bad' : ''}">${item.direction}</span>
      </div>`;
    }
    if (!portHtml) portHtml = '<span class="muted small">No direct holdings matched.</span>';
    document.getElementById('custom-impact-portfolio').innerHTML = portHtml;

    resultDiv.style.display = 'block';
  } catch(e) {
    alert('Analysis failed: ' + e.message);
  }
}

// ── Geo Risk Map ──────────────────────────────────────────────────────────────
let geoMap = null;
let geoMarkers = [];
let geoHotspots = [];
let geoActiveFilter = 'all';

async function initGeoMap() {
  const mapEl = document.getElementById('geo-map');
  if (!mapEl || typeof L === 'undefined') return;

  geoMap = L.map('geo-map', {center:[20,10],zoom:2,zoomControl:true});
  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution:'&copy; OpenStreetMap &copy; CARTO',
    subdomains:'abcd', maxZoom:19,
  }).addTo(geoMap);

  try {
    const resp = await fetch('/api/geo-risk');
    geoHotspots = await resp.json();
    renderGeoMarkers(geoHotspots);
  } catch(e) {
    console.error('Geo risk load failed:', e);
  }
}

function renderGeoMarkers(hotspots) {
  geoMarkers.forEach(m => geoMap.removeLayer(m));
  geoMarkers = [];

  const portfolioTickers = Object.keys(getPortfolioAWeights()).map(t => t.toUpperCase());

  hotspots.forEach(h => {
    const radius = 6 + (h.risk_score / 100) * 18;
    const marker = L.circleMarker([h.lat, h.lng], {
      radius, color: h.color, fillColor: h.color,
      fillOpacity: 0.45, weight: 2, opacity: 0.9,
    });
    marker.bindTooltip(`<strong>${h.title}</strong><br>${h.risk_level} — Score: ${h.risk_score}`, {
      permanent: false, direction: 'top',
    });
    marker.on('click', () => showGeoDetail(h, portfolioTickers));
    marker.addTo(geoMap);
    geoMarkers.push(marker);
  });
}

function showGeoDetail(h, portfolioTickers) {
  const detail = document.getElementById('geo-detail');
  const content = document.getElementById('geo-detail-content');
  if (!detail || !content) return;

  const matchedTickers = portfolioTickers.filter(t =>
    h.portfolio_keywords.some(kw => kw.toUpperCase() === t)
  );

  const assetTags = h.affected_assets.map(a => {
    const cls = matchedTickers.includes(a) ? 'asset-tag port-match' : 'asset-tag';
    return `<span class="${cls}">${a}</span>`;
  }).join('');

  const sectorTags = (h.affected_sectors || []).map(s =>
    `<span class="asset-tag">${s}</span>`).join('');

  const portMatch = matchedTickers.length
    ? `<div style="margin-top:10px;padding:10px;border-radius:10px;background:rgba(59,214,113,.06);border:1px solid rgba(59,214,113,.2);">
        <div class="muted small" style="margin-bottom:5px;">&#x2705; Portfolio Overlap</div>
        ${matchedTickers.map(t => `<span class="asset-tag port-match">${t}</span>`).join('')}
       </div>`
    : `<div style="margin-top:10px;font-size:12px;color:var(--muted);">No direct portfolio holdings matched for this hotspot.</div>`;

  content.innerHTML = `
    <div style="margin-bottom:12px;">
      <span style="font-size:17px;font-weight:800;">${h.title}</span>
      <span class="geo-risk-badge ${h.risk_level}">${h.risk_level}</span>
      <span class="muted small" style="margin-left:8px;">${h.region}</span>
      <span style="margin-left:8px;font-size:12px;color:var(--muted);">Risk Score: ${h.risk_score}/100</span>
    </div>
    <p style="font-size:13px;line-height:1.6;color:#c8d8e8;margin-bottom:10px;">${h.summary}</p>
    <div style="margin-bottom:10px;">
      <div class="muted small" style="margin-bottom:4px;">Market Impact</div>
      <div style="font-size:13px;">${h.market_impact}</div>
    </div>
    <div style="margin-bottom:8px;">
      <div class="muted small" style="margin-bottom:4px;">Affected Assets</div>
      ${assetTags}
    </div>
    <div style="margin-bottom:8px;">
      <div class="muted small" style="margin-bottom:4px;">Affected Sectors</div>
      ${sectorTags}
    </div>
    ${portMatch}
  `;
  detail.style.display = 'block';
}

function filterGeo(category, btn) {
  geoActiveFilter = category;
  document.querySelectorAll('.geo-filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');

  const catMap = {
    'conflict': ['conflict'],
    'tariffs': ['tariffs'],
    'oil': ['oil_chokepoint', 'shipping'],
    'commodity': ['commodity_shock', 'currency_pressure'],
    'all': null,
  };
  const allowed = catMap[category];
  const filtered = allowed ? geoHotspots.filter(h => allowed.includes(h.category)) : geoHotspots;
  renderGeoMarkers(filtered);
}

// ── AI Chat Panel ─────────────────────────────────────────────────────────────
let chatOpen = false;
let chatHistory = [];

function toggleChat() {
  chatOpen = !chatOpen;
  const panel = document.getElementById('chat-panel');
  panel.classList.toggle('open', chatOpen);
  if (chatOpen && chatHistory.length === 0) {
    appendChatMsg('ai', 'Hello! I\'m your AI Analyst. Ask me about your portfolio, macro conditions, risks, or anything market-related.');
  }
}

function appendChatMsg(role, text) {
  const msgs = document.getElementById('chat-messages');
  const div = document.createElement('div');
  div.className = `chat-msg ${role}`;
  div.textContent = text;
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
  chatHistory.push({role, text});
}

async function sendChat() {
  const input = document.getElementById('chat-input');
  const sendBtn = document.getElementById('chat-send');
  const loading = document.getElementById('chat-loading');
  const message = input?.value.trim();
  if (!message) return;

  appendChatMsg('user', message);
  input.value = '';
  sendBtn.disabled = true;
  loading.style.display = 'block';

  const context = {
    regime: document.querySelector('.regime-box strong')?.textContent || 'Unknown',
    fear_greed_score: document.querySelector('.fg-box div[style*="font-size:18px"]')?.textContent || 'Unknown',
    fear_greed_label: document.querySelector('.fg-box div[style*="font-size:12px"]')?.textContent || 'Unknown',
    macro_analysis: _macroData ? {
      regime: _macroData.macro_regime?.regime,
      regime_summary: _macroData.macro_regime?.regime_summary,
      morning_brief_headline: _macroData.morning_brief?.headline,
      biggest_risk: _macroData.morning_brief?.biggest_risk,
      cross_asset: _macroData.macro_regime?.cross_asset,
      rotation_signals: _macroData.sentiment?.rotation_signals,
    } : 'not loaded yet',
    note: 'Full AI macro analysis available above. Portfolio data in Portfolio Lab section.',
  };

  try {
    const resp = await fetch('/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message, context}),
    });
    const data = await resp.json();
    if (data.error) {
      appendChatMsg('error', 'Error: ' + data.error);
    } else {
      appendChatMsg('ai', data.response);
    }
  } catch(e) {
    appendChatMsg('error', 'Request failed: ' + e.message);
  } finally {
    sendBtn.disabled = false;
    loading.style.display = 'none';
  }
}

async function sendQuick(prompt) {
  document.getElementById('chat-input').value = prompt;
  await sendChat();
}

document.getElementById('chat-input')?.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChat(); }
});

// ── Macro AI Intel ────────────────────────────────────────────────────────────
let _macroData = null;

function colorClass(c) {
  return c === 'red' ? 'var(--red)' : c === 'green' ? 'var(--green)' : 'var(--amber)';
}

function renderMorningBrief(d) {
  const b = d.morning_brief || {};
  const paras = (b.paragraphs || []).map(p => `<p class="b-para">${p}</p>`).join('');
  const watchRows = (b.watchlist || []).map(w => `
    <tr>
      <td><strong>${w.event||''}</strong></td>
      <td style="color:var(--muted)">${w.why_now||''}</td>
      <td style="color:var(--green)">${w.bullish||''}</td>
      <td style="color:var(--red)">${w.bearish||''}</td>
      <td style="color:var(--blue)">${w.assets||''}</td>
    </tr>`).join('');
  document.getElementById('morning-brief-content').innerHTML = `
    <div class="brief-hero">
      <div class="b-headline">${b.headline||'Strategist Brief'}</div>
      ${paras}
      <div class="brief-meta">
        <div class="brief-meta-item"><div class="bm-lbl">Biggest Risk</div><span style="color:var(--red)">${b.biggest_risk||'—'}</span></div>
        <div class="brief-meta-item"><div class="bm-lbl">Most Sensitive Sector</div><span style="color:var(--amber)">${b.most_sensitive_sector||'—'}</span></div>
      </div>
    </div>
    <div class="card">
      <div class="card-header">&#x1F440; Watchlist</div>
      <div class="card-body" style="padding:0">
        <table class="watchlist-table">
          <thead><tr><th>Event</th><th>Why Now</th><th>Bull Case</th><th>Bear Case</th><th>Assets</th></tr></thead>
          <tbody>${watchRows}</tbody>
        </table>
      </div>
    </div>`;
}

function renderMacroRegime(d) {
  const r = d.macro_regime || {};
  const rc = r.regime_color || 'amber';
  const caHtml = (r.cross_asset || []).map(c => `
    <div class="ca-cell ${c.color||'amber'}">
      <div class="ca-sig">${c.signal||''}</div>
      <div class="ca-read">${c.reading||''}</div>
      <div class="ca-det">${c.detail||''}</div>
    </div>`).join('');
  const driversHtml = (r.top_drivers || []).map(dr => `
    <div class="driver-item">
      <div class="d-title">${dr.title||''}</div>
      <div class="d-change">${dr.what_changed||''}</div>
      <div class="d-impact">${dr.market_impact||''}</div>
      <div class="driver-wl">
        ${dr.winners ? `<span class="win">&#x2191; ${dr.winners}</span>` : ''}
        ${dr.losers  ? `<span class="lose">&#x2193; ${dr.losers}</span>`  : ''}
      </div>
    </div>`).join('');
  document.getElementById('macro-regime-content').innerHTML = `
    <div class="regime-hero ${rc}">
      <div class="r-label">Current Regime</div>
      <div class="r-name" style="color:${colorClass(rc)}">${r.regime||'Unknown'}</div>
      <div class="r-sum">${r.regime_summary||''}</div>
    </div>
    <div class="cross-asset-grid">${caHtml}</div>
    <div class="card">
      <div class="card-header">Top Macro Drivers</div>
      <div class="card-body">${driversHtml}</div>
    </div>`;
}

function renderMarketImpact(d) {
  const cards = (d.market_impact || []).map(c => {
    const sectorTags = (c.sectors||[]).map(s =>
      `<span class="ic-tag ${s.direction||'neutral'}">${s.name}</span>`).join('');
    return `<div class="impact-card">
      <div class="ic-title">${c.title||''}</div>
      <div class="ic-why">${c.why_it_matters||''}</div>
      <div class="ic-meta">
        <span class="ic-overall ${c.overall||'neutral'}">${(c.overall||'neutral').toUpperCase()}</span>
        <span style="color:var(--muted)">${c.horizon||''}</span>
        <span style="color:var(--blue);font-weight:700">Impact ${c.score||''}</span>
      </div>
      <div class="ic-tags">${sectorTags}</div>
    </div>`;
  }).join('');
  document.getElementById('market-impact-content').innerHTML =
    `<div class="grid-3">${cards}</div>`;
}

function renderCentralBanks(d) {
  const rows = (d.central_banks || []).map(cb => `
    <div class="cb-row">
      <div>
        <div class="cb-bank">${cb.bank||''}</div>
      </div>
      <div>
        <div class="cb-stance ${cb.stance_color||'amber'}">${cb.stance||''}</div>
      </div>
      <div class="cb-detail">
        ${cb.latest_signals||''} ${cb.market_interpretation||''}
        ${cb.key_risk ? `<div class="cb-risk">&#x26A0; ${cb.key_risk}</div>` : ''}
      </div>
    </div>`).join('');
  document.getElementById('central-banks-content').innerHTML = `<div>${rows}</div>`;
}

function renderAISectors(d) {
  const cards = (d.sectors || []).map(s => {
    const tw = (s.tailwinds||[]).map(b => `<li>${b}</li>`).join('');
    const hw = (s.headwinds||[]).map(b => `<li>${b}</li>`).join('');
    return `<div class="sector-ai-card">
      <div class="s-header">
        <div class="s-name">${s.sector||''}</div>
        <span class="s-sent ${s.sentiment_color||'amber'}">${s.sentiment||''}</span>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:8px">
        <div>
          <div style="font-size:10px;font-weight:700;color:var(--green);margin-bottom:4px">TAILWINDS</div>
          <ul class="bullet-list">${tw}</ul>
        </div>
        <div>
          <div style="font-size:10px;font-weight:700;color:var(--red);margin-bottom:4px">HEADWINDS</div>
          <ul class="bullet-list">${hw}</ul>
        </div>
      </div>
      ${s.key_catalyst ? `<div class="s-catalyst">&#x1F4CD; ${s.key_catalyst}</div>` : ''}
    </div>`;
  }).join('');
  document.getElementById('ai-sectors-content').innerHTML =
    `<div class="sector-ai-grid">${cards}</div>`;
}

function renderSentiment(d) {
  const sent = d.sentiment || {};
  const bullHtml = (sent.bullish_themes||[]).map(t => `
    <div class="theme-item theme-bull">
      <div class="t-title">&#x1F7E2; ${t.theme||''}</div>
      <div class="t-detail">${t.detail||''}</div>
      <div class="t-assets">Assets: ${t.assets||''}</div>
    </div>`).join('');
  const bearHtml = (sent.bearish_themes||[]).map(t => `
    <div class="theme-item theme-bear">
      <div class="t-title">&#x1F534; ${t.theme||''}</div>
      <div class="t-detail">${t.detail||''}</div>
      <div class="t-assets">Assets: ${t.assets||''}</div>
    </div>`).join('');
  const rotHtml = (sent.rotation_signals||[]).map(r => `
    <div class="rotation-item">
      <span class="r-from">&#x2193; ${r.from||''}</span>
      <span class="r-arrow">&#x2192;</span>
      <span class="r-to">&#x2191; ${r.to||''}</span>
      <span class="r-sig">${r.signal||''}</span>
    </div>`).join('');
  document.getElementById('ai-sentiment-content').innerHTML = `
    <div class="grid-2" style="margin-bottom:18px">
      <div>
        <div style="font-size:11px;font-weight:800;text-transform:uppercase;letter-spacing:.1em;color:var(--green);margin-bottom:10px">Bullish Themes</div>
        <div class="theme-col">${bullHtml}</div>
      </div>
      <div>
        <div style="font-size:11px;font-weight:800;text-transform:uppercase;letter-spacing:.1em;color:var(--red);margin-bottom:10px">Bearish Themes</div>
        <div class="theme-col">${bearHtml}</div>
      </div>
    </div>
    <div class="card">
      <div class="card-header">&#x1F504; Rotation Signals</div>
      <div class="card-body">${rotHtml}</div>
    </div>`;
}

function renderMacroError(msg) {
  const errHtml = `<div style="padding:24px;color:var(--red);border:1px solid rgba(255,95,109,.2);border-radius:12px;background:rgba(255,95,109,.05)">${msg}</div>`;
  ['morning-brief','macro-regime','market-impact','central-banks','ai-sectors','ai-sentiment'].forEach(id => {
    const el = document.getElementById(id + '-content');
    if (el && el.querySelector('.macro-loading')) el.innerHTML = errHtml;
  });
}

async function loadMacroAnalysis() {
  document.getElementById('macro-refresh-status').textContent = 'Loading macro analysis (may take ~30s first load)…';
  try {
    const resp = await fetch('/api/macro-analysis');
    const data = await resp.json();
    if (data.error) { renderMacroError('Analysis error: ' + data.error); return; }
    _macroData = data;
    renderMorningBrief(data);
    renderMacroRegime(data);
    renderMarketImpact(data);
    renderCentralBanks(data);
    renderAISectors(data);
    renderSentiment(data);
    document.getElementById('macro-refresh-status').textContent = 'AI analysis loaded ✓ (cached 2h)';
  } catch(e) {
    renderMacroError('Failed to load macro analysis: ' + e.message);
    document.getElementById('macro-refresh-status').textContent = 'Load failed';
  }
}

async function refreshMacro() {
  const btn = document.getElementById('refresh-macro-btn');
  const status = document.getElementById('macro-refresh-status');
  btn.disabled = true;
  btn.textContent = '⏳ Refreshing…';
  status.textContent = 'Running fresh analysis (~30-60s)…';
  ['morning-brief','macro-regime','market-impact','central-banks','ai-sectors','ai-sentiment'].forEach(id => {
    const el = document.getElementById(id + '-content');
    if (el) el.innerHTML = '<div class="macro-loading"><div class="spin"></div><br/>Refreshing…</div>';
  });
  try {
    const resp = await fetch('/api/refresh-macro', {method:'POST'});
    const result = await resp.json();
    if (result.data) {
      _macroData = result.data;
      renderMorningBrief(_macroData);
      renderMacroRegime(_macroData);
      renderMarketImpact(_macroData);
      renderCentralBanks(_macroData);
      renderAISectors(_macroData);
      renderSentiment(_macroData);
      status.textContent = 'Analysis refreshed ✓';
    }
  } catch(e) {
    renderMacroError('Refresh failed: ' + e.message);
    status.textContent = 'Refresh failed';
  } finally {
    btn.disabled = false;
    btn.textContent = '🔄 Refresh Analysis';
  }
}

// Load macro analysis on page load
loadMacroAnalysis();
</script>
</body>
</html>
'''


@app.route("/", methods=["GET", "POST"])
def index():
    form = {
        "start_date": DEFAULT_START,
        "risk_free_rate": str(RISK_FREE_RATE),
        "projection_value": "100000",
        "custom_benchmark_name": "Custom Benchmark",
        "portfolio_a": DEFAULT_PORTFOLIOS["Portfolio A"],
        "portfolio_b": DEFAULT_PORTFOLIOS["Portfolio B"],
        "portfolio_c": DEFAULT_PORTFOLIOS["Portfolio C"],
        "custom_benchmark": DEFAULT_CUSTOM_BENCHMARK,
    }
    if request.method == "POST":
        for k in form:
            form[k] = request.form.get(k, form[k])

    # Defaults
    error = None
    portfolios = {}
    comparison_headers = []
    colored_rows = []
    overlap_rows = []
    perf_series = json.dumps({})
    drawdown_series = json.dumps({})
    allocation_a = json.dumps({})
    sector_a = json.dumps({})
    projection_series = json.dumps({"labels": [], "base": [], "bull": [], "bear": []})
    rolling_corr_json = json.dumps({})
    corr_matrix_json = json.dumps({"labels": [], "matrix": []})
    macro_sens = {}
    projections = {
        s: {"median": "—", "p10": "—", "p90": "—", "prob_loss": "—", "prob_loss_raw": 0}
        for s in ["Base", "Bull", "Bear"]
    }
    fear_greed = {"score": 50, "label": "Neutral", "color": "#ffb44d",
                  "dash": 141.4, "total_arc": 282.7, "nx": 100.0, "ny": 10.0}

    # Parallel data fetching
    macro_cards_raw = get_macro_cards()
    global_heatmap = get_global_heatmap()
    sector_rotation = get_sector_rotation()
    news_items = get_news_feed()
    fear_greed = compute_fear_greed(macro_cards_raw)
    regime_label, regime_detail = detect_regime(macro_cards_raw)

    macro_cards = []
    spark_map = {}
    for idx, item in enumerate(macro_cards_raw):
        val = item["value"]
        display = "N/A" if val is None else f"{val:,.2f}".rstrip("0").rstrip(".")
        macro_cards.append({"label": item["label"], "display": display, "change": item["change"]})
        ticker = MACRO_TICKERS[item["label"]]
        try:
            hist = yf.Ticker(ticker).history(period="30d")
            spark_map[str(idx)] = {
                "labels": [d.strftime("%m-%d") for d in hist.index],
                "values": [round(float(v), 2) for v in hist["Close"].tolist()],
            }
        except Exception:
            spark_map[str(idx)] = {"labels": [], "values": []}

    try:
        raw_texts = {
            "Portfolio A": form["portfolio_a"],
            "Portfolio B": form["portfolio_b"],
            "Portfolio C": form["portfolio_c"],
        }
        parsed_portfolios = {}
        for name, text in raw_texts.items():
            if text.strip():
                parsed_portfolios[name] = normalize_holdings(parse_portfolio_text(text))

        if not parsed_portfolios:
            raise ValueError("All portfolios are empty.")

        custom_bm = parse_custom_benchmark(form["custom_benchmark"])
        start_date = form["start_date"]
        rf = float(form["risk_free_rate"])
        pv = float(form["projection_value"])

        universe = set()
        for w in parsed_portfolios.values():
            universe.update(t for t in w if t != "CASH")
        for b in BENCHMARKS.values():
            universe.update(t for t in b if t != "CASH")
        if custom_bm:
            universe.update(t for t in custom_bm if t != "CASH")

        metadata = get_ticker_metadata(sorted(universe | {"CASH"}))
        prices = download_price_data(sorted(universe), start=start_date)
        returns = price_to_returns(prices)
        bm_returns = build_benchmark_returns(returns, custom_bm)

        perf_curves: Dict[str, pd.Series] = {}
        dd_curves: Dict[str, pd.Series] = {}
        port_ret_map: Dict[str, pd.Series] = {}

        for name, weights in parsed_portfolios.items():
            pr = build_portfolio_returns(weights, returns)
            curve = cumulative_curve(pr)
            dd = curve / curve.cummax() - 1
            perf_curves[name] = curve / curve.iloc[0] - 1 if len(curve) else pd.Series(dtype=float)
            dd_curves[name] = dd
            port_ret_map[name] = pr

            se = exposure_breakdown(weights, metadata, "sector")
            ae = exposure_breakdown(weights, metadata, "assetClass")
            conc = concentration_risk(weights)
            ann_ret = annualized_return(pr)
            ann_vol = annualized_volatility(pr)
            sh = sharpe_ratio(pr, rf)
            mdd = max_drawdown(pr)
            beta_spy, alpha_spy, corr_spy = beta_alpha(pr, bm_returns["SPY"])
            rscore = risk_score(ann_ret, ann_vol, sh, mdd)
            blurb = portfolio_summary_blurb(weights, metadata, ann_vol, sh, conc)

            portfolios[name] = {
                "weights": {k: fmt_pct(v) for k, v in weights.items()},
                "sector_exposure": {k: fmt_pct(v) for k, v in se.items()},
                "asset_exposure": {k: fmt_pct(v) for k, v in ae.items()},
                "top_holdings": [
                    {"ticker": h["ticker"], "name": h["name"], "weight": fmt_pct(h["weight"])}
                    for h in top_holdings(weights, metadata)
                ],
                "concentration": {"label": conc["label"]},
                "metrics": {
                    "cumulative_return": fmt_pct(curve.iloc[-1] - 1) if len(curve) else "0.0%",
                    "annual_return": fmt_pct(ann_ret),
                    "volatility": fmt_pct(ann_vol),
                    "sharpe": f"{sh:.2f}",
                    "max_drawdown": fmt_pct(mdd),
                    "beta": f"{beta_spy:.2f}",
                    "alpha": fmt_pct(alpha_spy),
                    "alpha_raw": alpha_spy,
                    "correlation": f"{corr_spy:.2f}",
                },
                "returns": pr,
                "curve": curve,
                "sector_raw": se,
                "factor_tilt": {k: fmt_pct(v) for k, v in factor_tilt(weights, metadata).items()},
                "macro_sensitivity": macro_sensitivity(weights, metadata),
                "blurb": blurb,
                "risk_score": rscore,
                "factor_tilt_raw": factor_tilt(weights, metadata),
                "dna": classify_portfolio_dna(
                    weights, metadata, beta_spy, ann_vol,
                    factor_tilt(weights, metadata),
                    macro_sensitivity(weights, metadata),
                ),
                "ai_commentary": get_ai_commentary(
                    name, weights,
                    {"annual_return": fmt_pct(ann_ret), "volatility": fmt_pct(ann_vol),
                     "sharpe": f"{sh:.2f}", "max_drawdown": fmt_pct(mdd), "beta": f"{beta_spy:.2f}"},
                    {k: fmt_pct(v) for k, v in se.items()},
                    macro_sensitivity(weights, metadata),
                    regime_label,
                ),
            }

        for bname, br in bm_returns.items():
            curve = cumulative_curve(br)
            perf_curves[bname] = curve / curve.iloc[0] - 1 if len(curve) else pd.Series(dtype=float)
            dd_curves[bname] = curve / curve.cummax() - 1 if len(curve) else pd.Series(dtype=float)

        comparison_headers = list(portfolios.keys()) + list(bm_returns.keys())
        comp_targets = {n: portfolios[n]["returns"] for n in portfolios}
        comp_targets.update(bm_returns)

        metrics_fns = [
            ("Cumulative Return", lambda s: fmt_pct(cumulative_curve(s).iloc[-1] - 1) if len(s) else "0.0%"),
            ("Annualized Return", lambda s: fmt_pct(annualized_return(s))),
            ("Volatility",        lambda s: fmt_pct(annualized_volatility(s))),
            ("Sharpe",            lambda s: f"{sharpe_ratio(s, rf):.2f}"),
            ("Max Drawdown",      lambda s: fmt_pct(max_drawdown(s))),
            ("Beta vs SPY",       lambda s: f"{beta_alpha(s, bm_returns['SPY'])[0]:.2f}"),
            ("Alpha vs SPY",      lambda s: fmt_pct(beta_alpha(s, bm_returns['SPY'])[1])),
            ("Correlation vs SPY",lambda s: f"{beta_alpha(s, bm_returns['SPY'])[2]:.2f}"),
        ]
        raw_rows = [
            {"metric": mn, "cells": [fn(comp_targets[h]) for h in comparison_headers]}
            for mn, fn in metrics_fns
        ]
        colored_rows = color_comparison(raw_rows)

        overlap_rows = []
        names = list(parsed_portfolios.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                overlap_rows.append({
                    "name": f"{names[i]} vs {names[j]}",
                    "value": fmt_pct(sector_overlap(parsed_portfolios[names[i]], parsed_portfolios[names[j]], metadata)),
                })

        # Monte Carlo on first portfolio
        first_name = list(portfolios.keys())[0]
        proj = monte_carlo_projection(
            pv,
            annualized_return(portfolios[first_name]["returns"]),
            annualized_volatility(portfolios[first_name]["returns"]),
        )
        projections = {
            "Base": {"median": fmt_money(proj["base"]["median"]), "p10": fmt_money(proj["base"]["p10"]),
                     "p90": fmt_money(proj["base"]["p90"]), "prob_loss": fmt_pct(proj["base"]["prob_loss"]),
                     "prob_loss_raw": proj["base"]["prob_loss"]},
            "Bull": {"median": fmt_money(proj["bull"]["median"]), "p10": fmt_money(proj["bull"]["p10"]),
                     "p90": fmt_money(proj["bull"]["p90"]), "prob_loss": fmt_pct(proj["bull"]["prob_loss"]),
                     "prob_loss_raw": proj["bull"]["prob_loss"]},
            "Bear": {"median": fmt_money(proj["bear"]["median"]), "p10": fmt_money(proj["bear"]["p10"]),
                     "p90": fmt_money(proj["bear"]["p90"]), "prob_loss": fmt_pct(proj["bear"]["prob_loss"]),
                     "prob_loss_raw": proj["bear"]["prob_loss"]},
        }
        projection_series = json.dumps({
            "labels": [f"M{i+1}" for i in range(len(proj["base"]["sample_path"]))],
            "base": [round(x, 0) for x in proj["base"]["sample_path"]],
            "bull": [round(x, 0) for x in proj["bull"]["sample_path"]],
            "bear": [round(x, 0) for x in proj["bear"]["sample_path"]],
        })

        # Rolling correlations
        first_ret = portfolios[first_name]["returns"]
        rolling_corr_json = make_series_json(
            {"vs SPY": rolling_correlation(first_ret, bm_returns["SPY"]),
             "vs QQQ": rolling_correlation(first_ret, bm_returns["QQQ"])},
            pct=False,
        )

        # Correlation matrix
        corr_df = pd.DataFrame(port_ret_map).dropna()
        if not corr_df.empty:
            cm = corr_df.corr()
            corr_matrix_json = json.dumps({
                "labels": list(cm.columns),
                "matrix": [[round(float(v), 3) for v in row] for row in cm.values],
            })

        perf_series = make_series_json(perf_curves, pct=True)
        drawdown_series = make_series_json(dd_curves, pct=True)
        allocation_a = json.dumps(portfolios[first_name]["weights"])
        sector_a = json.dumps(portfolios[first_name]["sector_exposure"])
        macro_sens = portfolios[first_name]["macro_sensitivity"]

    except Exception as e:
        error = str(e)

    return render_template_string(
        TEMPLATE,
        updated_at=datetime.now().strftime("%A, %B %d %Y • %I:%M %p"),
        regime_label=regime_label,
        regime_detail=regime_detail,
        fear_greed=fear_greed,
        macro_cards=macro_cards,
        ticker_items=macro_cards,
        global_heatmap=global_heatmap,
        sector_rotation=sector_rotation,
        news_items=news_items,
        form=form,
        error=error,
        portfolios=portfolios,
        comparison_headers=comparison_headers,
        colored_rows=colored_rows,
        overlap_rows=overlap_rows,
        projections=projections,
        macro_sens=macro_sens,
        perf_series=perf_series,
        drawdown_series=drawdown_series,
        allocation_a=allocation_a,
        sector_a=sector_a,
        projection_series=projection_series,
        rolling_corr=rolling_corr_json,
        corr_matrix_json=corr_matrix_json,
        spark_data=json.dumps(spark_map),
        linkedin_url=LINKEDIN_URL,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5001)
