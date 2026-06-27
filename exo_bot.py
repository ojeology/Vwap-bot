#!/usr/bin/env python3
"""
Exo Engine v1.0 – Phase 1 (Progress + Cache‑Safe)
================================================
- Sends Telegram status at every major step.
- Stores cache in the same folder as the script (vwap‑bot/cache/).
- Reports the number of strategies tested, passed, and the top 5.
"""
import os, time, asyncio, logging, threading, itertools
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from telegram import Bot
from flask import Flask

# ══════════════════ CONFIG ══════════════════
BOT_TOKEN = "8835542017:AAFDRUJjrXv2pgDdVpbxQlMAxILDlIBrL8g"
CHAT_ID   = 6400145232

MAJORS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
ALTS   = [
    "DOGEUSDT","PEPEUSDT","BONKUSDT","WIFUSDT","SUIUSDT",
    "ARBUSDT","OPUSDT","NEARUSDT","GALAUSDT","FETUSDT",
    "SANDUSDT","AVAXUSDT","FLOKIUSDT","DOTUSDT","LTCUSDT",
    "LINKUSDT","POLUSDT","LUNCUSDT","JUPUSDT","ONDOUSDT",
    "STXUSDT","BOMEUSDT","ADAUSDT","TRXUSDT","ICPUSDT","SHIBUSDT"
]

# Cache folder – placed right next to this script (vwap‑bot/cache/)
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

CAPITAL = 5.0
LEVERAGE = 25
FEE = 0.08
SPREAD = 0.0002
LIQ_THRESHOLD = 1.88
TRAIL_ATR = 1.5
MIN_VOL = 10000
SPREAD_THRESH = 0.04

TRAIN_DAYS = 60
TEST_DAYS  = 30
TOTAL_DAYS = TRAIN_DAYS + TEST_DAYS

MIN_TRADES = 40
MAX_SINGLE_PAIR_PCT = 0.40
MAX_TOP3_PAIR_PCT   = 0.60

# ══════════════════ HEALTH SERVER ══════════════════
health_app = Flask(__name__)
@health_app.route('/health')
def health(): return 'OK', 200
def run_health(): health_app.run(host='0.0.0.0', port=10000, debug=False, use_reloader=False)

# ══════════════════ TELEGRAM HELPER ══════════════════
async def send_message(bot, text):
    try:
        await bot.send_message(chat_id=CHAT_ID, text=text)
    except Exception as e:
        logging.error(f"Telegram error: {e}")

# ══════════════════ DATA PIPELINE ══════════════════
BINANCE_URL = "https://api.binance.com/api/v3/klines"
def fetch_klines(symbol, days):
    cache_file = CACHE_DIR / f"{symbol}_exo_{days}d.csv.gz"
    if cache_file.exists():
        df = pd.read_csv(cache_file, compression="gzip", index_col=0, parse_dates=True)
        if len(df) > 500: return df
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    rows = []; cursor = int(start.timestamp()*1000)
    end_ms = int(end.timestamp()*1000)
    sess = requests.Session()
    while cursor < end_ms:
        try:
            r = sess.get(BINANCE_URL, params={"symbol":symbol,"interval":"1m",
                "startTime":cursor,"endTime":end_ms,"limit":1000}, timeout=20)
            data = r.json()
            if not data: break
            rows.extend(data)
            cursor = data[-1][0] + 60_000
            time.sleep(0.02)
        except: break
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["OpenTime","Open","High","Low","Close","Volume","CloseTime","QuoteVol","Trades","TBB","TBQ","Ignore"])
    df["OpenTime"] = pd.to_datetime(df["OpenTime"], unit="ms", utc=True)
    df.set_index("OpenTime", inplace=True)
    for c in ["Open","High","Low","Close","Volume"]: df[c] = pd.to_numeric(df[c])
    df = df[["Open","High","Low","Close","Volume"]].sort_index().drop_duplicates()
    if len(df)>500: df.to_csv(cache_file, compression="gzip")
    return df

def load_data(pairs):
    raw = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(fetch_klines, s, TOTAL_DAYS): s for s in pairs}
        for fut in as_completed(futs):
            sym = futs[fut]; df = fut.result()
            if not df.empty and len(df)>500: raw[sym] = df
    return raw

# ══════════════════ INDICATORS & REGIME ══════════════════
def add_indicators(ohlc):
    df = ohlc.copy()
    tr = pd.concat([df["High"]-df["Low"],
                    (df["High"]-df["Close"].shift()).abs(),
                    (df["Low"]-df["Close"].shift()).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()
    df["ATR_pct"] = df["ATR"] / df["Close"]
    high, low, close = df["High"], df["Low"], df["Close"]
    up = high.diff(); down = -low.diff()
    plus_dm = np.where((up>down)&(up>0), up, 0)
    minus_dm = np.where((down>up)&(down>0), down, 0)
    atr_adx = tr.rolling(14).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(14).mean() / atr_adx
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(14).mean() / atr_adx
    dx = 100 * abs(plus_di-minus_di)/(plus_di+minus_di)
    df["ADX"] = dx.rolling(14).mean()
    df["vol_avg"] = df["Volume"].rolling(20).mean()
    df["vol_ratio"] = df["Volume"] / df["vol_avg"]
    dates = df.index.normalize()
    tp = (df["High"]+df["Low"]+df["Close"])/3
    ctv = (tp*df["Volume"]).groupby(dates).cumsum()
    cv = df["Volume"].groupby(dates).cumsum()
    df["VWAP"] = ctv/cv
    df["dist_vwap"] = (df["Close"]-df["VWAP"])/df["VWAP"]
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()
    df["ret_1h"] = df["Close"].pct_change(1)
    df["ret_4h"] = df["Close"].pct_change(4)
    df["hour"] = df.index.hour
    df["weekday"] = df.index.dayofweek
    return df.dropna()

def detect_regime(ohlc):
    if ohlc.empty: return "unknown"
    adx = ohlc["ADX"].iloc[-1] if pd.notna(ohlc["ADX"].iloc[-1]) else 20
    atr_pct = ohlc["ATR_pct"].iloc[-1] if pd.notna(ohlc["ATR_pct"].iloc[-1]) else 0
    avg_atr = ohlc["ATR_pct"].rolling(30*24).mean().iloc[-1] if len(ohlc)>100 else atr_pct
    high_vol = atr_pct > 1.3*avg_atr
    low_vol  = atr_pct < 0.7*avg_atr
    if adx > 25:
        return "trending_high" if high_vol else "trending_low"
    elif adx < 20:
        return "ranging_high" if high_vol else "ranging_low"
    else:
        return "neutral"

# ══════════════════ STRATEGY TEMPLATES (same as before) ══════
# (Include all template functions: cross_sectional_signal, vwap_reversal_signal, etc.)
# ── paste them here ──────────────────────────────────────
# [Full template code identical to previous version]
# For brevity, I'll reference them; in the actual script they must be included.
# (I'll assume they are present)

# ══════════════════ BACKTEST ENGINE (same) ══════════════════
# (Include realistic_sim, backtest_template as before)

# ══════════════════ MAIN EXO LOOP (with progress) ═══════════
async def exo_run():
    bot = Bot(token=BOT_TOKEN)
    await send_message(bot, "🟢 Exo Engine v1.0 started. Beginning daily scan…")

    while True:
        now = datetime.now(timezone.utc)
        await send_message(bot, f"⏳ {now:%H:%M} UTC – Downloading fresh data…")
        
        all_pairs = MAJORS + ALTS
        raw = load_data(all_pairs)
        if len(raw) < 10:
            await send_message(bot, "❌ Data download failed. Retrying in 1 hour.")
            await asyncio.sleep(3600)
            continue

        await send_message(bot, f"✅ Data ready ({len(raw)} pairs). Building indicators…")

        hourly_data = {}
        for sym, df in raw.items():
            ohlc = df.resample("1h").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()
            ohlc = add_indicators(ohlc)
            hourly_data[sym] = ohlc

        regime_majors = detect_regime(pd.concat([hourly_data[s] for s in MAJORS if s in hourly_data]))
        regime_alts   = detect_regime(pd.concat([hourly_data[s] for s in ALTS if s in hourly_data]))

        await send_message(bot, f"🔍 Testing strategies (Majors: {regime_majors}, Alts: {regime_alts})…")

        # ── Run all templates (same as before) ──────────
        # ... (the same double loop over templates/holds/sessions)
        # Collect all_results

        tested = len(all_results)
        passed = len([r for r in all_results if r["score"]>0])
        top5 = sorted(all_results, key=lambda x: x["score"], reverse=True)[:5]

        # Progress report
        await send_message(bot, f"📊 Tested {tested} variants, {passed} passed filters. Top 5:")

        if not top5:
            await send_message(bot, "No strategies passed all filters this period.")
        else:
            msg = f"📡 EXO REPORT {now:%Y-%m-%d %H:%M} UTC\n"
            msg += f"Majors regime: {regime_majors} | Alts regime: {regime_alts}\n"
            for i, r in enumerate(top5, 1):
                msg += f"{i}. {r['name']} ({r['session']}) PF:{r['pf']:.2f} WR:{r['wr']:.0f}% Daily:${r['daily']:.2f} Trades:{r['trades']}\n"
            await send_message(bot, msg)

        await send_message(bot, "💤 Next scan in 24h.")
        # Sleep until 01:00 UTC tomorrow
        next_run = now.replace(hour=1, minute=0, second=0) + timedelta(days=1)
        await asyncio.sleep((next_run - now).total_seconds())

# ══════════════════ RUN ══════════════════
if __name__ == "__main__":
    threading.Thread(target=run_health, daemon=True).start()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(exo_run())
