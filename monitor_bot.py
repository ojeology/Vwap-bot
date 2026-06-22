#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VWAP Reversal Monitor Bot — Cloud Edition (Render + UptimeRobot)
"""
import time, asyncio, json, logging, threading, itertools, os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from telegram import Bot
from flask import Flask

# ═════════════════ CREDENTIALS ═════════════════
BOT_TOKEN = "8835542017:AAFDRUJjrXv2pgDdVpbxQlMAxILDlIBrL8g"
CHAT_ID   = 6400145232

# ═════════════════ HEALTH SERVER ═════════════════
health_app = Flask(__name__)

@health_app.route('/health')
def health():
    return 'OK', 200

def run_health_server():
    port = int(os.environ.get('PORT', 10000))
    health_app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

# ═════════════════ STRATEGY CONFIG ═════════════════
PAIRS = [
    "PEPEUSDT","BONKUSDT","WIFUSDT","SUIUSDT","ARBUSDT","OPUSDT",
    "GALAUSDT","FETUSDT","SANDUSDT","MEMEUSDT","AVAXUSDT",
    "BNBUSDT","DOTUSDT","LTCUSDT","LINKUSDT","INJUSDT",
    "POLUSDT","STXUSDT","BOMEUSDT","ETHUSDT","ICPUSDT"
]
SCAN_INTERVAL_MINUTES = 10
LOOKBACK_DAYS = 30
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
CACHE_TTL_H = 720

BLOCKED_HOURS = {0,1,3,4,6,9,10,14,18,19,20,21}
BODY_PCT_MIN = 0.50
RSI_LONG_MAX = 40
RSI_SHORT_MIN = 60
VOL_MULT = 1.2
RANGE_CAP = True
MAX_RANGE_ATR = 3.0

TP_GRID = [0.012,0.015,0.018,0.020,0.025,0.030,0.035,0.040]
SL_GRID = [0.007,0.010,0.012,0.015,0.018]

MEXC_URL = "https://api.mexc.com/api/v3/klines"
API_TIMEOUT = 30
API_RETRIES = 2

# ═════════════════ DATA FUNCTIONS ═════════════════
def _parse(rows):
    if not rows: return pd.DataFrame()
    nc = len(rows[0])
    base = ["OpenTime","Open","High","Low","Close","Volume","CloseTime","QuoteVol"]
    cols = (base + [f"x{i}" for i in range(8,nc)])[:nc]
    df = pd.DataFrame(rows, columns=cols)
    df["OpenTime"] = pd.to_datetime(df["OpenTime"], unit="ms", utc=True)
    df.set_index("OpenTime", inplace=True)
    for c in ["Open","High","Low","Close","Volume"]:
        df[c] = pd.to_numeric(df[c])
    return df[["Open","High","Low","Close","Volume"]].sort_index().drop_duplicates()

def _cache_path(symbol, days):
    return CACHE_DIR / f"{symbol}_{days}d.csv.gz"

def fetch_live_data(symbol):
    """Get the last ~5 hours of 1m candles for live indicators."""
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = end_ms - 300 * 60_000   # 5 hours
    rows = []
    cursor = start_ms
    sess = requests.Session()
    sess.headers.update({"User-Agent": "Mozilla/5.0"})
    try:
        while cursor < end_ms:
            data = None
            for attempt in range(API_RETRIES):
                try:
                    r = sess.get(MEXC_URL, params={
                        "symbol": symbol, "interval": "1m",
                        "startTime": cursor, "endTime": end_ms,
                        "limit": 500}, timeout=API_TIMEOUT)
                    r.raise_for_status()
                    data = r.json()
                    break
                except Exception:
                    if attempt < API_RETRIES: time.sleep(2)
            if data is None or not data: break
            rows.extend(data)
            last = data[-1][0]
            if last <= cursor: break
            cursor = last + 60_000
            if len(data) < 500: break
            time.sleep(0.05)
        return _parse(rows)
    except Exception as e:
        logging.error(f"Live fetch {symbol}: {e}")
        return pd.DataFrame()

def load_historical_data(symbol, days):
    """Load cached historical data (30 days) for indicator warmup and TP/SL."""
    cp = _cache_path(symbol, days)
    if cp.exists():
        try:
            return pd.read_csv(cp, compression="gzip", index_col=0, parse_dates=True)
        except:
            pass
    return pd.DataFrame()

# ═════════════════ INDICATORS ═════════════════
def ema_series(s, n): return s.ewm(span=n, adjust=False).mean()

def add_indicators(df):
    df = df.copy()
    d = df["Close"].diff()
    g = d.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    df["rsi"] = 100 - 100 / (1 + g / l.replace(0, np.nan))
    tr = pd.concat([(df["High"]-df["Low"]),
                    (df["High"]-df["Close"].shift()).abs(),
                    (df["Low"]-df["Close"].shift()).abs()], axis=1).max(1)
    df["atr"] = tr.rolling(14).mean()
    df["vol20"] = df["Volume"].rolling(20).mean()
    dates = df.index.normalize()
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    ctv = (tp * df["Volume"]).groupby(dates).cumsum()
    cv = df["Volume"].groupby(dates).cumsum()
    vwap = ctv / cv
    sq = (((tp - vwap) ** 2) * df["Volume"]).groupby(dates).cumsum()
    std = (sq / cv).apply(np.sqrt)
    df["vwap"] = vwap
    df["vwap_2up"] = vwap + 2 * std
    df["vwap_2dn"] = vwap - 2 * std
    df["body_pct"] = (df["Close"] - df["Open"]).abs() / df["atr"]
    return df.dropna()

# ═════════════════ SIGNAL & SIMULATION (for TP/SL) ═══
def get_signals(df):
    c, o = df["Close"], df["Open"]
    t2lo = (df["Low"] <= df["vwap_2dn"]) & (c > df["vwap_2dn"])
    t2hi = (df["High"] >= df["vwap_2up"]) & (c < df["vwap_2up"])
    strong = df["body_pct"] > BODY_PCT_MIN
    rsi_buy = df["rsi"] < RSI_LONG_MAX
    rsi_sell = df["rsi"] > RSI_SHORT_MIN
    buy = t2lo & (c > o) & strong & rsi_buy
    sell = t2hi & (c < o) & strong & rsi_sell
    buy &= df["Volume"] > VOL_MULT * df["vol20"]
    sell &= df["Volume"] > VOL_MULT * df["vol20"]
    if RANGE_CAP:
        range_ok = (df["High"] - df["Low"]) < MAX_RANGE_ATR * df["atr"]
        buy &= range_ok
        sell &= range_ok
    s = pd.Series(0, index=df.index)
    s[buy] = 1
    s[sell] = -1
    s[df.index.hour.isin(BLOCKED_HOURS)] = 0
    return s

def quick_simulate(df, sigs, tp_pct, sl_pct):
    ca = df["Close"].values; ha = df["High"].values
    la = df["Low"].values;   oa = df["Open"].values
    sa = sigs.values;        n = len(df)
    pnl = 0.0
    i = 0
    while i < n - 1:
        if sa[i] == 0:
            i += 1; continue
        ei = i + 1
        if ei >= n: break
        entry = oa[ei]; d = int(sa[i])
        tp_p = entry * (1 + d * tp_pct)
        sl_p = entry * (1 - d * sl_pct)
        rh = ha[ei:]; rl = la[ei:]
        tp_h = np.nonzero((rh >= tp_p) if d==1 else (rl <= tp_p))[0]
        sl_h = np.nonzero((rl <= sl_p) if d==1 else (rh >= sl_p))[0]
        ti = tp_h[0] if len(tp_h) else n
        si = sl_h[0] if len(sl_h) else n
        if ti <= si and ti < n:
            j = ei+ti; ep = tp_p
        elif si < ti and si < n:
            j = ei+si; ep = sl_p
        else:
            j = n-1; ep = ca[j]
        pnl += 5 * d * (ep - entry) / entry * 20 - 0.08
        i = j + 1
    return pnl

def optimize_tp_sl(symbol):
    """Find best TP/SL for this symbol using last 15 days of cached data."""
    hist = load_historical_data(symbol, LOOKBACK_DAYS)
    if hist.empty:
        return None, None
    hist = add_indicators(hist)
    train_start = hist.index.max() - pd.Timedelta(days=15)
    df_train = hist[hist.index >= train_start]
    if len(df_train) < 200:
        return None, None
    sigs_train = get_signals(df_train)
    best = {"pnl": -1e9, "tp": None, "sl": None}
    for tp, sl in itertools.product(TP_GRID, SL_GRID):
        if tp <= sl: continue
        pnl = quick_simulate(df_train, sigs_train, tp, sl)
        if pnl > best["pnl"]:
            best = {"pnl": pnl, "tp": tp, "sl": sl}
    if best["tp"] is None:
        return 0.025, 0.012   # fallback
    return best["tp"], best["sl"]

def check_signal(df):
    if df.empty: return 0, {"error": "no data"}
    latest = df.iloc[-1]
    c = latest["Close"]; o = latest["Open"]
    vol_ratio = latest["Volume"] / latest["vol20"] if latest["vol20"] else 0
    info = {
        "price": round(c, 6),
        "rsi": round(latest["rsi"], 1),
        "body/atr": round(latest["body_pct"], 2),
        "vol_ratio": round(vol_ratio, 2),
        "hour": latest.name.hour,
        "band_touch": "none"
    }
    if info["hour"] in BLOCKED_HOURS:
        return 0, {**info, "fail": "blocked hour"}

    touch_low = latest["Low"] <= latest["vwap_2dn"]
    close_inside_low = c > latest["vwap_2dn"]
    touch_high = latest["High"] >= latest["vwap_2up"]
    close_inside_high = c < latest["vwap_2up"]

    buy_sig = touch_low and close_inside_low and c > o and latest["body_pct"] > BODY_PCT_MIN and latest["rsi"] < RSI_LONG_MAX
    sell_sig = touch_high and close_inside_high and c < o and latest["body_pct"] > BODY_PCT_MIN and latest["rsi"] > RSI_SHORT_MIN

    if buy_sig:
        info["band_touch"] = "lower"
        if vol_ratio < VOL_MULT:
            info["fail"] = f"volume {vol_ratio:.2f}<{VOL_MULT}"
            return 0, info
        if RANGE_CAP and (latest["High"] - latest["Low"]) > MAX_RANGE_ATR * latest["atr"]:
            info["fail"] = "range cap"
            return 0, info
        return 1, info
    if sell_sig:
        info["band_touch"] = "upper"
        if vol_ratio < VOL_MULT:
            info["fail"] = f"volume {vol_ratio:.2f}<{VOL_MULT}"
            return 0, info
        if RANGE_CAP and (latest["High"] - latest["Low"]) > MAX_RANGE_ATR * latest["atr"]:
            info["fail"] = "range cap"
            return 0, info
        return -1, info

    info["fail"] = "no band setup"
    return 0, info

# ═════════════════ MAIN BOT ═════════════════
async def monitor():
    bot = Bot(token=BOT_TOKEN)
    await bot.send_message(chat_id=CHAT_ID, text="🟢 Cloud bot starting… optimizing TP/SL…")

    # Pre-compute TP/SL for all pairs using cached data
    tp_sl_map = {}
    for sym in PAIRS:
        tp, sl = optimize_tp_sl(sym)
        if tp:
            tp_sl_map[sym] = {"TP%": round(tp*100,2), "SL%": round(sl*100,2)}
        else:
            tp_sl_map[sym] = {"TP%": 0.0, "SL%": 0.0}
    logging.info("TP/SL table ready.")
    await bot.send_message(chat_id=CHAT_ID, text=f"✅ TP/SL ready for {len(tp_sl_map)} pairs.\nStarting scans every {SCAN_INTERVAL_MINUTES} min.")

    while True:
        scan_start = datetime.now(timezone.utc)
        status_lines = []
        trade_alerts = []

        for sym in PAIRS:
            live = fetch_live_data(sym)
            hist = load_historical_data(sym, LOOKBACK_DAYS)
            if not hist.empty:
                common = hist.index.intersection(live.index)
                hist = hist[~hist.index.isin(common)]
                df = pd.concat([hist, live]).sort_index()
                df = df.iloc[-2000:]
            else:
                df = live
            if df.empty:
                status_lines.append(f"{sym}: ❌ no data")
                continue

            df = add_indicators(df)
            signal, info = check_signal(df)
            tp_sl = tp_sl_map.get(sym, {"TP%": 0, "SL%": 0})
            price = info.get("price", 0)
            fail = info.get("fail", "ok")
            line = (f"{sym}: ${price:.6f} | RSI:{info['rsi']} | "
                    f"Body/ATR:{info['body/atr']} | Vol:{info['vol_ratio']}x | "
                    f"Band:{info.get('band_touch','?')} | {fail}")
            status_lines.append(line)

            if signal == 1:
                tp_price = price * (1 + tp_sl["TP%"]/100)
                sl_price = price * (1 - tp_sl["SL%"]/100)
                trade_alerts.append(
                    f"🟢 **BUY {sym}** at ${price:.6f}\n"
                    f"TP: ${tp_price:.6f} (+{tp_sl['TP%']}%)\n"
                    f"SL: ${sl_price:.6f} (-{tp_sl['SL%']}%)\n"
                    f"Vol: {info['vol_ratio']}x | RSI: {info['rsi']}"
                )
            elif signal == -1:
                tp_price = price * (1 - tp_sl["TP%"]/100)
                sl_price = price * (1 + tp_sl["SL%"]/100)
                trade_alerts.append(
                    f"🔴 **SELL {sym}** at ${price:.6f}\n"
                    f"TP: ${tp_price:.6f} (+{tp_sl['TP%']}%)\n"
                    f"SL: ${sl_price:.6f} (-{tp_sl['SL%']}%)\n"
                    f"Vol: {info['vol_ratio']}x | RSI: {info['rsi']}"
                )

        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        heartbeat = f"📡 **Heartbeat** {now}\n\n" + "\n".join(status_lines)
        if not trade_alerts:
            heartbeat += "\n\n❌ No valid trades this scan."
        else:
            heartbeat += f"\n\n🔥 {len(trade_alerts)} trade signal(s) detected!"

        await bot.send_message(chat_id=CHAT_ID, text=heartbeat, parse_mode="Markdown")

        for alert in trade_alerts:
            await bot.send_message(chat_id=CHAT_ID, text=alert, parse_mode="Markdown")

        elapsed = (datetime.now(timezone.utc) - scan_start).total_seconds()
        sleep_time = max(0, SCAN_INTERVAL_MINUTES * 60 - elapsed)
        logging.info(f"Sleeping {sleep_time:.0f}s...")
        await asyncio.sleep(sleep_time)

# ═════════════════ RUN ═════════════════
if __name__ == "__main__":
    threading.Thread(target=run_health_server, daemon=True).start()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(monitor())
