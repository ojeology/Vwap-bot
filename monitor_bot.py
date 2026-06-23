#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VWAP Reversal Bot – Institutional Grade (Thread‑Safe, Daily Backtest Report)
"""
import time, asyncio, logging, threading, itertools, os
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
from pathlib import Path

from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from flask import Flask

# ════════════════ CREDENTIALS (hardcoded) ════════════════
BOT_TOKEN = "8835542017:AAFDRUJjrXv2pgDdVpbxQlMAxILDlIBrL8g"
CHAT_ID   = 6400145232

# ════════════════ HEALTH SERVER ════════════════
health_app = Flask(__name__)

@health_app.route('/health')
def health():
    return 'OK', 200

def run_health_server():
    port = int(os.environ.get('PORT', 10000))
    health_app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

# ════════════════ CONFIG ════════════════
ALL_PAIRS = [
    "XRPUSDT","DOGEUSDT","PEPEUSDT","BONKUSDT","WIFUSDT",
    "SUIUSDT","ARBUSDT","OPUSDT","NEARUSDT","GALAUSDT",
    "FETUSDT","SANDUSDT","MEMEUSDT","AVAXUSDT","FLOKIUSDT",
    "BNBUSDT","DOTUSDT","LTCUSDT","LINKUSDT","INJUSDT",
    "POLUSDT","LUNCUSDT","JUPUSDT","ONDOUSDT","STXUSDT",
    "BOMEUSDT","ETHUSDT","ADAUSDT","TRXUSDT","ICPUSDT",
    "SOLUSDT","SHIBUSDT"
]
PAIRS = ALL_PAIRS[:21]   # initial set, will be updated daily

SCAN_INTERVAL_MINUTES = 10
LOOKBACK_DAYS = 30
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

BLOCKED_HOURS = {0,1,3,4,6,9,10,14,18,19,20,21}
BODY_PCT_MIN = 0.50
RSI_LONG_MAX = 40
RSI_SHORT_MIN = 60
VOL_MULT = 1.2
RANGE_CAP = True
MAX_RANGE_ATR = 3.0

SL_COOLDOWN = 5          # minutes
DAILY_LOSS_LIMIT = 0.08  # 8% of starting balance

USE_EMA_REGIME = True

STARTING_BALANCE = 100.0
RISK_PER_TRADE = 0.01    # 1% of equity
MAX_CONCURRENT_TRADES = 3

SECTORS = {
    "meme": ["PEPEUSDT","BONKUSDT","SHIBUSDT","FLOKIUSDT","WIFUSDT","BOMEUSDT"],
    "l1":   ["ETHUSDT","SOLUSDT","BNBUSDT","ADAUSDT","AVAXUSDT","ICPUSDT"],
    "defi": ["UNIUSDT","LINKUSDT","INJUSDT","ONDOUSDT"],
    "mid":  ["XRPUSDT","DOGEUSDT","TRXUSDT","LTCUSDT","DOTUSDT","POLUSDT","SANDUSDT","MEMEUSDT","GALAUSDT","NEARUSDT","FETUSDT","OPUSDT","ARBUSDT","SUIUSDT","STXUSDT","JUPUSDT","LUNCUSDT"]
}
MAX_SECTOR_EXPOSURE = 2

MEXC_URL = "https://api.mexc.com/api/v3/klines"
API_TIMEOUT = 30
API_RETRIES = 2

# ════════════════ GLOBAL STATE (thread‑safe) ════════════════
cached_data = {}
cached_indicators = {}
last_hist_fetch = {}

trade_log = []
signal_log = []
signal_counter = 0
open_trades = {}
cooldowns = {}
daily_pnl = 0.0
sector_counts = {}

state_lock = asyncio.Lock()   # protects open_trades, cooldowns, sector_counts, trade_log, signal_log

OPTIMIZED_TP_SL = {}

# ════════════════ DATA HELPERS ════════════════
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

def fetch_klines(symbol, start_ms, end_ms, limit=500):
    rows = []
    cursor = start_ms
    sess = requests.Session()
    sess.headers.update({"User-Agent": "Mozilla/5.0"})
    while cursor < end_ms:
        data = None
        for attempt in range(API_RETRIES):
            try:
                r = sess.get(MEXC_URL, params={
                    "symbol": symbol, "interval": "1m",
                    "startTime": cursor, "endTime": end_ms,
                    "limit": limit}, timeout=API_TIMEOUT)
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
        if len(data) < limit: break
        time.sleep(0.05)
    return _parse(rows)

def fetch_live_data(symbol):
    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end_ms = int(now.timestamp() * 1000)
    start_ms = int((now - timedelta(hours=5)).timestamp() * 1000)
    df_today = fetch_klines(symbol, int(today_start.timestamp()*1000), end_ms)
    df_recent = fetch_klines(symbol, start_ms, end_ms)
    if df_today.empty:
        return df_recent
    common = df_recent.index.intersection(df_today.index)
    return pd.concat([df_today[~df_today.index.isin(common)], df_recent]).sort_index()

def load_historical_data(symbol, days):
    now = datetime.now(timezone.utc)
    last = last_hist_fetch.get(symbol)
    if last and (now - last).days < 1:
        return cached_data.get(symbol, pd.DataFrame())
    cp = _cache_path(symbol, days)
    if cp.exists():
        try:
            df = pd.read_csv(cp, compression="gzip", index_col=0, parse_dates=True)
            last_hist_fetch[symbol] = now
            return df
        except: pass
    start_ms = int((now - timedelta(days=days)).timestamp() * 1000)
    end_ms = int(now.timestamp() * 1000)
    df = fetch_klines(symbol, start_ms, end_ms)
    if len(df) > 500:
        df.to_csv(cp, compression="gzip")
    last_hist_fetch[symbol] = now
    return df

# ════════════════ INDICATORS ════════════════
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
    df["ema50"] = df["Close"].ewm(span=50).mean()
    df["ema200"] = df["Close"].ewm(span=200).mean()
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

def update_pair_cache(sym):
    try:
        live = fetch_live_data(sym)
        if live.empty: return sym, None
        if sym in cached_data:
            old = cached_data[sym]
            common = old.index.intersection(live.index)
            old = old[~old.index.isin(common)]
            df = pd.concat([old, live]).sort_index()
        else:
            df = live
        cached_data[sym] = df.tail(2000)
        now = datetime.now(timezone.utc)
        today = df[df.index >= now.replace(hour=0, minute=0, second=0, microsecond=0)]
        if len(today) >= 200:
            ind = add_indicators(today)
        else:
            ind = add_indicators(df.tail(500))
        cached_indicators[sym] = ind
        return sym, ind
    except Exception as e:
        logging.error(f"update_pair_cache {sym}: {e}")
        return sym, None

# ════════════════ REGIME & RISK CHECKS ════════════════
def volatility_regime_ok(df):
    atr = df["atr"].iloc[-1]
    hist = df["atr"].tail(100)
    if len(hist) < 50:
        return True
    low, high = np.percentile(hist, [25, 85])
    return low <= atr <= high

def get_sector(sym):
    for sec, pairs in SECTORS.items():
        if sym in pairs:
            return sec
    return "mid"

async def can_trade(sym):
    async with state_lock:
        if len(open_trades) >= MAX_CONCURRENT_TRADES:
            return False
        sec = get_sector(sym)
        if sector_counts.get(sec, 0) >= MAX_SECTOR_EXPOSURE:
            return False
        return True

def get_equity():
    """Single source of truth: starting balance + all logged PnL"""
    return STARTING_BALANCE + sum(t.get("pnl", 0) for t in trade_log)

def get_latest_price(sym):
    """Consistent price source: cached indicators first, else None"""
    df = cached_indicators.get(sym)
    if df is not None and not df.empty:
        return df.iloc[-1]["Close"]
    return None

# ════════════════ SIGNAL LOGIC ════════════════
def check_signal(df, sym):
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
    now_ts = latest.name
    if info["hour"] in BLOCKED_HOURS:
        return 0, {**info, "fail": "blocked hour"}
    # cooldown check will be done outside with lock
    if not volatility_regime_ok(df):
        return 0, {**info, "fail": "volatility regime"}

    touch_low = latest["Low"] <= latest["vwap_2dn"]
    close_inside_low = c > latest["vwap_2dn"]
    touch_high = latest["High"] >= latest["vwap_2up"]
    close_inside_high = c < latest["vwap_2up"]

    buy_sig = touch_low and close_inside_low and c > o and latest["body_pct"] > BODY_PCT_MIN and latest["rsi"] < RSI_LONG_MAX
    sell_sig = touch_high and close_inside_high and c < o and latest["body_pct"] > BODY_PCT_MIN and latest["rsi"] > RSI_SHORT_MIN

    if USE_EMA_REGIME:
        ema50 = latest.get("ema50", 0)
        ema200 = latest.get("ema200", 0)
        if ema50 and ema200:
            trend_up = ema50 > ema200
            trend_down = ema50 < ema200
            if buy_sig and not trend_up:
                return 0, {**info, "fail": "regime (no long in downtrend)"}
            if sell_sig and not trend_down:
                return 0, {**info, "fail": "regime (no short in uptrend)"}

    if buy_sig:
        info["band_touch"] = "lower"
        if vol_ratio < VOL_MULT:
            return 0, {**info, "fail": f"volume {vol_ratio:.2f}<{VOL_MULT}"}
        if RANGE_CAP and (latest["High"] - latest["Low"]) > MAX_RANGE_ATR * latest["atr"]:
            return 0, {**info, "fail": "range cap"}
        return 1, info
    if sell_sig:
        info["band_touch"] = "upper"
        if vol_ratio < VOL_MULT:
            return 0, {**info, "fail": f"volume {vol_ratio:.2f}<{VOL_MULT}"}
        if RANGE_CAP and (latest["High"] - latest["Low"]) > MAX_RANGE_ATR * latest["atr"]:
            return 0, {**info, "fail": "range cap"}
        return -1, info

    return 0, {**info, "fail": "no band setup"}

def condition_score(df):
    if df.empty: return 0, 0, False, "none", {}
    latest = df.iloc[-1]
    c = latest["Close"]; o = latest["Open"]
    vol_ratio = latest["Volume"] / latest["vol20"] if latest["vol20"] != 0 else 0
    rsi = latest["rsi"]
    body = latest["body_pct"]
    hour_ok = latest.name.hour not in BLOCKED_HOURS
    band_touch = "none"
    if latest["Low"] <= latest["vwap_2dn"] and c > latest["vwap_2dn"]:
        band_touch = "LOWER"
    elif latest["High"] >= latest["vwap_2up"] and c < latest["vwap_2up"]:
        band_touch = "UPPER"

    rsi_ok = rsi < RSI_LONG_MAX or rsi > RSI_SHORT_MIN
    body_ok = body > BODY_PCT_MIN
    if rsi < RSI_LONG_MAX:
        dir_ok = c > o
    elif rsi > RSI_SHORT_MIN:
        dir_ok = c < o
    else:
        dir_ok = False
    vol_ok = vol_ratio >= VOL_MULT
    range_ok = (latest["High"] - latest["Low"]) < MAX_RANGE_ATR * latest["atr"] if RANGE_CAP else True
    max_score = 5
    score = sum([rsi_ok, body_ok, dir_ok, vol_ok, range_ok])
    detail = {
        "price": c, "rsi": rsi, "body": body, "vol_ratio": vol_ratio,
        "rsi_ok": rsi_ok, "body_ok": body_ok, "dir_ok": dir_ok, "vol_ok": vol_ok, "range_ok": range_ok
    }
    return score, max_score, hour_ok, band_touch, detail

# ════════════════ BACKTEST HELPERS ════════════════
TP_GRID = [0.012,0.015,0.018,0.020,0.025,0.030,0.035,0.040]
SL_GRID = [0.007,0.010,0.012,0.015,0.018]

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

def get_best_tp_sl(df_train):
    if len(df_train) < 200: return 0.025, 0.012
    sigs = get_signals_for_backtest(df_train)
    best = {"pnl": -1e9, "tp": None, "sl": None}
    for tp, sl in itertools.product(TP_GRID, SL_GRID):
        if tp <= sl: continue
        pnl = quick_simulate(df_train, sigs, tp, sl)
        if pnl > best["pnl"]:
            best = {"pnl": pnl, "tp": tp, "sl": sl}
    if best["tp"] is None: return 0.025, 0.012
    return best["tp"], best["sl"]

def get_signals_for_backtest(df):
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
        buy &= range_ok; sell &= range_ok
    s = pd.Series(0, index=df.index)
    s[buy] = 1; s[sell] = -1
    s[df.index.hour.isin(BLOCKED_HOURS)] = 0
    return s

# ════════════════ TELEGRAM HANDLERS ════════════════
async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    try:
        parts = data.split('|')
        outcome, pair, side = parts[0], parts[1], int(parts[2])
        entry, tp, sl = float(parts[3]), float(parts[4]), float(parts[5])
        signal_id = int(parts[6]) if len(parts) > 6 else 0

        if side == 1:
            exit_price = tp if outcome == "TP" else sl
            pnl = 5 * (exit_price - entry) / entry * 20 - 0.08
        else:
            exit_price = tp if outcome == "TP" else sl
            pnl = 5 * (entry - exit_price) / entry * 20 - 0.08
        pnl = round(pnl, 2)

        async with state_lock:
            trade_log.append({
                "pair": pair, "side": side, "entry": entry, "tp": tp, "sl": sl,
                "outcome": outcome, "pnl": pnl, "timestamp": datetime.now(timezone.utc)
            })
            if len(trade_log) > 1000: del trade_log[:-1000]

            if signal_id and 0 < signal_id <= len(signal_log):
                signal_log[signal_id - 1]["outcome"] = outcome
                signal_log[signal_id - 1]["pnl"] = pnl
            if len(signal_log) > 1000: del signal_log[:-1000]

            global daily_pnl
            daily_pnl += pnl
            if outcome == "SL":
                cooldowns[pair] = datetime.now(timezone.utc) + timedelta(minutes=SL_COOLDOWN)

        new_text = query.message.text + f"\n\n✅ Outcome: {outcome} | PnL: ${pnl:.2f}"
        await query.edit_message_text(text=new_text, reply_markup=None)
    except Exception as e:
        logging.error(f"Button error: {e}")

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    async with state_lock:
        if not trade_log:
            await update.message.reply_text("No trades yet."); return
        wins = [t for t in trade_log if t['pnl'] > 0]
        losses = [t for t in trade_log if t['pnl'] <= 0]
        win_rate = len(wins)/len(trade_log)*100 if trade_log else 0
        total_pnl = sum(t['pnl'] for t in trade_log)
        gross_win = sum(t['pnl'] for t in wins) if wins else 0
        gross_loss = abs(sum(t['pnl'] for t in losses)) if losses else 0
        pf = gross_win/gross_loss if gross_loss > 0 else float('inf')
        today = datetime.now(timezone.utc).date()
        today_trades = [t for t in trade_log if t['timestamp'].date() == today]
        today_pnl = sum(t['pnl'] for t in today_trades)
    text = (f"📊 **Cumulative Stats**\n"
            f"Trades: {len(trade_log)}\n"
            f"Win Rate: {win_rate:.1f}%\n"
            f"Total PnL: ${total_pnl:.2f}\n"
            f"Profit Factor: {pf:.2f}\n"
            f"Today: ${today_pnl:.2f} ({len(today_trades)} trades)")
    await update.message.reply_text(text, parse_mode='Markdown')

async def daily_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    today = datetime.now(timezone.utc).date()
    async with state_lock:
        today_trades = [t for t in trade_log if t['timestamp'].date() == today]
    if not today_trades:
        await update.message.reply_text("No trades today."); return
    wins = sum(1 for t in today_trades if t['pnl'] > 0)
    losses = len(today_trades) - wins
    total_pnl = sum(t['pnl'] for t in today_trades)
    ending_balance = STARTING_BALANCE + total_pnl
    total_return = (ending_balance - STARTING_BALANCE) / STARTING_BALANCE * 100
    win_rate = wins / len(today_trades) * 100 if today_trades else 0
    gross_win = sum(t['pnl'] for t in today_trades if t['pnl'] > 0)
    gross_loss = abs(sum(t['pnl'] for t in today_trades if t['pnl'] <= 0))
    pf = gross_win / gross_loss if gross_loss > 0 else float('inf')
    cum_pnl = 0; max_dd = 0; peak = 0
    for t in today_trades:
        cum_pnl += t['pnl']
        if cum_pnl > peak: peak = cum_pnl
        dd = cum_pnl - peak
        if dd < max_dd: max_dd = dd
    text = (
        f"📊 DAILY PERFORMANCE REPORT\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 Starting Balance: ${STARTING_BALANCE:.2f}\n"
        f"💰 Ending Balance:   ${ending_balance:.2f}\n"
        f"📈 Total Return: {total_return:+.2f}%\n"
        f"🎯 Win Rate: {win_rate:.1f}%\n"
        f"🔁 Total Trades: {len(today_trades)}\n"
        f"📊 Profit Factor: {pf:.2f}\n"
        f"📉 Max Drawdown: {max_dd:+.2f} USD\n"
        f"🟢 Winning Trades: {wins}\n"
        f"🔴 Losing Trades: {losses}\n"
        f"⚙️ Strategy: VWAP Reversal + RSI + Volume\n"
        f"⏱ Timeframe: 1m\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━"
    )
    await update.message.reply_text(text)

async def optimize_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Running daily pair selection & TP/SL optimisation...")
    asyncio.create_task(daily_pair_selection_and_optimize())

# ════════════════ CONTINUOUS TP MONITOR (thread‑safe) ════════════════
async def continuous_tp_monitor():
    await asyncio.sleep(5)
    while True:
        async with state_lock:
            closed_ids = []
            for trade_id, t in list(open_trades.items()):
                sym = t["pair"]
                price = get_latest_price(sym)
                if price is None:
                    continue
                side = t["side"]
                hit = None
                if side == 1:
                    if price >= t["tp"]: hit = "TP"
                    elif price <= t["sl"]: hit = "SL"
                else:
                    if price <= t["tp"]: hit = "TP"
                    elif price >= t["sl"]: hit = "SL"
                if not hit:
                    continue

                entry = t["entry"]
                exit_price = t["tp"] if hit == "TP" else t["sl"]
                if side == 1:
                    pnl = 5 * (exit_price - entry) / entry * 20 - 0.08
                else:
                    pnl = 5 * (entry - exit_price) / entry * 20 - 0.08
                pnl = round(pnl, 2)

                trade_log.append({
                    "pair": sym, "side": side, "entry": entry, "tp": t["tp"], "sl": t["sl"],
                    "outcome": hit, "pnl": pnl, "timestamp": datetime.now(timezone.utc)
                })
                if len(trade_log) > 1000: del trade_log[:-1000]

                if "signal_id" in t and 0 < t["signal_id"] <= len(signal_log):
                    signal_log[t["signal_id"] - 1]["outcome"] = hit
                    signal_log[t["signal_id"] - 1]["pnl"] = pnl
                if len(signal_log) > 1000: del signal_log[:-1000]

                global daily_pnl
                daily_pnl += pnl
                if hit == "SL":
                    cooldowns[sym] = datetime.now(timezone.utc) + timedelta(minutes=SL_COOLDOWN)

                sec = get_sector(sym)
                sector_counts[sec] = max(0, sector_counts.get(sec, 0) - 1)
                closed_ids.append(trade_id)

                try:
                    new_text = t["alert_msg"].text + f"\n\n✅ Auto‑close: {hit} | PnL: ${pnl:.2f}"
                    await application.bot.edit_message_text(
                        chat_id=t["chat_id"],
                        message_id=t["alert_msg"].message_id,
                        text=new_text)
                except: pass
                await application.bot.send_message(chat_id=CHAT_ID, text=f"🔔 {sym} {hit} hit! PnL: ${pnl:.2f}")

            for tid in closed_ids:
                del open_trades[tid]

        await asyncio.sleep(30)

# ════════════════ DAILY PAIR SELECTION & BACKTEST REPORT ════════════════
async def daily_pair_selection_and_optimize():
    global PAIRS, OPTIMIZED_TP_SL
    await application.bot.send_message(chat_id=CHAT_ID, text="⏳ Daily pair selection & TP/SL optimisation starting...")
    now = datetime.now(timezone.utc)
    end_ms = int(now.timestamp() * 1000)
    start_ms = int((now - timedelta(days=30)).timestamp() * 1000)

    results = {}
    for sym in ALL_PAIRS:
        df = fetch_klines(sym, start_ms, end_ms)
        if len(df) < 500: continue
        df = add_indicators(df)
        train_end = df.index.max()
        train_start = train_end - timedelta(days=15)
        test_end = train_end
        test_start = train_end - timedelta(days=5)
        df_train = df[(df.index >= train_start) & (df.index < test_start)]
        df_test  = df[(df.index >= test_start) & (df.index <= test_end)]
        if len(df_train) < 200 or df_test.empty: continue
        best_tp, best_sl = get_best_tp_sl(df_train)
        sigs_test = get_signals_for_backtest(df_test)
        oos_pnl = quick_simulate(df_test, sigs_test, best_tp, best_sl)
        results[sym] = {"oos_pnl": oos_pnl, "tp": best_tp, "sl": best_sl}

    if not results:
        await application.bot.send_message(chat_id=CHAT_ID, text="❌ Not enough data for daily optimisation."); return

    # Build full breakdown message
    sorted_all = sorted(results.items(), key=lambda x: x[1]['oos_pnl'], reverse=True)
    lines = ["📋 *Daily Backtest Results (OOS PnL)*", "```"]
    for sym, d in sorted_all:
        pnl_str = f"${d['oos_pnl']:.2f}"
        tp_str = f"{d['tp']*100:.2f}%"
        sl_str = f"{d['sl']*100:.2f}%"
        lines.append(f"{sym:<12} {pnl_str:>8}  TP:{tp_str}  SL:{sl_str}")
    lines.append("```")

    top_pairs = [sym for sym, d in sorted_all if d['oos_pnl'] > 0][:20]
    if len(top_pairs) < 5:
        top_pairs = [sym for sym, d in sorted_all][:20]

    async with state_lock:
        PAIRS = top_pairs
        OPTIMIZED_TP_SL = {sym: (results[sym]['tp'], results[sym]['sl']) for sym in top_pairs}

    summary = (f"✅ Daily optimisation done.\n"
               f"🟢 New pair list ({len(PAIRS)} pairs): {', '.join(PAIRS[:10])}...\n"
               f"📊 Selected pairs OOS PnL total: ${sum(results[s]['oos_pnl'] for s in top_pairs):.2f}")
    full_msg = "\n".join(lines) + f"\n{summary}"
    await application.bot.send_message(chat_id=CHAT_ID, text=full_msg, parse_mode='Markdown')

# ════════════════ MAIN LOOP ════════════════
async def monitor():
    global application, signal_counter, daily_pnl
    application = Application.builder().token(BOT_TOKEN).build()
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(CommandHandler("daily", daily_command))
    application.add_handler(CommandHandler("optimize", optimize_command))
    application.add_handler(CallbackQueryHandler(button_callback))
    await application.initialize()
    await application.start()
    asyncio.create_task(application.updater.start_polling())
    asyncio.create_task(continuous_tp_monitor())

    bot = application.bot
    await bot.send_message(chat_id=CHAT_ID, text="🏦 Institutional VWAP Bot started.\nAll features active, daily backtest report enabled.")

    # Run initial optimisation
    await daily_pair_selection_and_optimize()

    last_date = datetime.now(timezone.utc).date()
    executor = ThreadPoolExecutor(max_workers=10)

    while True:
        now = datetime.now(timezone.utc)
        if now.date() != last_date:
            async with state_lock:
                daily_pnl = 0.0
                cooldowns.clear()
                sector_counts.clear()
            last_date = now.date()

        if daily_pnl <= -DAILY_LOSS_LIMIT * STARTING_BALANCE:
            await bot.send_message(chat_id=CHAT_ID, text="⚠️ Daily loss limit hit. Trading halted until midnight.")
            await asyncio.sleep(SCAN_INTERVAL_MINUTES * 60)
            continue

        # Clean expired cooldowns
        async with state_lock:
            for pair, until in list(cooldowns.items()):
                if now >= until:
                    del cooldowns[pair]

        # Scan all pairs in parallel
        futures = [executor.submit(update_pair_cache, sym) for sym in PAIRS]
        strong = []; watch = []; weak = []; no_trade = 0
        lines = []

        for future in as_completed(futures):
            sym, ind_df = future.result()
            if ind_df is None or ind_df.empty:
                no_trade += 1; continue

            # Check cooldown with lock
            async with state_lock:
                cooldown_until = cooldowns.get(sym)

            signal, info = check_signal(ind_df, sym)
            # Apply cooldown if exists
            if cooldown_until and ind_df.index[-1] < cooldown_until:
                signal = 0
                info["fail"] = "cooldown"

            score, max_score, hour_ok, band_touch, det = condition_score(ind_df)
            tp_sl = OPTIMIZED_TP_SL.get(sym, (0.025, 0.012))
            price = info.get("price", 0)

            if score == 5: strong.append(sym)
            elif score == 4: watch.append(sym)
            elif score == 3: weak.append(sym)
            else: no_trade += 1

            if score >= 3:
                icon = "🟢" if score == 5 else "⚠️"
                band_str = f"Band {band_touch}" if band_touch != "none" else "Band none"
                missing = []
                if score < 5:
                    if not det['rsi_ok']: missing.append("RSI")
                    if not det['body_ok']: missing.append("body")
                    if not det['dir_ok']: missing.append("direction")
                    if not det['vol_ok']: missing.append("volume")
                    if not det['range_ok']: missing.append("range")
                line = f"{icon} *{sym}*   {price:.6f}   RSI {det['rsi']:.0f}   Score {score}/5   {band_str}"
                if missing:
                    line += f"\n     Missing: {', '.join(missing)}"
                lines.append(line)

            # Execute trade if signal valid and risk checks pass
            if signal != 0:
                if not await can_trade(sym):
                    await bot.send_message(chat_id=CHAT_ID,
                                           text=f"⛔ {sym} signal blocked – concurrency or sector limit.")
                    continue

                async with state_lock:
                    signal_counter += 1
                    sig_id = signal_counter
                    side = signal
                    tp, sl = tp_sl
                    tp_price = price * (1 + tp) if side == 1 else price * (1 - tp)
                    sl_price = price * (1 - sl) if side == 1 else price * (1 + sl)
                    direction = "LONG" if side == 1 else "SHORT"

                    equity = get_equity()
                    risk_capital = equity * RISK_PER_TRADE
                    notional = risk_capital / sl

                    reason = (f"VWAP±2σ {band_touch} band touch, RSI {info['rsi']} "
                              f"{'<' if side==1 else '>'} {'40' if side==1 else '60'}, "
                              f"body/ATR {info['body/atr']}, volume {info['vol_ratio']}x > {VOL_MULT}x")

                    alert_text = (
                        f"{'🟢 ENTRY SIGNAL' if side==1 else '🔴 ENTRY SIGNAL'}\n"
                        f"━━━━━━━━━━━━━━━━━\n"
                        f"📊 Pair: ${sym}\n"
                        f"📈 Direction: {direction}\n"
                        f"💰 Entry Price: ${price:.6f}\n"
                        f"⚙️ Strategy: VWAP Reversal + RSI + Volume\n"
                        f"⏱ Timeframe: 1m\n"
                        f"💸 Fees: 0.08%\n"
                        f"📉 Slippage: 0.02%\n"
                        f"🧠 Reason:\n{reason}\n"
                        f"━━━━━━━━━━━━━━━━━\n"
                        f"✅ TP: ${tp_price:.6f} (+{tp*100:.2f}%)\n"
                        f"❌ SL: ${sl_price:.6f} (-{sl*100:.2f}%)\n"
                        f"📊 Suggested Size: ${notional:.2f} (risk: ${risk_capital:.2f})"
                    )

                    keyboard = InlineKeyboardMarkup([
                        [InlineKeyboardButton("✅ TP Hit", callback_data=f"TP|{sym}|{side}|{price}|{tp_price}|{sl_price}|{sig_id}"),
                         InlineKeyboardButton("❌ SL Hit", callback_data=f"SL|{sym}|{side}|{price}|{tp_price}|{sl_price}|{sig_id}")]
                    ])

                    sent_msg = await bot.send_message(chat_id=CHAT_ID, text=alert_text,
                                                       reply_markup=keyboard, parse_mode='Markdown')
                    signal_log.append({
                        "id": sig_id, "pair": sym, "side": side, "entry": price,
                        "tp": tp_price, "sl": sl_price, "timestamp": now,
                        "outcome": None, "pnl": None
                    })
                    if len(signal_log) > 1000: del signal_log[:-1000]

                    open_trades[f"{sym}_{time.time()}"] = {
                        "pair": sym, "side": side, "entry": price, "tp": tp_price, "sl": sl_price,
                        "alert_msg": sent_msg, "chat_id": CHAT_ID, "signal_id": sig_id
                    }
                    sec = get_sector(sym)
                    sector_counts[sec] = sector_counts.get(sec, 0) + 1

        # Heartbeat
        summary = f"📊 SCANNED: {len(PAIRS)} | 🟢 {len(strong)} | ⚠️ {len(watch)+len(weak)} | ❌ {no_trade}"
        heartbeat = f"📡 {now:%Y-%m-%d %H:%M} UTC | MARKET SCAN\n" + "\n".join(lines) + f"\n\n{summary}"
        await bot.send_message(chat_id=CHAT_ID, text=heartbeat, parse_mode='Markdown')

        # Daily optimisation at 12:00 UTC
        if now.hour == 12 and now.minute == 0:
            asyncio.create_task(daily_pair_selection_and_optimize())

        elapsed = (datetime.now(timezone.utc) - now).total_seconds()
        sleep_seconds = max(1, SCAN_INTERVAL_MINUTES * 60 - elapsed)
        await asyncio.sleep(sleep_seconds)

# ════════════════ RUN ════════════════
if __name__ == "__main__":
    threading.Thread(target=run_health_server, daemon=True).start()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(monitor())
