#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VWAP Reversal Monitor Bot – Final Edition
Auto‑track, optimized TP/SL, clean heartbeat, /check command.
"""
import time, asyncio, logging, threading, itertools, os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
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

# Hardcoded optimized TP/SL from E16 backtest (extracted 2026-06-23)
OPTIMIZED_TP_SL = {
    "PEPEUSDT":  (0.020, 0.015),
    "BONKUSDT":  (0.030, 0.012),
    "WIFUSDT":   (0.040, 0.018),
    "SUIUSDT":   (0.040, 0.012),
    "ARBUSDT":   (0.040, 0.010),
    "OPUSDT":    (0.040, 0.018),
    "GALAUSDT":  (0.030, 0.007),
    "FETUSDT":   (0.040, 0.010),
    "SANDUSDT":  (0.020, 0.015),
    "MEMEUSDT":  (0.030, 0.010),
    "AVAXUSDT":  (0.020, 0.015),
    "BNBUSDT":   (0.040, 0.012),
    "DOTUSDT":   (0.020, 0.007),
    "LTCUSDT":   (0.040, 0.010),
    "LINKUSDT":  (0.040, 0.010),
    "INJUSDT":   (0.035, 0.010),
    "POLUSDT":   (0.035, 0.012),
    "STXUSDT":   (0.040, 0.018),
    "BOMEUSDT":  (0.040, 0.010),
    "ETHUSDT":   (0.035, 0.018),
    "ICPUSDT":   (0.035, 0.007),
}

MEXC_URL = "https://api.mexc.com/api/v3/klines"
API_TIMEOUT = 30
API_RETRIES = 2

# ═════════════════ DATA & INDICATORS ═════════════════
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
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = end_ms - 300 * 60_000
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
    cp = _cache_path(symbol, days)
    if cp.exists():
        try:
            return pd.read_csv(cp, compression="gzip", index_col=0, parse_dates=True)
        except:
            pass
    return pd.DataFrame()

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

def optimize_tp_sl(symbol):
    """Return optimized TP/SL from hardcoded E16 values, with fallback."""
    if symbol in OPTIMIZED_TP_SL:
        return OPTIMIZED_TP_SL[symbol]
    # Fallback for any missing pair
    if symbol in ("BOMEUSDT", "INJUSDT", "ICPUSDT"):
        return 0.035, 0.012
    return 0.025, 0.012

# ═════════════════ TRADE LOGGING & AUTO‑TRACK ═════════════════
trade_log = []
open_trades = {}   # trade_id -> {pair, side, entry, tp, sl, alert_msg, chat_id}

def get_pair_checklist(df, symbol):
    if df.empty:
        return f"{symbol}: no data"
    latest = df.iloc[-1]
    c, o = latest["Close"], latest["Open"]
    vol_ratio = latest["Volume"] / latest["vol20"] if latest["vol20"] != 0 else 0
    rsi = latest["rsi"]
    body = latest["body_pct"]
    band_low_touch = latest["Low"] <= latest["vwap_2dn"]
    band_high_touch = latest["High"] >= latest["vwap_2up"]
    close_inside_low = c > latest["vwap_2dn"]
    close_inside_high = c < latest["vwap_2up"]
    hour_ok = latest.name.hour not in BLOCKED_HOURS
    body_ok = body > BODY_PCT_MIN
    rsi_buy_ok = rsi < RSI_LONG_MAX
    rsi_sell_ok = rsi > RSI_SHORT_MIN
    dir_buy = c > o
    dir_sell = c < o
    vol_ok = vol_ratio >= VOL_MULT
    range_ok = (latest["High"] - latest["Low"]) < MAX_RANGE_ATR * latest["atr"] if RANGE_CAP else True

    buy_possible = band_low_touch and close_inside_low
    sell_possible = band_high_touch and close_inside_high

    parts = [f"{symbol}: ${c:.6f}"]
    parts.append("Hour" + ("✅" if hour_ok else "❌"))

    if buy_possible:
        parts.append(f"RSI{rsi:.0f}" + ("✅" if rsi_buy_ok else "❌"))
        parts.append("Dir" + ("✅" if dir_buy else "❌"))
        parts.append(f"Body{body:.2f}" + ("✅" if body_ok else "❌"))
        parts.append(f"Vol{vol_ratio:.1f}x" + ("✅" if vol_ok else "❌"))
        if RANGE_CAP:
            parts.append("Range" + ("✅" if range_ok else "❌"))
    elif sell_possible:
        parts.append(f"RSI{rsi:.0f}" + ("✅" if rsi_sell_ok else "❌"))
        parts.append("Dir" + ("✅" if dir_sell else "❌"))
        parts.append(f"Body{body:.2f}" + ("✅" if body_ok else "❌"))
        parts.append(f"Vol{vol_ratio:.1f}x" + ("✅" if vol_ok else "❌"))
        if RANGE_CAP:
            parts.append("Range" + ("✅" if range_ok else "❌"))
    else:
        parts.append("Band❌")
    return " | ".join(parts)

# ═════════════════ TELEGRAM HANDLERS ═════════════════
application = None

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data  # "outcome|pair|side|entry|tp|sl"
    try:
        parts = data.split('|')
        outcome, pair, side, entry, tp, sl = parts[0], parts[1], int(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
        if side == 1:
            pnl = 5 * ((tp - entry) / entry) * 20 - 0.08 if outcome == "TP" else 5 * ((sl - entry) / entry) * 20 - 0.08
        else:
            pnl = 5 * ((entry - tp) / entry) * 20 - 0.08 if outcome == "TP" else 5 * ((entry - sl) / entry) * 20 - 0.08
        pnl = round(pnl, 2)
        trade = {"pair": pair, "side": side, "entry": entry, "tp": tp, "sl": sl, "outcome": outcome, "pnl": pnl, "timestamp": datetime.now(timezone.utc)}
        trade_log.append(trade)
        new_text = query.message.text + f"\n\n✅ Outcome: {outcome} | PnL: ${pnl:.2f}"
        await query.edit_message_text(text=new_text, reply_markup=None)
    except Exception as e:
        logging.error(f"Button error: {e}")

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not trade_log:
        await update.message.reply_text("No trades yet.")
        return
    wins = [t for t in trade_log if t['pnl'] > 0]
    losses = [t for t in trade_log if t['pnl'] <= 0]
    win_rate = len(wins)/len(trade_log)*100
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

async def today_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    today = datetime.now(timezone.utc).date()
    today_trades = [t for t in trade_log if t['timestamp'].date() == today]
    if not today_trades:
        await update.message.reply_text("No trades today yet.")
        return
    wins = sum(1 for t in today_trades if t['pnl'] > 0)
    pnl = sum(t['pnl'] for t in today_trades)
    await update.message.reply_text(f"📅 Today: {len(today_trades)} trades | {wins} wins | PnL: ${pnl:.2f}")

async def history_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not trade_log:
        await update.message.reply_text("No history.")
        return
    recent = trade_log[-5:]
    lines = "\n".join(f"{t['pair']} {'L' if t['side']==1 else 'S'} → {t['outcome']} ${t['pnl']:.2f}" for t in reversed(recent))
    await update.message.reply_text(f"Last 5 trades:\n{lines}")

async def check_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lines = []
    for sym in PAIRS:
        live = fetch_live_data(sym)
        hist = load_historical_data(sym, LOOKBACK_DAYS)
        if not hist.empty:
            common = hist.index.intersection(live.index)
            hist = hist[~hist.index.isin(common)]
            df = pd.concat([hist, live]).sort_index().iloc[-2000:]
        else:
            df = live
        if df.empty:
            lines.append(f"{sym}: no data")
            continue
        df = add_indicators(df)
        lines.append(get_pair_checklist(df, sym))
    await update.message.reply_text("\n".join(lines))

# ═════════════════ MAIN BOT LOOP ═════════════════
async def monitor():
    global application
    application = Application.builder().token(BOT_TOKEN).build()
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(CommandHandler("today", today_command))
    application.add_handler(CommandHandler("history", history_command))
    application.add_handler(CommandHandler("check", check_command))
    application.add_handler(CallbackQueryHandler(button_callback))
    await application.initialize()
    await application.start()
    asyncio.create_task(application.updater.start_polling())

    bot = application.bot
    await bot.send_message(chat_id=CHAT_ID, text="🟢 VWAP Bot started with optimized TP/SL.\nCommands: /stats, /today, /history, /check")

    tp_sl_map = {}
    for sym in PAIRS:
        tp, sl = optimize_tp_sl(sym)
        tp_sl_map[sym] = {"TP%": round(tp*100,2), "SL%": round(sl*100,2)}

    while True:
        scan_start = datetime.now(timezone.utc)

        # --- Auto‑track open trades ---
        closed_ids = []
        for trade_id, t in list(open_trades.items()):
            sym = t["pair"]
            live = fetch_live_data(sym)
            if live.empty:
                continue
            latest_price = live.iloc[-1]["Close"]
            side = t["side"]
            hit = None
            if side == 1:
                if latest_price >= t["tp"]: hit = "TP"
                elif latest_price <= t["sl"]: hit = "SL"
            else:
                if latest_price <= t["tp"]: hit = "TP"
                elif latest_price >= t["sl"]: hit = "SL"

            if hit:
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
                try:
                    new_text = t["alert_msg"].text + f"\n\n✅ Auto‑close: {hit} | PnL: ${pnl:.2f}"
                    await bot.edit_message_text(chat_id=t["chat_id"], message_id=t["alert_msg"].message_id, text=new_text)
                except:
                    pass
                await bot.send_message(chat_id=CHAT_ID, text=f"🔔 {sym} {hit} hit! PnL: ${pnl:.2f}")
                closed_ids.append(trade_id)

        for tid in closed_ids:
            del open_trades[tid]

        # --- Scan pairs for new signals ---
        near_miss = []
        trade_alerts = []

        for sym in PAIRS:
            live = fetch_live_data(sym)
            hist = load_historical_data(sym, LOOKBACK_DAYS)
            if not hist.empty:
                common = hist.index.intersection(live.index)
                hist = hist[~hist.index.isin(common)]
                df = pd.concat([hist, live]).sort_index().iloc[-2000:]
            else:
                df = live
            if df.empty:
                continue
            df = add_indicators(df)
            signal, info = check_signal(df)
            tp_sl = tp_sl_map.get(sym, {"TP%": 2.5, "SL%": 1.2})
            price = info.get("price", 0)

            if signal == 1:
                tp_price = price * (1 + tp_sl["TP%"]/100)
                sl_price = price * (1 - tp_sl["SL%"]/100)
                alert_text = (
                    f"🟢 **BUY {sym}**\n"
                    f"Entry: ${price:.6f}\n"
                    f"TP: ${tp_price:.6f} (+{tp_sl['TP%']}%)\n"
                    f"SL: ${sl_price:.6f} (-{tp_sl['SL%']}%)\n"
                    f"Vol: {info['vol_ratio']}x | RSI: {info['rsi']}\n"
                    f"Time: {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
                )
                keyboard = InlineKeyboardMarkup([
                    [InlineKeyboardButton("✅ TP Hit", callback_data=f"TP|{sym}|1|{price}|{tp_price}|{sl_price}"),
                     InlineKeyboardButton("❌ SL Hit", callback_data=f"SL|{sym}|1|{price}|{tp_price}|{sl_price}")]
                ])
                sent_msg = await bot.send_message(chat_id=CHAT_ID, text=alert_text,
                                                   reply_markup=keyboard, parse_mode='Markdown')
                trade_id = f"{sym}_{int(time.time())}"
                open_trades[trade_id] = {
                    "pair": sym, "side": 1, "entry": price, "tp": tp_price, "sl": sl_price,
                    "alert_msg": sent_msg, "chat_id": CHAT_ID
                }

            elif signal == -1:
                tp_price = price * (1 - tp_sl["TP%"]/100)
                sl_price = price * (1 + tp_sl["SL%"]/100)
                alert_text = (
                    f"🔴 **SELL {sym}**\n"
                    f"Entry: ${price:.6f}\n"
                    f"TP: ${tp_price:.6f} (+{tp_sl['TP%']}%)\n"
                    f"SL: ${sl_price:.6f} (-{tp_sl['SL%']}%)\n"
                    f"Vol: {info['vol_ratio']}x | RSI: {info['rsi']}\n"
                    f"Time: {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
                )
                keyboard = InlineKeyboardMarkup([
                    [InlineKeyboardButton("✅ TP Hit", callback_data=f"TP|{sym}|3|{price}|{tp_price}|{sl_price}"),
                     InlineKeyboardButton("❌ SL Hit", callback_data=f"SL|{sym}|3|{price}|{tp_price}|{sl_price}")]
                ])
                sent_msg = await bot.send_message(chat_id=CHAT_ID, text=alert_text,
                                                   reply_markup=keyboard, parse_mode='Markdown')
                trade_id = f"{sym}_{int(time.time())}"
                open_trades[trade_id] = {
                    "pair": sym, "side": 3, "entry": price, "tp": tp_price, "sl": sl_price,
                    "alert_msg": sent_msg, "chat_id": CHAT_ID
                }

            # Build near-miss list
            latest = df.iloc[-1]
            c, o = latest["Close"], latest["Open"]
            vol_ratio = latest["Volume"] / latest["vol20"] if latest["vol20"] != 0 else 0
            rsi = latest["rsi"]
            body_ok = latest["body_pct"] > BODY_PCT_MIN
            rsi_ok = (rsi < RSI_LONG_MAX) or (rsi > RSI_SHORT_MIN)
            dir_ok = (c > o) if rsi < RSI_LONG_MAX else (c < o)
            vol_ok = vol_ratio >= VOL_MULT
            range_ok = (latest["High"] - latest["Low"]) < MAX_RANGE_ATR * latest["atr"] if RANGE_CAP else True
            met = sum([body_ok, rsi_ok, dir_ok, vol_ok, range_ok])
            if met >= 3:
                near_miss.append(f"{sym}: RSI{rsi:.0f} B{latest['body_pct']:.2f} V{vol_ratio:.2f}x Band❌")

        # Heartbeat
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        heartbeat = f"📡 {now} | Near‑miss: {len(near_miss)}"
        if near_miss:
            heartbeat += "\n" + "\n".join(near_miss)
        await bot.send_message(chat_id=CHAT_ID, text=heartbeat)

        elapsed = (datetime.now(timezone.utc) - scan_start).total_seconds()
        sleep_time = max(0, SCAN_INTERVAL_MINUTES * 60 - elapsed)
        await asyncio.sleep(sleep_time)

# ═════════════════ RUN ═════════════════
if __name__ == "__main__":
    threading.Thread(target=run_health_server, daemon=True).start()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(monitor())
