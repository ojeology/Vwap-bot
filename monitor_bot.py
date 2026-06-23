#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VWAP Reversal Monitor Bot – Final Professional Edition
Clean heartbeat, professional signals, daily reports, auto‑track, logging.
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
SIGNAL_LOG_CHANNEL = os.environ.get("SIGNAL_LOG_CHANNEL", None)   # optional

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

STARTING_BALANCE = 100.0   # paper trading starting balance

# Optimized TP/SL from E16 backtest (extracted 2026-06-23)
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
    if symbol in OPTIMIZED_TP_SL:
        return OPTIMIZED_TP_SL[symbol]
    if symbol in ("BOMEUSDT", "INJUSDT", "ICPUSDT"):
        return 0.035, 0.012
    return 0.025, 0.012

# ═════════════════ TRADE LOGGING & SIGNAL COUNTER ═════════════════
trade_log = []
signal_log = []
signal_counter = 0
open_trades = {}

def condition_score(df):
    """Return (score, max_score, hour_ok, band_touch, detail_dict)."""
    if df.empty:
        return 0, 0, False, "none", {}
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
        "price": c,
        "rsi": rsi,
        "body": body,
        "vol_ratio": vol_ratio,
        "rsi_ok": rsi_ok,
        "body_ok": body_ok,
        "dir_ok": dir_ok,
        "vol_ok": vol_ok,
        "range_ok": range_ok,
    }
    return score, max_score, hour_ok, band_touch, detail

# ═════════════════ TELEGRAM HANDLERS ═════════════════
application = None

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

        trade_log.append({
            "pair": pair, "side": side, "entry": entry, "tp": tp, "sl": sl,
            "outcome": outcome, "pnl": pnl, "timestamp": datetime.now(timezone.utc)
        })
        if signal_id and 0 < signal_id <= len(signal_log):
            signal_log[signal_id - 1]["outcome"] = outcome
            signal_log[signal_id - 1]["pnl"] = pnl

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

async def signals_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    open_signals = [s for s in signal_log if s.get('outcome') is None]
    if not open_signals:
        await update.message.reply_text("No open signals.")
        return
    lines = []
    for s in open_signals:
        lines.append(f"#{s['id']} {s['pair']} {'LONG' if s['side']==1 else 'SHORT'} Entry: {s['entry']:.6f} TP: {s['tp']:.6f} SL: {s['sl']:.6f}")
    await update.message.reply_text("📌 Open Signals:\n" + "\n".join(lines))

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
        score, max_score, hour_ok, band_touch, det = condition_score(df)
        hour_icon = "🟢" if hour_ok else "⏰"
        rsi_str = f"RSI {det['rsi']:.0f}"
        body_str = f"Body {det['body']:.2f}"
        vol_str = f"Vol {det['vol_ratio']:.1f}x"
        line = (
            f"{sym}: {det['price']:.6f} {hour_icon} "
            f"{rsi_str}{'✅' if det['rsi_ok'] else '❌'} "
            f"{body_str}{'✅' if det['body_ok'] else '❌'} "
            f"Dir{'✅' if det['dir_ok'] else '❌'} "
            f"{vol_str}{'✅' if det['vol_ok'] else '❌'} "
            f"Rng{'✅' if det['range_ok'] else '❌'} "
            f"Band {band_touch} "
            f"[{score}/{max_score}]"
        )
        lines.append(line)
    await update.message.reply_text("📋 Full Pair Status:\n" + "\n".join(lines))

async def daily_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    today = datetime.now(timezone.utc).date()
    today_trades = [t for t in trade_log if t['timestamp'].date() == today]
    if not today_trades:
        await update.message.reply_text("No trades today.")
        return
    wins = sum(1 for t in today_trades if t['pnl'] > 0)
    losses = len(today_trades) - wins
    total_pnl = sum(t['pnl'] for t in today_trades)
    ending_balance = STARTING_BALANCE + total_pnl
    total_return = (ending_balance - STARTING_BALANCE) / STARTING_BALANCE * 100
    win_rate = wins / len(today_trades) * 100 if today_trades else 0
    gross_win = sum(t['pnl'] for t in today_trades if t['pnl'] > 0)
    gross_loss = abs(sum(t['pnl'] for t in today_trades if t['pnl'] <= 0))
    pf = gross_win / gross_loss if gross_loss > 0 else float('inf')
    # Simulated max drawdown from daily PnL
    cum_pnl = 0
    max_dd = 0
    peak = 0
    for t in today_trades:
        cum_pnl += t['pnl']
        if cum_pnl > peak:
            peak = cum_pnl
        dd = cum_pnl - peak
        if dd < max_dd:
            max_dd = dd

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

# ═════════════════ MAIN BOT LOOP ═════════════════
async def monitor():
    global application, signal_counter
    application = Application.builder().token(BOT_TOKEN).build()
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(CommandHandler("today", today_command))
    application.add_handler(CommandHandler("history", history_command))
    application.add_handler(CommandHandler("check", check_command))
    application.add_handler(CommandHandler("signals", signals_command))
    application.add_handler(CommandHandler("daily", daily_command))
    application.add_handler(CallbackQueryHandler(button_callback))
    await application.initialize()
    await application.start()
    asyncio.create_task(application.updater.start_polling())

    bot = application.bot
    await bot.send_message(chat_id=CHAT_ID, text="🟢 Professional VWAP Bot started.\nCommands: /stats, /today, /history, /check, /signals, /daily")

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
                if "signal_id" in t and 0 < t["signal_id"] <= len(signal_log):
                    signal_log[t["signal_id"] - 1]["outcome"] = hit
                    signal_log[t["signal_id"] - 1]["pnl"] = pnl

                try:
                    new_text = t["alert_msg"].text + f"\n\n✅ Auto‑close: {hit} | PnL: ${pnl:.2f}"
                    await bot.edit_message_text(chat_id=t["chat_id"], message_id=t["alert_msg"].message_id, text=new_text)
                except:
                    pass
                await bot.send_message(chat_id=CHAT_ID, text=f"🔔 {sym} {hit} hit! PnL: ${pnl:.2f}")
                closed_ids.append(trade_id)

        for tid in closed_ids:
            del open_trades[tid]

        # --- Scan pairs for new signals & build heartbeat report ---
        report_rows = []
        strong_pairs = []
        watch_pairs = []
        weak_pairs = []
        no_trade_pairs = []

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
                no_trade_pairs.append(sym)
                report_rows.append((sym, "N/A", "N/A", "0/5", "❌ NO DATA"))
                continue

            df = add_indicators(df)
            signal, info = check_signal(df)
            score, max_score, hour_ok, band_touch, det = condition_score(df)
            tp_sl = tp_sl_map.get(sym, {"TP%": 2.5, "SL%": 1.2})
            price = info.get("price", 0)

            # Determine signal tag
            if score == 5:
                tag = "🟢 STRONG"
                strong_pairs.append(sym)
            elif score == 4:
                tag = "⚠️ WATCH"
                watch_pairs.append(sym)
            elif score == 3:
                tag = "⚠️ WEAK"
                weak_pairs.append(sym)
            else:
                tag = "❌ NO TRADE"
                no_trade_pairs.append(sym)

            rsi_val = f"{det['rsi']:.0f}"
            price_str = f"${price:.6f}" if price else "N/A"
            report_rows.append((sym, price_str, rsi_val, f"{score}/{max_score}", tag))

            # If a valid signal fires
            if signal in (1, -1):
                signal_counter += 1
                sig_id = signal_counter
                side = signal
                tp_price = price * (1 + tp_sl["TP%"]/100) if side == 1 else price * (1 - tp_sl["TP%"]/100)
                sl_price = price * (1 - tp_sl["SL%"]/100) if side == 1 else price * (1 + tp_sl["SL%"]/100)
                direction = "LONG" if side == 1 else "SHORT"
                fees_pct = 0.08
                slippage_pct = 0.02
                reason = (f"VWAP±2σ {band_touch} band touch, RSI {info['rsi']} "
                          f"{'<' if side==1 else '>'} {'40' if side==1 else '60'}, "
                          f"body/ATR {info['body/atr']}, volume {info['vol_ratio']}x > {VOL_MULT}x")

                alert_text = (
                    f"{'🟢 ENTRY SIGNAL' if side == 1 else '🔴 ENTRY SIGNAL'}\n"
                    f"━━━━━━━━━━━━━━━━━\n"
                    f"📊 Pair: ${sym}\n"
                    f"📈 Direction: {direction}\n"
                    f"💰 Entry Price: ${price:.6f}\n"
                    f"⚙️ Strategy: VWAP Reversal + RSI + Volume\n"
                    f"⏱ Timeframe: 1m\n"
                    f"💸 Fees: {fees_pct}%\n"
                    f"📉 Slippage: {slippage_pct}%\n"
                    f"🧠 Reason:\n{reason}\n"
                    f"━━━━━━━━━━━━━━━━━\n"
                    f"✅ TP: ${tp_price:.6f} (+{tp_sl['TP%']}%)\n"
                    f"❌ SL: ${sl_price:.6f} (-{tp_sl['SL%']}%)"
                )

                keyboard = InlineKeyboardMarkup([
                    [InlineKeyboardButton("✅ TP Hit", callback_data=f"TP|{sym}|{side}|{price}|{tp_price}|{sl_price}|{sig_id}"),
                     InlineKeyboardButton("❌ SL Hit", callback_data=f"SL|{sym}|{side}|{price}|{tp_price}|{sl_price}|{sig_id}")]
                ])

                sent_msg = await bot.send_message(chat_id=CHAT_ID, text=alert_text,
                                                   reply_markup=keyboard, parse_mode='Markdown')
                signal_log.append({
                    "id": sig_id, "pair": sym, "side": side, "entry": price,
                    "tp": tp_price, "sl": sl_price, "timestamp": datetime.now(timezone.utc),
                    "outcome": None, "pnl": None
                })
                trade_id = f"{sym}_{int(time.time())}"
                open_trades[trade_id] = {
                    "pair": sym, "side": side, "entry": price, "tp": tp_price, "sl": sl_price,
                    "alert_msg": sent_msg, "chat_id": CHAT_ID, "signal_id": sig_id
                }
                if SIGNAL_LOG_CHANNEL:
                    try:
                        await bot.send_message(chat_id=SIGNAL_LOG_CHANNEL, text=alert_text, parse_mode='Markdown')
                    except:
                        pass

        # Compose clean table heartbeat
        header = f"📡 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} | MARKET SCAN REPORT"
        table = "```\n"
        table += f"{'SYMBOL':<14} {'PRICE':<12} {'RSI':<6} {'SCORE':<7} {'SIGNAL':<12}\n"
        table += "─" * 60 + "\n"
        for sym, price_str, rsi_str, score_str, tag in report_rows:
            table += f"{sym:<14} {price_str:<12} {rsi_str:<6} {score_str:<7} {tag:<12}\n"
        table += "```\n"

        summary = ""
        if strong_pairs:
            summary += f"🟢 STRONG SETUPS: {', '.join(strong_pairs)}\n"
        if watch_pairs:
            summary += f"⚠️ WATCHLIST: {', '.join(watch_pairs)}\n"
        summary += f"📊 TOTAL ASSETS SCANNED: {len(PAIRS)}"

        full_report = header + "\n" + table + summary
        await bot.send_message(chat_id=CHAT_ID, text=full_report)

        # Send daily report at 23:59 UTC automatically
        now = datetime.now(timezone.utc)
        if now.hour == 23 and now.minute == 59:
            await daily_command(None, None)   # we'll call the daily function directly with a trick? Actually need bot/update context. Simpler: we use the existing daily_command but need a fake update. Better: just call the internal logic and send via bot.
            # We'll implement a simplified version:
            today = now.date()
            today_trades = [t for t in trade_log if t['timestamp'].date() == today]
            if today_trades:
                wins = sum(1 for t in today_trades if t['pnl'] > 0)
                losses = len(today_trades) - wins
                total_pnl = sum(t['pnl'] for t in today_trades)
                ending_balance = STARTING_BALANCE + total_pnl
                total_return = (ending_balance - STARTING_BALANCE) / STARTING_BALANCE * 100
                win_rate = wins / len(today_trades) * 100 if today_trades else 0
                gross_win = sum(t['pnl'] for t in today_trades if t['pnl'] > 0)
                gross_loss = abs(sum(t['pnl'] for t in today_trades if t['pnl'] <= 0))
                pf = gross_win / gross_loss if gross_loss > 0 else float('inf')
                cum_pnl = 0
                max_dd = 0
                peak = 0
                for t in today_trades:
                    cum_pnl += t['pnl']
                    if cum_pnl > peak:
                        peak = cum_pnl
                    dd = cum_pnl - peak
                    if dd < max_dd:
                        max_dd = dd
                daily_text = (
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
                await bot.send_message(chat_id=CHAT_ID, text=daily_text)

        elapsed = (datetime.now(timezone.utc) - scan_start).total_seconds()
        sleep_time = max(0, SCAN_INTERVAL_MINUTES * 60 - elapsed)
        await asyncio.sleep(sleep_time)

# ═════════════════ RUN ═════════════════
if __name__ == "__main__":
    threading.Thread(target=run_health_server, daemon=True).start()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(monitor())
