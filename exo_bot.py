#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║           DERIV SNIPER BOT v2  –  Professional Edition              ║
║   Rich terminal dashboard · Advanced Telegram · Termux-ready        ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import json, time, sqlite3, threading, queue, logging, asyncio, os, sys, pickle
from datetime import datetime, timedelta, timezone
from collections import deque
from typing import Optional

import numpy as np
import pandas as pd
import websocket
from sklearn.ensemble import RandomForestClassifier

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.align import Align
from rich import box

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler, ContextTypes,
)

# ══════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════
DERIV_APP_TOKEN    = os.environ.get("DERIV_TOKEN",    "m8MRwwwroJy6YQw")
TELEGRAM_BOT_TOKEN = os.environ.get("TG_BOT_TOKEN",   "8908331931:AAHIbTDn67QLSODEhQo0EHWUvx6mYOkU_-o")
TELEGRAM_CHAT_ID   = os.environ.get("TG_CHAT_ID",     "6400145232")

# Loud warning if still using hardcoded fallbacks (set env vars for production)
if not os.environ.get("DERIV_TOKEN"):
    print("⚠   DERIV_TOKEN not set — using hardcoded fallback. Set env var for production.")
if not os.environ.get("TG_BOT_TOKEN"):
    print("⚠   TG_BOT_TOKEN not set — using hardcoded fallback. Set env var for production.")

SYNTH_VOLATILITY    = ["R_100", "R_75", "R_50", "R_25", "R_10"]
SYNTH_VOLATILITY_1S = ["1HZ100V", "1HZ90V", "1HZ75V", "1HZ50V", "1HZ30V", "1HZ25V", "1HZ15V", "1HZ10V"]
SYNTH_RANGE_BREAK   = ["RDBULL", "RDBEAR"]

SYMBOLS = SYNTH_VOLATILITY + SYNTH_VOLATILITY_1S + SYNTH_RANGE_BREAK
ALL_TOUCH_SYMBOLS = SYMBOLS

# ── Risk parameters ───────────────────────────────────────────────────
STAKE                  = 3.0          # fixed risk per trade
TARGET_PROFIT          = 0.60         # display only; real profit depends on market
# Barrier = ATR × ATR_BARRIER_MULT above/below spot.
ATR_BARRIER_MULT       = 0.40
DURATION               = 15
CONTRACT_TYPE          = "ONETOUCH"
COOLDOWN_MINUTES       = 20
MAX_CONSECUTIVE_LOSSES = 3
PAUSE_MINUTES          = 30
DAILY_LOSS_LIMIT       = -20.0        # stop-loss floor per session
DAILY_PROFIT_TARGET    = 9.0          # take-profit per session
MAX_DAILY_TRADES       = 9999         # effectively no cap

EMA_FAST               = 50
EMA_SLOW               = 200
ATR_PERIOD             = 14
ATR_MA_PERIOD          = 30
RSI_PERIOD             = 14
SCORE_THRESHOLD        = 80
HEARTBEAT_INTERVAL_SEC = 900

# ── ML filter ────────────────────────────────────────────────────────
# The model trains in *observe-only* mode (never blocks a trade) until
# ML_MIN_TRADES real trades are recorded. After that it retrains every
# ML_RETRAIN_EVERY trades and starts filtering low-confidence trades.
# A progress bar is shown in Telegram during each retrain cycle.
MODEL_PATH        = "ml_model.pkl"
ML_MIN_TRADES     = 100
ML_RETRAIN_EVERY  = 100
ML_CONFIDENCE_MIN = 0.5
ML_FEATURE_COLS   = [
    "score", "wick_atr_ratio", "atr", "atr_ma",
    "ema_fast_slope", "ema_slow_slope", "ema_distance",
]

logging.basicConfig(
    filename="sniper.log", level=logging.INFO,
    format="%(asctime)s [%(threadName)s] %(levelname)s %(message)s",
)
logger = logging.getLogger("sniper")

# ══════════════════════════════════════════════════════════════════════
#  GLOBALS
# ══════════════════════════════════════════════════════════════════════
_lock = threading.RLock()
ohlcv, indicators, cooldown_until, current_candle, last_price = {}, {}, {}, {}, {}
pending_signals, active_contracts = {}, {}
locked_symbols: dict = {}
unconfirmed_buys: dict = {}
total_pnl = 0.0
peak_equity = 0.0
max_drawdown = 0.0
win_count = loss_count = daily_trades = consecutive_losses = 0
paused = False
pause_until = datetime.min.replace(tzinfo=timezone.utc)
session_start = datetime.now(timezone.utc)
signal_log: deque[str] = deque(maxlen=20)
session_symbol_stats: dict = {}   # symbol -> {"wins": int, "losses": int, "pnl": float}

# Daily session log — appended on every TP or SL hit; cleared at calendar midnight
# Each entry: {reason, pnl, wins, losses, trades, time, best_sym, best_pnl, worst_sym, worst_pnl, duration_min}
daily_session_log: list = []
_daily_session_log_date: str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

# Per-symbol score history for sparklines — {symbol: deque(maxlen=20)}
symbol_score_history: dict = {sym: deque(maxlen=20) for sym in SYMBOLS}

telegram_app = None
_tg_loop = None
_auto_resume_active = False
_test_trade_sem = threading.Semaphore(1)
_test_trade_active: dict = {}

# ── ML filter state ─────────────────────────────────────────────────
ml_model = None
ml_trained_on = 0
ml_total_trades = 0
ml_lock = threading.Lock()
ml_training_active = False   # True while a retrain is in progress (for the progress bar)


def _ml_load():
    global ml_model, ml_trained_on
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                payload = pickle.load(f)
            ml_model = payload.get("model")
            ml_trained_on = payload.get("trained_on", 0)
            logger.info(f"ML model loaded from {MODEL_PATH} (trained_on={ml_trained_on})")
        except Exception as e:
            logger.error(f"Failed to load ML model: {e}")


def _ml_save():
    try:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({"model": ml_model, "trained_on": ml_trained_on}, f)
    except Exception as e:
        logger.error(f"Failed to save ML model: {e}")


def _ml_export_csv(total: int):
    """Fetch all trades from DB, write to CSV bytes, send as Telegram document."""
    import io, csv as _csv
    try:
        conn = sqlite3.connect("trades.db")
        cols = [
            "id", "timestamp", "symbol", "direction", "barrier",
            "stake", "payout", "profit", "win", "score",
            "wick_atr_ratio", "atr", "atr_ma",
            "ema_fast_slope", "ema_slow_slope", "ema_distance",
        ]
        rows = conn.execute(f"SELECT {', '.join(cols)} FROM trades ORDER BY id").fetchall()
        conn.close()

        buf = io.StringIO()
        writer = _csv.writer(buf)
        writer.writerow(cols)
        writer.writerows(rows)
        csv_bytes = buf.getvalue().encode("utf-8")

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        filename = f"ml_training_{total}trades_{ts}.csv"
        caption = (
            f"📊 <b>ML Training Export</b>\n"
            f"Trained on <b>{total}</b> trades.\n"
            f"All trade features + outcomes — ready for external analysis."
        )
        _send_tg_document(csv_bytes, filename, caption)
        logger.info(f"ML CSV export sent ({total} rows, {len(csv_bytes)} bytes)")
    except Exception as e:
        logger.error(f"ML CSV export failed: {e}")
        _send_tg(f"⚠️ <b>ML CSV export failed</b>\n<code>{e}</code>")


def _ml_progress_bar(pct: float, width: int = 12) -> str:
    filled = int(width * min(1.0, pct))
    return "█" * filled + "░" * (width - filled)


def _ml_train():
    """Train (or retrain) the RandomForest.
    Sends a Telegram progress bar at start → complete (or skipped).
    ml_training_active is already set to True by the caller before this
    thread is spawned; this function clears it in a finally block so the
    flag is always reset even on error."""
    global ml_model, ml_trained_on, ml_training_active

    try:
        try:
            conn = sqlite3.connect("trades.db")
            rows = conn.execute(
                f"SELECT {', '.join(ML_FEATURE_COLS)}, win FROM trades"
            ).fetchall()
            conn.close()
        except Exception as e:
            logger.error(f"ML training query failed: {e}")
            return

        total = len(rows)
        bar_start = _ml_progress_bar(0.0)

        _send_tg(
            f"🤖 <b>ML RETRAINING</b> — started\n"
            f"Training on <b>{total}</b> trades…\n"
            f"<code>[{bar_start}]   0%</code>"
        )

        if total < ML_MIN_TRADES:
            _send_tg(
                f"🤖 <b>ML RETRAINING</b> — skipped\n"
                f"Need {ML_MIN_TRADES} trades, only {total} recorded.\n"
                f"<code>[{_ml_progress_bar(total / ML_MIN_TRADES)}] {int(total / ML_MIN_TRADES * 100)}%</code> (observe-only)"
            )
            return

        X = [[r[i] if r[i] is not None else 0.0 for i in range(len(ML_FEATURE_COLS))] for r in rows]
        y = [r[-1] for r in rows]

        if len(set(y)) < 2:
            _send_tg(
                f"🤖 <b>ML RETRAINING</b> — skipped\n"
                f"Need both wins and losses in history. Only one class found."
            )
            logger.info("ML training skipped: need both wins and losses in trades.db")
            return

        try:
            clf = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
            clf.fit(X, y)
            with ml_lock:
                ml_model = clf
                ml_trained_on = total
            _ml_save()
            logger.info(f"ML model trained on {total} trades")
            bar_done = _ml_progress_bar(1.0)
            _send_tg(
                f"🤖 <b>ML RETRAINING COMPLETE</b> ✅\n"
                f"<code>[{bar_done}] 100%</code>\n"
                f"Trained on <b>{total}</b> trades.\n"
                f"Low-confidence trades will now be filtered."
            )
            # Export all training rows to CSV and send to Telegram
            _ml_export_csv(total)
        except Exception as e:
            logger.error(f"ML training failed: {e}")
            _send_tg(f"🤖 <b>ML RETRAINING FAILED</b>\n<code>{e}</code>")
    finally:
        # Always clear the in-progress flag, even on early returns or exceptions
        with ml_lock:
            ml_training_active = False


def _ml_maybe_retrain(total_trades: int):
    global ml_total_trades, ml_training_active
    ml_total_trades = total_trades
    spawn = False
    with ml_lock:
        # Never spawn a second retrain while one is already running
        if not ml_training_active:
            if ml_trained_on == 0 and total_trades >= ML_MIN_TRADES:
                spawn = True
            elif ml_trained_on and total_trades - ml_trained_on >= ML_RETRAIN_EVERY:
                spawn = True
        if spawn:
            # Set the flag while still holding the lock so no second caller
            # can race in before the thread starts.
            ml_training_active = True
    if spawn:
        threading.Thread(target=_ml_train, daemon=True, name="MLTrain").start()


def _ml_should_trade(details: dict) -> bool:
    with ml_lock:
        model = ml_model
    if model is None:
        return True
    try:
        feats = [[
            details.get("total_score", 0),
            details.get("extension_atr", 0),
            details.get("atr", 0),
            details.get("atr_ma", 0),
            details.get("ema_fast_sl", 0),
            details.get("ema_slow_sl", 0),
            details.get("ema_distance", 0),
        ]]
        proba = model.predict_proba(feats)[0]
        classes = list(model.classes_)
        win_idx = classes.index(1) if 1 in classes else (len(classes) - 1)
        confidence = proba[win_idx]
        return confidence >= ML_CONFIDENCE_MIN
    except Exception as e:
        logger.error(f"ML prediction failed, allowing trade: {e}")
        return True


def _ml_progress_text() -> str:
    with ml_lock:
        trained_on, model, training = ml_trained_on, ml_model, ml_training_active
    total = ml_total_trades
    bar_len = 10

    if training:
        bar = _ml_progress_bar(0.5, bar_len)
        return f"ML Filter  : ⏳ RETRAINING [{bar}] in progress…"

    if model is None:
        pct = min(1.0, total / ML_MIN_TRADES) if ML_MIN_TRADES else 1.0
        filled = int(bar_len * pct)
        bar = "█" * filled + "░" * (bar_len - filled)
        return f"ML Filter  : warming up [{bar}] {total}/{ML_MIN_TRADES} trades (observe-only)"
    else:
        since = max(0, total - trained_on)
        pct = min(1.0, since / ML_RETRAIN_EVERY) if ML_RETRAIN_EVERY else 1.0
        filled = int(bar_len * pct)
        bar = "█" * filled + "░" * (bar_len - filled)
        return f"ML Filter  : ACTIVE [{bar}] next retrain in {max(0, ML_RETRAIN_EVERY - since)} trades"


# ══════════════════════════════════════════════════════════════════════
#  SESSION BEST / WORST ASSET
# ══════════════════════════════════════════════════════════════════════
def _best_worst_session():
    """Return (best_sym, best_stats, worst_sym, worst_stats) for the current
    session. Returns (None, None, None, None) if no trades yet."""
    with _lock:
        stats = {s: dict(v) for s, v in session_symbol_stats.items()
                 if v["wins"] + v["losses"] > 0}
    if not stats:
        return None, None, None, None
    best_sym  = max(stats, key=lambda s: stats[s]["pnl"])
    worst_sym = min(stats, key=lambda s: stats[s]["pnl"])
    return best_sym, stats[best_sym], worst_sym, stats[worst_sym]


def _best_worst_line() -> str:
    """One-liner summary for dashboards."""
    b_sym, b_st, w_sym, w_st = _best_worst_session()
    if b_sym is None:
        return "Best/Worst  : — (no trades yet)"
    b_wr = b_st["wins"] / (b_st["wins"] + b_st["losses"]) * 100 if (b_st["wins"] + b_st["losses"]) else 0
    w_wr = w_st["wins"] / (w_st["wins"] + w_st["losses"]) * 100 if (w_st["wins"] + w_st["losses"]) else 0
    return (
        f"🏅 Best  : {b_sym}  {'+' if b_st['pnl'] >= 0 else ''}${b_st['pnl']:.2f}  {b_wr:.0f}% WR\n"
        f"💔 Worst : {w_sym}  {'+' if w_st['pnl'] >= 0 else ''}${w_st['pnl']:.2f}  {w_wr:.0f}% WR"
    )


def _init_symbol(sym):
    ohlcv[sym] = pd.DataFrame()
    cooldown_until[sym] = datetime.min.replace(tzinfo=timezone.utc)
    current_candle[sym] = None
    last_price[sym] = 0.0
    indicators[sym] = {
        "ema_slow": None, "ema_fast": None,
        "ema_slow_rising": False, "ema_fast_rising": False,
        "ema_dist_increasing": False,
        "atr": None, "atr_ma": None, "atr_rising": False,
        "rsi": None,
        "macd_bullish": False, "macd_hist_rising": False, "macd_hist": None,
        "adx": None, "di_bullish": False,
        "bb_upper": None, "bb_lower": None, "bb_mid": None,
        "bb_squeeze": False, "bb_position": None,
        "ready": False,
    }

for sym in SYMBOLS:
    _init_symbol(sym)

db_queue: queue.Queue = queue.Queue()


def _db_writer():
    conn = sqlite3.connect("trades.db", check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT,    symbol          TEXT,
            direction       TEXT,    barrier         REAL,
            stake           REAL,    payout          REAL,
            profit          REAL,    win             INTEGER,
            score           REAL,    wick_atr_ratio  REAL,
            atr             REAL,    atr_ma          REAL,
            ema_fast_slope  REAL,    ema_slow_slope  REAL,
            ema_distance    REAL
        )
    """)
    conn.commit()
    while True:
        item = db_queue.get()
        if item is None:
            break
        conn.execute("INSERT INTO trades VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", item)
        conn.commit()
        total = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        _ml_maybe_retrain(total)
    conn.close()


def _watch_thread(target, args=(), name="Worker", restartable=True):
    def _wrapper():
        restart_count = 0
        while True:
            try:
                target(*args)
                if not restartable:
                    return
                raise RuntimeError(f"{name} exited unexpectedly")
            except Exception as e:
                restart_count += 1
                logger.error(f"Thread '{name}' crashed: {e}", exc_info=True)
                try:
                    _send_tg(
                        f"⚠️ <b>Worker thread crashed</b>: <code>{name}</code>\n"
                        f"Error: <code>{e}</code>\n"
                        f"Restarting (attempt {restart_count})…"
                    )
                except Exception:
                    pass
                time.sleep(min(30, 3 * restart_count))
    threading.Thread(target=_wrapper, daemon=True, name=name).start()

# DBWriter is started inside main() after _ml_load() so startup order is guaranteed


def get_recent_trades(limit=8):
    conn = sqlite3.connect("trades.db")
    rows = conn.execute(
        "SELECT timestamp, symbol, direction, profit, win, score FROM trades ORDER BY id DESC LIMIT ?",
        (limit,)
    ).fetchall()
    conn.close()
    return rows


def get_db_summary():
    conn = sqlite3.connect("trades.db")
    row = conn.execute(
        "SELECT COUNT(*), SUM(profit), SUM(win), SUM(1-win) FROM trades"
    ).fetchone()
    conn.close()
    return row


def get_alltime_symbol_stats(limit=10):
    conn = sqlite3.connect("trades.db")
    rows = conn.execute(
        "SELECT symbol, COUNT(*), SUM(win), SUM(profit) "
        "FROM trades GROUP BY symbol ORDER BY SUM(profit) DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return rows


def get_alltime_daily_stats(limit=7):
    conn = sqlite3.connect("trades.db")
    rows = conn.execute(
        "SELECT substr(timestamp, 1, 10) AS day, COUNT(*), SUM(win), SUM(profit) "
        "FROM trades GROUP BY day ORDER BY day DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return rows


# ══════════════════════════════════════════════════════════════════════
#  INDICATORS
# ══════════════════════════════════════════════════════════════════════
def update_indicators(symbol: str) -> bool:
    df = ohlcv[symbol]
    if len(df) < 500:
        return False
    df["EMA_SLOW"] = df["Close"].ewm(span=EMA_SLOW, adjust=False).mean()
    df["EMA_FAST"] = df["Close"].ewm(span=EMA_FAST, adjust=False).mean()
    hi, lo, pc = df["High"], df["Low"], df["Close"].shift(1)
    tr = np.maximum(hi - lo, np.maximum(abs(hi - pc), abs(lo - pc)))
    df["ATR"] = tr.rolling(ATR_PERIOD).mean()
    df["ATR_MA"] = df["ATR"].rolling(ATR_MA_PERIOD).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI"] = df["RSI"].fillna(100)

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]

    up_move = hi.diff()
    down_move = -lo.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr14 = tr.ewm(alpha=1 / ATR_PERIOD, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / ATR_PERIOD, adjust=False).mean() / atr14.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / ATR_PERIOD, adjust=False).mean() / atr14.replace(0, np.nan)
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0)
    df["ADX"] = dx.ewm(alpha=1 / ATR_PERIOD, adjust=False).mean()
    df["PLUS_DI"] = plus_di
    df["MINUS_DI"] = minus_di

    bb_mid = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    df["BB_MID"] = bb_mid
    df["BB_UPPER"] = bb_mid + 2 * bb_std
    df["BB_LOWER"] = bb_mid - 2 * bb_std
    df["BB_WIDTH"] = (df["BB_UPPER"] - df["BB_LOWER"]) / bb_mid.replace(0, np.nan)

    with _lock:
        ind = indicators[symbol]
        ind["ema_slow"] = df["EMA_SLOW"].iloc[-1]
        ind["ema_fast"] = df["EMA_FAST"].iloc[-1]
        ind["atr"] = df["ATR"].iloc[-1]
        ind["atr_ma"] = df["ATR_MA"].iloc[-1]
        ind["rsi"] = df["RSI"].iloc[-1]
        ind["ema_slow_rising"] = (df["EMA_SLOW"].diff().iloc[-10:] > 0).sum() >= 7
        ind["ema_fast_rising"] = (df["EMA_FAST"].diff().iloc[-10:] > 0).sum() >= 7
        df["EMADist"] = df["EMA_FAST"] - df["EMA_SLOW"]
        ind["ema_dist_increasing"] = (df["EMADist"].diff().iloc[-5:] > 0).sum() >= 4
        ind["atr_rising"] = (df["ATR"].diff().iloc[-5:] > 0).sum() >= 3

        macd_val = df["MACD"].iloc[-1]
        macd_sig = df["MACD_SIGNAL"].iloc[-1]
        hist = df["MACD_HIST"].iloc[-1]
        ind["macd_bullish"] = bool(macd_val > macd_sig)
        ind["macd_hist"] = round(float(hist), 5) if pd.notna(hist) else None
        ind["macd_hist_rising"] = (df["MACD_HIST"].diff().iloc[-3:] > 0).sum() >= 2

        adx_val = df["ADX"].iloc[-1]
        ind["adx"] = round(float(adx_val), 1) if pd.notna(adx_val) else None
        pdi, mdi = df["PLUS_DI"].iloc[-1], df["MINUS_DI"].iloc[-1]
        ind["di_bullish"] = bool(pd.notna(pdi) and pd.notna(mdi) and pdi > mdi)

        bb_u, bb_l, bb_m = df["BB_UPPER"].iloc[-1], df["BB_LOWER"].iloc[-1], df["BB_MID"].iloc[-1]
        ind["bb_upper"], ind["bb_lower"], ind["bb_mid"] = bb_u, bb_l, bb_m
        bb_width_now = df["BB_WIDTH"].iloc[-1]
        bb_width_avg = df["BB_WIDTH"].rolling(50).mean().iloc[-1]
        ind["bb_squeeze"] = bool(pd.notna(bb_width_now) and pd.notna(bb_width_avg) and bb_width_now < bb_width_avg)
        band_range = (bb_u - bb_l) if pd.notna(bb_u) and pd.notna(bb_l) and (bb_u - bb_l) > 0 else None
        last_close = df["Close"].iloc[-1]
        ind["bb_position"] = float((last_close - bb_l) / band_range) if band_range else None

        ind["ready"] = True
    return True


def score_signal(symbol: str, candle: dict) -> tuple[int, str, dict]:
    with _lock:
        ind = dict(indicators[symbol])
    score, details = 0, {}

    price = candle["Close"]
    ema_slow = ind["ema_slow"] or 0
    ema_fast = ind["ema_fast"] or 0
    direction = "UP" if price > ema_slow else "DOWN"

    trend = 0
    if direction == "UP":
        if ind["ema_fast_rising"] and ind["ema_slow_rising"]:
            trend += 15
        if (ema_fast or 0) > (ema_slow or 0):
            trend += 4
        if ind["ema_dist_increasing"]:
            trend += 6
    else:
        if not ind["ema_fast_rising"] and not ind["ema_slow_rising"]:
            trend += 15
        if (ema_fast or 0) < (ema_slow or 0):
            trend += 4
        if not ind["ema_dist_increasing"]:
            trend += 6
    score += trend
    details["trend"] = trend

    atr = ind["atr"] or 1
    ema_fast_val = ema_fast or price
    extension = abs(price - ema_fast_val) / atr if atr else 0
    rsi = ind.get("rsi")

    entry_quality = 0
    if extension <= 0.5:
        entry_quality += 15
    elif extension <= 1.0:
        entry_quality += 8

    if rsi is not None:
        if direction == "UP":
            if 40 <= rsi <= 65:
                entry_quality += 15
            elif 30 <= rsi < 40 or 65 < rsi <= 75:
                entry_quality += 7
        else:
            if 35 <= rsi <= 60:
                entry_quality += 15
            elif 25 <= rsi < 35 or 60 < rsi <= 70:
                entry_quality += 7
    score += entry_quality
    details["entry_quality"] = entry_quality
    details["extension_atr"] = round(extension, 2)
    details["rsi"] = round(rsi, 1) if rsi is not None else None

    vol_ok = ind["atr_rising"] and (ind["atr"] or 0) > (ind["atr_ma"] or 0)
    vol = 0
    if vol_ok:
        vol += 10
    if ind["atr_rising"]:
        vol += 5
    score += vol
    details["volatility"] = vol

    if direction == "UP":
        mom = 10 if price > ema_fast else 0
    else:
        mom = 10 if price < ema_fast else 0
    score += mom
    details["momentum"] = mom

    macd_agree = ind["macd_bullish"] if direction == "UP" else not ind["macd_bullish"]
    macd_agree = macd_agree and ind["macd_hist_rising"]

    adx_val = ind.get("adx") or 0
    di_agree = ind["di_bullish"] if direction == "UP" else not ind["di_bullish"]
    adx_agree = adx_val >= 20 and di_agree

    bb_pos = ind.get("bb_position")
    if bb_pos is None:
        bb_agree = False
    elif direction == "UP":
        bb_agree = bb_pos <= 0.75
    else:
        bb_agree = bb_pos >= 0.25

    confluence_count = int(macd_agree) + int(adx_agree) + int(bb_agree)
    confluence = 0
    if macd_agree:
        confluence += 8
    if adx_agree:
        confluence += 6
    if bb_agree:
        confluence += 6
    score += confluence
    details["confluence"] = confluence
    details["confluence_count"] = confluence_count

    if confluence_count < 2:
        score = int(score * 0.5)
    details["confluence_gate_passed"] = confluence_count >= 2

    details.update({
        "total_score": score,
        "atr": ind["atr"],
        "atr_ma": ind["atr_ma"],
        "ema_fast_sl": 1 if ind["ema_fast_rising"] else 0,
        "ema_slow_sl": 1 if ind["ema_slow_rising"] else 0,
        "ema_distance": (ema_fast or 0) - (ema_slow or 0),
        "adx": ind.get("adx"),
        "macd_hist": ind.get("macd_hist"),
        "bb_position": round(bb_pos, 2) if bb_pos is not None else None,
    })
    return score, direction, details


# ══════════════════════════════════════════════════════════════════════
#  BARRIER HELPER
# ══════════════════════════════════════════════════════════════════════
def _compute_barrier(symbol: str, direction: str = "UP") -> str:
    with _lock:
        atr = (indicators.get(symbol) or {}).get("atr") or 0.0
    if atr <= 0:
        return "+0.20" if direction == "UP" else "-0.20"
    offset = atr * ATR_BARRIER_MULT
    sign = "+" if direction == "UP" else "-"
    return f"{sign}{offset:.2f}"


# ══════════════════════════════════════════════════════════════════════
#  TELEGRAM HELPERS
# ══════════════════════════════════════════════════════════════════════
def _send_tg(text: str, reply_markup=None, parse_mode: str = "HTML"):
    if not telegram_app or not _tg_loop:
        return

    async def _inner():
        try:
            await telegram_app.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID, text=text,
                reply_markup=reply_markup, parse_mode=parse_mode,
            )
        except Exception as e:
            logger.error(f"Telegram send: {e}")

    fut = asyncio.run_coroutine_threadsafe(_inner(), _tg_loop)
    return fut


def _send_tg_document(file_bytes: bytes, filename: str, caption: str = ""):
    """Send a file/document to Telegram (runs on the bot's async loop)."""
    if not telegram_app or not _tg_loop:
        return

    async def _inner():
        try:
            import io
            await telegram_app.bot.send_document(
                chat_id=TELEGRAM_CHAT_ID,
                document=io.BytesIO(file_bytes),
                filename=filename,
                caption=caption,
                parse_mode="HTML",
            )
        except Exception as e:
            logger.error(f"Telegram send_document: {e}")

    asyncio.run_coroutine_threadsafe(_inner(), _tg_loop)


def _send_tg_wait(text: str, reply_markup=None, parse_mode: str = "HTML", timeout: float = 8.0):
    fut = _send_tg(text, reply_markup=reply_markup, parse_mode=parse_mode)
    if fut is not None:
        try:
            fut.result(timeout=timeout)
        except Exception as e:
            logger.debug(f"_send_tg_wait timeout/error: {e}")


def _score_bar_str(score: int, width: int = 10) -> str:
    filled = max(0, min(width, int(score / 100 * width)))
    return "█" * filled + "░" * (width - filled)


def _component_bar(pts: int, max_pts: int, width: int = 8) -> str:
    filled = round(width * pts / max_pts) if max_pts else 0
    return "█" * filled + "░" * (width - filled)


def _signal_card(sym: str, score: int, direction: str, details: dict) -> str:
    trend  = details.get("trend",         0)
    eq     = details.get("entry_quality", 0)
    vol    = details.get("volatility",    0)
    mom    = details.get("momentum",      0)
    conf   = details.get("confluence",    0)
    conf_n = details.get("confluence_count", 0)

    overall_bar = _score_bar_str(score, width=12)
    t_bar = _component_bar(trend, 25)
    e_bar = _component_bar(eq,    30)
    v_bar = _component_bar(vol,   15)
    m_bar = _component_bar(mom,   10)
    c_bar = _component_bar(conf,  20)

    extension  = details.get("extension_atr", 0)
    rsi        = details.get("rsi")
    adx        = details.get("adx")
    macd_hist  = details.get("macd_hist")
    bb_pos     = details.get("bb_position")
    barrier    = _compute_barrier(sym, direction)
    arrow      = "🟢 BUY" if direction == "UP" else "🔴 SELL"
    gate_ok    = details.get("confluence_gate_passed", True)

    with _lock:
        wc, lc, pnl = win_count, loss_count, total_pnl
    total   = wc + lc
    wr      = f"{wc / total * 100:.0f}%" if total else "—"
    pnl_str = f"{'+'if pnl>=0 else ''}${pnl:.2f}"
    session_line = f"#{total + 1}  |  {wc}W/{lc}L  {wr}  |  P&L {pnl_str}"

    combo_bits = []
    if macd_hist is not None:
        combo_bits.append(f"MACD hist {macd_hist:+.4f}")
    if adx is not None:
        combo_bits.append(f"ADX {adx:.0f}")
    if bb_pos is not None:
        combo_bits.append(f"BB pos {bb_pos:.2f}")
    combo_line = f"   {'  |  '.join(combo_bits)}\n" if combo_bits else ""

    lines = [
        f"🔫 <b>SIGNAL DETECTED</b>  {arrow}",
        f"━━━━━━━━━━━━━━━━━━━━",
        f"📌 Symbol   : <code>{sym}</code>",
        f"🏅 Score    : <b>{score}/100</b>  <code>[{overall_bar}]</code>",
        f"━━━━━━━━━━━━━━━━━━━━",
        f"📈 Trend    : <code>[{t_bar}]</code> {trend:>2}/25"
        f"  {'🟢' if trend >= 15 else '🟡' if trend >= 8 else '🔴'}",
        f"🎯 Entry    : <code>[{e_bar}]</code> {eq:>2}/30"
        f"  ext={extension:.2f}×ATR"
        f"{f', RSI {rsi:.0f}' if rsi is not None else ''}",
        f"⚡ Volatility: <code>[{v_bar}]</code> {vol:>2}/15"
        f"  {'🟢' if vol >= 10 else '🟡' if vol >= 5 else '🔴'}",
        f"🚀 Momentum : <code>[{m_bar}]</code> {mom:>2}/10"
        f"  {'✅' if mom else '—'}",
        f"🧩 Confluence: <code>[{c_bar}]</code> {conf:>2}/20"
        f"  {conf_n}/3 combo  {'🟢' if gate_ok else '🔴 GATED'}",
    ]
    text = "\n".join(lines) + "\n" + combo_line + (
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"💵 Stake  : ${STAKE:.2f}  →  win ~${STAKE + TARGET_PROFIT:.2f}\n"
        f"📍 Barrier: {barrier}  ({DURATION} min)\n"
        f"📋 Session : {session_line}\n"
    )
    return text


def _result_card(sym: str, profit: float, win: bool, details: dict) -> str:
    with _lock:
        wc, lc, pnl, cl = win_count, loss_count, total_pnl, consecutive_losses
    total   = wc + lc
    wr      = wc / total * 100 if total else 0
    pnl_str = f"+${profit:.2f}" if profit > 0 else f"${profit:.2f}"
    streak  = ("✅" * max(0, 3 - cl)) if win else ("❌" * cl)
    return (
        f"{'🏆' if win else '💀'} <b>TRADE {'WIN' if win else 'LOSS'}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📌 <b>Symbol  :</b> <code>{sym}</code>\n"
        f"💵 <b>P&L     :</b> <b>{pnl_str}</b>\n"
        f"📊 <b>Session :</b> {'+' if pnl >= 0 else ''}${pnl:.2f}  ({wc}W / {lc}L  {wr:.0f}%)\n"
        f"🔥 <b>Streak  :</b> {streak or '—'}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Score: {details.get('total_score', '?')}/100  |  "
        f"Entry: {details.get('entry_quality', 0)}/30 "
        f"(ext {details.get('extension_atr', 0):.2f}×ATR)\n"
    )


def _session_summary_text(snap: dict) -> str:
    reason = snap["reason"]
    reason_line = {
        "TP":     f"🎯 <b>DAILY TAKE-PROFIT HIT</b>  (target ${DAILY_PROFIT_TARGET:.2f})",
        "SL":     f"🛑 <b>DAILY STOP-LOSS HIT</b>  (floor ${DAILY_LOSS_LIMIT:.2f})",
        "MANUAL": "📄 <b>SESSION REPORT</b>",
    }.get(reason, "🔁 <b>SESSION LIMIT REACHED</b>")

    pnl   = snap["pnl"]
    wc    = snap["wins"]
    lc    = snap["losses"]
    total = wc + lc
    wr    = wc / total * 100 if total else 0
    dur   = snap["duration"]
    h, r  = divmod(int(dur.total_seconds()), 3600)
    mi    = r // 60
    mdd_pct = (snap["max_dd"] / snap["peak"] * 100) if snap["peak"] > 0 else 0.0

    lines = [
        reason_line,
        "━━━━━━━━━━━━━━━━━━━━",
        f"💵 <b>Session P&L</b> : {'+' if pnl >= 0 else ''}${pnl:.2f}",
        f"📊 <b>Trades</b>      : {total}  ({wc}W / {lc}L  {wr:.0f}%)",
        f"⏱ <b>Duration</b>    : {h}h {mi}m",
        "",
        "<b>Drawdown</b>",
        f"  Peak Equity : ${snap['peak']:.2f}",
        f"  Max DD      : -${snap['max_dd']:.2f}  ({mdd_pct:.1f}%)",
        "",
    ]

    # Best / worst asset for the completed session
    symbols = snap.get("symbols") or {}
    if symbols:
        traded = {s: v for s, v in symbols.items() if v["wins"] + v["losses"] > 0}
        if traded:
            best_sym  = max(traded, key=lambda s: traded[s]["pnl"])
            worst_sym = min(traded, key=lambda s: traded[s]["pnl"])
            bv, wv = traded[best_sym], traded[worst_sym]
            b_wr = bv["wins"] / (bv["wins"] + bv["losses"]) * 100 if (bv["wins"] + bv["losses"]) else 0
            w_wr = wv["wins"] / (wv["wins"] + wv["losses"]) * 100 if (wv["wins"] + wv["losses"]) else 0
            lines += [
                "<b>Best / Worst Asset</b>",
                f"  🏅 Best  : <code>{best_sym}</code>  {'+' if bv['pnl'] >= 0 else ''}${bv['pnl']:.2f}  ({b_wr:.0f}% WR)",
                f"  💔 Worst : <code>{worst_sym}</code>  {'+' if wv['pnl'] >= 0 else ''}${wv['pnl']:.2f}  ({w_wr:.0f}% WR)",
                "",
            ]

        lines.append("<b>Asset Breakdown</b>")
        for sym, s in sorted(symbols.items(), key=lambda kv: kv[1]["pnl"], reverse=True):
            s_total = s["wins"] + s["losses"]
            s_wr = s["wins"] / s_total * 100 if s_total else 0
            sign = "+" if s["pnl"] >= 0 else ""
            lines.append(
                f"  <code>{sym:<10}</code> {s['wins']}W/{s['losses']}L "
                f"({s_wr:.0f}%)  {sign}${s['pnl']:.2f}"
            )
    else:
        lines.append("  — no trades this session —")

    lines += [
        "━━━━━━━━━━━━━━━━━━━━",
        "🔄 <b>New session started — bot keeps trading.</b>",
    ]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
#  ORDER MANAGEMENT
# ══════════════════════════════════════════════════════════════════════
def _reserve_trade_slot(symbol: str, now: datetime) -> bool:
    global daily_trades
    with _lock:
        if (
            not paused
            and now >= pause_until
            and now >= cooldown_until[symbol]
            and daily_trades < MAX_DAILY_TRADES
            and total_pnl > DAILY_LOSS_LIMIT
        ):
            daily_trades += 1
            cooldown_until[symbol] = now + timedelta(minutes=COOLDOWN_MINUTES)
            return True
        return False


def _release_trade_slot(symbol: str):
    global daily_trades
    with _lock:
        daily_trades = max(0, daily_trades - 1)
        cooldown_until[symbol] = datetime.min.replace(tzinfo=timezone.utc)


def request_proposal(ws, symbol: str, details: dict, direction: str):
    details["direction"] = direction
    details["proposal_time"] = datetime.now(timezone.utc)
    with _lock:
        pending_signals[symbol] = details
    ws.send(json.dumps({
        "proposal": 1,
        "amount": STAKE,
        "basis": "stake",
        "contract_type": CONTRACT_TYPE,
        "currency": "USD",
        "duration": DURATION,
        "duration_unit": "m",
        "symbol": symbol,
        "barrier": _compute_barrier(symbol, direction),
    }))


def on_proposal(ws, msg: dict, symbol: str):
    prop = msg.get("proposal", {})
    pid = prop.get("id")
    with _lock:
        if symbol not in pending_signals:
            return
        if pending_signals[symbol].get("proposal_id"):
            return
        pending_signals[symbol]["proposal_id"] = pid
    if pid:
        ws.send(json.dumps({"buy": pid, "price": STAKE}))
        with _lock:
            if symbol in pending_signals:
                pending_signals[symbol]["buy_sent_at"] = datetime.now(timezone.utc)


def on_buy(ws, msg: dict, symbol: str):
    buy = msg.get("buy", {})
    cid = buy.get("contract_id")
    with _lock:
        details = pending_signals.pop(symbol, None)
        if details is None:
            ub = unconfirmed_buys.pop(symbol, None)
            if ub:
                details = ub["details"]
            else:
                return
    direction = details.get("direction", "UP")
    if not cid:
        _release_trade_slot(symbol)
        return
    with _lock:
        active_contracts[cid] = {
            "symbol": symbol,
            "direction": direction,
            "barrier": buy.get("barrier"),
            "stake": buy.get("buy_price", STAKE),
            "payout": buy.get("payout"),
            "entry_time": datetime.now(timezone.utc),
            "entry_price": last_price.get(symbol, 0),
            "details": details,
            "settled": False,
        }
    ws.send(json.dumps({
        "proposal_open_contract": 1,
        "contract_id": int(cid),
        "subscribe": 1,
    }))
    _send_tg(_signal_card(symbol, details.get("total_score", 0), direction, details))
    _log(f"🔥 OPEN  {symbol} {direction}  score={details.get('total_score','?')}/100  cid={cid}")


def on_contract_update(ws, msg: dict, symbol: str):
    global total_pnl, win_count, loss_count, consecutive_losses, paused, pause_until
    global daily_trades, _auto_resume_active, peak_equity, max_drawdown, session_start
    global session_symbol_stats, daily_session_log, _daily_session_log_date

    contract = msg.get("proposal_open_contract", {})
    cid = contract.get("contract_id")
    if not cid:
        return

    with _lock:
        if cid not in active_contracts:
            return
        info = active_contracts[cid]
        if info.get("settled"):
            return

        spot = contract.get("current_spot")
        if spot:
            last_price[symbol] = float(spot)

        if not contract.get("is_expired"):
            return

        info["settled"] = True
        profit = float(contract.get("profit", 0))
        win = profit > 0
        d = info.get("details", {})

        if win:
            win_count += 1
            consecutive_losses = 0
        else:
            loss_count += 1
            consecutive_losses += 1

        total_pnl += profit
        peak_equity = max(peak_equity, total_pnl)
        max_drawdown = max(max_drawdown, peak_equity - total_pnl)

        stats = session_symbol_stats.setdefault(symbol, {"wins": 0, "losses": 0, "pnl": 0.0})
        stats["pnl"] += profit
        if win:
            stats["wins"] += 1
        else:
            stats["losses"] += 1

        cl        = consecutive_losses
        pnl_snap  = total_pnl
        dt_snap   = daily_trades

        del active_contracts[cid]

        trigger_consec = cl >= MAX_CONSECUTIVE_LOSSES
        trigger_floor  = pnl_snap <= DAILY_LOSS_LIMIT
        trigger_profit = DAILY_PROFIT_TARGET > 0 and pnl_snap >= DAILY_PROFIT_TARGET
        session_reset_trigger = trigger_floor or trigger_profit

        session_snapshot = None
        needs_resume_thread = False

        if session_reset_trigger:
            reason = "TP" if trigger_profit else "SL"
            _sym_snap = {s: dict(v) for s, v in session_symbol_stats.items()}
            _dur      = datetime.now(timezone.utc) - session_start
            session_snapshot = {
                "reason": reason,
                "pnl": pnl_snap,
                "wins": win_count,
                "losses": loss_count,
                "trades": dt_snap,
                "peak": peak_equity,
                "max_dd": max_drawdown,
                "symbols": _sym_snap,
                "duration": _dur,
            }
            # --- persist to daily session log ---
            _traded = {s: v for s, v in _sym_snap.items() if v["wins"] + v["losses"] > 0}
            _best_s  = max(_traded, key=lambda s: _traded[s]["pnl"]) if _traded else None
            _worst_s = min(_traded, key=lambda s: _traded[s]["pnl"]) if _traded else None
            today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if today_str != _daily_session_log_date:
                daily_session_log.clear()
                _daily_session_log_date = today_str
            daily_session_log.append({
                "reason":       reason,
                "pnl":          pnl_snap,
                "wins":         win_count,
                "losses":       loss_count,
                "trades":       dt_snap,
                "time":         datetime.now(timezone.utc).strftime("%H:%M UTC"),
                "duration_min": max(1, int(_dur.total_seconds() // 60)),
                "best_sym":     _best_s,
                "best_pnl":     _traded[_best_s]["pnl"] if _best_s else 0.0,
                "worst_sym":    _worst_s,
                "worst_pnl":    _traded[_worst_s]["pnl"] if _worst_s else 0.0,
            })
            # --- reset session ---
            total_pnl = 0.0
            win_count = 0
            loss_count = 0
            daily_trades = 0
            consecutive_losses = 0
            peak_equity = 0.0
            max_drawdown = 0.0
            session_symbol_stats = {}
            session_start = datetime.now(timezone.utc)
            paused = False
            pause_until = datetime.min.replace(tzinfo=timezone.utc)
            _auto_resume_active = False
        elif trigger_consec and not paused:
            paused = True
            pause_until = datetime.now(timezone.utc) + timedelta(seconds=PAUSE_MINUTES * 60)
            needs_resume_thread = not _auto_resume_active
            if needs_resume_thread:
                _auto_resume_active = True
        resume_at = pause_until

    db_queue.put((
        datetime.now(timezone.utc).isoformat(), symbol, info["direction"],
        info["barrier"], info["stake"], info["payout"], profit, int(win),
        d.get("total_score", 0), d.get("extension_atr", 0),
        d.get("atr", 0), d.get("atr_ma", 0),
        d.get("ema_fast_sl", 0), d.get("ema_slow_sl", 0), d.get("ema_distance", 0),
    ))

    _send_tg(_result_card(symbol, profit, win, d))
    _log(f"{'WIN' if win else 'LOSS'}  {symbol}  ${profit:+.2f}  total=${pnl_snap:+.2f}")

    if session_snapshot:
        _send_tg(_session_summary_text(session_snapshot))
        _log(f"🔁 SESSION RESET ({session_snapshot['reason']})  "
             f"final=${session_snapshot['pnl']:+.2f}  — fresh session started")
    elif trigger_consec:
        _send_tg(
            f"⛔ <b>BOT PAUSED</b> – {MAX_CONSECUTIVE_LOSSES} consecutive losses.\n"
            f"Auto-resuming in {PAUSE_MINUTES} minutes."
        )

    if needs_resume_thread:
        def _auto_resume():
            global paused, consecutive_losses, _auto_resume_active
            wait = max(0, (resume_at - datetime.now(timezone.utc)).total_seconds())
            time.sleep(wait + 1)
            with _lock:
                _auto_resume_active = False
                if paused and datetime.now(timezone.utc) >= pause_until:
                    paused = False
                    consecutive_losses = 0
            _send_tg("▶ <b>Bot auto-resumed.</b>")
            _log("▶ Auto-resumed from timed pause")
        threading.Thread(target=_auto_resume, daemon=True, name="AutoResume").start()


# ══════════════════════════════════════════════════════════════════════
#  HISTORY LOADER
# ══════════════════════════════════════════════════════════════════════
def fetch_history(symbol: str, hard_timeout: int = 60):
    _log(f"Loading history: {symbol}…")
    all_candles, end_epoch = [], int(datetime.now(timezone.utc).timestamp())
    wall_end = time.time() + hard_timeout
    ws_obj = None
    try:
        ws_obj = websocket.WebSocket()
        ws_obj.connect("wss://ws.derivws.com/websockets/v3?app_id=1089", timeout=10)
        ws_obj.send(json.dumps({"authorize": DERIV_APP_TOKEN}))
        ws_obj.settimeout(8)
        while time.time() < wall_end:
            try:
                r = json.loads(ws_obj.recv())
            except Exception:
                break
            if r.get("msg_type") == "authorize":
                break
            if "error" in r:
                _log(f"⚠  {symbol}: auth denied")
                return
        for _ in range(3):
            if time.time() >= wall_end:
                break
            ws_obj.send(json.dumps({
                "ticks_history": symbol, "end": end_epoch,
                "style": "candles", "granularity": 60,
                "count": 500, "adjust_start_time": 1,
            }))
            candles = []
            t0 = time.time()
            while time.time() - t0 < 12 and time.time() < wall_end:
                ws_obj.settimeout(max(1, min(12, wall_end - time.time())))
                try:
                    r = json.loads(ws_obj.recv())
                except websocket.WebSocketTimeoutException:
                    break
                except Exception:
                    break
                if r.get("msg_type") == "candles":
                    candles = r.get("candles", [])
                    break
                if "error" in r:
                    _log(f"⚠  {symbol}: {r['error'].get('message', 'API error')} – skipping history")
                    candles = []
                    break
            if not candles:
                break
            df = pd.DataFrame(candles)
            df["epoch"] = df["epoch"].astype(int)
            all_candles.append(df.sort_values("epoch"))
            if len(candles) < 500:
                break
            end_epoch = int(df["epoch"].min()) - 1
    except Exception as e:
        logger.error(f"History {symbol}: {e}")
    finally:
        try:
            if ws_obj:
                ws_obj.close()
        except Exception:
            pass
    if all_candles:
        full = (
            pd.concat(all_candles, ignore_index=True)
            .drop_duplicates("epoch")
            .sort_values("epoch")
        )
        full["time"] = pd.to_datetime(full["epoch"], unit="s", utc=True)
        full.set_index("time", inplace=True)
        full = full.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
        with _lock:
            ohlcv[symbol] = full[["Open", "High", "Low", "Close"]].astype(float)
        update_indicators(symbol)
        _log(f"✓ {symbol}: {len(full)} candles loaded")
    else:
        _log(f"⚠  {symbol}: no history – will accumulate live")


# ══════════════════════════════════════════════════════════════════════
#  LIVE WEBSOCKET
# ══════════════════════════════════════════════════════════════════════
def _on_open(ws, symbol: str):
    ws.send(json.dumps({"authorize": DERIV_APP_TOKEN}))
    threading.Timer(1.0, lambda: ws.send(json.dumps({
        "ticks_history": symbol, "subscribe": 1,
        "granularity": 60, "style": "candles", "end": "latest",
    }))).start()
    _log(f"⚡ {symbol} connected")


def _on_message(ws, message: str, symbol: str):
    try:
        msg   = json.loads(message)
        mtype = msg.get("msg_type")
        if mtype == "candles":
            candles = msg.get("candles", [])
            if not candles:
                return
            c = candles[-1]
            if c.get("close"):
                with _lock:
                    last_price[symbol] = float(c["close"])
            with _lock:
                current_candle[symbol] = dict(c)

        elif mtype == "ohlc":
            ohlc = msg.get("ohlc", {})
            if not ohlc:
                return
            c = {
                "epoch": int(ohlc.get("open_time", 0)),
                "open":  ohlc.get("open"),
                "high":  ohlc.get("high"),
                "low":   ohlc.get("low"),
                "close": ohlc.get("close"),
            }
            if c.get("close"):
                with _lock:
                    last_price[symbol] = float(c["close"])
            with _lock:
                prev = current_candle[symbol]

            if prev is not None and prev["epoch"] != c["epoch"]:
                closed  = prev
                new_row = pd.DataFrame(
                    [{"Open":  float(closed["open"]),
                      "High":  float(closed["high"]),
                      "Low":   float(closed["low"]),
                      "Close": float(closed["close"])}],
                    index=[pd.to_datetime(closed["epoch"], unit="s", utc=True)],
                )
                with _lock:
                    ohlcv[symbol] = pd.concat([ohlcv[symbol], new_row]).iloc[-5000:]
                if update_indicators(symbol):
                    now = datetime.now(timezone.utc)
                    row_c = {
                        "Close": float(closed["close"]),
                        "Open":  float(closed["open"]),
                        "High":  float(closed["high"]),
                        "Low":   float(closed["low"]),
                    }
                    score, direction, details = score_signal(symbol, row_c)
                    details["direction"] = direction

                    if score >= SCORE_THRESHOLD:
                        with _lock:
                            already_locked = symbol in locked_symbols
                            cooldown_left = max(0, int((cooldown_until[symbol] - now).total_seconds() // 60))
                            locked_symbols[symbol] = {
                                "score": score,
                                "details": details,
                                "direction": direction,
                                "lock_price": float(closed["close"]),
                                "lock_time": now,
                            }
                        if not already_locked:
                            _send_tg(
                                f"🔒 <b>LOCKED</b> — {symbol}\n"
                                f"Score <b>{score}/100</b> {direction}  |  "
                                f"Entry {details.get('entry_quality',0)}/30 "
                                f"(ext {details.get('extension_atr',0):.2f}×ATR)  |  "
                                f"{('cooldown ' + str(cooldown_left) + 'm' if cooldown_left else 'armed for tick entry')}"
                            )
                    else:
                        with _lock:
                            old_lock = locked_symbols.pop(symbol, None)
                        if old_lock:
                            _send_tg(f"🔓 {symbol} unlocked — score dropped to {score}/100")

                with _lock:
                    current_candle[symbol] = dict(c)
            else:
                with _lock:
                    lock = locked_symbols.get(symbol)
                if lock:
                    now = datetime.now(timezone.utc)
                    if (now - lock["lock_time"]).total_seconds() > 120:
                        with _lock:
                            locked_symbols.pop(symbol, None)
                        lock = None
                if lock:
                    row_c = {
                        "Close": float(c["close"]),
                        "Open":  float(c["open"]),
                        "High":  float(c["high"]),
                        "Low":   float(c["low"]),
                    }
                    score, direction, details = score_signal(symbol, row_c)
                    details["direction"] = direction
                    if score >= SCORE_THRESHOLD and direction == lock["direction"]:
                        atr = (indicators.get(symbol) or {}).get("atr") or 0.0
                        lock_price = lock["lock_price"]
                        price = row_c["Close"]
                        pullback = False
                        if direction == "UP" and lock_price - price >= 0.1 * atr:
                            pullback = True
                        elif direction == "DOWN" and price - lock_price >= 0.1 * atr:
                            pullback = True
                        entry_passed = pullback or details.get("entry_quality", 0) >= 15
                        if entry_passed:
                            with _lock:
                                locked_symbols.pop(symbol, None)
                            if not _ml_should_trade(details):
                                _log(f"🤖 {symbol} {direction} skipped — ML predicted low confidence")
                            elif _reserve_trade_slot(symbol, datetime.now(timezone.utc)):
                                request_proposal(ws, symbol, details, direction)
                                _log(f"🎯 {symbol} {direction} TICK-ENTRY  score={score}/100  "
                                     f"entry_quality={details.get('entry_quality',0)}/30")
                            else:
                                _send_tg(f"⚠️ {symbol} tick-entry passed but risk gate closed — skipped")

                with _lock:
                    current_candle[symbol] = dict(c)

        elif mtype == "proposal":
            on_proposal(ws, msg, symbol)
        elif mtype == "buy":
            on_buy(ws, msg, symbol)
        elif mtype == "proposal_open_contract":
            on_contract_update(ws, msg, symbol)
        elif mtype == "error":
            err = msg.get("error", {})
            logger.error(f"{symbol} API error: {err.get('message', err)}")
            with _lock:
                if symbol in pending_signals:
                    det = pending_signals[symbol]
                    if det.get("proposal_id"):
                        pass
                    else:
                        pending_signals.pop(symbol, None)
                        _release_trade_slot(symbol)
    except json.JSONDecodeError as e:
        logger.error(f"{symbol} bad JSON: {e}")
    except Exception as e:
        logger.exception(f"{symbol} on_message exception: {e}")


def _on_error(ws, error):
    logger.error(f"WS error: {error}")


def _pending_trade_timeout_loop():
    while True:
        time.sleep(10)
        try:
            now = datetime.now(timezone.utc)
            with _lock:
                for symbol in list(pending_signals.keys()):
                    det = pending_signals[symbol]
                    buy_sent_at = det.get("buy_sent_at")
                    if buy_sent_at:
                        if (now - buy_sent_at).total_seconds() > 60:
                            det = pending_signals.pop(symbol, None)
                            unconfirmed_buys[symbol] = {
                                "details": det,
                                "expires_at": now + timedelta(seconds=120),
                            }
                            _log(f"⏰ {symbol} buy ack missing for 60s — holding slot for 120s in case of late ack")
                        continue
                    if (now - det.get("proposal_time", now)).total_seconds() > 30:
                        pending_signals.pop(symbol, None)
                        _release_trade_slot(symbol)
                        _log(f"⏰ {symbol} pending proposal timed out — slot released")
                for symbol in list(unconfirmed_buys.keys()):
                    if now >= unconfirmed_buys[symbol]["expires_at"]:
                        unconfirmed_buys.pop(symbol, None)
                        _release_trade_slot(symbol)
                        _send_tg(
                            f"⚠️ <b>UNCONFIRMED TRADE EXPIRED</b> — {symbol}\n"
                            f"A buy order was sent but no ack was received within 3 minutes.\n"
                            f"The trade slot has been released; please check your Deriv account."
                        )
                        _log(f"⏰ {symbol} unconfirmed buy expired — slot released")
        except Exception as e:
            logger.error(f"pending_trade_timeout_loop: {e}")


def _ws_thread(symbol: str):
    while True:
        try:
            ws_app = websocket.WebSocketApp(
                "wss://ws.derivws.com/websockets/v3?app_id=1089",
                on_open    = lambda ws:      _on_open(ws, symbol),
                on_message = lambda ws, msg: _on_message(ws, msg, symbol),
                on_error   = _on_error,
            )
            ws_app.run_forever(ping_interval=20, ping_timeout=10)
        except Exception as e:
            logger.error(f"{symbol} WS thread exception: {e}")
        _log(f"⚠  {symbol} disconnected. Reconnecting in 5s…")
        time.sleep(5)


def _log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    with _lock:
        signal_log.appendleft(f"[{ts}] {msg}")
    logger.info(msg)


# ══════════════════════════════════════════════════════════════════════
#  TELEGRAM KEYBOARD BUILDERS
# ══════════════════════════════════════════════════════════════════════
def _main_kb() -> InlineKeyboardMarkup:
    with _lock:
        is_paused = paused
    pause_lbl = "▶ Resume" if is_paused else "⏸ Pause"
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📊 Status",         callback_data="status"),
         InlineKeyboardButton("💰 P&L",            callback_data="pnl")],
        [InlineKeyboardButton("📜 History",        callback_data="history"),
         InlineKeyboardButton("📋 Log",            callback_data="signals")],
        [InlineKeyboardButton("📄 Session Report", callback_data="session_report"),
         InlineKeyboardButton("🏆 All-Time",       callback_data="alltime")],
        [InlineKeyboardButton("📅 Day History",    callback_data="daily_history"),
         InlineKeyboardButton("🏅 Best/Worst",     callback_data="best_worst")],
        [InlineKeyboardButton("📈 Scores",         callback_data="score_sparklines"),
         InlineKeyboardButton("⚙ Settings",       callback_data="settings")],
        [InlineKeyboardButton(pause_lbl,           callback_data="toggle_pause"),
         InlineKeyboardButton("⏭ Skip Symbol",    callback_data="skip_menu")],
        [InlineKeyboardButton("🔄 Refresh",        callback_data="refresh"),
         InlineKeyboardButton("🧪 Test Trade",     callback_data="test_menu")],
    ])


def _skip_kb() -> InlineKeyboardMarkup:
    btns = [[InlineKeyboardButton(s, callback_data=f"skip_{s}")] for s in SYMBOLS]
    btns.append([InlineKeyboardButton("🔙 Back", callback_data="main_menu")])
    return InlineKeyboardMarkup(btns)


def _test_group_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📊 Volatility  (R_*)",      callback_data="tg_vol")],
        [InlineKeyboardButton("⚡ Volatility 1s (1HZ*)",  callback_data="tg_vol1s")],
        [InlineKeyboardButton("📉 Range Break (RD*)",     callback_data="tg_rdb")],
        [InlineKeyboardButton("🔙 Back",                  callback_data="main_menu")],
    ])


def _test_sym_kb(group: str) -> InlineKeyboardMarkup:
    groups = {
        "tg_vol":   SYNTH_VOLATILITY,
        "tg_vol1s": SYNTH_VOLATILITY_1S,
        "tg_rdb":   SYNTH_RANGE_BREAK,
    }
    syms = groups.get(group, SYNTH_VOLATILITY)
    rows = []
    for i in range(0, len(syms), 2):
        if i + 1 >= len(syms):
            rows.append([InlineKeyboardButton(syms[i], callback_data=f"test_sym_{syms[i]}")])
        else:
            rows.append([
                InlineKeyboardButton(syms[i],     callback_data=f"test_sym_{syms[i]}"),
                InlineKeyboardButton(syms[i + 1], callback_data=f"test_sym_{syms[i + 1]}"),
            ])
    rows.append([InlineKeyboardButton("🔙 Back", callback_data="test_menu")])
    return InlineKeyboardMarkup(rows)


# ══════════════════════════════════════════════════════════════════════
#  TEXT BUILDERS
# ══════════════════════════════════════════════════════════════════════
def _status_text():
    now = datetime.now(timezone.utc)
    with _lock:
        contracts = {cid: dict(c) for cid, c in active_contracts.items()}
        cds       = {s: t for s, t in cooldown_until.items() if t > now}
        is_paused = paused
        trades    = daily_trades
    lines = [
        "📊 <b>BOT STATUS</b>\n━━━━━━━━━━━━━━━━━━━━",
        f"State      : {'⏸ PAUSED' if is_paused else '▶ RUNNING'}",
        f"Active     : {len(contracts)} trade(s)",
        f"Day trades : {trades}",
        _ml_progress_text(),
        "",
        _best_worst_line(),
    ]
    for cid, c in contracts.items():
        exp  = c["entry_time"] + timedelta(minutes=DURATION)
        left = max(0, int((exp - now).total_seconds()))
        m, s = divmod(left, 60)
        lines.append(f"  • {c['symbol']} #{cid} {c['direction']} {m:02d}:{s:02d} left")
    lines.append(f"Cooldowns  : {len(cds)}")
    for sym, t in cds.items():
        lines.append(f"  • {sym} — {int((t - now).total_seconds() // 60)}m left")
    return "\n".join(lines)


def _pnl_text():
    with _lock:
        pnl, wc, lc, cl, dt, is_paused = total_pnl, win_count, loss_count, consecutive_losses, daily_trades, paused
        peak, mdd = peak_equity, max_drawdown
    total  = wc + lc
    wr     = wc / total * 100 if total else 0
    db_sum = get_db_summary()
    at, ap, aw, al = db_sum if db_sum[0] else (0, 0, 0, 0)
    at_wr  = aw / at * 100 if at else 0
    cur_dd     = max(0.0, peak - pnl)
    mdd_pct    = (mdd / peak * 100) if peak > 0 else 0.0

    b_sym, b_st, w_sym, w_st = _best_worst_session()
    bw_lines = ""
    if b_sym:
        b_wr = b_st["wins"] / (b_st["wins"] + b_st["losses"]) * 100 if (b_st["wins"] + b_st["losses"]) else 0
        w_wr = w_st["wins"] / (w_st["wins"] + w_st["losses"]) * 100 if (w_st["wins"] + w_st["losses"]) else 0
        bw_lines = (
            f"\n<b>Best / Worst Asset (session)</b>\n"
            f"  🏅 Best  : <code>{b_sym}</code>  {'+' if b_st['pnl'] >= 0 else ''}${b_st['pnl']:.2f}  ({b_wr:.0f}% WR)\n"
            f"  💔 Worst : <code>{w_sym}</code>  {'+' if w_st['pnl'] >= 0 else ''}${w_st['pnl']:.2f}  ({w_wr:.0f}% WR)\n"
        )

    return (
        f"💰 <b>P&L REPORT</b>\n━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>Session</b>\n"
        f"  P&L      : <b>{'+' if pnl >= 0 else ''}${pnl:.2f}</b>\n"
        f"  Trades   : {total}  ({wc}W / {lc}L)\n"
        f"  Win Rate : {wr:.1f}%\n"
        f"  Streak   : {'🔴' * cl if cl else '🟢 No losses'}\n"
        f"{bw_lines}"
        f"\n<b>Drawdown</b>\n"
        f"  Peak Equity  : ${peak:.2f}\n"
        f"  Current DD   : -${cur_dd:.2f}\n"
        f"  Max DD       : -${mdd:.2f}  ({mdd_pct:.1f}%)\n\n"
        f"<b>All-time (DB)</b>\n"
        f"  P&L      : {'+' if (ap or 0) >= 0 else ''}${(ap or 0):.2f}\n"
        f"  Trades   : {at}  ({aw}W / {al}L)\n"
        f"  Win Rate : {at_wr:.1f}%\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Session TP: ${DAILY_PROFIT_TARGET:.2f}  |  Session SL: ${DAILY_LOSS_LIMIT}\n"
        f"<i>Hitting either auto-resets the session and keeps trading.</i>"
    )


def _live_session_report_text():
    with _lock:
        snap = {
            "reason": "MANUAL",
            "pnl": total_pnl,
            "wins": win_count,
            "losses": loss_count,
            "trades": daily_trades,
            "peak": peak_equity,
            "max_dd": max_drawdown,
            "symbols": {s: dict(v) for s, v in session_symbol_stats.items()},
            "duration": datetime.now(timezone.utc) - session_start,
        }
    text = _session_summary_text(snap)
    text = text.replace(
        "🔄 <b>New session started — bot keeps trading.</b>",
        "📄 <i>Live snapshot — session continues, nothing was reset.</i>",
    )
    return text


def _alltime_text():
    db_sum = get_db_summary()
    at, ap, aw, al = db_sum if db_sum[0] else (0, 0, 0, 0)
    ap = ap or 0.0
    at_wr = aw / at * 100 if at else 0.0
    avg_trade = ap / at if at else 0.0

    lines = [
        "🏆 <b>ALL-TIME SCOREBOARD</b>",
        "━━━━━━━━━━━━━━━━━━━━",
        f"💵 <b>Total P&L</b>  : {'+' if ap >= 0 else ''}${ap:.2f}",
        f"📊 <b>Trades</b>     : {at}  ({aw}W / {al}L)",
        f"🎯 <b>Win Rate</b>   : {at_wr:.1f}%",
        f"📈 <b>Avg/Trade</b>  : {'+' if avg_trade >= 0 else ''}${avg_trade:.2f}",
        "",
        "<b>By Symbol (all-time)</b>",
    ]
    sym_rows = get_alltime_symbol_stats()
    if sym_rows:
        for sym, cnt, wins, pnl in sym_rows:
            wins = wins or 0
            pnl = pnl or 0.0
            wr = wins / cnt * 100 if cnt else 0
            sign = "+" if pnl >= 0 else ""
            lines.append(f"  <code>{sym:<10}</code> {cnt} trades  {wins}W ({wr:.0f}%)  {sign}${pnl:.2f}")
    else:
        lines.append("  — no trades recorded yet —")

    lines += ["", "<b>Last 7 Days</b>"]
    day_rows = get_alltime_daily_stats()
    if day_rows:
        for day, cnt, wins, pnl in day_rows:
            wins = wins or 0
            pnl = pnl or 0.0
            sign = "+" if pnl >= 0 else ""
            lines.append(f"  {day}  {cnt} trades  {wins}W  {sign}${pnl:.2f}")
    else:
        lines.append("  — no trades recorded yet —")

    lines += [
        "━━━━━━━━━━━━━━━━━━━━",
        "<i>Persists across restarts, sessions, and days (stored in trades.db).</i>",
    ]
    return "\n".join(lines)


def _history_text():
    rows = get_recent_trades(8)
    if not rows:
        return "📜 <b>No trade history yet.</b>"
    lines = ["📜 <b>Last 8 Trades</b>\n━━━━━━━━━━━━━━━━━━━━"]
    for ts, sym, direction, profit, win, score in rows:
        sign = "+" if profit > 0 else ""
        lines.append(
            f"{'🏆' if win else '💀'}  {ts[11:16]}  <code>{sym:<7}</code> {direction} "
            f"{sign}${profit:.2f}  score={score:.0f}"
        )
    return "\n".join(lines)


def _signals_text():
    with _lock:
        lines = list(signal_log)
    if not lines:
        return "📋 <b>No signals logged yet this session.</b>"
    return "📋 <b>Recent Signal Log</b>\n━━━━━━━━━━━━━━━━━━━━\n" + "\n".join(lines[:15])


def _daily_history_text() -> str:
    """Show every TP / SL hit recorded today.
    Also clears the log if UTC date has rolled past the stored date,
    so the view is never stale from a prior calendar day."""
    global daily_session_log, _daily_session_log_date
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with _lock:
        if today != _daily_session_log_date:
            daily_session_log = []
            _daily_session_log_date = today
        log = list(daily_session_log)

    if not log:
        return (
            "📅 <b>Daily Session History</b>\n━━━━━━━━━━━━━━━━━━━━\n"
            f"Date: {today}\n\n"
            "— No TP or SL hits yet today. —\n\n"
            "<i>Each time the bot hits its daily TP or SL it resets and "
            "this record is updated.</i>"
        )

    tp_hits = sum(1 for e in log if e["reason"] == "TP")
    sl_hits = sum(1 for e in log if e["reason"] == "SL")
    total_day_pnl = sum(e["pnl"] for e in log)
    total_day_trades = sum(e["trades"] for e in log)
    total_day_wins   = sum(e["wins"]   for e in log)
    total_day_losses = sum(e["losses"] for e in log)
    total_possible   = total_day_wins + total_day_losses
    day_wr = total_day_wins / total_possible * 100 if total_possible else 0

    lines = [
        "📅 <b>Daily Session History</b>",
        "━━━━━━━━━━━━━━━━━━━━",
        f"Date   : {today}",
        f"🎯 TP hits  : <b>{tp_hits}</b>   🛑 SL hits : <b>{sl_hits}</b>",
        f"Day P&L     : <b>{'+' if total_day_pnl >= 0 else ''}${total_day_pnl:.2f}</b>",
        f"Day trades  : {total_day_trades}  ({total_day_wins}W / {total_day_losses}L  {day_wr:.0f}%)",
        "━━━━━━━━━━━━━━━━━━━━",
    ]

    for i, e in enumerate(log, 1):
        icon = "🎯" if e["reason"] == "TP" else "🛑"
        sign = "+" if e["pnl"] >= 0 else ""
        h, m_rem = divmod(e["duration_min"], 60)
        dur_str  = f"{h}h {m_rem}m" if h else f"{m_rem}m"
        entry = (
            f"{icon} <b>Session {i}</b>  [{e['time']}]  {dur_str}\n"
            f"   P&L: <b>{sign}${e['pnl']:.2f}</b>  ·  "
            f"Trades: {e['trades']} ({e['wins']}W/{e['losses']}L)"
        )
        if e["best_sym"]:
            b_sign = "+" if e["best_pnl"] >= 0 else ""
            w_sign = "+" if e["worst_pnl"] >= 0 else ""
            entry += (
                f"\n   🏅 {e['best_sym']} {b_sign}${e['best_pnl']:.2f}  "
                f"· 💔 {e['worst_sym']} {w_sign}${e['worst_pnl']:.2f}"
            )
        lines.append(entry)

    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"<i>Resets at midnight UTC. Today: {today}</i>")
    return "\n".join(lines)


def _best_worst_text() -> str:
    """Full ranked asset breakdown for the current session."""
    with _lock:
        stats = {s: dict(v) for s, v in session_symbol_stats.items()
                 if v["wins"] + v["losses"] > 0}

    if not stats:
        return (
            "🏅 <b>Best / Worst Asset</b>\n━━━━━━━━━━━━━━━━━━━━\n"
            "— No trades in this session yet. —"
        )

    ranked = sorted(stats.items(), key=lambda kv: kv[1]["pnl"], reverse=True)
    best_sym, best_st   = ranked[0]
    worst_sym, worst_st = ranked[-1]

    def _fmt(sym, st):
        total = st["wins"] + st["losses"]
        wr    = st["wins"] / total * 100 if total else 0
        sign  = "+" if st["pnl"] >= 0 else ""
        return f"  <code>{sym:<10}</code>  {st['wins']}W/{st['losses']}L ({wr:.0f}%)  {sign}${st['pnl']:.2f}"

    lines = [
        "🏅 <b>Best / Worst Asset — Current Session</b>",
        "━━━━━━━━━━━━━━━━━━━━",
        "",
        f"🏅 <b>Best :</b>  {_fmt(best_sym, best_st).strip()}",
        f"💔 <b>Worst:</b>  {_fmt(worst_sym, worst_st).strip()}",
        "",
        "<b>All Assets Ranked</b>",
    ]

    medals = ["🥇", "🥈", "🥉"]
    for rank, (sym, st) in enumerate(ranked):
        medal = medals[rank] if rank < 3 else "  "
        lines.append(f"{medal}{_fmt(sym, st)}")

    lines += [
        "━━━━━━━━━━━━━━━━━━━━",
        "<i>Resets each time TP or SL is hit. Live session data.</i>",
    ]
    return "\n".join(lines)


# ── Sparklines ────────────────────────────────────────────────────────
_SPARK_CHARS = "▁▂▃▄▅▆▇█"

def _sparkline(values) -> str:
    """Return an 8-level ASCII sparkline string from a sequence of numbers."""
    vals = list(values)
    if not vals:
        return "—"
    lo, hi = min(vals), max(vals)
    rng = hi - lo or 1
    return "".join(_SPARK_CHARS[min(7, int((v - lo) / rng * 7.99))] for v in vals)


def _score_sparklines_text() -> str:
    """Per-symbol score trend sparklines (last 20 scans) for Telegram."""
    with _lock:
        snap = {sym: list(symbol_score_history.get(sym, [])) for sym in SYMBOLS}

    lines = [
        "📈 <b>Score Sparklines</b>  <i>(last ≤20 scans, newest right)</i>",
        "━━━━━━━━━━━━━━━━━━━━",
    ]
    for sym in SYMBOLS:
        vals = snap[sym]
        if not vals:
            lines.append(f"  <code>{sym:<10}</code>  — no data yet")
            continue
        spark   = _sparkline(vals)
        current = vals[-1]
        avg     = sum(vals) / len(vals)
        trend   = "↑" if len(vals) >= 3 and vals[-1] > vals[-3] else ("↓" if len(vals) >= 3 and vals[-1] < vals[-3] else "→")
        lines.append(
            f"  <code>{sym:<10}</code> {trend} "
            f"<code>{spark}</code>  "
            f"now <b>{current}</b>  avg {avg:.0f}"
        )
    lines += [
        "━━━━━━━━━━━━━━━━━━━━",
        f"<i>Threshold: {SCORE_THRESHOLD}/100 to trigger a trade.</i>",
    ]
    return "\n".join(lines)


def _settings_text():
    return (
        f"⚙ <b>Settings</b>\n━━━━━━━━━━━━━━━━━━━━\n"
        f"Stake           : ${STAKE}  (fixed risk per trade)\n"
        f"Barrier         : ATR × {ATR_BARRIER_MULT}  (auto-scales per symbol)\n"
        f"Duration        : {DURATION} min\n"
        f"Contract        : {CONTRACT_TYPE}\n"
        f"Min Confidence  : {SCORE_THRESHOLD}/100\n"
        f"Cooldown        : {COOLDOWN_MINUTES} min\n"
        f"Max Consec Loss : {MAX_CONSECUTIVE_LOSSES}\n"
        f"Pause Duration  : {PAUSE_MINUTES} min\n"
        f"Session TP      : ${DAILY_PROFIT_TARGET:.2f}\n"
        f"Session SL      : ${DAILY_LOSS_LIMIT}\n"
        f"Trade Cap       : none\n"
        f"{_ml_progress_text()}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<i>Hitting TP or SL sends a full session summary,\n"
        f"then auto-resets and keeps trading (demo mode — no pause).</i>\n"
        f"<i>Edit STAKE / ATR_BARRIER_MULT / SCORE_THRESHOLD in bot.py to change.</i>"
    )


# ══════════════════════════════════════════════════════════════════════
#  TEST TRADE — independent WS connection, bypasses signal filter
# ══════════════════════════════════════════════════════════════════════
def _run_test_trade(symbol: str):
    tag = f"[TEST {symbol}]"

    acquired = _test_trade_sem.acquire(blocking=False)
    if not acquired:
        with _lock:
            running = list(_test_trade_active.keys())
        who = running[0] if running else "another symbol"
        _send_tg_wait(
            f"⏳ <b>Test Trade Queued</b>\n"
            f"A test trade on <b>{who}</b> is still running (up to {DURATION} min).\n"
            f"Please wait for it to finish before starting another."
        )
        return

    with _lock:
        _test_trade_active[symbol] = datetime.now(timezone.utc)

    _log(f"🧪 {tag} starting one-shot test trade…")

    def tg(text: str):
        _send_tg_wait(text)

    tg(
        f"🧪 <b>Test Trade Started</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Symbol   : <code>{symbol}</code>\n"
        f"Stake    : ${STAKE:.2f}\n"
        f"Target   : ${STAKE + TARGET_PROFIT:.2f}\n"
        f"Expiry   : {DURATION} min\n"
        f"Type     : {CONTRACT_TYPE}\n"
        f"<i>Step 1/4 – Connecting to Deriv…</i>"
    )

    try:
        ws = websocket.WebSocket()
        ws.connect("wss://ws.derivws.com/websockets/v3?app_id=1089", timeout=15)
    except Exception as e:
        tg(f"🧪 <b>Test Trade FAILED</b>\n❌ Connection error: <code>{e}</code>")
        return

    def recv_typed(*types, timeout: int = 10) -> Optional[dict]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                ws.settimeout(max(1, deadline - time.time()))
                raw = ws.recv()
                msg = json.loads(raw)
                if msg.get("msg_type") in types:
                    return msg
                if "error" in msg:
                    return msg
            except Exception:
                break
        return None

    try:
        ws.send(json.dumps({"authorize": DERIV_APP_TOKEN}))
        auth = recv_typed("authorize", timeout=10)
        if not auth or "error" in auth:
            err = (auth or {}).get("error", {}).get("message", "timeout")
            tg(f"🧪 <b>Test Trade FAILED</b>\n❌ Auth error: <code>{err}</code>")
            return
        account = auth.get("authorize", {}).get("loginid", "?")

        ws.send(json.dumps({"ticks": symbol}))
        tick_msg = recv_typed("tick", timeout=8)
        spot_now = float((tick_msg or {}).get("tick", {}).get("quote", 0)) if tick_msg else 0

        with _lock:
            ema_slow = (indicators.get(symbol) or {}).get("ema_slow") or spot_now
        direction = "UP" if spot_now > ema_slow else "DOWN"

        tg(
            f"🧪 <b>Test Trade</b>  –  ✅ Authorised\n"
            f"Account  : <code>{account}</code>\n"
            f"Spot now : {spot_now}\n"
            f"Direction: {'🟢 BUY' if direction == 'UP' else '🔴 SELL'}\n"
            f"Barrier  : {_compute_barrier(symbol, direction)}  (ATR×{ATR_BARRIER_MULT})\n"
            f"<i>Step 2/4 – Requesting proposal…</i>"
        )
        ws.send(json.dumps({
            "proposal": 1,
            "amount": STAKE,
            "basis": "stake",
            "contract_type": CONTRACT_TYPE,
            "currency": "USD",
            "duration": DURATION,
            "duration_unit": "m",
            "symbol": symbol,
            "barrier": _compute_barrier(symbol, direction),
        }))
        prop_msg = recv_typed("proposal", timeout=10)
        if not prop_msg or "error" in prop_msg:
            err = (prop_msg or {}).get("error", {}).get("message", "timeout")
            tg(f"🧪 <b>Test Trade FAILED</b>\n❌ Proposal error: <code>{err}</code>")
            return
        prop    = prop_msg["proposal"]
        pid     = prop["id"]
        barrier = prop.get("spot") or prop.get("barrier") or "—"
        ask     = prop.get("ask_price", STAKE)
        payout  = prop.get("payout", "?")
        tg(
            f"🧪 <b>Test Trade</b>  –  ✅ Proposal OK\n"
            f"Proposal ID : <code>{pid}</code>\n"
            f"Ask Price   : ${ask}\n"
            f"Payout      : ${payout}\n"
            f"Spot        : {barrier}\n"
            f"<i>Step 3/4 – Buying contract…</i>"
        )

        ws.send(json.dumps({"buy": pid, "price": STAKE}))
        buy_msg = recv_typed("buy", timeout=10)
        if not buy_msg or "error" in buy_msg:
            err = (buy_msg or {}).get("error", {}).get("message", "timeout")
            tg(f"🧪 <b>Test Trade FAILED</b>\n❌ Buy error: <code>{err}</code>")
            return
        buy       = buy_msg["buy"]
        cid       = buy["contract_id"]
        bought_at = buy.get("buy_price", STAKE)
        paid_out  = buy.get("payout", "?")
        tg(
            f"🧪 <b>Test Trade</b>  –  ✅ Contract Bought!\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Contract ID : <code>{cid}</code>\n"
            f"Paid        : ${bought_at}\n"
            f"Max Payout  : ${paid_out}\n"
            f"Expires     : {DURATION} min\n"
            f"<i>Step 4/4 – Waiting for result…  ⏳</i>"
        )

        ws.send(json.dumps({
            "proposal_open_contract": 1,
            "contract_id": int(cid),
            "subscribe": 1,
        }))
        wait_deadline = time.time() + DURATION * 60 + 60
        contract_data: dict = {}
        while time.time() < wait_deadline:
            try:
                ws.settimeout(5)
                raw = ws.recv()
                msg = json.loads(raw)
                mtype = msg.get("msg_type")
                if mtype == "proposal_open_contract":
                    poc = msg.get("proposal_open_contract", {})
                    contract_data = poc
                    if poc.get("is_expired") or poc.get("is_sold"):
                        break
                elif mtype == "error":
                    err = msg.get("error", {}).get("message", "?")
                    tg(f"🧪 <b>Test Trade</b> – API error during monitoring: <code>{err}</code>")
                    break
            except websocket.WebSocketTimeoutException:
                continue
            except Exception as e:
                logger.error(f"🧪 {tag} recv error: {e}")
                break

        profit = float(contract_data.get("profit", 0))
        win    = profit > 0
        emoji  = "🏆" if win else "💀"
        label  = "WIN" if win else "LOSS"
        pstr   = f"+${profit:.2f}" if win else f"${profit:.2f}"
        _log(f"🧪 {tag} RESULT={label}  profit={pstr}")
        tg(
            f"{emoji} <b>Test Trade RESULT: {label}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Symbol      : <code>{symbol}</code>\n"
            f"Contract ID : <code>{cid}</code>\n"
            f"P&L         : <b>{pstr}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"✅ Full pipeline confirmed: proposal → buy → settlement → result\n"
            f"<i>Test trade does NOT affect session stats or signal cooldowns.</i>"
        )
    except Exception as e:
        logger.exception(f"🧪 {tag} unexpected error: {e}")
        tg(f"🧪 <b>Test Trade Error</b>\n<code>{e}</code>")
    finally:
        try:
            ws.close()
        except Exception:
            pass
        with _lock:
            _test_trade_active.pop(symbol, None)
        _test_trade_sem.release()


# ══════════════════════════════════════════════════════════════════════
#  TELEGRAM COMMAND HANDLERS
# ══════════════════════════════════════════════════════════════════════
async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 <b>Deriv Sniper Bot v2</b>\nProfessional edition – use the menu below.",
        reply_markup=_main_kb(), parse_mode="HTML",
    )

async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(_status_text(), reply_markup=_main_kb(), parse_mode="HTML")

async def cmd_pnl(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(_pnl_text(), reply_markup=_main_kb(), parse_mode="HTML")

async def cmd_pause(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    global paused, pause_until
    with _lock:
        paused = True
        pause_until = datetime.now(timezone.utc) + timedelta(hours=24)
    await update.message.reply_text("⏸ <b>Bot paused.</b>  /resume to restart.", parse_mode="HTML")

async def cmd_resume(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    global paused, pause_until, consecutive_losses
    with _lock:
        paused = False
        consecutive_losses = 0
        pause_until = datetime.min.replace(tzinfo=timezone.utc)
    await update.message.reply_text("▶ <b>Bot resumed.</b>", parse_mode="HTML")

async def cmd_history(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(_history_text(), reply_markup=_main_kb(), parse_mode="HTML")

async def cmd_session(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(_live_session_report_text(), reply_markup=_main_kb(), parse_mode="HTML")

async def cmd_alltime(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(_alltime_text(), reply_markup=_main_kb(), parse_mode="HTML")


# ══════════════════════════════════════════════════════════════════════
#  TELEGRAM BUTTON HANDLER
# ══════════════════════════════════════════════════════════════════════
async def btn_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    global paused, pause_until, consecutive_losses
    q = update.callback_query
    d = q.data
    print(f"[BTN] received callback_data={d!r}", flush=True)
    logger.info(f"Button pressed: {d!r}")
    try:
        await q.answer()
    except Exception as e:
        print(f"[BTN] q.answer() failed: {e}", flush=True)
        return

    try:
        if d in ("main_menu", "refresh"):
            await q.edit_message_text(
                "🤖 <b>Deriv Sniper Bot v2</b>  –  Select an option:",
                reply_markup=_main_kb(), parse_mode="HTML",
            )
        elif d == "status":
            await q.edit_message_text(_status_text(),   reply_markup=_main_kb(), parse_mode="HTML")
        elif d == "pnl":
            await q.edit_message_text(_pnl_text(),      reply_markup=_main_kb(), parse_mode="HTML")
        elif d == "history":
            await q.edit_message_text(_history_text(),  reply_markup=_main_kb(), parse_mode="HTML")
        elif d == "session_report":
            await q.edit_message_text(_live_session_report_text(), reply_markup=_main_kb(), parse_mode="HTML")
        elif d == "alltime":
            await q.edit_message_text(_alltime_text(),  reply_markup=_main_kb(), parse_mode="HTML")
        elif d == "signals":
            await q.edit_message_text(_signals_text(),  reply_markup=_main_kb(), parse_mode="HTML")
        elif d == "daily_history":
            await q.edit_message_text(_daily_history_text(), reply_markup=_main_kb(), parse_mode="HTML")
        elif d == "best_worst":
            await q.edit_message_text(_best_worst_text(), reply_markup=_main_kb(), parse_mode="HTML")
        elif d == "score_sparklines":
            await q.edit_message_text(_score_sparklines_text(), reply_markup=_main_kb(), parse_mode="HTML")
        elif d == "settings":
            await q.edit_message_text(
                _settings_text(),
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back", callback_data="main_menu")]]),
                parse_mode="HTML",
            )
        elif d == "toggle_pause":
            with _lock:
                paused = not paused
                if paused:
                    pause_until = datetime.now(timezone.utc) + timedelta(hours=24)
                else:
                    consecutive_losses = 0
                    pause_until = datetime.min.replace(tzinfo=timezone.utc)
                is_now_paused = paused
            label = "⏸ Paused" if is_now_paused else "▶ Resumed"
            await q.edit_message_text(
                f"{label} – use the menu to continue.",
                reply_markup=_main_kb(), parse_mode="HTML",
            )
        elif d == "skip_menu":
            await q.edit_message_text(
                "⏭ Choose a symbol to put on 60-minute cooldown:",
                reply_markup=_skip_kb(), parse_mode="HTML",
            )
        elif d.startswith("skip_"):
            sym = d[5:]
            if sym in SYMBOLS:
                with _lock:
                    cooldown_until[sym] = datetime.now(timezone.utc) + timedelta(minutes=60)
                _log(f"⏭ {sym} skipped via Telegram (60min)")
            await q.edit_message_text(
                f"⏭ <b>{sym}</b> skipped for 60 minutes.",
                reply_markup=_main_kb(), parse_mode="HTML",
            )
        elif d == "test_menu":
            await q.edit_message_text(
                f"🧪 <b>Test Trade</b>  –  Pick a symbol group\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"Stake: <b>${STAKE:.2f}</b>  ·  "
                f"Target: <b>${STAKE + TARGET_PROFIT:.2f}</b>  ·  "
                f"Expiry: <b>{DURATION} min</b>\n"
                f"Bypasses signal filters — fires immediately.",
                reply_markup=_test_group_kb(), parse_mode="HTML",
            )
        elif d in ("tg_vol", "tg_vol1s", "tg_rdb"):
            labels = {
                "tg_vol":   "📊 Volatility (R_*)",
                "tg_vol1s": "⚡ Volatility 1s (1HZ*)",
                "tg_rdb":   "📉 Range Break",
            }
            await q.edit_message_text(
                f"🧪 <b>Test Trade  –  {labels[d]}</b>\n━━━━━━━━━━━━━━━━━━━━\nSelect a symbol:",
                reply_markup=_test_sym_kb(d), parse_mode="HTML",
            )
        elif d.startswith("test_sym_"):
            sym = d[9:]
            if sym not in ALL_TOUCH_SYMBOLS:
                await q.answer("Unknown symbol.", show_alert=True)
                return
            await q.edit_message_text(
                f"🧪 <b>Test Trade Launched</b>\n"
                f"Symbol: <code>{sym}</code>  ·  "
                f"Stake: ${STAKE:.2f}  ·  "
                f"Target: ${STAKE + TARGET_PROFIT:.2f}  ·  "
                f"Expiry: {DURATION} min\n\n"
                f"⏳ Connecting to Deriv and placing order…\n"
                f"You'll get step-by-step updates and a final result.",
                reply_markup=InlineKeyboardMarkup([[
                    InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")
                ]]),
                parse_mode="HTML",
            )
            threading.Thread(
                target=_run_test_trade, args=(sym,),
                daemon=True, name=f"TestTrade-{sym}"
            ).start()
        else:
            await q.edit_message_text(
                "🤖 <b>Deriv Sniper Bot v2</b>  –  Select an option:",
                reply_markup=_main_kb(), parse_mode="HTML",
            )
    except Exception as e:
        print(f"[BTN] ERROR in btn_handler d={d!r}: {e}", flush=True)
        logger.error(f"btn_handler error d={d!r}: {e}")
        import traceback; traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
#  HEARTBEAT JOB
# ══════════════════════════════════════════════════════════════════════
async def heartbeat_job(ctx: ContextTypes.DEFAULT_TYPE):
    with _lock:
        pnl, wc, lc, cl, ac, is_paused = (
            total_pnl, win_count, loss_count,
            consecutive_losses, len(active_contracts), paused,
        )
        peak, mdd = peak_equity, max_drawdown
    total  = wc + lc
    wr     = wc / total * 100 if total else 0
    cur_dd = max(0.0, peak - pnl)
    msg    = (
        f"❤ <b>Heartbeat  –  {datetime.now(timezone.utc).strftime('%H:%M UTC')}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"State  : {'⏸ PAUSED' if is_paused else '▶ RUNNING'}\n"
        f"P&L    : {'+' if pnl >= 0 else ''}${pnl:.2f}\n"
        f"Trades : {total}  ({wc}W / {lc}L  {wr:.1f}%)\n"
        f"Active : {ac}\n"
        f"Streak : {'🔴' * cl if cl else '🟢 None'}\n"
        f"Drawdown: -${cur_dd:.2f}  (max -${mdd:.2f})\n"
        f"{_best_worst_line()}"
    )
    _send_tg(msg, reply_markup=_main_kb())


# ══════════════════════════════════════════════════════════════════════
#  RICH TERMINAL DASHBOARD
# ══════════════════════════════════════════════════════════════════════
console = Console()

def _make_header() -> Panel:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d  %H:%M:%S  UTC")
    with _lock:
        is_paused = paused
    mode = Text("⏸  PAUSED", style="bold yellow") if is_paused else Text("▶  RUNNING", style="bold green")
    t = Text("  DERIV SNIPER BOT  v2.0   ", style="bold cyan")
    t.append(f"  {now}  ", style="dim white")
    t.append("    ")
    t.append(mode)
    return Panel(Align.center(t), style="bold blue", box=box.DOUBLE_EDGE)


def _make_symbols_table() -> Panel:
    tbl = Table(
        box=box.SIMPLE_HEAD, show_header=True, header_style="bold magenta",
        expand=True, min_width=68,
    )
    tbl.add_column("Symbol",   style="cyan",    min_width=7)
    tbl.add_column("Price",    style="white",   min_width=11)
    tbl.add_column("EMA Fast", style="yellow",  min_width=10)
    tbl.add_column("EMA Slow", style="yellow",  min_width=10)
    tbl.add_column("ATR",      style="magenta", min_width=8)
    tbl.add_column("Trend",    style="green",   min_width=10)
    tbl.add_column("Ready",    style="bold",    min_width=6)
    tbl.add_column("Cooldown", style="red",     min_width=10)

    now = datetime.now(timezone.utc)
    with _lock:
        ind_snap = {s: dict(indicators[s]) for s in SYMBOLS}
        cd_snap  = dict(cooldown_until)
        pr_snap  = dict(last_price)
        ac_syms  = {c["symbol"] for c in active_contracts.values()}

    for sym in SYMBOLS:
        ind   = ind_snap[sym]
        price = pr_snap.get(sym, 0)
        cd    = cd_snap[sym]
        cd_str   = f"{int((cd - now).total_seconds() // 60)}m left" if cd > now else "Ready"
        cd_style = "red" if cd > now else "green"
        if ind["ema_fast_rising"] and ind["ema_slow_rising"]:
            trend = "↑↑ Strong"
        elif ind["ema_fast_rising"]:
            trend = "↑  Weak"
        elif ind["ema_fast"] is not None:
            trend = "↓  Down"
        else:
            trend = "--"
        tbl.add_row(
            Text(sym, style="bold green" if sym in ac_syms else "cyan"),
            f"{price:.5f}" if price else "loading…",
            f"{ind['ema_fast']:.4f}" if ind["ema_fast"] else "--",
            f"{ind['ema_slow']:.4f}" if ind["ema_slow"] else "--",
            f"{ind['atr']:.5f}"      if ind["atr"]      else "--",
            trend,
            Text("✓", style="bold green") if ind["ready"] else Text("…", style="dim"),
            Text(cd_str, style=cd_style),
        )
    return Panel(tbl, title="[bold blue]📊 Symbol Monitor", border_style="blue")


def _bar(pct: int, width: int = 20) -> str:
    filled = max(0, min(width, int(pct / 100 * width)))
    return "█" * filled + "░" * (width - filled)


def _make_trades_panel() -> Panel:
    now = datetime.now(timezone.utc)
    with _lock:
        contracts = {cid: dict(c) for cid, c in active_contracts.items()}
    if not contracts:
        return Panel(
            Align.center(Text("\n  No active trades  \n", style="dim"), vertical="middle"),
            title="[bold green]📈 Active Trades  ·  TP / Time Bars",
            border_style="green", height=12,
        )
    content = Text()
    for cid, info in contracts.items():
        sym     = info["symbol"]
        barrier = info.get("barrier") or 0
        entry   = info.get("entry_time", now)
        stake   = info.get("stake", STAKE)
        payout  = info.get("payout", stake + TARGET_PROFIT)
        expires = entry + timedelta(minutes=DURATION)
        price   = last_price.get(sym, 0)
        ep      = info.get("entry_price", price)
        score   = info.get("details", {}).get("total_score", "?")
        direction = info.get("direction", "UP")

        if barrier and price and ep:
            span   = abs(float(barrier) - ep)
            moved  = abs(price - ep)
            tp_pct = min(100, int(moved / span * 100)) if span else 0
        else:
            tp_pct = 0

        elapsed  = (now - entry).total_seconds()
        time_pct = max(0, 100 - int(elapsed / (DURATION * 60) * 100))
        left_secs = max(0, int((expires - now).total_seconds()))
        m, s = divmod(left_secs, 60)

        tp_col   = "green" if tp_pct >= 70 else ("yellow" if tp_pct >= 40 else "red")
        time_col = "green" if time_pct >= 50 else ("yellow" if time_pct >= 20 else "red")

        content.append(f"  {sym:<7}", style="bold cyan")
        content.append(f"  #{cid}  {direction}  score={score}/100  ", style="dim")
        content.append(f"stake=${stake:.2f}  pay=${payout:.2f}\n", style="white")
        content.append("  TP   [", style="dim")
        content.append(_bar(tp_pct), style=tp_col)
        content.append(f"]  {tp_pct:3d}%  barrier={barrier}\n", style="dim")
        content.append("  Time [", style="dim")
        content.append(_bar(time_pct), style=time_col)
        content.append(f"]  {time_pct:3d}%  {m:02d}:{s:02d} left\n\n", style="dim")

    return Panel(
        content,
        title=f"[bold green]📈 Active Trades ({len(contracts)})  ·  TP / Time Bars",
        border_style="green",
    )


def _make_pnl_panel() -> Panel:
    with _lock:
        pnl, wc, lc, cl, dt, is_paused = (
            total_pnl, win_count, loss_count,
            consecutive_losses, daily_trades, paused,
        )
        peak, mdd = peak_equity, max_drawdown
    cur_dd = max(0.0, peak - pnl)
    total = wc + lc
    wr    = wc / total * 100 if total else 0
    dur   = datetime.now(timezone.utc) - session_start
    h, r  = divmod(int(dur.total_seconds()), 3600)
    mi    = r // 60
    col   = "green" if pnl >= 0 else "red"
    risk_used = (
        min(100, int(abs(min(0, pnl)) / abs(DAILY_LOSS_LIMIT) * 100))
        if DAILY_LOSS_LIMIT < 0 else 0
    )
    risk_col = "red" if risk_used >= 80 else ("yellow" if risk_used >= 50 else "green")

    b_sym, b_st, w_sym, w_st = _best_worst_session()

    t = Text()
    t.append("  Session P&L : ", style="dim"); t.append(f"${pnl:+.2f}\n", style=f"bold {col}")
    t.append("  Wins / Losses: ", style="dim")
    t.append(f"{wc}", style="bold green"); t.append(" / ")
    t.append(f"{lc}\n", style="bold red")
    t.append("  Win Rate    : ", style="dim")
    t.append(f"{wr:.1f}%\n", style="bold yellow" if wr >= 50 else "bold red")
    t.append("  Consec Loss : ", style="dim")
    t.append(f"{cl}/{MAX_CONSECUTIVE_LOSSES}\n", style="bold red" if cl > 0 else "white")
    t.append("  Day Trades  : ", style="dim"); t.append(f"{dt}\n", style="white")
    t.append("  Session Up  : ", style="dim"); t.append(f"{h}h {mi}m\n", style="white")
    t.append("  Drawdown    : ", style="dim")
    t.append(f"-${cur_dd:.2f}", style="bold red" if cur_dd > 0 else "white")
    t.append("  (max ", style="dim"); t.append(f"-${mdd:.2f}", style="bold red"); t.append(")\n", style="dim")

    # Best / Worst asset block
    if b_sym:
        b_wr = b_st["wins"] / (b_st["wins"] + b_st["losses"]) * 100 if (b_st["wins"] + b_st["losses"]) else 0
        w_wr = w_st["wins"] / (w_st["wins"] + w_st["losses"]) * 100 if (w_st["wins"] + w_st["losses"]) else 0
        t.append("\n  🏅 Best  : ", style="dim")
        t.append(f"{b_sym}", style="bold green")
        t.append(f"  {'+' if b_st['pnl'] >= 0 else ''}${b_st['pnl']:.2f}  ({b_wr:.0f}% WR)\n", style="green")
        t.append("  💔 Worst : ", style="dim")
        t.append(f"{w_sym}", style="bold red")
        t.append(f"  {'+' if w_st['pnl'] >= 0 else ''}${w_st['pnl']:.2f}  ({w_wr:.0f}% WR)\n", style="red")

    t.append(f"\n  Risk Limit  : [", style="dim")
    t.append(_bar(risk_used), style=risk_col)
    t.append(f"]  {risk_used}%  (floor ${DAILY_LOSS_LIMIT:.0f})", style="dim")
    return Panel(t, title="[bold yellow]💰 P&L  &  Risk", border_style="yellow")


def _make_log_panel() -> Panel:
    with _lock:
        lines = list(signal_log)
    t = Text()
    for line in lines:
        t.append(line + "\n")
    return Panel(t, title="[bold white]📋 Signal Log", border_style="white")


def _make_footer() -> Panel:
    return Panel(
        Align.center(Text(
            f"  Stake ${STAKE}  ·  Session TP ${DAILY_PROFIT_TARGET}  ·  Session SL ${DAILY_LOSS_LIMIT}  ·  "
            f"Duration {DURATION}min  ·  Cooldown {COOLDOWN_MINUTES}min  ·  "
            f"Min Confidence ≥ {SCORE_THRESHOLD}  ·  {_ml_progress_text()}",
            style="dim",
        )),
        style="dim blue",
    )


_layout = Layout()
_layout.split_column(
    Layout(name="header", size=3),
    Layout(name="body",   ratio=1),
    Layout(name="footer", size=3),
)
_layout["body"].split_row(
    Layout(name="left",  ratio=2),
    Layout(name="right", ratio=1),
)
_layout["left"].split_column(
    Layout(name="symbols", ratio=2),
    Layout(name="trades",  ratio=3),
)

def refresh_dashboard():
    try:
        _layout["header"].update(_make_header())
        _layout["symbols"].update(_make_symbols_table())
        _layout["trades"].update(_make_trades_panel())
        _layout["right"].split_column(
            Layout(_make_pnl_panel(), name="pnl", ratio=1),
            Layout(_make_log_panel(), name="log", ratio=2),
        )
        _layout["footer"].update(_make_footer())
    except Exception as e:
        logger.debug(f"dashboard refresh error: {e}")


def _terminal_loop():
    while True:
        try:
            refresh_dashboard()
        except Exception as e:
            logger.debug(f"terminal loop error: {e}")
        time.sleep(1)


# ══════════════════════════════════════════════════════════════════════
#  SCAN STATUS — periodic digest sent every 5 minutes
# ══════════════════════════════════════════════════════════════════════
def _score_sparkline_loop():
    """Hourly thread: sends per-symbol score sparklines to Telegram."""
    time.sleep(3600)  # first report after 1 hour
    while True:
        try:
            _send_tg(_score_sparklines_text())
        except Exception as e:
            logger.error(f"score_sparkline_loop: {e}")
        time.sleep(3600)


def _scan_status_loop():
    time.sleep(90)
    while True:
        try:
            now = datetime.now(timezone.utc)
            lines = [f"📡 <b>SCAN STATUS</b>  {now.strftime('%H:%M')} UTC\n━━━━━━━━━━━━━━━━━━━━"]
            for sym in SYMBOLS:
                with _lock:
                    ind   = dict(indicators[sym])
                    cc    = current_candle[sym]
                    cd    = cooldown_until[sym]
                    psd   = paused
                    dtrd  = daily_trades
                    ppnl  = total_pnl

                if not ind.get("ready"):
                    with _lock:
                        nrows = len(ohlcv[sym])
                    lines.append(f"⏳ <code>{sym:<10}</code>  loading… ({nrows}/500 candles)")
                    continue

                if cc:
                    row_c = {"Close": float(cc.get("close", 0)),
                             "Open":  float(cc.get("open", 0)),
                             "High":  float(cc.get("high", 0)),
                             "Low":   float(cc.get("low", 0))}
                    score, direction, det = score_signal(sym, row_c)
                else:
                    score, direction, det = 0, "UP", {}

                # Track score history for sparklines
                with _lock:
                    if sym in symbol_score_history:
                        symbol_score_history[sym].append(score)
                    else:
                        symbol_score_history[sym] = deque([score], maxlen=20)

                heat = "🔥🔥" if score >= SCORE_THRESHOLD else "🔥" if score >= SCORE_THRESHOLD - 20 else "  "
                locked = " 🔒" if sym in locked_symbols else ""

                blocked = ""
                if score >= SCORE_THRESHOLD:
                    if psd:
                        blocked = " ⏸paused"
                    elif now < cd:
                        left = max(0, int((cd - now).total_seconds() // 60))
                        blocked = f" ⏱{left}m cd"
                    elif ppnl <= DAILY_LOSS_LIMIT:
                        blocked = " 🚫floor"

                t = det.get("trend", 0)
                eq = det.get("entry_quality", 0)
                v = det.get("volatility", 0)
                m = det.get("momentum", 0)
                bar = f"T{t} E{eq} V{v} M{m}"
                lines.append(
                    f"{heat}<code>{sym:<10}</code>  <b>{score:>3}/100</b> {direction} {bar}{blocked}{locked}"
                )

            lines.append("")
            lines.append(_best_worst_line())
            _send_tg("\n".join(lines))
        except Exception as e:
            logger.error(f"scan_status_loop: {e}")
        time.sleep(300)


# ══════════════════════════════════════════════════════════════════════
#  TELEGRAM STARTUP
# ══════════════════════════════════════════════════════════════════════
def _start_telegram():
    async def _run():
        global telegram_app, _tg_loop
        _tg_loop = asyncio.get_running_loop()
        app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
        telegram_app = app

        async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
            import traceback
            msg = f"{datetime.now()} TG-ERROR: {context.error}\n"
            logger.error(msg)
            with open("tg_errors.log", "a") as f:
                f.write(msg)
                traceback.print_exception(
                    type(context.error), context.error,
                    context.error.__traceback__, file=f,
                )

        app.add_error_handler(error_handler)
        app.add_handler(CommandHandler("start",   cmd_start))
        app.add_handler(CommandHandler("status",  cmd_status))
        app.add_handler(CommandHandler("pnl",     cmd_pnl))
        app.add_handler(CommandHandler("pause",   cmd_pause))
        app.add_handler(CommandHandler("resume",  cmd_resume))
        app.add_handler(CommandHandler("history", cmd_history))
        app.add_handler(CommandHandler("session", cmd_session))
        app.add_handler(CommandHandler("alltime", cmd_alltime))
        app.add_handler(CallbackQueryHandler(btn_handler))

        if app.job_queue is not None:
            app.job_queue.run_repeating(
                heartbeat_job, interval=HEARTBEAT_INTERVAL_SEC, first=15,
            )
        else:
            async def _hb_loop():
                await asyncio.sleep(15)
                while True:
                    try:
                        await heartbeat_job(None)
                    except Exception as exc:
                        logger.error(f"Heartbeat error: {exc}")
                    await asyncio.sleep(HEARTBEAT_INTERVAL_SEC)
            asyncio.ensure_future(_hb_loop())

        _log("📱 Telegram bot started  (PTB " +
             __import__("telegram").__version__ + ")")
        async with app:
            await app.start()
            await app.updater.start_polling(
                allowed_updates=["message", "callback_query"],
            )
            while True:
                await asyncio.sleep(3600)

    asyncio.run(_run())


# ══════════════════════════════════════════════════════════════════════
#  HEALTH CHECK SERVER
# ══════════════════════════════════════════════════════════════════════
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

class _HealthHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        with _lock:
            pnl, wc, lc, is_paused = total_pnl, win_count, loss_count, paused
            peak, mdd = peak_equity, max_drawdown
        uptime = datetime.now(timezone.utc) - session_start
        body = json.dumps({
            "status": "ok",
            "state": "paused" if is_paused else "running",
            "uptime_seconds": int(uptime.total_seconds()),
            "session_pnl": round(pnl, 2),
            "wins": wc,
            "losses": lc,
            "max_drawdown": round(mdd, 2),
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except Exception:
            pass

    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()


def _start_health_server():
    port = int(os.environ.get("PORT", "8099"))
    try:
        server = ThreadingHTTPServer(("0.0.0.0", port), _HealthHandler)
        _log(f"🩺 Health check server listening on :{port} (for UptimeRobot)")
        server.serve_forever()
    except Exception as e:
        logger.error(f"Health server failed to start on :{port}: {e}")


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    is_tty = sys.stdout.isatty()
    console.print(Panel(
        Align.center(Text(
            f"\n  DERIV SNIPER BOT  v2.0  –  Professional Edition\n"
            f"  Loading history for {len(SYMBOLS)} symbols…\n",
            style="bold cyan",
        )),
        border_style="blue", box=box.DOUBLE_EDGE,
    ))

    BATCH = 5
    for i in range(0, len(SYMBOLS), BATCH):
        batch   = SYMBOLS[i:i + BATCH]
        threads = [
            threading.Thread(target=fetch_history, args=(sym,), daemon=True, name=f"Hist-{sym}")
            for sym in batch
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=70)

    console.print("[green]✓ History phase done.  Starting live feeds…[/green]")

    _ml_load()

    _watch_thread(_db_writer,                  name="DBWriter")   # must be after _ml_load()

    for sym in SYMBOLS:
        _watch_thread(_ws_thread, args=(sym,), name=f"WS-{sym}")

    _watch_thread(_start_telegram,             name="Telegram")
    _watch_thread(_terminal_loop,              name="TermRefresh")
    _watch_thread(_scan_status_loop,           name="ScanStatus")
    _watch_thread(_score_sparkline_loop,       name="ScoreSparkline")
    _watch_thread(_pending_trade_timeout_loop, name="PendingTimeout")
    _watch_thread(_start_health_server,        name="HealthServer")

    if is_tty:
        with Live(_layout, refresh_per_second=1, screen=True):
            try:
                while True:
                    time.sleep(60)
            except KeyboardInterrupt:
                pass
    else:
        console.print(
            "[yellow]Non-TTY detected: Rich dashboard disabled.\n"
            "Watch sniper.log or Telegram for status.[/yellow]"
        )
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            pass

    console.print("[yellow]Shutting down…[/yellow]")
    db_queue.put(None)


def _run_forever():
    restart_count = 0
    while True:
        try:
            main()
            break
        except KeyboardInterrupt:
            break
        except Exception as e:
            restart_count += 1
            logger.error(f"FATAL: unhandled exception in main(): {e}", exc_info=True)
            try:
                _send_tg(
                    f"💥 <b>BOT CRASHED</b>\n"
                    f"Error: <code>{e}</code>\n"
                    f"Auto-restarting (attempt {restart_count})…"
                )
            except Exception as tg_err:
                logger.error(f"Failed to send crash notification: {tg_err}")
            backoff = min(60, 5 * restart_count)
            time.sleep(backoff)


if __name__ == "__main__":
    _run_forever()

