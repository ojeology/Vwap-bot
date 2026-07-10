#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║        DERIV SNIPER BOT v3  –  Professional Edition                  ║
║  Supertrend · Market Sessions · Fast ML · 93% Confidence Gate        ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import json, time, sqlite3, threading, queue, logging, asyncio, os, sys, pickle, io, csv as _csv
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from collections import deque
from typing import Optional

import numpy as np
import pandas as pd

# PostgreSQL support (optional – falls back to SQLite when DATABASE_URL is unset)
DATABASE_URL = os.environ.get("DATABASE_URL")
USE_PG = bool(DATABASE_URL)
if USE_PG:
    try:
        import psycopg2
        import psycopg2.extras
    except ImportError:
        # psycopg2 not installed — fall back to SQLite and warn
        import logging as _lg
        _lg.getLogger("sniper").warning(
            "DATABASE_URL is set but psycopg2-binary is not installed. "
            "Falling back to SQLite. Run: pip install psycopg2-binary"
        )
        DATABASE_URL = None
        USE_PG = False
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
    Application, CommandHandler, CallbackQueryHandler,
    MessageHandler, filters, ContextTypes,
)

# ══════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════
DERIV_APP_TOKEN    = os.environ.get("DERIV_TOKEN",    "m8MRwwwroJy6YQw")
TELEGRAM_BOT_TOKEN = os.environ.get("TG_BOT_TOKEN",   "")
TELEGRAM_CHAT_ID   = os.environ.get("TG_CHAT_ID",     "6400145232")

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
STAKE                  = 3.0          # $3 per trade
TARGET_PROFIT          = 0.70         # display only
PROFIT_MIN             = 0.55         # reject proposal if profit-per-trade < this
PROFIT_MAX             = 0.90         # reject proposal if profit-per-trade > this
ATR_BARRIER_MULT       = 0.60        # tuned for ~$0.60-0.90 profit/trade across all symbols
DURATION               = 10
CONTRACT_TYPE          = "ONETOUCH"
COOLDOWN_MINUTES       = 20
MAX_CONSECUTIVE_LOSSES = 3
PAUSE_MINUTES          = 30
DAILY_LOSS_LIMIT       = -10.0        # session SL
DAILY_PROFIT_TARGET    = 5.0          # session TP
MAX_DAILY_TRADES       = 9999

# ── Indicator parameters ──────────────────────────────────────────────
EMA_SLOW               = 200          # kept for direction fallback only
ATR_PERIOD             = 14
ATR_MA_PERIOD          = 30
RSI_PERIOD             = 14
SUPERTREND_PERIOD      = 10
SUPERTREND_ATR_MULT    = 3.0
SCORE_THRESHOLD        = 93           # minimum score to trade (raised from 80)
HEARTBEAT_INTERVAL_SEC = 3600      # richer hourly heartbeat

# ── Market sessions (UTC hours) ───────────────────────────────────────
# Maps session name → (start_hour_utc, end_hour_utc).
# Midnight wraps across 00:00, handled in _get_session_name().
MARKET_SESSIONS = {
    "Midnight": (22, 2),
    "Asian":    (2,  10),
    "London":   (10, 13),
    "New York": (13, 22),
}
SESSION_EMOJIS = {
    "Midnight": "🌙",
    "Asian":    "🌏",
    "London":   "🇬🇧",
    "New York": "🗽",
}

# ── ML filter ────────────────────────────────────────────────────────
MODEL_PATH        = "ml_model.pkl"
ML_MIN_TRADES     = 100
ML_RETRAIN_EVERY  = 50           # retrain every 50 trades (continuous rolling)
ML_CONFIDENCE_MIN = 0.75              # require ≥75% confidence (adjustable via Telegram)
ML_FEATURE_COLS   = [
    "score", "wick_atr_ratio", "atr", "atr_ma",
    "ema_fast_slope", "ema_slow_slope", "ema_distance",
]

# ── Per-class ML: separate model per volatility family ───────────────
ML_SYMBOL_CLASSES = {
    "standard_vol": ["R_100", "R_75", "R_50", "R_25", "R_10"],
    "tick_vol":     ["1HZ100V", "1HZ90V", "1HZ75V", "1HZ50V",
                     "1HZ30V", "1HZ25V", "1HZ15V", "1HZ10V"],
    "range_break":  ["RDBULL", "RDBEAR"],
}
TICK_MOMENTUM_MIN = 3   # consecutive confirming ticks required before entry
# Note: DB columns keep original names for compat; we store:
#   ema_fast_slope  → supertrend direction (1 or -1)
#   ema_slow_slope  → ADX value
#   ema_distance    → Supertrend distance / ATR ratio

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
ohlcv_m5:     dict = {}    # 5-min resampled OHLCV per symbol
indicators_m5: dict = {}   # M5 Supertrend direction per symbol
ohlcv_m15:     dict = {}   # 15-min resampled OHLCV per symbol
indicators_m15: dict = {}  # M15 Supertrend direction per symbol
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
session_symbol_stats: dict = {}   # symbol → {wins, losses, pnl}

# Daily session log (TP/SL hits) ──────────────────────────────────────
daily_session_log: list = []
_daily_session_log_date: str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

# Market-session stats (Midnight/Asian/London/New York) ────────────────
# Persists across TP/SL resets within a calendar day; resets at midnight UTC.
market_session_stats: dict = {
    name: {"wins": 0, "losses": 0, "pnl": 0.0, "trades": 0}
    for name in MARKET_SESSIONS
}
_market_session_date: str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

# Score sparklines ─────────────────────────────────────────────────────
symbol_score_history: dict = {sym: deque(maxlen=20) for sym in SYMBOLS}

telegram_app = None
_tg_loop = None
_tg_ready = threading.Event()   # set only after app.start() + polling confirmed up
_auto_resume_active = False
_bot_ready = False          # set True after history + DB + WS threads are up
_pinned_msg_id: Optional[int] = None   # ID of the auto-pinned dashboard message
_current_session_name: str = ""        # for session-change auto-reports
_test_trade_sem = threading.Semaphore(1)
_test_trade_active: dict = {}

# ML state ─────────────────────────────────────────────────────────────
ml_model = None          # global combined model (all symbols)
ml_trained_on = 0
ml_total_trades = 0
ml_lock = threading.Lock()
ml_training_active = False
ml_models_per_class: dict = {cls: None for cls in ML_SYMBOL_CLASSES}   # per-class models
ml_trained_per_class: dict = {cls: 0   for cls in ML_SYMBOL_CLASSES}

# Tick momentum ─────────────────────────────────────────────────────────
tick_history: dict = {}  # sym → deque(maxlen=10) of intra-candle close prices

# ── Performance: indicator computation serialiser & tick throttle ────────
_indicator_executor: Optional[ThreadPoolExecutor] = None  # initialised in main()
_last_tick_score_time: dict = {}   # sym → float  – throttle intra-candle score_signal
_last_computed_epoch:  dict = {}   # sym → int    – skip redundant indicator recomputes

# ══════════════════════════════════════════════════════════════════════
#  SESSION HELPERS
# ══════════════════════════════════════════════════════════════════════
def _get_session_name(dt: Optional[datetime] = None) -> str:
    """Return the market session name for a given UTC datetime."""
    if dt is None:
        dt = datetime.now(timezone.utc)
    h = dt.hour
    if h >= 22 or h < 2:
        return "Midnight"
    elif h < 10:
        return "Asian"
    elif h < 13:
        return "London"
    else:
        return "New York"


def _next_session_start(dt: Optional[datetime] = None) -> datetime:
    """Return the UTC datetime when the NEXT market session begins."""
    if dt is None:
        dt = datetime.now(timezone.utc)
    current = _get_session_name(dt)
    # Each session ends (next session starts) at this UTC hour
    next_start_hour = {"Midnight": 2, "Asian": 10, "London": 13, "New York": 22}
    h = next_start_hour[current]
    candidate = dt.replace(hour=h, minute=0, second=0, microsecond=0)
    if candidate <= dt:
        candidate += timedelta(days=1)
    return candidate


def _roll_market_session_stats_if_needed():
    """Clear market_session_stats if UTC date has rolled over."""
    global market_session_stats, _market_session_date
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with _lock:
        if today != _market_session_date:
            market_session_stats = {
                name: {"wins": 0, "losses": 0, "pnl": 0.0, "trades": 0}
                for name in MARKET_SESSIONS
            }
            _market_session_date = today


# ══════════════════════════════════════════════════════════════════════
#  ML HELPERS
# ══════════════════════════════════════════════════════════════════════
def _get_symbol_class(symbol: str) -> str:
    for cls, syms in ML_SYMBOL_CLASSES.items():
        if symbol in syms:
            return cls
    return "standard_vol"


def _ml_load():
    global ml_model, ml_trained_on, ml_models_per_class, ml_trained_per_class
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                payload = pickle.load(f)
            if isinstance(payload, dict):
                ml_model = payload.get("model")
                ml_trained_on = payload.get("trained_on", 0)
                ml_models_per_class  = payload.get("per_class_models",  ml_models_per_class)
                ml_trained_per_class = payload.get("per_class_trained", ml_trained_per_class)
            else:
                # Legacy: bare model object saved without wrapper dict
                ml_model = payload
                ml_trained_on = 0
            logger.info(f"ML model loaded (trained_on={ml_trained_on}, "
                        f"classes={[k for k,v in ml_models_per_class.items() if v]})")
        except Exception as e:
            logger.error(f"Failed to load ML model: {e}")


def _ml_save():
    try:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({
                "model":              ml_model,
                "trained_on":         ml_trained_on,
                "per_class_models":   ml_models_per_class,
                "per_class_trained":  ml_trained_per_class,
            }, f)
    except Exception as e:
        logger.error(f"Failed to save ML model: {e}")


def _ml_get_confidence(details: dict, symbol: str = "") -> float:
    """Return ML win-probability (0.0–1.0) using the best available model for this symbol."""
    cls = _get_symbol_class(symbol) if symbol else "standard_vol"
    with ml_lock:
        # Prefer per-class model; fall back to global model
        model = ml_models_per_class.get(cls) or ml_model
    if model is None:
        return 1.0
    try:
        feats = [[
            details.get("total_score", 0),
            details.get("wick_atr_ratio", 0),
            details.get("atr", 0) or 0,
            details.get("atr_ma", 0) or 0,
            details.get("ema_fast_sl", 0) or 0,
            details.get("ema_slow_sl", 0) or 0,
            details.get("ema_distance", 0) or 0,
        ]]
        proba = model.predict_proba(feats)[0]
        classes = list(model.classes_)
        win_idx = classes.index(1) if 1 in classes else len(classes) - 1
        return float(proba[win_idx])
    except Exception as e:
        logger.error(f"ML confidence failed, allowing trade: {e}")
        return 1.0


def _ml_should_trade(details: dict, symbol: str = "") -> bool:
    """Filter trade by ML confidence. Stores 'ml_confidence' in details for display."""
    conf = _ml_get_confidence(details, symbol)
    details["ml_confidence"] = conf
    cls = _get_symbol_class(symbol) if symbol else "standard_vol"
    with ml_lock:
        has_model = (ml_models_per_class.get(cls) or ml_model) is not None
    if not has_model:
        return True   # observe-only until first model
    return conf >= ML_CONFIDENCE_MIN


def _ml_export_csv(total: int):
    try:
        conn = sqlite3.connect("trades.db")
        cols = [
            "id", "timestamp", "symbol", "direction", "barrier",
            "stake", "payout", "profit", "win", "score",
            "wick_atr_ratio", "atr", "atr_ma",
            "ema_fast_slope", "ema_slow_slope", "ema_distance",
            "market_session",
        ]
        # Gracefully handle missing market_session column in older DBs
        try:
            rows = conn.execute(
                f"SELECT {', '.join(cols)} FROM trades ORDER BY id"
            ).fetchall()
        except sqlite3.OperationalError:
            cols = cols[:-1]  # drop market_session
            rows = conn.execute(
                f"SELECT {', '.join(cols)} FROM trades ORDER BY id"
            ).fetchall()
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
            f"Features: Supertrend Dir | ADX | ST-Distance + RSI/MACD/BB confluence."
        )
        _send_tg_document(csv_bytes, filename, caption)
        logger.info(f"ML CSV export sent ({len(rows)} rows)")
    except Exception as e:
        logger.error(f"ML CSV export failed: {e}")
        _send_tg(f"⚠️ <b>ML CSV export failed</b>\n<code>{e}</code>")


def _ml_progress_bar(pct: float, width: int = 12) -> str:
    filled = int(width * min(1.0, pct))
    return "█" * filled + "░" * (width - filled)


def _ml_train():
    """Train / retrain RandomForest (global + per-class). Sets ml_training_active; clears in finally."""
    global ml_model, ml_trained_on, ml_training_active, ml_models_per_class, ml_trained_per_class
    try:
        try:
            conn = sqlite3.connect("trades.db")
            # Fetch with symbol so we can split by class
            rows_sym = conn.execute(
                f"SELECT symbol, {', '.join(ML_FEATURE_COLS)}, win FROM trades"
            ).fetchall()
            conn.close()
        except Exception as e:
            logger.error(f"ML training query failed: {e}")
            return

        rows  = [r[1:] for r in rows_sym]   # drop symbol column for global model
        total = len(rows)
        _send_tg(
            f"🤖 <b>ML RETRAINING</b> — started\n"
            f"Training on <b>{total}</b> real trades (global + 3 class models)…\n"
            f"<code>[{_ml_progress_bar(0.0)}]   0%</code>"
        )

        if total < ML_MIN_TRADES:
            _send_tg(
                f"🤖 <b>ML RETRAINING</b> — skipped\n"
                f"Need {ML_MIN_TRADES} trades, only {total} recorded.\n"
                f"<code>[{_ml_progress_bar(total / ML_MIN_TRADES)}] "
                f"{int(total / ML_MIN_TRADES * 100)}%</code> (observe-only)"
            )
            return

        X = [[r[i] if r[i] is not None else 0.0 for i in range(len(ML_FEATURE_COLS))] for r in rows]
        y = [r[-1] for r in rows]

        if len(set(y)) < 2:
            _send_tg("🤖 <b>ML RETRAINING</b> — skipped\nNeed both wins AND losses in history.")
            return

        try:
            # ── Global model (all symbols) ─────────────────────────────────
            clf = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
            clf.fit(X, y)
            with ml_lock:
                ml_model = clf
                ml_trained_on = total

            # ── Per-class models ───────────────────────────────────────────
            new_cls_models   = {}
            new_cls_trained  = {}
            cls_lines = []
            for cls, syms in ML_SYMBOL_CLASSES.items():
                cls_rows = [r for r in rows_sym if r[0] in syms]
                if len(cls_rows) < 20:
                    cls_lines.append(f"  {cls}: only {len(cls_rows)} trades — skipped")
                    continue
                Xc = [[r[i+1] if r[i+1] is not None else 0.0 for i in range(len(ML_FEATURE_COLS))]
                      for r in cls_rows]
                yc = [r[-1] for r in cls_rows]
                if len(set(yc)) < 2:
                    cls_lines.append(f"  {cls}: need both outcomes — skipped")
                    continue
                clf_c = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
                clf_c.fit(Xc, yc)
                new_cls_models[cls]  = clf_c
                new_cls_trained[cls] = len(cls_rows)
                wr_c = sum(yc) / len(yc) * 100
                cls_lines.append(f"  {cls}: {len(cls_rows)} trades  {wr_c:.0f}%WR ✅")

            with ml_lock:
                ml_models_per_class.update(new_cls_models)
                ml_trained_per_class.update(new_cls_trained)

            _ml_save()
            _send_tg(
                f"🤖 <b>ML RETRAINING COMPLETE</b> ✅\n"
                f"<code>[{_ml_progress_bar(1.0)}] 100%</code>\n"
                f"Global: <b>{total}</b> trades  |  Gate ≥<b>{ML_CONFIDENCE_MIN*100:.0f}%</b>\n"
                f"Per-class:\n" + "\n".join(cls_lines)
            )
            _ml_export_csv(total)
        except Exception as e:
            logger.error(f"ML training failed: {e}")
            _send_tg(f"🤖 <b>ML RETRAINING FAILED</b>\n<code>{e}</code>")
    finally:
        with ml_lock:
            ml_training_active = False


def _ml_maybe_retrain(total_trades: int):
    global ml_total_trades, ml_training_active
    ml_total_trades = total_trades
    spawn = False
    with ml_lock:
        if not ml_training_active:
            if ml_trained_on == 0 and total_trades >= ML_MIN_TRADES:
                spawn = True
            elif ml_trained_on and total_trades - ml_trained_on >= ML_RETRAIN_EVERY:
                spawn = True
        if spawn:
            ml_training_active = True
    if spawn:
        threading.Thread(target=_ml_train, daemon=True, name="MLTrain").start()


def _ml_bootstrap_from_history():
    """
    Build initial ML training data from historical candle data.
    Simulates ONETOUCH outcomes: did price touch the barrier in next DURATION candles?
    This lets the ML filter work from the very first real trade.
    """
    global ml_model, ml_trained_on

    with ml_lock:
        if ml_model is not None:
            _send_tg(
                f"🤖 ML model already loaded (trained on {ml_trained_on} samples) — skipping bootstrap."
            )
            return

    _send_tg("🤖 <b>ML Bootstrap</b> — building training data from candle history…")

    all_X, all_y = [], []

    for sym in SYMBOLS:
        with _lock:
            if sym not in ohlcv or len(ohlcv[sym]) < 100:
                continue
            df = ohlcv[sym].copy()
            ind = dict(indicators.get(sym, {}))

        if not ind.get("ready"):
            continue
        atr = ind.get("atr") or 0
        if atr <= 0:
            continue

        barrier_offset = atr * ATR_BARRIER_MULT
        st_dir  = float(ind.get("supertrend_dir", 0))
        adx_v   = float(ind.get("adx") or 0)
        atr_ma  = float(ind.get("atr_ma") or atr)
        # ema_distance slot = ST-distance / ATR (barrier_offset / atr = ATR_BARRIER_MULT constant)
        # This matches the 'ema_distance' column used at live-inference time (score_signal line 964).
        st_dist_f = barrier_offset / atr  # = ATR_BARRIER_MULT; consistent proxy across all bootstrap rows

        check_from = max(30, len(df) - 400)
        check_to   = len(df) - DURATION - 1
        if check_to <= check_from:
            continue

        for i in range(check_from, check_to):
            candle = df.iloc[i]
            future = df.iloc[i + 1:i + DURATION + 1]
            close  = float(candle["Close"])

            direction = "UP" if st_dir >= 0 else "DOWN"
            if direction == "UP":
                target  = close + barrier_offset
                touched = bool((future["High"] >= target).any())
            else:
                target  = close - barrier_offset
                touched = bool((future["Low"] <= target).any())

            hi   = float(candle["High"])
            lo   = float(candle["Low"])
            wick = (hi - lo) / (atr or 1)

            all_X.append([
                float(SCORE_THRESHOLD), wick, float(atr), atr_ma,
                st_dir, adx_v, st_dist_f,
            ])
            all_y.append(1 if touched else 0)

    total = len(all_X)
    wins  = sum(all_y)

    if total < ML_MIN_TRADES or len(set(all_y)) < 2:
        _send_tg(
            f"🤖 Bootstrap: only {total} samples (need {ML_MIN_TRADES} with both outcomes).\n"
            f"Bot will train on real trades."
        )
        return

    try:
        clf = RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=42
        )
        clf.fit(all_X, all_y)
        with ml_lock:
            ml_model = clf
            ml_trained_on = total
        _ml_save()
        logger.info(f"ML bootstrap done: {total} samples, {wins} wins ({100*wins/total:.1f}%)")
        _send_tg(
            f"🤖 <b>ML Bootstrap Complete</b> ✅\n"
            f"Trained on <b>{total}</b> historical candle simulations\n"
            f"Simulated win rate: <b>{100*wins/total:.1f}%</b>\n"
            f"Confidence gate: ≥ <b>{ML_CONFIDENCE_MIN*100:.0f}%</b> — ACTIVE now\n"
            f"<i>Per-class models activate after {ML_MIN_TRADES} real trades each.</i>"
        )
    except Exception as e:
        logger.error(f"ML bootstrap failed: {e}")
        _send_tg(f"⚠️ ML bootstrap failed: <code>{e}</code>")


def _ml_progress_text() -> str:
    with ml_lock:
        trained_on, model, training = ml_trained_on, ml_model, ml_training_active
    total = ml_total_trades
    bar_len = 10
    if training:
        bar = _ml_progress_bar(0.5, bar_len)
        return f"ML Filter  : ⏳ RETRAINING [{bar}] in progress…"
    if model is None:
        pct    = min(1.0, total / ML_MIN_TRADES) if ML_MIN_TRADES else 1.0
        filled = int(bar_len * pct)
        bar    = "█" * filled + "░" * (bar_len - filled)
        return f"ML Filter  : warming up [{bar}] {total}/{ML_MIN_TRADES} (observe-only)"
    since  = max(0, total - trained_on)
    pct    = min(1.0, since / ML_RETRAIN_EVERY) if ML_RETRAIN_EVERY else 1.0
    filled = int(bar_len * pct)
    bar    = "█" * filled + "░" * (bar_len - filled)
    return f"ML Filter  : ACTIVE [{bar}] retrain in {max(0, ML_RETRAIN_EVERY - since)} trades  (≥{ML_CONFIDENCE_MIN*100:.0f}% conf)"


# ══════════════════════════════════════════════════════════════════════
#  SYMBOL STATE INIT
# ══════════════════════════════════════════════════════════════════════
def _init_symbol(sym):
    ohlcv[sym]          = pd.DataFrame(columns=["Open", "High", "Low", "Close"])
    ohlcv_m5[sym]       = pd.DataFrame(columns=["Open", "High", "Low", "Close"])
    indicators_m5[sym]  = {"supertrend_dir": 0, "ready": False}
    ohlcv_m15[sym]      = pd.DataFrame(columns=["Open", "High", "Low", "Close"])
    indicators_m15[sym] = {"supertrend_dir": 0, "ready": False}
    current_candle[sym] = None
    last_price[sym]     = 0.0
    cooldown_until[sym] = datetime.min.replace(tzinfo=timezone.utc)
    tick_history[sym]   = deque(maxlen=10)
    indicators[sym] = {
        "ema_slow": None,       # EMA 200 value (direction fallback + confluence)
        # Supertrend
        "supertrend_val":   None,
        "supertrend_dir":   0,     # 1=bullish, -1=bearish
        "supertrend_upper": None,
        "supertrend_lower": None,
        # Volatility
        "atr": None, "atr_ma": None, "atr_rising": False,
        # Momentum / oscillators
        "rsi": None,
        "stochrsi_k": None, "stochrsi_d": None,   # StochRSI (K and D lines, 0-100)
        "macd_bullish": False, "macd_hist": None, "macd_hist_rising": False,
        # ADX / DI
        "adx": None, "di_bullish": False,
        # Bollinger Bands
        "bb_upper": None, "bb_lower": None, "bb_mid": None, "bb_position": None,
        # Candle quality
        "body_ratio":     None,   # abs(close-open)/(high-low) — 0=doji, 1=full body
        "upper_wick_atr": None,   # upper wick size in ATR units
        "lower_wick_atr": None,   # lower wick size in ATR units
        "ready": False,
    }


# ══════════════════════════════════════════════════════════════════════
#  DATABASE
# ══════════════════════════════════════════════════════════════════════
db_queue: queue.Queue = queue.Queue()

_CREATE_TABLE_SQLITE = """
    CREATE TABLE IF NOT EXISTS trades (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp       TEXT,
        symbol          TEXT,
        direction       TEXT,
        barrier         TEXT,
        stake           REAL,    payout          REAL,
        profit          REAL,    win             INTEGER,
        score           REAL,    wick_atr_ratio  REAL,
        atr             REAL,    atr_ma          REAL,
        ema_fast_slope  REAL,    ema_slow_slope  REAL,
        ema_distance    REAL,
        market_session  TEXT
    )
"""

_CREATE_TABLE_PG = """
    CREATE TABLE IF NOT EXISTS trades (
        id              SERIAL PRIMARY KEY,
        timestamp       TEXT,
        symbol          TEXT,
        direction       TEXT,
        barrier         TEXT,
        stake           REAL,    payout          REAL,
        profit          REAL,    win             INTEGER,
        score           REAL,    wick_atr_ratio  REAL,
        atr             REAL,    atr_ma          REAL,
        ema_fast_slope  REAL,    ema_slow_slope  REAL,
        ema_distance    REAL,
        market_session  TEXT
    )
"""

_INSERT_COLS = (
    "timestamp,symbol,direction,barrier,stake,payout,profit,win,score,"
    "wick_atr_ratio,atr,atr_ma,ema_fast_slope,ema_slow_slope,ema_distance,market_session"
)


def _db_fetch(sql: str, params: tuple = ()) -> list:
    """Run a SELECT and return rows. sql uses ? placeholders (auto-converted for PG)."""
    try:
        if USE_PG:
            conn = psycopg2.connect(DATABASE_URL)
            cur  = conn.cursor()
            cur.execute(sql.replace("?", "%s"), params)
            rows = cur.fetchall()
            cur.close(); conn.close()
        else:
            conn = sqlite3.connect("trades.db")
            rows = conn.execute(sql, params).fetchall()
            conn.close()
        return rows
    except Exception as e:
        logger.error(f"DB fetch error: {e}")
        return []


def _db_writer():
    if USE_PG:
        conn = psycopg2.connect(DATABASE_URL)
        conn.autocommit = True
        cur  = conn.cursor()
        cur.execute(_CREATE_TABLE_PG)
        # Add market_session column if missing (older PG schema)
        cur.execute("""
            DO $ BEGIN
                ALTER TABLE trades ADD COLUMN market_session TEXT;
            EXCEPTION WHEN duplicate_column THEN NULL;
            END $;
        """)
        def _write(item):
            cur.execute(
                f"INSERT INTO trades ({_INSERT_COLS}) "
                f"VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                item,
            )
            cur.execute("SELECT COUNT(*) FROM trades")
            return cur.fetchone()[0]
    else:
        conn = sqlite3.connect("trades.db")
        conn.execute(_CREATE_TABLE_SQLITE)
        try:
            conn.execute("ALTER TABLE trades ADD COLUMN market_session TEXT")
        except sqlite3.OperationalError:
            pass
        conn.commit()

    # ── PostgreSQL path: write immediately (autocommit) ──────────────────
    if USE_PG:
        while True:
            item = db_queue.get()
            if item is None:
                break
            try:
                total = _write(item)
                _ml_maybe_retrain(total)
            except Exception as e:
                logger.error(f"DB write error: {e}")
        return

    # ── SQLite path: batch writes to reduce commit overhead ──────────────
    _BATCH_MAX      = 5
    _FLUSH_INTERVAL = 2.0      # seconds between forced flushes
    _batch: list    = []
    _last_flush     = [time.time()]

    def _sqlite_flush() -> int:
        """Commit pending batch; return new total row count (0 if nothing flushed)."""
        if not _batch:
            return 0
        try:
            with conn:
                for it in _batch:
                    conn.execute(
                        f"INSERT INTO trades ({_INSERT_COLS}) "
                        f"VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        it,
                    )
            _batch.clear()
            _last_flush[0] = time.time()
            return conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        except Exception as exc:
            logger.error(f"DB batch flush error: {exc}")
            return 0

    while True:
        try:
            item = db_queue.get(timeout=_FLUSH_INTERVAL)
        except queue.Empty:
            # Timeout — flush whatever has accumulated
            try:
                total = _sqlite_flush()
                if total:
                    _ml_maybe_retrain(total)
            except Exception as e:
                logger.error(f"DB flush error: {e}")
            continue

        if item is None:
            # Shutdown sentinel — flush remaining rows and exit
            try:
                _sqlite_flush()
            except Exception:
                pass
            break

        _batch.append(item)
        now = time.time()
        if len(_batch) >= _BATCH_MAX or (now - _last_flush[0]) >= _FLUSH_INTERVAL:
            try:
                total = _sqlite_flush()
                if total:
                    _ml_maybe_retrain(total)
            except Exception as e:
                logger.error(f"DB write error: {e}")


def _watch_thread(target, args=(), name="Worker", restartable=True):
    def _wrapper():
        while True:
            try:
                target(*args)
            except Exception as e:
                logger.error(f"Thread {name} crashed: {e}", exc_info=True)
            if not restartable:
                break
            logger.info(f"Thread {name} restarting in 5 s…")
            time.sleep(5)
    t = threading.Thread(target=_wrapper, daemon=True, name=name)
    t.start()
    return t


# ══════════════════════════════════════════════════════════════════════
#  DB QUERY HELPERS  (SQLite + PostgreSQL via _db_fetch)
# ══════════════════════════════════════════════════════════════════════
def get_recent_trades(limit=8):
    return _db_fetch(
        "SELECT timestamp, symbol, direction, profit, win, score "
        "FROM trades ORDER BY id DESC LIMIT ?", (limit,)
    )


def get_db_summary():
    rows = _db_fetch(
        "SELECT COUNT(*), SUM(profit), "
        "SUM(CASE WHEN win=1 THEN 1 ELSE 0 END), "
        "SUM(CASE WHEN win=0 THEN 1 ELSE 0 END) FROM trades"
    )
    return rows[0] if rows else (0, 0, 0, 0)


def get_alltime_symbol_stats(limit=10):
    return _db_fetch(
        "SELECT symbol, COUNT(*), SUM(CASE WHEN win=1 THEN 1 ELSE 0 END), SUM(profit) "
        "FROM trades GROUP BY symbol ORDER BY SUM(profit) DESC LIMIT ?", (limit,)
    )


def get_alltime_daily_stats(limit=7):
    return _db_fetch(
        "SELECT date(timestamp) as day, COUNT(*), "
        "SUM(CASE WHEN win=1 THEN 1 ELSE 0 END), SUM(profit) "
        "FROM trades GROUP BY day ORDER BY day DESC LIMIT ?", (limit,)
    )


def get_7day_full_breakdown():
    """Return per-day, per-session breakdown for the last 7 days."""
    if USE_PG:
        sql = (
            "SELECT DATE(timestamp), market_session, COUNT(*), "
            "SUM(CASE WHEN win=1 THEN 1 ELSE 0 END), SUM(profit) "
            "FROM trades "
            "WHERE DATE(timestamp) >= CURRENT_DATE - INTERVAL '7 days' "
            "GROUP BY DATE(timestamp), market_session "
            "ORDER BY DATE(timestamp) DESC, SUM(profit) DESC"
        )
    else:
        sql = (
            "SELECT date(timestamp), market_session, COUNT(*), "
            "SUM(CASE WHEN win=1 THEN 1 ELSE 0 END), SUM(profit) "
            "FROM trades "
            "WHERE date(timestamp) >= date('now', '-7 days') "
            "GROUP BY date(timestamp), market_session "
            "ORDER BY date(timestamp) DESC, SUM(profit) DESC"
        )
    return _db_fetch(sql)


def get_session_alltime_stats():
    """Return per-market-session lifetime stats."""
    return _db_fetch(
        "SELECT market_session, COUNT(*), "
        "SUM(CASE WHEN win=1 THEN 1 ELSE 0 END), SUM(profit) "
        "FROM trades WHERE market_session IS NOT NULL "
        "GROUP BY market_session ORDER BY SUM(profit) DESC"
    )


# ══════════════════════════════════════════════════════════════════════
#  SUPERTREND COMPUTATION
# ══════════════════════════════════════════════════════════════════════
def _compute_supertrend(df: pd.DataFrame, period: int = SUPERTREND_PERIOD,
                        multiplier: float = SUPERTREND_ATR_MULT):
    """
    Vectorised Supertrend indicator.
    Returns (supertrend_line, direction_series, final_upper, final_lower)
    direction: 1 = bullish (price above ST), -1 = bearish (price below ST)
    """
    hl2   = (df["High"] + df["Low"]) / 2
    hi, lo, pc = df["High"], df["Low"], df["Close"].shift(1)
    tr    = np.maximum(hi - lo, np.maximum(abs(hi - pc), abs(lo - pc)))
    atr_s = tr.rolling(period).mean()

    basic_upper = hl2 + multiplier * atr_s
    basic_lower = hl2 - multiplier * atr_s

    n     = len(df)
    close = df["Close"]

    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    st          = pd.Series(np.nan, index=df.index)
    direction   = pd.Series(0,      index=df.index)

    # Find first row where ATR is valid; skip NaN-polluted early rows entirely
    valid_atr = atr_s.notna()
    if not valid_atr.any():
        return st, direction, final_upper, final_lower
    first_valid_i = int(np.where(valid_atr.values)[0][0])
    start_i = max(1, first_valid_i)

    for i in range(start_i, n):
        # Skip if current bands are still NaN (shouldn't happen after first_valid_i, but guard anyway)
        if pd.isna(basic_upper.iloc[i]) or pd.isna(basic_lower.iloc[i]):
            continue

        prev_fu = final_upper.iloc[i - 1]
        prev_fl = final_lower.iloc[i - 1]
        prev_c  = close.iloc[i - 1]

        # Final upper band
        if pd.isna(prev_fu) or basic_upper.iloc[i] < prev_fu or prev_c > prev_fu:
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = prev_fu

        # Final lower band
        if pd.isna(prev_fl) or basic_lower.iloc[i] > prev_fl or prev_c < prev_fl:
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = prev_fl

        # Supertrend direction
        prev_st = st.iloc[i - 1]
        if pd.isna(prev_st) or pd.isna(prev_fu):
            # Initialise: direction from whether close is above upper band
            if close.iloc[i] > final_upper.iloc[i]:
                st.iloc[i]        = final_lower.iloc[i]
                direction.iloc[i] = 1
            else:
                st.iloc[i]        = final_upper.iloc[i]
                direction.iloc[i] = -1
        elif prev_st == prev_fu:
            # Was bearish (sitting on upper band)
            if close.iloc[i] > final_upper.iloc[i]:
                st.iloc[i]        = final_lower.iloc[i]
                direction.iloc[i] = 1
            else:
                st.iloc[i]        = final_upper.iloc[i]
                direction.iloc[i] = -1
        else:
            # Was bullish (sitting on lower band)
            if close.iloc[i] < final_lower.iloc[i]:
                st.iloc[i]        = final_upper.iloc[i]
                direction.iloc[i] = -1
            else:
                st.iloc[i]        = final_lower.iloc[i]
                direction.iloc[i] = 1

    return st, direction, final_upper, final_lower


# ══════════════════════════════════════════════════════════════════════
#  INDICATOR UPDATE
# ══════════════════════════════════════════════════════════════════════
def update_indicators(symbol: str) -> bool:
    # Take a locked snapshot of OHLCV-only columns so we never hold the lock
    # during heavy computation, and we don't pollute ohlcv with indicator columns.
    with _lock:
        raw = ohlcv[symbol]
        if len(raw) < max(500, SUPERTREND_PERIOD + 5):
            return False
        df = raw[["Open", "High", "Low", "Close"]].copy()

    # EMA Slow (for fallback direction only)
    df["EMA_SLOW"] = df["Close"].ewm(span=EMA_SLOW, adjust=False).mean()

    # ATR
    hi, lo, pc = df["High"], df["Low"], df["Close"].shift(1)
    tr = np.maximum(hi - lo, np.maximum(abs(hi - pc), abs(lo - pc)))
    df["ATR"]    = tr.rolling(ATR_PERIOD).mean()
    df["ATR_MA"] = df["ATR"].rolling(ATR_MA_PERIOD).mean()

    # RSI
    delta    = df["Close"].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = (100 - (100 / (1 + rs))).fillna(100)

    # StochRSI — (14-period) with K=3, D=3 smoothing
    rsi_min = df["RSI"].rolling(14).min()
    rsi_max = df["RSI"].rolling(14).max()
    rsi_rng = (rsi_max - rsi_min).replace(0, np.nan)
    df["SRSI_K"] = ((df["RSI"] - rsi_min) / rsi_rng).ewm(span=3, adjust=False).mean() * 100
    df["SRSI_D"] = df["SRSI_K"].ewm(span=3, adjust=False).mean()

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_HIST"]   = df["MACD"] - df["MACD_SIGNAL"]

    # ADX / DI
    up_move  = hi.diff()
    down_move = -lo.diff()
    plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr14    = tr.ewm(alpha=1 / ATR_PERIOD, adjust=False).mean()
    plus_di  = 100 * pd.Series(plus_dm,  index=df.index).ewm(alpha=1 / ATR_PERIOD, adjust=False).mean() / atr14.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / ATR_PERIOD, adjust=False).mean() / atr14.replace(0, np.nan)
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0)
    df["ADX"]      = dx.ewm(alpha=1 / ATR_PERIOD, adjust=False).mean()
    df["PLUS_DI"]  = plus_di
    df["MINUS_DI"] = minus_di

    # Bollinger Bands
    bb_mid = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    df["BB_MID"]   = bb_mid
    df["BB_UPPER"] = bb_mid + 2 * bb_std
    df["BB_LOWER"] = bb_mid - 2 * bb_std
    df["BB_WIDTH"] = (df["BB_UPPER"] - df["BB_LOWER"]) / bb_mid.replace(0, np.nan)

    # Supertrend
    st_line, st_dir_series, st_upper, st_lower = _compute_supertrend(df)
    df["ST"]       = st_line
    df["ST_DIR"]   = st_dir_series
    df["ST_UPPER"] = st_upper
    df["ST_LOWER"] = st_lower

    with _lock:
        ind = indicators[symbol]
        ind["ema_slow"]         = df["EMA_SLOW"].iloc[-1]
        ind["atr"]              = df["ATR"].iloc[-1]
        ind["atr_ma"]           = df["ATR_MA"].iloc[-1]
        ind["atr_rising"]       = (df["ATR"].diff().iloc[-5:] > 0).sum() >= 3
        ind["rsi"]              = df["RSI"].iloc[-1]
        srsi_k = df["SRSI_K"].iloc[-1]; srsi_d = df["SRSI_D"].iloc[-1]
        ind["stochrsi_k"] = round(float(srsi_k), 1) if pd.notna(srsi_k) else None
        ind["stochrsi_d"] = round(float(srsi_d), 1) if pd.notna(srsi_d) else None

        # Supertrend
        ind["supertrend_val"]   = df["ST"].iloc[-1]   if pd.notna(df["ST"].iloc[-1])   else None
        ind["supertrend_dir"]   = int(df["ST_DIR"].iloc[-1])
        ind["supertrend_upper"] = df["ST_UPPER"].iloc[-1] if pd.notna(df["ST_UPPER"].iloc[-1]) else None
        ind["supertrend_lower"] = df["ST_LOWER"].iloc[-1] if pd.notna(df["ST_LOWER"].iloc[-1]) else None

        macd_val = df["MACD"].iloc[-1]
        macd_sig = df["MACD_SIGNAL"].iloc[-1]
        hist     = df["MACD_HIST"].iloc[-1]
        ind["macd_bullish"]     = bool(macd_val > macd_sig)
        ind["macd_hist"]        = round(float(hist), 5) if pd.notna(hist) else None
        ind["macd_hist_rising"] = (df["MACD_HIST"].diff().iloc[-3:] > 0).sum() >= 2

        adx_val = df["ADX"].iloc[-1]
        pdi, mdi = df["PLUS_DI"].iloc[-1], df["MINUS_DI"].iloc[-1]
        ind["adx"]        = round(float(adx_val), 1) if pd.notna(adx_val) else None
        ind["di_bullish"] = bool(pd.notna(pdi) and pd.notna(mdi) and pdi > mdi)

        bb_u = df["BB_UPPER"].iloc[-1];  bb_l = df["BB_LOWER"].iloc[-1]
        bb_m = df["BB_MID"].iloc[-1]
        ind["bb_upper"] = bb_u;  ind["bb_lower"] = bb_l;  ind["bb_mid"] = bb_m
        band_rng = (bb_u - bb_l) if pd.notna(bb_u) and pd.notna(bb_l) and (bb_u - bb_l) > 0 else None
        last_close = df["Close"].iloc[-1]
        ind["bb_position"] = float((last_close - bb_l) / band_rng) if band_rng else None

        # Candle quality: body ratio + wick sizes in ATR units
        last_open  = df["Open"].iloc[-1]
        last_high  = df["High"].iloc[-1]
        last_low   = df["Low"].iloc[-1]
        candle_rng = max(float(last_high - last_low), 1e-10)
        atr_safe   = max(float(ind.get("atr") or df["ATR"].iloc[-1] or 1.0), 1e-10)
        body_size  = abs(float(last_close) - float(last_open))
        ind["body_ratio"]     = round(body_size / candle_rng, 3)
        ind["upper_wick_atr"] = round((float(last_high) - max(float(last_open), float(last_close))) / atr_safe, 3)
        ind["lower_wick_atr"] = round((min(float(last_open), float(last_close)) - float(last_low)) / atr_safe, 3)

        ind["ready"] = True
    return True


# ══════════════════════════════════════════════════════════════════════
#  SIGNAL SCORING  (Supertrend-based)
# ══════════════════════════════════════════════════════════════════════
def score_signal(symbol: str, candle: dict) -> tuple:
    with _lock:
        ind = dict(indicators[symbol])
    score, details = 0, {}

    price    = float(candle["Close"])
    ema_slow = ind.get("ema_slow") or price
    st_dir   = ind.get("supertrend_dir", 0)
    st_val   = ind.get("supertrend_val")
    atr      = ind.get("atr") or 1.0

    # ─── Direction ────────────────────────────────────────────────────
    if st_dir != 0:
        direction = "UP" if st_dir == 1 else "DOWN"
    else:
        direction = "UP" if price > ema_slow else "DOWN"

    # ─── Trend: Supertrend (max 30 pts) ───────────────────────────────
    trend = 0
    if st_dir != 0:
        if (direction == "UP" and st_dir == 1) or (direction == "DOWN" and st_dir == -1):
            trend += 25   # ST confirms direction
        if st_val is not None:
            st_dist_atr = abs(price - st_val) / atr
            if st_dist_atr >= 0.5:
                trend += 5   # strong trend (price well above/below ST)
    else:
        # Supertrend not ready → use EMA_SLOW as fallback
        if (direction == "UP" and price > ema_slow) or (direction == "DOWN" and price < ema_slow):
            trend += 15
    score += trend
    details["trend"] = trend

    # ─── Entry quality: RSI + distance from Supertrend ────────────────
    extension = abs(price - (st_val or price)) / atr
    rsi       = ind.get("rsi")

    entry_quality = 0
    if extension <= 1.0:
        entry_quality += 15
    elif extension <= 2.0:
        entry_quality += 8

    # StochRSI oscillator scoring (15 pts) — falls back to plain RSI if not ready
    srsi_k   = ind.get("stochrsi_k")
    srsi_d   = ind.get("stochrsi_d")
    if srsi_k is not None:
        if direction == "UP":
            if srsi_k < 25:                  entry_quality += 15   # oversold → prime entry
            elif srsi_k < 50:                entry_quality += 8    # neutral-bearish → ok
            elif srsi_k < 70:                entry_quality += 4    # neutral-bullish → caution
            # srsi_k >= 70: overbought → 0 pts
        else:
            if srsi_k > 75:                  entry_quality += 15   # overbought → prime entry
            elif srsi_k > 50:                entry_quality += 8    # neutral-bullish → ok
            elif srsi_k > 30:                entry_quality += 4    # neutral-bearish → caution
            # srsi_k <= 30: oversold → 0 pts
    elif rsi is not None:
        # Fallback: plain RSI while StochRSI warms up
        if direction == "UP":
            if 40 <= rsi <= 65:              entry_quality += 15
            elif 30 <= rsi < 40 or 65 < rsi <= 75: entry_quality += 7
        else:
            if 35 <= rsi <= 60:              entry_quality += 15
            elif 25 <= rsi < 35 or 60 < rsi <= 70: entry_quality += 7

    # Wick rejection scoring (max 8 pts) — strong rejection wick confirms direction
    uw = ind.get("upper_wick_atr") or 0.0
    lw = ind.get("lower_wick_atr") or 0.0
    wick_pts = 0
    if direction == "UP":
        if lw >= 0.5:   wick_pts = 8    # strong bullish rejection (buyers stepped in)
        elif lw >= 0.25: wick_pts = 4
    else:
        if uw >= 0.5:   wick_pts = 8    # strong bearish rejection (sellers stepped in)
        elif uw >= 0.25: wick_pts = 4
    entry_quality += wick_pts

    # Candle body quality gate — doji/spinning-top candles are unreliable
    body_r = ind.get("body_ratio")
    if body_r is not None and body_r < 0.20:
        entry_quality = int(entry_quality * 0.6)   # weak candle: penalise entry score

    score += entry_quality
    details["entry_quality"]  = entry_quality
    details["extension_atr"]  = round(extension, 2)
    details["rsi"]            = round(rsi, 1) if rsi is not None else None
    details["stochrsi_k"]     = round(srsi_k, 1) if srsi_k is not None else None
    details["wick_pts"]       = wick_pts
    details["body_ratio"]     = round(body_r, 2) if body_r is not None else None

    # ─── Volatility (max 15 pts) ──────────────────────────────────────
    vol_ok = ind.get("atr_rising") and (ind.get("atr") or 0) > (ind.get("atr_ma") or 0)
    vol = 0
    if vol_ok:           vol += 10
    if ind.get("atr_rising"): vol += 5
    score += vol
    details["volatility"] = vol

    # ─── Momentum (max 10 pts) ────────────────────────────────────────
    if direction == "UP":
        mom = 10 if (st_val is None or price > st_val) else 0
    else:
        mom = 10 if (st_val is None or price < st_val) else 0
    score += mom
    details["momentum"] = mom

    # ─── Confluence: MACD + ADX + BB + EMA200 (max 25 pts) ──────────
    macd_agree = ind.get("macd_bullish") if direction == "UP" else not ind.get("macd_bullish")
    macd_agree = bool(macd_agree) and bool(ind.get("macd_hist_rising"))

    adx_val  = ind.get("adx") or 0
    di_agree = ind.get("di_bullish") if direction == "UP" else not ind.get("di_bullish")
    adx_agree = adx_val >= 25 and bool(di_agree)

    bb_pos = ind.get("bb_position")
    if bb_pos is None:
        bb_agree = False
    elif direction == "UP":
        bb_agree = bb_pos <= 0.75
    else:
        bb_agree = bb_pos >= 0.25

    # EMA 200 confluence: price on the correct side of the long-term trend
    ema200_agree = (direction == "UP"   and price > ema_slow) or \
                   (direction == "DOWN" and price < ema_slow)

    confluence_count = int(macd_agree) + int(adx_agree) + int(bb_agree) + int(ema200_agree)
    confluence = 0
    if macd_agree:    confluence += 8
    if adx_agree:     confluence += 6
    if bb_agree:      confluence += 5
    if ema200_agree:  confluence += 6   # EMA 200 alignment bonus
    score += confluence
    details["confluence"]             = confluence
    details["confluence_count"]       = confluence_count
    details["confluence_gate_passed"] = confluence_count >= 2
    details["ema200_agree"]           = ema200_agree

    # Gate: need ≥2 confluence OR very strong ST + volatility
    if confluence_count < 2:
        if not (trend >= 25 and vol >= 10):
            score = int(score * 0.5)

    # Cap at 100 — max theoretical is 105, display always shows /100
    score = min(score, 100)

    # Build ML feature values (stored in DB compat column names)
    st_dir_f  = float(st_dir)
    st_dist_f = abs(price - (st_val or price)) / atr
    wick_atr  = (float(candle.get("High", price)) - float(candle.get("Low", price))) / atr

    details.update({
        "total_score":   score,
        "atr":           ind.get("atr"),
        "atr_ma":        ind.get("atr_ma"),
        "adx":           adx_val,
        "macd_hist":     ind.get("macd_hist"),
        "bb_position":   round(bb_pos, 2) if bb_pos is not None else None,
        # DB compat columns (new meaning):
        "ema_fast_sl":   st_dir_f,     # supertrend direction
        "ema_slow_sl":   adx_val,      # ADX value
        "ema_distance":  st_dist_f,    # ST distance / ATR
        "wick_atr_ratio": round(wick_atr, 3),
    })
    return score, direction, details


# ══════════════════════════════════════════════════════════════════════
#  TICK MOMENTUM HELPER
# ══════════════════════════════════════════════════════════════════════
def _has_tick_momentum(symbol: str, direction: str) -> bool:
    """Return True if the last TICK_MOMENTUM_MIN intra-candle closes confirm direction.
    Returns True (allow) when there isn't enough data yet — we never block on uncertainty."""
    with _lock:
        hist = list(tick_history.get(symbol, []))
    n = TICK_MOMENTUM_MIN
    if len(hist) < n:
        return True   # not enough ticks accumulated — don't block
    recent = hist[-n:]
    if direction == "UP":
        return all(recent[i] <= recent[i + 1] for i in range(len(recent) - 1))
    else:
        return all(recent[i] >= recent[i + 1] for i in range(len(recent) - 1))


# ══════════════════════════════════════════════════════════════════════
#  M5 MULTI-TIMEFRAME INDICATORS
# ══════════════════════════════════════════════════════════════════════
def update_m5_indicators(symbol: str) -> bool:
    """Resample M1 candles → 5-min OHLCV and compute M5 Supertrend direction.
    Stores result in indicators_m5[symbol]. Returns True when M5 is ready."""
    try:
        with _lock:
            df_m1 = ohlcv[symbol].copy()
        if len(df_m1) < 60:          # need at least 60 M1 bars (1 h) to form useful M5
            return False
        df_m5 = (
            df_m1
            .resample("5min")
            .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
            .dropna()
        )
        if len(df_m5) < 20:          # need enough M5 bars for Supertrend
            return False
        _, st_dir_s, _, _ = _compute_supertrend(df_m5, period=SUPERTREND_PERIOD,
                                                 multiplier=SUPERTREND_ATR_MULT)
        m5_dir = int(st_dir_s.iloc[-1]) if len(st_dir_s) > 0 else 0
        with _lock:
            ohlcv_m5[symbol]                        = df_m5
            indicators_m5[symbol]["supertrend_dir"] = m5_dir
            indicators_m5[symbol]["ready"]          = True
        return True
    except Exception as e:
        logger.debug(f"update_m5_indicators {symbol}: {e}")
        return False


def update_m15_indicators(symbol: str) -> bool:
    """Resample M1 candles → 15-min OHLCV and compute M15 Supertrend direction.
    Stores result in indicators_m15[symbol]. Returns True when M15 is ready.
    With 500 M1 bars this yields ~33 M15 bars — enough for Supertrend(10)."""
    try:
        with _lock:
            df_m1 = ohlcv[symbol].copy()
        if len(df_m1) < 15:          # absolute minimum
            return False
        df_m15 = (
            df_m1
            .resample("15min")
            .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
            .dropna()
        )
        if len(df_m15) < 15:         # need enough bars for Supertrend period
            return False
        _, st_dir_s, _, _ = _compute_supertrend(df_m15, period=SUPERTREND_PERIOD,
                                                  multiplier=SUPERTREND_ATR_MULT)
        m15_dir = int(st_dir_s.iloc[-1]) if len(st_dir_s) > 0 else 0
        with _lock:
            ohlcv_m15[symbol]                        = df_m15
            indicators_m15[symbol]["supertrend_dir"] = m15_dir
            indicators_m15[symbol]["ready"]          = True
        return True
    except Exception as e:
        logger.debug(f"update_m15_indicators {symbol}: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════
#  CLOSED-CANDLE PROCESSOR  (runs in _indicator_executor, max 2 at once)
# ══════════════════════════════════════════════════════════════════════
def _process_closed_candle(symbol: str, closed: dict, ws) -> None:
    """
    Heavy indicator computation + signal evaluation for a just-closed candle.
    Submitted to _indicator_executor so at most 2 symbols compute simultaneously,
    preventing CPU spikes when all 15 candles close at the same time.
    """
    try:
        update_m5_indicators(symbol)
        update_m15_indicators(symbol)
        if not update_indicators(symbol):
            return
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
                cooldown_left  = max(0, int((cooldown_until[symbol] - now).total_seconds() // 60))
                locked_symbols[symbol] = {
                    "score": score, "details": details,
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
                    f"{('cooldown ' + str(cooldown_left) + 'm' if cooldown_left else 'armed')}"
                )
        else:
            with _lock:
                old_lock = locked_symbols.pop(symbol, None)
            if old_lock:
                _send_tg(f"🔓 {symbol} unlocked — score dropped to {score}/100")
    except Exception as e:
        logger.exception(f"_process_closed_candle {symbol}: {e}")


# ══════════════════════════════════════════════════════════════════════
#  BARRIER HELPER
# ══════════════════════════════════════════════════════════════════════
def _compute_barrier(symbol: str, direction: str = "UP") -> str:
    with _lock:
        atr = (indicators.get(symbol) or {}).get("atr") or 0.0
    if atr <= 0:
        return "+0.20" if direction == "UP" else "-0.20"
    offset = atr * ATR_BARRIER_MULT
    sign   = "+" if direction == "UP" else "-"
    return f"{sign}{offset:.2f}"


# ══════════════════════════════════════════════════════════════════════
#  TELEGRAM HELPERS
# ══════════════════════════════════════════════════════════════════════
def _tg_loop_ok() -> bool:
    """Return True only when the Telegram event loop exists and is still running."""
    return (
        telegram_app is not None
        and _tg_loop is not None
        and not _tg_loop.is_closed()
    )


def _send_tg(text: str, reply_markup=None, parse_mode: str = "HTML"):
    async def _inner():
        try:
            await telegram_app.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID, text=text,
                reply_markup=reply_markup, parse_mode=parse_mode,
            )
        except Exception as e:
            logger.error(f"send_tg failed: {e}")
    if _tg_loop_ok():
        try:
            asyncio.run_coroutine_threadsafe(_inner(), _tg_loop)
        except RuntimeError as e:
            logger.error(f"send_tg schedule failed: {e}")


def _send_tg_document(file_bytes: bytes, filename: str, caption: str = ""):
    async def _inner():
        try:
            await telegram_app.bot.send_document(
                chat_id=TELEGRAM_CHAT_ID,
                document=(filename, file_bytes),
                caption=caption,
                parse_mode="HTML",
            )
        except Exception as e:
            logger.error(f"send_tg_document failed: {e}")
    if _tg_loop_ok():
        try:
            asyncio.run_coroutine_threadsafe(_inner(), _tg_loop)
        except RuntimeError as e:
            logger.error(f"send_tg_document schedule failed: {e}")


def _send_tg_wait(text: str, reply_markup=None, parse_mode: str = "HTML", timeout: float = 8.0):
    async def _inner():
        try:
            await telegram_app.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID, text=text,
                reply_markup=reply_markup, parse_mode=parse_mode,
            )
        except Exception as e:
            logger.error(f"send_tg_wait failed: {e}")
    if _tg_loop_ok():
        try:
            fut = asyncio.run_coroutine_threadsafe(_inner(), _tg_loop)
            fut.result(timeout=timeout)
        except Exception as e:
            logger.debug(f"send_tg_wait timeout/error: {e}")


def _send_rejection(symbol: str, direction: str, score: int, reason: str):
    """Send a concise trade-rejected card to Telegram and write to log."""
    _log(f"❌ {symbol} {direction} rejected — {reason}")
    _send_tg(
        f"🚫 <b>REJECTED</b> — <code>{symbol}</code> {direction}\n"
        f"Score: <b>{score}/100</b>\n"
        f"Reason: <i>{reason}</i>"
    )


def _score_bar_str(score: int, width: int = 10) -> str:
    filled = max(0, min(width, int(score / 100 * width)))
    return "█" * filled + "░" * (width - filled)


def _component_bar(pts: int, max_pts: int, width: int = 8) -> str:
    filled = max(0, min(width, int(pts / max_pts * width))) if max_pts else 0
    return "█" * filled + "░" * (width - filled)


def _conf_str(conf: Optional[float]) -> str:
    """Format ML confidence for display."""
    if conf is None:
        return "—"
    pct = conf * 100
    if pct >= 97:
        emoji = "🟢"
    elif pct >= 93:
        emoji = "🟡"
    else:
        emoji = "🔴"
    return f"{emoji} {pct:.1f}%"


def _signal_card(sym: str, score: int, direction: str, details: dict) -> str:
    with _lock:
        wc, lc, pnl = win_count, loss_count, total_pnl
    total = wc + lc
    wr    = wc / total * 100 if total else 0
    pnl_str = f"{'+'if pnl>=0 else ''}${pnl:.2f}"
    session_line = f"#{total + 1}  |  {wc}W/{lc}L  {wr:.0f}%WR  |  P&L {pnl_str}"
    mkt_session  = _get_session_name()
    conf = details.get("ml_confidence")
    conf_line = f"🤖 ML Conf : <b>{_conf_str(conf)}</b>\n" if conf is not None else ""

    t  = details.get("trend", 0)
    eq = details.get("entry_quality", 0)
    v  = details.get("volatility", 0)
    m  = details.get("momentum", 0)
    cf = details.get("confluence", 0)
    cc = details.get("confluence_count", 0)

    st_dir  = details.get("ema_fast_sl", 0)
    adx_val = details.get("ema_slow_sl", 0) or 0

    return (
        f"{'🟢' if direction == 'UP' else '🔴'} <b>SIGNAL</b>  <code>{sym}</code>  "
        f"{'📈 UP' if direction == 'UP' else '📉 DOWN'}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🎯 Score   : <b>{score}/100</b>  [{_score_bar_str(score)}]\n"
        f"   Trend   : {t}/30  [{_component_bar(t,30)}]  ST {'↑Bullish' if st_dir>0 else '↓Bearish'}\n"
        f"   Entry   : {eq}/30  [{_component_bar(eq,30)}]  RSI {details.get('rsi','—')}\n"
        f"   Volatility: {v}/15  [{_component_bar(v,15)}]  ADX {adx_val:.0f}\n"
        f"   Momentum: {m}/10  [{_component_bar(m,10)}]\n"
        f"   Confluence: {cf}/20  [{_component_bar(cf,20)}]  ({cc}/3 factors)\n"
        f"{conf_line}"
        f"💵 Stake  : ${STAKE:.2f}  →  win ~${STAKE + TARGET_PROFIT:.2f}\n"
        f"🕐 Session: {SESSION_EMOJIS.get(mkt_session,'')} {mkt_session}\n"
        f"📋 Session : {session_line}\n"
    )


def _make_footer_text() -> str:
    return (
        f"  Stake ${STAKE}  ·  Session TP ${DAILY_PROFIT_TARGET}  ·  Session SL ${DAILY_LOSS_LIMIT}  ·  "
        f"Duration {DURATION}min  ·  Cooldown {COOLDOWN_MINUTES}min  ·  "
        f"Min Score ≥{SCORE_THRESHOLD}  ·  {_ml_progress_text()}"
    )


def _result_card(sym: str, profit: float, win: bool, details: dict) -> str:
    with _lock:
        wc, lc, pnl, cl = win_count, loss_count, total_pnl, consecutive_losses
    total  = wc + lc
    wr     = wc / total * 100 if total else 0
    pnl_str = f"+${profit:.2f}" if profit > 0 else f"${profit:.2f}"
    conf    = details.get("ml_confidence")
    conf_line = f"🤖 ML Conf : <b>{_conf_str(conf)}</b>\n" if conf is not None else ""

    if win:
        header = f"🎊 <b>Ibrahim congratulations 🎊</b>  <code>{sym}</code>"
    else:
        header = f"💀 <b>Lost bryme</b>  <code>{sym}</code>"

    return (
        f"{header}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"💵 <b>P&L     :</b> <b>{pnl_str}</b>\n"
        f"📊 <b>Session :</b> {'+' if pnl >= 0 else ''}${pnl:.2f}  ({wc}W / {lc}L  {wr:.0f}%)\n"
        f"🔴 <b>Streak  :</b> {'🔴' * cl if cl else '🟢 none'}\n"
        f"🕐 Session : {SESSION_EMOJIS.get(_get_session_name(),'')} {_get_session_name()}\n"
        f"{conf_line}"
    )


def _session_summary_text(snap: dict) -> str:
    reason = snap.get("reason", "MANUAL")
    ses    = snap.get("market_session", "")
    header = {
        "TP":          f"🎯 <b>SESSION TP HIT</b>  (target ${DAILY_PROFIT_TARGET:.2f})",
        "SL":          f"🛑 <b>SESSION SL HIT</b>  (floor ${DAILY_LOSS_LIMIT:.2f})",
        "MANUAL":      "📄 <b>SESSION REPORT</b>",
        "SESSION_END": f"🔔 <b>SESSION ENDED</b> — {SESSION_EMOJIS.get(ses,'')} {ses}",
    }.get(reason, "🔁 <b>SESSION LIMIT REACHED</b>")

    pnl  = snap["pnl"]
    wc   = snap["wins"]
    lc   = snap["losses"]
    total = wc + lc
    wr    = wc / total * 100 if total else 0
    dur   = snap.get("duration", timedelta(0))
    h, r  = divmod(int(dur.total_seconds()), 3600)
    mi    = r // 60
    dur_str = f"{h}h {mi}m" if h else f"{mi}m"
    mkt_s   = snap.get("market_session", _get_session_name())

    lines = [
        header, "━━━━━━━━━━━━━━━━━━━━",
        f"💵 <b>Session P&L</b>  : {'+' if pnl >= 0 else ''}${pnl:.2f}",
        f"📊 <b>Trades</b>       : {total}  ({wc}W / {lc}L  {wr:.0f}%WR)",
        f"📈 <b>Peak Equity</b>  : ${snap.get('peak', 0):.2f}",
        f"📉 <b>Max Drawdown</b> : -${snap.get('max_dd', 0):.2f}",
        f"⏱  <b>Duration</b>     : {dur_str}",
        f"🕐 <b>Market Session</b>: {SESSION_EMOJIS.get(mkt_s,'')} {mkt_s}",
        "",
    ]

    symbols = snap.get("symbols", {})
    traded  = {s: v for s, v in symbols.items() if v["wins"] + v["losses"] > 0}
    if traded:
        best_sym  = max(traded, key=lambda s: traded[s]["pnl"])
        worst_sym = min(traded, key=lambda s: traded[s]["pnl"])
        bv, wv = traded[best_sym], traded[worst_sym]
        b_wr = bv["wins"] / (bv["wins"] + bv["losses"]) * 100 if (bv["wins"] + bv["losses"]) else 0
        w_wr = wv["wins"] / (wv["wins"] + wv["losses"]) * 100 if (wv["wins"] + wv["losses"]) else 0
        lines += [
            f"  🏅 Best  : <code>{best_sym}</code>  {'+' if bv['pnl'] >= 0 else ''}${bv['pnl']:.2f}  ({b_wr:.0f}% WR)",
            f"  💔 Worst : <code>{worst_sym}</code>  {'+' if wv['pnl'] >= 0 else ''}${wv['pnl']:.2f}  ({w_wr:.0f}% WR)",
            "",
            "<b>All Assets Traded</b>",
        ]
        for sym, s in sorted(traded.items(), key=lambda kv: kv[1]["pnl"], reverse=True):
            s_wr  = s["wins"] / (s["wins"] + s["losses"]) * 100 if (s["wins"] + s["losses"]) else 0
            sign  = "+" if s["pnl"] >= 0 else ""
            lines.append(
                f"  <code>{sym:<10}</code>  {s['wins']}W/{s['losses']}L "
                f"({s_wr:.0f}%)  {sign}${s['pnl']:.2f}"
            )
    else:
        lines.append("  — no trades this session —")

    # Countdown to next session
    next_dt   = snap.get("next_session_at")
    next_name = snap.get("next_session_name", "")
    if next_dt:
        now = datetime.now(timezone.utc)
        secs_left = max(0, int((next_dt - now).total_seconds()))
        hh, rem = divmod(secs_left, 3600)
        mm = rem // 60
        if hh:
            wait_str = f"{hh}h {mm}m"
        else:
            wait_str = f"{mm}m"
        next_emoji = SESSION_EMOJIS.get(next_name, "")
        lines += [
            "",
            f"⏳ <b>Waiting for next session</b>",
            f"   {next_emoji} <b>{next_name}</b> starts in <b>{wait_str}</b>  ({next_dt.strftime('%H:%M UTC')})",
            f"   Bot will resume automatically. 🤖",
        ]
    else:
        lines += ["", "🔄 <b>New session started — bot keeps trading.</b>"]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
#  TRADE SLOT MANAGEMENT
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


# ══════════════════════════════════════════════════════════════════════
#  PROPOSAL / BUY
# ══════════════════════════════════════════════════════════════════════
def request_proposal(ws, symbol: str, details: dict, direction: str):
    details["direction"]     = direction
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
    pid  = prop.get("id")
    with _lock:
        if symbol not in pending_signals:
            return
        if pending_signals[symbol].get("proposal_id"):
            return
        pending_signals[symbol]["proposal_id"] = pid
    if not pid:
        return

    # ── Payout quality gate ────────────────────────────────────────────
    try:
        offered_payout = float(prop.get("payout", 0))
        offered_profit = round(offered_payout - STAKE, 4)
    except (TypeError, ValueError):
        offered_profit = 0.0

    if offered_profit < PROFIT_MIN or offered_profit > PROFIT_MAX:
        # Payout outside band — cancel cleanly
        with _lock:
            pending_signals.pop(symbol, None)
        _release_trade_slot(symbol)
        _send_tg(
            f"💸 <b>Payout rejected</b> — <code>{symbol}</code>\n"
            f"Offered profit: <b>${offered_profit:.2f}</b>  "
            f"(target ${PROFIT_MIN:.2f}–${PROFIT_MAX:.2f})\n"
            f"Barrier too {'close' if offered_profit < PROFIT_MIN else 'far'} — skipping."
        )
        _log(f"💸 {symbol} proposal rejected — profit ${offered_profit:.2f} outside "
             f"[${PROFIT_MIN}–${PROFIT_MAX}]")
        return

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
            "symbol":      symbol,
            "direction":   direction,
            "barrier":     buy.get("barrier"),
            "stake":       buy.get("buy_price", STAKE),
            "payout":      buy.get("payout"),
            "entry_time":  datetime.now(timezone.utc),
            "entry_price": last_price.get(symbol, 0),
            "details":     details,
            "settled":     False,
        }
    ws.send(json.dumps({
        "proposal_open_contract": 1,
        "contract_id": int(cid),
        "subscribe": 1,
    }))
    _send_tg(_signal_card(symbol, details.get("total_score", 0), direction, details))
    _log(f"🔥 OPEN  {symbol} {direction}  score={details.get('total_score','?')}/100  "
         f"conf={details.get('ml_confidence', 1.0)*100:.0f}%  cid={cid}")


# ══════════════════════════════════════════════════════════════════════
#  CONTRACT UPDATE  (result checking + sessions)
# ══════════════════════════════════════════════════════════════════════
def on_contract_update(ws, msg: dict, symbol: str):
    global total_pnl, win_count, loss_count, consecutive_losses, paused, pause_until
    global daily_trades, _auto_resume_active, peak_equity, max_drawdown, session_start
    global session_symbol_stats, daily_session_log, _daily_session_log_date
    global market_session_stats, _market_session_date

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

        # ── RESULT FIX: accept is_expired OR is_sold OR explicit status ──
        status = contract.get("status", "")
        is_settled = (
            bool(contract.get("is_expired"))
            or bool(contract.get("is_sold"))
            or status in ("won", "lost")
        )
        if not is_settled:
            return

        info["settled"] = True

        # Extract profit robustly
        profit = float(contract.get("profit") or 0)
        if profit == 0:
            if status == "won":
                profit = float(info.get("payout", STAKE) or STAKE) - float(info.get("stake", STAKE) or STAKE)
            elif status == "lost":
                profit = -float(info.get("stake", STAKE) or STAKE)

        win = profit > 0 or status == "won"
        d   = info.get("details", {})
        mkt_session = _get_session_name(info.get("entry_time"))

        # ── Session P&L counters ──────────────────────────────────────
        if win:
            win_count += 1
            consecutive_losses = 0
        else:
            loss_count += 1
            consecutive_losses += 1

        total_pnl   += profit
        peak_equity  = max(peak_equity, total_pnl)
        max_drawdown = max(max_drawdown, peak_equity - total_pnl)

        # Per-symbol session stats
        stats = session_symbol_stats.setdefault(symbol, {"wins": 0, "losses": 0, "pnl": 0.0})
        stats["pnl"] += profit
        if win: stats["wins"]   += 1
        else:   stats["losses"] += 1

        # Per-market-session stats (reset at midnight UTC)
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today_str != _market_session_date:
            market_session_stats  = {n: {"wins":0,"losses":0,"pnl":0.0,"trades":0} for n in MARKET_SESSIONS}
            _market_session_date  = today_str
        ms = market_session_stats.setdefault(mkt_session, {"wins":0,"losses":0,"pnl":0.0,"trades":0})
        ms["trades"] += 1
        ms["pnl"]    += profit
        if win: ms["wins"]   += 1
        else:   ms["losses"] += 1

        cl        = consecutive_losses
        pnl_snap  = total_pnl
        dt_snap   = daily_trades

        del active_contracts[cid]

        trigger_consec = cl >= MAX_CONSECUTIVE_LOSSES
        trigger_floor  = pnl_snap <= DAILY_LOSS_LIMIT
        trigger_profit = DAILY_PROFIT_TARGET > 0 and pnl_snap >= DAILY_PROFIT_TARGET
        session_reset_trigger = trigger_floor or trigger_profit

        session_snapshot     = None
        needs_resume_thread  = False
        needs_session_wait   = False
        _next_sess_dt        = None

        if session_reset_trigger:
            reason      = "TP" if trigger_profit else "SL"
            _sym_snap   = {s: dict(v) for s, v in session_symbol_stats.items()}
            _dur        = datetime.now(timezone.utc) - session_start
            _next_sess_dt = _next_session_start()
            session_snapshot = {
                "reason": reason, "pnl": pnl_snap, "wins": win_count,
                "losses": loss_count, "trades": dt_snap, "peak": peak_equity,
                "max_dd": max_drawdown, "symbols": _sym_snap, "duration": _dur,
                "market_session": mkt_session,
                "next_session_at": _next_sess_dt,
                "next_session_name": _get_session_name(_next_sess_dt),
            }
            _traded  = {s: v for s, v in _sym_snap.items() if v["wins"] + v["losses"] > 0}
            _best_s  = max(_traded, key=lambda s: _traded[s]["pnl"]) if _traded else None
            _worst_s = min(_traded, key=lambda s: _traded[s]["pnl"]) if _traded else None

            if today_str != _daily_session_log_date:
                daily_session_log.clear()
                _daily_session_log_date = today_str
            daily_session_log.append({
                "reason": reason, "pnl": pnl_snap, "wins": win_count,
                "losses": loss_count, "trades": dt_snap,
                "time": datetime.now(timezone.utc).strftime("%H:%M UTC"),
                "duration_min": max(1, int(_dur.total_seconds() // 60)),
                "best_sym": _best_s,
                "best_pnl": _traded[_best_s]["pnl"] if _best_s else 0.0,
                "worst_sym": _worst_s,
                "worst_pnl": _traded[_worst_s]["pnl"] if _worst_s else 0.0,
                "market_session": mkt_session,
            })
            # Reset session counters
            total_pnl = 0.0; win_count = 0; loss_count = 0; daily_trades = 0
            consecutive_losses = 0; peak_equity = 0.0; max_drawdown = 0.0
            session_symbol_stats = {}
            session_start  = datetime.now(timezone.utc)
            # Pause trading until next session starts
            paused = True
            pause_until = _next_sess_dt
            _auto_resume_active = True
            needs_session_wait = True

        elif trigger_consec and not paused:
            paused = True
            pause_until = datetime.now(timezone.utc) + timedelta(seconds=PAUSE_MINUTES * 60)
            needs_resume_thread = not _auto_resume_active
            if needs_resume_thread:
                _auto_resume_active = True
        resume_at = pause_until

    # DB write — includes market_session
    db_queue.put((
        datetime.now(timezone.utc).isoformat(), symbol, info["direction"],
        info["barrier"], info["stake"], info["payout"], profit, int(win),
        d.get("total_score", 0), d.get("wick_atr_ratio", 0),
        d.get("atr", 0), d.get("atr_ma", 0),
        d.get("ema_fast_sl", 0), d.get("ema_slow_sl", 0), d.get("ema_distance", 0),
        mkt_session,
    ))

    _send_tg(_result_card(symbol, profit, win, d))
    _log(f"{'WIN' if win else 'LOSS'}  {symbol}  ${profit:+.2f}  total=${pnl_snap:+.2f}  "
         f"session={mkt_session}")

    if session_snapshot:
        _send_tg(_session_summary_text(session_snapshot))

    elif trigger_consec:
        _send_tg(
            f"⛔ <b>BOT PAUSED</b> – {MAX_CONSECUTIVE_LOSSES} consecutive losses.\n"
            f"Auto-resuming in {PAUSE_MINUTES} minutes."
        )

    if needs_session_wait and _next_sess_dt:
        _wait_until = _next_sess_dt  # captured in closure
        def _session_wait():
            global paused, _auto_resume_active
            secs = max(0, (_wait_until - datetime.now(timezone.utc)).total_seconds())
            time.sleep(secs + 2)
            with _lock:
                _auto_resume_active = False
                if paused and datetime.now(timezone.utc) >= _wait_until:
                    paused = False
            sess_name  = _get_session_name()
            sess_emoji = SESSION_EMOJIS.get(sess_name, "")
            _send_tg(
                f"🔔 <b>New Session Started!</b>\n"
                f"{sess_emoji} <b>{sess_name}</b> session is now active.\n"
                f"▶ Bot is back to trading. Good luck Ibrahim! 💪"
            )
        threading.Thread(target=_session_wait, daemon=True, name="SessionWait").start()

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
        threading.Thread(target=_auto_resume, daemon=True, name="AutoResume").start()


# ══════════════════════════════════════════════════════════════════════
#  STALE CONTRACT WATCHDOG
# ══════════════════════════════════════════════════════════════════════
def _stale_contract_watchdog():
    """Force-settle contracts that are past expiry + 5 min grace (result-fix for lost trades)."""
    while True:
        time.sleep(30)
        try:
            now = datetime.now(timezone.utc)
            grace = timedelta(minutes=DURATION + 5)
            with _lock:
                stale = [
                    (cid, dict(info))
                    for cid, info in active_contracts.items()
                    if not info.get("settled")
                    and (now - info.get("entry_time", now)) > grace
                ]
            for cid, info in stale:
                sym    = info["symbol"]
                stake  = info.get("stake", STAKE)
                profit = -float(stake)   # assume loss if no settlement received
                logger.warning(f"⏰ Force-settling stale contract {cid} ({sym}) as LOSS")
                _send_tg(
                    f"⚠️ <b>STALE CONTRACT SETTLED</b> – {sym}\n"
                    f"Contract #{cid} had no result after {DURATION+5} min.\n"
                    f"Recording as LOSS (${profit:.2f}).\n"
                    f"<i>Check your Deriv account to verify.</i>"
                )
                # Inject a synthetic settlement message
                synthetic_msg = {
                    "proposal_open_contract": {
                        "contract_id": cid,
                        "is_expired": True,
                        "profit": profit,
                        "status": "lost",
                    }
                }
                on_contract_update(None, synthetic_msg, sym)
        except Exception as e:
            logger.error(f"stale_contract_watchdog: {e}")


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
                    _log(f"⚠  {symbol}: {r['error'].get('message','API error')} – skipping")
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
        full = full.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close"})
        with _lock:
            ohlcv[symbol] = full[["Open","High","Low","Close"]].astype(float)
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
                    ohlcv[symbol] = pd.concat([ohlcv[symbol], new_row]).iloc[-500:]
                    # Reset intra-candle tick history for fresh momentum window
                    tick_history.setdefault(symbol, deque(maxlen=10)).clear()
                # Clear any existing lock immediately on candle close so that the
                # intra-candle path never acts on a stale pre-close lock while
                # the executor worker is still recomputing indicators.
                with _lock:
                    locked_symbols.pop(symbol, None)
                    current_candle[symbol] = dict(c)

                # Submit heavy indicator computation (+re-arming lock) to the
                # serialising executor so at most 2 symbols compute simultaneously,
                # preventing CPU spikes when all 15 candles close at the same time.
                _closed_snap = dict(closed)
                if _indicator_executor is not None:
                    _indicator_executor.submit(_process_closed_candle, symbol, _closed_snap, ws)
                else:
                    _process_closed_candle(symbol, _closed_snap, ws)

            else:
                # ── Intra-candle tick: track velocity + check locked entries ──
                close_px = float(c["close"]) if c.get("close") else None
                if close_px is not None:
                    with _lock:
                        tick_history.setdefault(symbol, deque(maxlen=10)).append(close_px)

                with _lock:
                    lock = locked_symbols.get(symbol)
                if lock:
                    now = datetime.now(timezone.utc)
                    age = (now - lock["lock_time"]).total_seconds()
                    if age > 120:
                        with _lock:
                            locked_symbols.pop(symbol, None)
                        _send_tg(
                            f"⏰ <b>Lock expired</b> — <code>{symbol}</code> {lock['direction']}\n"
                            f"Score held but tick momentum never confirmed in 120 s"
                        )
                        lock = None
                if lock:
                    # Throttle: at most one score_signal call per 2 s per symbol
                    # to avoid burning CPU on rapid-fire 1HZ* tick streams.
                    _now_t = time.time()
                    if _now_t - _last_tick_score_time.get(symbol, 0) < 2.0:
                        with _lock:
                            current_candle[symbol] = dict(c)
                        return
                    _last_tick_score_time[symbol] = _now_t

                    row_c = {
                        "Close": float(c["close"]),
                        "Open":  float(c["open"]),
                        "High":  float(c["high"]),
                        "Low":   float(c["low"]),
                    }
                    score, direction, details = score_signal(symbol, row_c)
                    details["direction"] = direction

                    if score >= SCORE_THRESHOLD and direction == lock["direction"]:
                        # ── Tick momentum gate (soft: skip this tick, retry next) ───
                        if not _has_tick_momentum(symbol, direction):
                            _log(f"⏳ {symbol} {direction} waiting for tick momentum "
                                 f"({len(tick_history.get(symbol,[]))} ticks so far)…")
                            # lock stays — will re-check on next OHLC update
                        else:
                            # Confirmed momentum: attempt entry
                            # entry_passed is always True for ONETOUCH contracts —
                            # a pullback gate hurts ONETOUCH (price moves AWAY from barrier);
                            # real quality control is the M5 + ML gates below.
                            with _lock:
                                locked_symbols.pop(symbol, None)

                            # ── M5 + M15 multi-timeframe gates ────────────────────
                            m5_ind    = indicators_m5.get(symbol, {})
                            m5_ready  = m5_ind.get("ready", False)
                            m5_dir    = m5_ind.get("supertrend_dir", 0)
                            m15_ind   = indicators_m15.get(symbol, {})
                            m15_ready = m15_ind.get("ready", False)
                            m15_dir   = m15_ind.get("supertrend_dir", 0)
                            if (m5_ready and m5_dir != 0
                                    and ((direction == "UP" and m5_dir != 1)
                                         or (direction == "DOWN" and m5_dir != -1))):
                                _send_rejection(symbol, direction, score,
                                                f"M5 Supertrend disagrees "
                                                f"({'↑' if m5_dir==1 else '↓'} on M5 vs {direction} on M1)")
                            elif (m15_ready and m15_dir != 0
                                    and ((direction == "UP" and m15_dir != 1)
                                         or (direction == "DOWN" and m15_dir != -1))):
                                _send_rejection(symbol, direction, score,
                                                f"M15 Supertrend disagrees "
                                                f"({'↑' if m15_dir==1 else '↓'} on M15 vs {direction} on M1)")
                            elif not _ml_should_trade(details, symbol):
                                conf = details.get("ml_confidence", 0)
                                _send_rejection(symbol, direction, score,
                                                f"ML confidence {conf*100:.1f}% < "
                                                f"{ML_CONFIDENCE_MIN*100:.0f}% gate "
                                                f"[{_get_symbol_class(symbol)}]")
                            elif _reserve_trade_slot(symbol, datetime.now(timezone.utc)):
                                request_proposal(ws, symbol, details, direction)
                                _log(f"🎯 {symbol} {direction} TICK-ENTRY  score={score}/100  "
                                     f"momentum={len(tick_history.get(symbol,[]))}ticks  "
                                     f"conf={details.get('ml_confidence',1.0)*100:.0f}%  "
                                     f"class={_get_symbol_class(symbol)}")
                            else:
                                _send_rejection(symbol, direction, score,
                                                "Risk gate closed (paused / cooldown / daily limit)")

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
                    if not det.get("proposal_id"):
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
                            _log(f"⏰ {symbol} buy ack missing for 60s — holding slot")
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
                            f"Buy order sent but no ack within 3 min.\n"
                            f"Slot released — check your Deriv account."
                        )
        except Exception as e:
            logger.error(f"pending_trade_timeout_loop: {e}")


def _ws_thread(symbol: str):
    while True:
        try:
            ws_app = websocket.WebSocketApp(
                "wss://ws.derivws.com/websockets/v3?app_id=1089",
                on_open    = lambda ws: _on_open(ws, symbol),
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
def _ml_conf_label() -> str:
    pct = int(ML_CONFIDENCE_MIN * 100)
    return f"🤖 ML Gate: {pct}%  →  {'90%' if pct == 75 else '75%'}"


def _main_kb() -> InlineKeyboardMarkup:
    with _lock:
        is_paused = paused
    pause_lbl = "▶ Resume" if is_paused else "⏸ Pause"
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📊 Status",          callback_data="status"),
         InlineKeyboardButton("💰 P&L",             callback_data="pnl")],
        [InlineKeyboardButton("📜 History",         callback_data="history"),
         InlineKeyboardButton("📋 Log",             callback_data="signals")],
        [InlineKeyboardButton("📄 Session Report",  callback_data="session_report"),
         InlineKeyboardButton("🏆 All-Time",        callback_data="alltime")],
        [InlineKeyboardButton("📅 Day History",     callback_data="daily_history"),
         InlineKeyboardButton("🏅 Best/Worst",      callback_data="best_worst")],
        [InlineKeyboardButton("📈 Scores",          callback_data="score_sparklines"),
         InlineKeyboardButton("🌏 Sessions",        callback_data="market_sessions")],
        [InlineKeyboardButton("📆 7-Day P&L",       callback_data="seven_day_pnl"),
         InlineKeyboardButton("⚙ Settings",        callback_data="settings")],
        [InlineKeyboardButton(pause_lbl,            callback_data="toggle_pause"),
         InlineKeyboardButton("⏭ Skip Symbol",     callback_data="skip_menu")],
        [InlineKeyboardButton("🔄 Refresh",         callback_data="refresh"),
         InlineKeyboardButton("🧪 Test Trade",      callback_data="test_menu")],
        [InlineKeyboardButton("📦 Backup",           callback_data="backup"),
         InlineKeyboardButton(_ml_conf_label(),     callback_data="ml_conf_toggle")],
    ])


def _skip_kb() -> InlineKeyboardMarkup:
    btns = [[InlineKeyboardButton(s, callback_data=f"skip_{s}")] for s in SYMBOLS]
    btns.append([InlineKeyboardButton("🔙 Back", callback_data="main_menu")])
    return InlineKeyboardMarkup(btns)


def _test_group_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📊 Volatility  (R_*)",     callback_data="tg_vol")],
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
    mkt_sess = _get_session_name(now)
    lines = [
        "📊 <b>BOT STATUS</b>\n━━━━━━━━━━━━━━━━━━━━",
        f"State      : {'⏸ PAUSED' if is_paused else '▶ RUNNING'}",
        f"Market Sess: {SESSION_EMOJIS.get(mkt_sess,'')} {mkt_sess}",
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
    avg_trade = ap / at if at else 0.0
    cur_dd    = max(0.0, peak - pnl)
    mdd_pct   = (mdd / peak * 100) if peak > 0 else 0.0

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
        f"  Avg/Trade: {'+' if avg_trade >= 0 else ''}${avg_trade:.2f}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Session TP: ${DAILY_PROFIT_TARGET:.2f}  |  Session SL: ${DAILY_LOSS_LIMIT}\n"
        f"<i>Hitting either auto-resets the session and keeps trading.</i>"
    )


def _live_session_report_text():
    with _lock:
        snap = {
            "reason": "MANUAL",
            "pnl": total_pnl, "wins": win_count, "losses": loss_count,
            "trades": daily_trades, "peak": peak_equity, "max_dd": max_drawdown,
            "symbols": {s: dict(v) for s, v in session_symbol_stats.items()},
            "duration": datetime.now(timezone.utc) - session_start,
            "market_session": _get_session_name(),
        }
    text = _session_summary_text(snap)
    return text.replace(
        "🔄 <b>New session started — bot keeps trading.</b>",
        "📄 <i>Live snapshot — session continues, nothing was reset.</i>",
    )


def _alltime_text():
    db_sum = get_db_summary()
    at, ap, aw, al = db_sum if db_sum[0] else (0, 0, 0, 0)
    ap    = ap or 0.0
    at_wr = aw / at * 100 if at else 0.0
    avg   = ap / at if at else 0.0

    lines = [
        "🏆 <b>ALL-TIME SCOREBOARD</b>",
        "━━━━━━━━━━━━━━━━━━━━",
        f"💵 <b>Total P&L</b>  : {'+' if ap >= 0 else ''}${ap:.2f}",
        f"📊 <b>Trades</b>     : {at}  ({aw}W / {al}L)",
        f"🎯 <b>Win Rate</b>   : {at_wr:.1f}%",
        f"📈 <b>Avg/Trade</b>  : {'+' if avg >= 0 else ''}${avg:.2f}",
        "",
        "<b>By Symbol (all-time)</b>",
    ]
    sym_rows = get_alltime_symbol_stats()
    if sym_rows:
        for sym, cnt, wins, pnl in sym_rows:
            wins = wins or 0; pnl = pnl or 0.0
            wr   = wins / cnt * 100 if cnt else 0
            sign = "+" if pnl >= 0 else ""
            lines.append(f"  <code>{sym:<10}</code> {cnt} trades  {wins}W ({wr:.0f}%)  {sign}${pnl:.2f}")
    else:
        lines.append("  — no trades recorded yet —")

    lines += ["", "<b>Last 7 Days (summary)</b>"]
    day_rows = get_alltime_daily_stats()
    if day_rows:
        for day, cnt, wins, pnl in day_rows:
            wins = wins or 0; pnl = pnl or 0.0
            sign = "+" if pnl >= 0 else ""
            lines.append(f"  {day}  {cnt} trades  {wins}W  {sign}${pnl:.2f}")
    else:
        lines.append("  — no trades recorded yet —")

    lines += [
        "━━━━━━━━━━━━━━━━━━━━",
        "<i>Persists across restarts (stored in trades.db).</i>",
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
    global daily_session_log, _daily_session_log_date
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with _lock:
        if today != _daily_session_log_date:
            daily_session_log    = []
            _daily_session_log_date = today
        log = list(daily_session_log)

    if not log:
        return (
            "📅 <b>Daily Session History</b>\n━━━━━━━━━━━━━━━━━━━━\n"
            f"Date: {today}\n\n"
            "— No TP or SL hits yet today. —\n\n"
            "<i>Each time the bot hits its daily TP or SL it resets and this record is updated.</i>"
        )

    tp_hits = sum(1 for e in log if e["reason"] == "TP")
    sl_hits = sum(1 for e in log if e["reason"] == "SL")
    total_day_pnl    = sum(e["pnl"]    for e in log)
    total_day_trades = sum(e["trades"] for e in log)
    total_day_wins   = sum(e["wins"]   for e in log)
    total_day_losses = sum(e["losses"] for e in log)
    total_possible   = total_day_wins + total_day_losses
    day_wr = total_day_wins / total_possible * 100 if total_possible else 0

    lines = [
        "📅 <b>Daily Session History</b>",
        "━━━━━━━━━━━━━━━━━━━━",
        f"Date   : {today}",
        f"🎯 TP hits: <b>{tp_hits}</b>   🛑 SL hits: <b>{sl_hits}</b>",
        f"Day P&L   : <b>{'+' if total_day_pnl >= 0 else ''}${total_day_pnl:.2f}</b>",
        f"Day trades: {total_day_trades}  ({total_day_wins}W / {total_day_losses}L  {day_wr:.0f}%)",
        "━━━━━━━━━━━━━━━━━━━━",
    ]

    for i, e in enumerate(log, 1):
        icon  = "🎯" if e["reason"] == "TP" else "🛑"
        sign  = "+" if e["pnl"] >= 0 else ""
        h, mr = divmod(e["duration_min"], 60)
        dur_str  = f"{h}h {mr}m" if h else f"{mr}m"
        ms_name  = e.get("market_session", "—")
        entry = (
            f"{icon} <b>Session {i}</b>  [{e['time']}]  {dur_str}  "
            f"{SESSION_EMOJIS.get(ms_name,'')} {ms_name}\n"
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

    lines += ["━━━━━━━━━━━━━━━━━━━━", f"<i>Resets at midnight UTC. Today: {today}</i>"]
    return "\n".join(lines)


def _market_sessions_text() -> str:
    """Current-day per-market-session breakdown."""
    _roll_market_session_stats_if_needed()
    now = datetime.now(timezone.utc)
    current = _get_session_name(now)
    with _lock:
        ms_snap = {k: dict(v) for k, v in market_session_stats.items()}

    # All-time session stats from DB
    at_rows = {row[0]: row for row in get_session_alltime_stats()}

    lines = [
        "🌏 <b>Market Session Performance</b>",
        "━━━━━━━━━━━━━━━━━━━━",
        f"Now: {SESSION_EMOJIS.get(current,'')} <b>{current}</b>  "
        f"({now.strftime('%H:%M UTC')})",
        "",
        "<b>Today (in-memory)</b>",
    ]

    for name in ["Midnight", "Asian", "London", "New York"]:
        s    = ms_snap.get(name, {"wins": 0, "losses": 0, "pnl": 0.0, "trades": 0})
        tot  = s["wins"] + s["losses"]
        wr   = s["wins"] / tot * 100 if tot else 0
        sign = "+" if s["pnl"] >= 0 else ""
        tag  = " ◄ LIVE" if name == current else ""
        lines.append(
            f"  {SESSION_EMOJIS.get(name,'')} <b>{name:<10}</b>"
            f"{tag}\n"
            f"    Trades: {tot}  {s['wins']}W/{s['losses']}L ({wr:.0f}%WR)  "
            f"P&L: <b>{sign}${s['pnl']:.2f}</b>"
        )

    lines += ["", "<b>All-Time (DB)</b>"]
    if at_rows:
        for name in ["Midnight", "Asian", "London", "New York"]:
            row = at_rows.get(name)
            if row:
                _, cnt, wins, pnl = row
                wins = wins or 0; pnl = pnl or 0.0
                wr   = wins / cnt * 100 if cnt else 0
                sign = "+" if pnl >= 0 else ""
                lines.append(
                    f"  {SESSION_EMOJIS.get(name,'')} {name:<10}  "
                    f"{cnt} trades  {wins}W ({wr:.0f}%)  {sign}${pnl:.2f}"
                )
    else:
        lines.append("  — no data in DB yet —")

    lines += [
        "━━━━━━━━━━━━━━━━━━━━",
        "<i>Today stats reset at midnight UTC. DB stats are lifetime totals.</i>",
    ]
    return "\n".join(lines)


def _7day_pnl_text() -> str:
    """Full 7-day P&L breakdown — per day, per market session."""
    rows = get_7day_full_breakdown()
    day_rows = get_alltime_daily_stats(7)

    lines = [
        "📆 <b>Last 7 Days — Full P&L Breakdown</b>",
        "━━━━━━━━━━━━━━━━━━━━",
    ]

    if not day_rows:
        lines.append("  — no trades in the last 7 days —")
        return "\n".join(lines)

    # Summarise per day
    day_totals: dict = {}
    for day, cnt, wins, pnl in day_rows:
        pnl  = pnl or 0.0
        wins = wins or 0
        day_totals[day] = {"cnt": cnt, "wins": wins, "pnl": pnl}

    # Session breakdown per day
    sess_by_day: dict = {}
    for row in rows:
        day, sess, cnt, wins, pnl = row
        pnl  = pnl or 0.0
        wins = wins or 0
        sess_by_day.setdefault(day, []).append((sess or "?", cnt, wins, pnl))

    week_pnl   = sum(v["pnl"] for v in day_totals.values())
    week_trades = sum(v["cnt"] for v in day_totals.values())
    week_wins   = sum(v["wins"] for v in day_totals.values())
    week_wr     = week_wins / week_trades * 100 if week_trades else 0
    week_sign   = "+" if week_pnl >= 0 else ""
    lines += [
        f"Week Total : <b>{week_sign}${week_pnl:.2f}</b>",
        f"Trades     : {week_trades}  ({week_wins}W  {week_wr:.0f}%WR)",
        "━━━━━━━━━━━━━━━━━━━━",
    ]

    for day in sorted(day_totals.keys(), reverse=True):
        dt = day_totals[day]
        wr   = dt["wins"] / dt["cnt"] * 100 if dt["cnt"] else 0
        sign = "+" if dt["pnl"] >= 0 else ""
        lines.append(
            f"\n📅 <b>{day}</b>  —  "
            f"<b>{sign}${dt['pnl']:.2f}</b>  "
            f"({dt['cnt']} trades  {dt['wins']}W  {wr:.0f}%WR)"
        )
        # Per-session breakdown for this day
        if day in sess_by_day:
            for sess, cnt, wins, pnl in sorted(sess_by_day[day], key=lambda r: r[3], reverse=True):
                s_wr   = wins / cnt * 100 if cnt else 0
                s_sign = "+" if pnl >= 0 else ""
                em     = SESSION_EMOJIS.get(sess, "")
                lines.append(
                    f"    {em} {sess:<10}  {cnt}T  {wins}W ({s_wr:.0f}%)  {s_sign}${pnl:.2f}"
                )

    lines += [
        "━━━━━━━━━━━━━━━━━━━━",
        "<i>All figures from trades.db — persists across restarts.</i>",
    ]
    return "\n".join(lines)


# ── Session best / worst ─────────────────────────────────────────────
def _best_worst_session():
    with _lock:
        stats = {s: dict(v) for s, v in session_symbol_stats.items()
                 if v["wins"] + v["losses"] > 0}
    if len(stats) < 1:
        return None, None, None, None
    best_sym  = max(stats, key=lambda s: stats[s]["pnl"])
    worst_sym = min(stats, key=lambda s: stats[s]["pnl"])
    return best_sym, stats[best_sym], worst_sym, stats[worst_sym]


def _best_worst_line() -> str:
    b_sym, b_st, w_sym, w_st = _best_worst_session()
    if not b_sym:
        return "🏅 Best/Worst: no trades this session"
    b_wr = b_st["wins"] / (b_st["wins"] + b_st["losses"]) * 100 if (b_st["wins"] + b_st["losses"]) else 0
    w_wr = w_st["wins"] / (w_st["wins"] + w_st["losses"]) * 100 if (w_st["wins"] + w_st["losses"]) else 0
    return (
        f"🏅 Best  : {b_sym}  {'+' if b_st['pnl'] >= 0 else ''}${b_st['pnl']:.2f}  {b_wr:.0f}% WR\n"
        f"💔 Worst : {w_sym}  {'+' if w_st['pnl'] >= 0 else ''}${w_st['pnl']:.2f}  {w_wr:.0f}% WR"
    )


def _best_worst_text() -> str:
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
    lines += ["━━━━━━━━━━━━━━━━━━━━", "<i>Resets each TP/SL. Live session data.</i>"]
    return "\n".join(lines)


# ── Sparklines ────────────────────────────────────────────────────────
_SPARK_CHARS = "▁▂▃▄▅▆▇█"


def _sparkline(values) -> str:
    vals = list(values)
    if not vals:
        return "—"
    lo, hi = min(vals), max(vals)
    rng = hi - lo or 1
    return "".join(_SPARK_CHARS[min(7, int((v - lo) / rng * 7.99))] for v in vals)


def _score_sparklines_text() -> str:
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
            f"<code>{spark}</code>  now <b>{current}</b>  avg {avg:.0f}"
        )
    lines += ["━━━━━━━━━━━━━━━━━━━━", f"<i>Threshold: {SCORE_THRESHOLD}/100 to trigger a trade.</i>"]
    return "\n".join(lines)


def _settings_text():
    return (
        f"⚙ <b>Settings</b>\n━━━━━━━━━━━━━━━━━━━━\n"
        f"Stake           : ${STAKE}  (fixed risk per trade)\n"
        f"Barrier         : ATR × {ATR_BARRIER_MULT}  (auto-scales per symbol)\n"
        f"Duration        : {DURATION} min\n"
        f"Contract        : {CONTRACT_TYPE}\n"
        f"Min Score       : {SCORE_THRESHOLD}/100  (93+ gate)\n"
        f"ML Confidence   : ≥{ML_CONFIDENCE_MIN*100:.0f}%  (hard filter)\n"
        f"Cooldown        : {COOLDOWN_MINUTES} min\n"
        f"Max Consec Loss : {MAX_CONSECUTIVE_LOSSES}\n"
        f"Pause Duration  : {PAUSE_MINUTES} min\n"
        f"Session TP      : ${DAILY_PROFIT_TARGET:.2f}\n"
        f"Session SL      : ${DAILY_LOSS_LIMIT}\n"
        f"Trend Engine    : Supertrend (period={SUPERTREND_PERIOD}, mult={SUPERTREND_ATR_MULT})\n"
        f"{_ml_progress_text()}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<i>TP/SL auto-resets session and keeps trading.</i>"
    )


# ══════════════════════════════════════════════════════════════════════
#  TEST TRADE
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
            f"Please wait for it to finish."
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
        f"Expiry   : {DURATION} min\n"
        f"Type     : {CONTRACT_TYPE}\n"
        f"<i>Step 1/4 – Connecting to Deriv…</i>"
    )

    try:
        ws = websocket.WebSocket()
        ws.connect("wss://ws.derivws.com/websockets/v3?app_id=1089", timeout=15)
    except Exception as e:
        tg(f"🧪 <b>Test Trade FAILED</b>\n❌ Connection error: <code>{e}</code>")
        _test_trade_sem.release()
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
        tick_msg  = recv_typed("tick", timeout=8)
        spot_now  = float((tick_msg or {}).get("tick", {}).get("quote", 0)) if tick_msg else 0

        with _lock:
            ema_slow  = (indicators.get(symbol) or {}).get("ema_slow") or spot_now
            st_dir_v  = (indicators.get(symbol) or {}).get("supertrend_dir", 0)
        if st_dir_v != 0:
            direction = "UP" if st_dir_v == 1 else "DOWN"
        else:
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
            "proposal": 1, "amount": STAKE, "basis": "stake",
            "contract_type": CONTRACT_TYPE, "currency": "USD",
            "duration": DURATION, "duration_unit": "m",
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
        ask     = prop.get("ask_price", STAKE)
        payout  = prop.get("payout", 0)
        try:
            offered_profit = round(float(payout) - STAKE, 4)
        except (TypeError, ValueError):
            offered_profit = 0.0

        # ── Payout gate (same rule as live trading) ───────────────────
        if offered_profit < PROFIT_MIN or offered_profit > PROFIT_MAX:
            tg(
                f"🧪 <b>Test Trade</b>  –  💸 Payout rejected\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"Offered payout : ${payout}  (profit ${offered_profit:.2f})\n"
                f"Target band    : ${PROFIT_MIN:.2f} – ${PROFIT_MAX:.2f}\n"
                f"Barrier too {'close (easy touch → low payout)' if offered_profit < PROFIT_MIN else 'far (hard touch → high payout)'}.\n"
                f"<i>Tune ATR_BARRIER_MULT ({ATR_BARRIER_MULT}) to move into band.</i>"
            )
            return

        tg(
            f"🧪 <b>Test Trade</b>  –  ✅ Proposal OK\n"
            f"Proposal ID : <code>{pid}</code>\n"
            f"Ask Price   : ${ask}\n"
            f"Payout      : ${payout}  (profit <b>${offered_profit:.2f}</b>)\n"
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
            "proposal_open_contract": 1, "contract_id": int(cid), "subscribe": 1,
        }))
        wait_deadline  = time.time() + DURATION * 60 + 60
        contract_data: dict = {}
        while time.time() < wait_deadline:
            try:
                ws.settimeout(5)
                raw  = ws.recv()
                msg  = json.loads(raw)
                mtype = msg.get("msg_type")
                if mtype == "proposal_open_contract":
                    poc = msg.get("proposal_open_contract", {})
                    contract_data = poc
                    status = poc.get("status", "")
                    if poc.get("is_expired") or poc.get("is_sold") or status in ("won", "lost"):
                        break
                elif mtype == "error":
                    err = msg.get("error", {}).get("message", "?")
                    tg(f"🧪 API error during monitoring: <code>{err}</code>")
                    break
            except websocket.WebSocketTimeoutException:
                continue
            except Exception as e:
                logger.error(f"🧪 {tag} recv error: {e}")
                break

        profit = float(contract_data.get("profit", 0))
        win    = profit > 0 or contract_data.get("status") == "won"
        emoji  = "🏆" if win else "💀"
        label  = "WIN" if win else "LOSS"
        pstr   = f"+${profit:.2f}" if win else f"${profit:.2f}"
        tg(
            f"{emoji} <b>Test Trade RESULT: {label}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Symbol      : <code>{symbol}</code>\n"
            f"Contract ID : <code>{cid}</code>\n"
            f"P&L         : <b>{pstr}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"✅ Full pipeline confirmed: proposal → buy → settlement → result\n"
            f"<i>Test trade does NOT affect session stats.</i>"
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
        "🤖 <b>Deriv Sniper Bot v3</b>\nSupertrend · Sessions · 93% Confidence Gate",
        reply_markup=_main_kb(), parse_mode="HTML",
    )

async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(_status_text(), reply_markup=_main_kb(), parse_mode="HTML")

_NOT_READY_MSG = "⏳ <b>Bot is still warming up</b> (loading candle history…)\nTry again in 30–60 seconds."

async def cmd_pnl(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _bot_ready:
        await update.message.reply_text(_NOT_READY_MSG, parse_mode="HTML"); return
    await update.message.reply_text(_pnl_text(), reply_markup=_main_kb(), parse_mode="HTML")

async def cmd_pause(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _bot_ready:
        await update.message.reply_text(_NOT_READY_MSG, parse_mode="HTML"); return
    global paused, pause_until
    with _lock:
        paused = True
        pause_until = datetime.now(timezone.utc) + timedelta(hours=24)
    await update.message.reply_text("⏸ <b>Bot paused.</b>  /resume to restart.", parse_mode="HTML")

async def cmd_resume(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _bot_ready:
        await update.message.reply_text(_NOT_READY_MSG, parse_mode="HTML"); return
    global paused, pause_until, consecutive_losses
    with _lock:
        paused = False
        consecutive_losses = 0
        pause_until = datetime.min.replace(tzinfo=timezone.utc)
    await update.message.reply_text("▶ <b>Bot resumed.</b>", parse_mode="HTML")

async def cmd_history(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _bot_ready:
        await update.message.reply_text(_NOT_READY_MSG, parse_mode="HTML"); return
    await update.message.reply_text(_history_text(), reply_markup=_main_kb(), parse_mode="HTML")

async def cmd_session(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _bot_ready:
        await update.message.reply_text(_NOT_READY_MSG, parse_mode="HTML"); return
    await update.message.reply_text(_live_session_report_text(), reply_markup=_main_kb(), parse_mode="HTML")

async def cmd_alltime(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _bot_ready:
        await update.message.reply_text(_NOT_READY_MSG, parse_mode="HTML"); return
    await update.message.reply_text(_alltime_text(), reply_markup=_main_kb(), parse_mode="HTML")

async def cmd_export(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Send full trades.db CSV on demand."""
    if not _bot_ready:
        await update.message.reply_text(_NOT_READY_MSG, parse_mode="HTML"); return
    await update.message.reply_text("⏳ Generating CSV export…", parse_mode="HTML")
    try:
        rows = _db_fetch("SELECT * FROM trades ORDER BY id")
        if USE_PG:
            cols = ["id","timestamp","symbol","direction","barrier","stake","payout",
                    "profit","win","score","wick_atr_ratio","atr","atr_ma",
                    "ema_fast_slope","ema_slow_slope","ema_distance","market_session"]
        else:
            conn = sqlite3.connect("trades.db")
            desc = conn.execute("PRAGMA table_info(trades)").fetchall()
            conn.close()
            cols = [d[1] for d in desc]
        buf  = io.StringIO()
        w    = _csv.writer(buf)
        w.writerow(cols)
        w.writerows(rows)
        csv_bytes = buf.getvalue().encode("utf-8")
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        await update.message.reply_document(
            document=(f"trades_{ts}.csv", csv_bytes),
            caption=f"📊 <b>Trades Export</b> — {len(rows)} trades",
            parse_mode="HTML",
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Export failed: <code>{e}</code>", parse_mode="HTML")


async def cmd_backup(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Send the raw trades.db file (SQLite) or a PG dump notice."""
    if not _bot_ready:
        await update.message.reply_text(_NOT_READY_MSG, parse_mode="HTML"); return
    if USE_PG:
        row = get_db_summary()
        total = row[0] if row else 0
        await update.message.reply_text(
            f"☁️ <b>Trades stored in PostgreSQL</b>\n"
            f"Total trades: <b>{total}</b>\n"
            f"Use /export to download a CSV of all trades.",
            parse_mode="HTML",
        )
    elif os.path.exists("trades.db"):
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        with open("trades.db", "rb") as f:
            await update.message.reply_document(
                document=f,
                filename=f"trades_{ts}.db",
                caption="🗄️ <b>SQLite backup</b> — raw trades database",
                parse_mode="HTML",
            )
    else:
        await update.message.reply_text("⚠️ No trades database found yet.", parse_mode="HTML")


async def cmd_upload_csv(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """
    Handle a file upload from the user:
    • .csv  — trades backup  → restore rows to DB, then retrain ML
    • .csv  — ML-only CSV   → train ML model (legacy path, no timestamp col)
    • .pkl  — ML model      → restore model directly
    """
    doc = update.message.document
    fname = (doc.file_name or "").lower() if doc else ""

    # ── .pkl upload: restore ML model (admin chat only — pickle is unsafe from unknown senders) ──
    if fname.endswith(".pkl"):
        # Only accept from the configured chat to prevent arbitrary code execution via pickle
        if str(update.effective_chat.id) != str(TELEGRAM_CHAT_ID):
            await update.message.reply_text("⛔ Model restore is only available from the admin chat.")
            return
        await update.message.reply_text("⏳ Loading ML model from file…", parse_mode="HTML")
        try:
            file    = await doc.get_file()
            raw     = await file.download_as_bytearray()
            payload = pickle.loads(bytes(raw))
            global ml_model, ml_trained_on, ml_models_per_class, ml_trained_per_class
            if isinstance(payload, dict):
                with ml_lock:
                    ml_model             = payload.get("model")
                    ml_trained_on        = payload.get("trained_on", 0)
                    ml_models_per_class  = payload.get("per_class_models",  ml_models_per_class)
                    ml_trained_per_class = payload.get("per_class_trained", ml_trained_per_class)
            else:
                with ml_lock:
                    ml_model      = payload
                    ml_trained_on = 0
            _ml_save()
            with ml_lock:
                trained = ml_trained_on
            await update.message.reply_text(
                f"🤖 <b>ML Model Restored</b> ✅\n"
                f"Trained on <b>{trained}</b> trades.\n"
                f"Confidence gate: ≥ <b>{ML_CONFIDENCE_MIN*100:.0f}%</b> — ACTIVE now",
                parse_mode="HTML",
            )
            logger.info(f"ML model restored from pkl upload (trained_on={trained})")
        except Exception as e:
            logger.exception(f"pkl upload error: {e}")
            await update.message.reply_text(
                f"❌ <b>Model restore failed</b>\n<code>{e}</code>", parse_mode="HTML"
            )
        return

    # Guard: we catch Document.ALL so non-CSV/pkl files reach here — ignore silently.
    if not fname.endswith(".csv"):
        return

    await update.message.reply_text("⏳ Downloading and processing your CSV…", parse_mode="HTML")
    try:
        file   = await doc.get_file()
        raw    = await file.download_as_bytearray()
        text   = raw.decode("utf-8")
        reader = _csv.DictReader(io.StringIO(text))
        rows   = list(reader)
        if not rows:
            await update.message.reply_text("❌ CSV is empty.", parse_mode="HTML")
            return

        col_keys = set(rows[0].keys())

        # ── Trades backup restore path ──────────────────────────────────
        # Detected when CSV has the core trade columns (exported by 📦 Backup).
        _TRADE_RESTORE_REQUIRED = {"timestamp", "symbol", "direction", "profit", "win"}
        if _TRADE_RESTORE_REQUIRED.issubset(col_keys):
            await update.message.reply_text(
                f"📂 Detected <b>trades backup</b> ({len(rows)} rows).\n"
                f"⏳ Restoring to database…",
                parse_mode="HTML",
            )
            inserted = skipped = errors = 0
            try:
                if USE_PG:
                    import psycopg2 as _pg2
                    _rc = _pg2.connect(DATABASE_URL)
                    _rc.autocommit = False
                else:
                    _rc = sqlite3.connect("trades.db")

                _db_cols = [
                    "timestamp", "symbol", "direction", "barrier",
                    "stake", "payout", "profit", "win", "score",
                    "wick_atr_ratio", "atr", "atr_ma",
                    "ema_fast_slope", "ema_slow_slope", "ema_distance", "market_session",
                ]
                for row in rows:
                    try:
                        ts_val     = row.get("timestamp", "")
                        sym_val    = row.get("symbol", "")
                        dir_val    = row.get("direction", "")
                        stake_val  = float(row.get("stake") or 0)
                        # Deduplication: skip if exact (timestamp, symbol, direction, stake) exists
                        if USE_PG:
                            dup = _rc.cursor()
                            dup.execute(
                                "SELECT 1 FROM trades WHERE timestamp=%s AND symbol=%s "
                                "AND direction=%s AND stake=%s LIMIT 1",
                                (ts_val, sym_val, dir_val, stake_val),
                            )
                        else:
                            dup = _rc.execute(
                                "SELECT 1 FROM trades WHERE timestamp=? AND symbol=? "
                                "AND direction=? AND stake=? LIMIT 1",
                                (ts_val, sym_val, dir_val, stake_val),
                            )
                        if dup.fetchone():
                            skipped += 1
                            continue

                        def _cast(c, v):
                            if c in ("timestamp","symbol","direction","barrier","market_session"):
                                return v or None
                            if c == "win":
                                return int(float(v)) if v not in (None, "") else 0
                            return float(v) if v not in (None, "") else None
                        vals = tuple(_cast(c, row.get(c)) for c in _db_cols)
                        ph = ",".join(["%s" if USE_PG else "?"] * len(_db_cols))
                        _ins = (
                            f"INSERT INTO trades ({','.join(_db_cols)}) VALUES ({ph})"
                        )
                        if USE_PG:
                            cur = _rc.cursor(); cur.execute(_ins, vals)
                        else:
                            _rc.execute(_ins, vals)
                        inserted += 1
                    except Exception as row_err:
                        errors += 1
                        logger.debug(f"restore row skip: {row_err}")
                if USE_PG:
                    _rc.commit()
                else:
                    _rc.commit()
                _rc.close()
            except Exception as db_err:
                logger.error(f"trades restore DB error: {db_err}")
                await update.message.reply_text(
                    f"⚠️ DB error during restore: <code>{db_err}</code>", parse_mode="HTML"
                )
                return

            await update.message.reply_text(
                f"✅ <b>Trades Restored</b>\n"
                f"Inserted : <b>{inserted}</b> trades\n"
                f"Skipped  : {skipped} (already in DB)\n"
                f"Errors   : {errors}\n\n"
                f"<i>ML model will retrain automatically once enough trades are recorded.</i>",
                parse_mode="HTML",
            )
            logger.info(f"Trades restore: inserted={inserted} skipped={skipped} errors={errors}")

            # Trigger ML retrain if we now have enough data (guard against concurrent runs)
            total_now = _db_fetch("SELECT COUNT(*) FROM trades")
            if total_now and total_now[0][0] >= ML_MIN_TRADES:
                if not ml_training_active:
                    threading.Thread(target=_ml_train, daemon=True, name="MLRetrain").start()
            return

        # ── ML-only CSV path (legacy: no timestamp column) ──────────────
        req_cols = ML_FEATURE_COLS + ["win"]
        found = [c for c in req_cols if c in col_keys]
        if "win" not in found or len(found) < len(ML_FEATURE_COLS):
            await update.message.reply_text(
                f"❌ CSV must contain columns: {', '.join(req_cols)}\n"
                f"Found: {list(col_keys)}",
                parse_mode="HTML",
            )
            return

        X, y = [], []
        for row in rows:
            try:
                X.append([float(row.get(c, 0) or 0) for c in ML_FEATURE_COLS])
                y.append(int(float(row.get("win", 0) or 0)))
            except (ValueError, KeyError):
                continue

        if len(X) < 10 or len(set(y)) < 2:
            await update.message.reply_text(
                f"❌ Need ≥10 rows with both win=0 and win=1. Got {len(X)} rows.",
                parse_mode="HTML",
            )
            return

        clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        clf.fit(X, y)
        wins = sum(y)
        # global already declared at top of function (pkl path); no re-declaration needed
        with ml_lock:
            ml_model      = clf
            ml_trained_on = len(X)
        _ml_save()
        await update.message.reply_text(
            f"🤖 <b>ML Model Trained from Your CSV</b> ✅\n"
            f"Rows: <b>{len(X)}</b>  |  Wins: <b>{wins}</b>  ({100*wins/len(X):.1f}%)\n"
            f"Confidence gate: ≥ <b>{ML_CONFIDENCE_MIN*100:.0f}%</b> — ACTIVE now",
            parse_mode="HTML",
        )
        logger.info(f"ML trained from user CSV upload: {len(X)} rows, {wins} wins")
    except Exception as e:
        logger.exception(f"CSV upload error: {e}")
        await update.message.reply_text(f"❌ Processing failed: <code>{e}</code>", parse_mode="HTML")


# ══════════════════════════════════════════════════════════════════════
#  BACKUP HELPER  (runs in a background thread — safe to call from btn_handler)
# ══════════════════════════════════════════════════════════════════════
def _do_backup(what: str) -> None:
    """Send trades CSV and/or ML model to Telegram.  Runs in a daemon thread."""
    try:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")

        if what in ("backup_csv", "backup_all"):
            rows = _db_fetch("SELECT * FROM trades ORDER BY id")
            if USE_PG:
                cols = ["id","timestamp","symbol","direction","barrier","stake","payout",
                        "profit","win","score","wick_atr_ratio","atr","atr_ma",
                        "ema_fast_slope","ema_slow_slope","ema_distance","market_session"]
            else:
                conn = sqlite3.connect("trades.db")
                desc = conn.execute("PRAGMA table_info(trades)").fetchall()
                conn.close()
                cols = [d[1] for d in desc]
            buf = io.StringIO()
            w   = _csv.writer(buf)
            w.writerow(cols)
            w.writerows(rows)
            csv_bytes = buf.getvalue().encode("utf-8")
            _send_tg_document(
                csv_bytes,
                f"trades_backup_{ts}.csv",
                f"📊 <b>Trades Backup</b> — {len(rows)} trade(s)\n"
                f"<i>After a redeploy: send this file back to this chat and the bot will "
                f"restore all trades automatically.</i>",
            )

        if what in ("backup_ml", "backup_all"):
            if os.path.exists(MODEL_PATH):
                with open(MODEL_PATH, "rb") as f:
                    pkl_bytes = f.read()
                with ml_lock:
                    trained = ml_trained_on
                _send_tg_document(
                    pkl_bytes,
                    f"ml_model_{ts}.pkl",
                    f"🤖 <b>ML Model Backup</b> — trained on {trained} trades\n"
                    f"<i>Send this .pkl file back to this chat to restore the ML model "
                    f"after a redeploy.</i>",
                )
            else:
                _send_tg("⚠️ No ML model file yet — bot needs more trades to train first.")

    except Exception as exc:
        logger.error(f"_do_backup error: {exc}")
        _send_tg(f"❌ <b>Backup failed</b>\n<code>{exc}</code>")


# ══════════════════════════════════════════════════════════════════════
#  TELEGRAM BUTTON HANDLER
# ══════════════════════════════════════════════════════════════════════
async def btn_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    global paused, pause_until, consecutive_losses
    q = update.callback_query
    d = q.data
    logger.info(f"Button pressed: {d!r}")
    try:
        await q.answer()
    except Exception as e:
        logger.debug(f"q.answer() failed: {e}")
        return

    try:
        if d in ("main_menu", "refresh"):
            await q.edit_message_text(
                "🤖 <b>Deriv Sniper Bot v3</b>  –  Select an option:",
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
        elif d == "market_sessions":
            await q.edit_message_text(_market_sessions_text(), reply_markup=_main_kb(), parse_mode="HTML")
        elif d == "seven_day_pnl":
            await q.edit_message_text(_7day_pnl_text(),  reply_markup=_main_kb(), parse_mode="HTML")
        elif d == "ml_conf_toggle":
            global ML_CONFIDENCE_MIN
            ML_CONFIDENCE_MIN = 0.90 if ML_CONFIDENCE_MIN < 0.85 else 0.75
            pct = int(ML_CONFIDENCE_MIN * 100)
            await q.edit_message_text(
                f"🤖 <b>ML Gate → {pct}%</b>\n"
                f"Trades now require ≥{pct}% ML confidence.\n"
                f"{'More selective — fewer but higher-quality trades.' if pct == 90 else 'More permissive — more trades, model still learning.'}",
                reply_markup=_main_kb(), parse_mode="HTML",
            )
            _log(f"🤖 ML_CONFIDENCE_MIN set to {pct}% via Telegram")
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
                f"Stake: <b>${STAKE:.2f}</b>  ·  Expiry: <b>{DURATION} min</b>\n"
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
                f"Symbol: <code>{sym}</code>  ·  Stake: ${STAKE:.2f}  ·  Expiry: {DURATION} min\n\n"
                f"⏳ Connecting to Deriv and placing order…\n"
                f"You'll get step-by-step updates.",
                reply_markup=InlineKeyboardMarkup([[
                    InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")
                ]]),
                parse_mode="HTML",
            )
            threading.Thread(
                target=_run_test_trade, args=(sym,),
                daemon=True, name=f"TestTrade-{sym}"
            ).start()
        elif d == "backup":
            await q.edit_message_text(
                "📦 <b>Backup & Restore</b>\n━━━━━━━━━━━━━━━━━━━━\n"
                "Export your trade history and ML model as files.\n\n"
                "<b>To restore after a redeploy:</b>\n"
                "• Send a <code>trades_backup_*.csv</code> file here → trades are "
                "re-inserted into the database automatically\n"
                "• Send a <code>ml_model_*.pkl</code> file here → ML model is restored\n\n"
                "<i>Files are sent directly to this Telegram chat — no cloud storage needed.</i>",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("📊 Export Trades CSV",  callback_data="backup_csv")],
                    [InlineKeyboardButton("🤖 Export ML Model",    callback_data="backup_ml")],
                    [InlineKeyboardButton("📦 Export Both",        callback_data="backup_all")],
                    [InlineKeyboardButton("🔙 Back",               callback_data="main_menu")],
                ]),
                parse_mode="HTML",
            )
        elif d in ("backup_csv", "backup_ml", "backup_all"):
            await q.edit_message_text(
                "⏳ Preparing files — you'll receive them in a moment…",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔙 Back", callback_data="backup")]
                ]),
                parse_mode="HTML",
            )
            threading.Thread(
                target=_do_backup, args=(d,), daemon=True, name="BotBackup"
            ).start()
        else:
            await q.edit_message_text(
                "🤖 <b>Deriv Sniper Bot v3</b>  –  Select an option:",
                reply_markup=_main_kb(), parse_mode="HTML",
            )
    except Exception as e:
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
    now    = datetime.now(timezone.utc)
    mkt_s  = _get_session_name(now)

    # Best session today
    _roll_market_session_stats_if_needed()
    with _lock:
        ms_snap = {k: dict(v) for k, v in market_session_stats.items()
                   if v["trades"] > 0}
    if ms_snap:
        best_ms = max(ms_snap, key=lambda k: ms_snap[k]["pnl"])
        bms = ms_snap[best_ms]
        bms_wr = bms["wins"] / bms["trades"] * 100 if bms["trades"] else 0
        session_line = (
            f"Best Session: {SESSION_EMOJIS.get(best_ms,'')} {best_ms}  "
            f"{'+' if bms['pnl'] >= 0 else ''}${bms['pnl']:.2f}  ({bms_wr:.0f}%WR)\n"
        )
    else:
        session_line = ""

    # ML status line
    with ml_lock:
        ml_m, ml_t, ml_act = ml_model, ml_trained_on, ml_training_active
    ml_total = ml_total_trades
    if ml_act:
        ml_line = "⏳ Retraining in progress…"
    elif ml_m is None:
        ml_line = f"🔄 Warming up ({ml_total}/{ML_MIN_TRADES} trades needed)"
    else:
        nxt = max(0, ML_RETRAIN_EVERY - (ml_total - ml_t))
        ml_line = f"✅ Active · trained on {ml_t} · next retrain in {nxt} trades"

    # Top 5 hottest symbols right now
    scored = []
    for sym in SYMBOLS:
        with _lock:
            cc = current_candle.get(sym)
        if cc:
            try:
                row_c = {"Close": float(cc.get("close", 0)), "Open": float(cc.get("open", 0)),
                         "High":  float(cc.get("high",  0)), "Low":  float(cc.get("low",  0))}
                s, d, _ = score_signal(sym, row_c)
                scored.append((s, sym, d))
            except Exception:
                pass
    scored.sort(reverse=True)
    hot_lines = ""
    for s, sym, d in scored[:5]:
        heat = "🔥🔥" if s >= SCORE_THRESHOLD else "🔥" if s >= SCORE_THRESHOLD - 15 else "  "
        m5d   = indicators_m5.get(sym,  {}).get("supertrend_dir", 0)
        m15d  = indicators_m15.get(sym, {}).get("supertrend_dir", 0)
        m5ic  = ("↑" if m5d  == 1 else "↓" if m5d  == -1 else "→") + "M5"
        m15ic = ("↑" if m15d == 1 else "↓" if m15d == -1 else "→") + "M15"
        hot_lines += f"  {heat}<code>{sym:<10}</code> {s:>3}/100 {d} {m5ic} {m15ic}\n"
    if not hot_lines:
        hot_lines = "  — no data yet —\n"

    msg = (
        f"❤️ <b>Hourly Heartbeat  –  {now.strftime('%H:%M UTC')}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"State   : {'⏸ PAUSED' if is_paused else '▶ RUNNING'}\n"
        f"Market  : {SESSION_EMOJIS.get(mkt_s,'')} {mkt_s}\n"
        f"P&L     : <b>{'+' if pnl >= 0 else ''}${pnl:.2f}</b>\n"
        f"Trades  : {total}  ({wc}W / {lc}L  {wr:.1f}%)\n"
        f"Active  : {ac}  |  Streak: {'🔴×' + str(cl) if cl else '🟢 None'}\n"
        f"Drawdown: -${cur_dd:.2f}  (max -${mdd:.2f})\n"
        f"{session_line}"
        f"🤖 ML: {ml_line}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>Top Symbols:</b>\n{hot_lines}"
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
    t = Text("  DERIV SNIPER BOT  v3.0   ", style="bold cyan")
    t.append(f"  {now}  ", style="dim white")
    t.append(f"  [{_get_session_name()}]  ", style="bold magenta")
    t.append("    ")
    t.append(mode)
    return Panel(Align.center(t), style="bold blue", box=box.DOUBLE_EDGE)


def _make_symbols_table() -> Panel:
    tbl = Table(
        box=box.SIMPLE_HEAD, show_header=True, header_style="bold magenta",
        expand=True, min_width=68,
    )
    tbl.add_column("Symbol",      style="cyan",    min_width=7)
    tbl.add_column("Price",       style="white",   min_width=11)
    tbl.add_column("Supertrend",  style="yellow",  min_width=10)
    tbl.add_column("Direction",   style="green",   min_width=10)
    tbl.add_column("ATR",         style="magenta", min_width=8)
    tbl.add_column("ADX",         style="blue",    min_width=6)
    tbl.add_column("Ready",       style="bold",    min_width=6)
    tbl.add_column("Cooldown",    style="red",     min_width=10)

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
        cd_str   = f"{int((cd - now).total_seconds() // 60)}m" if cd > now else "Ready"
        cd_style = "red" if cd > now else "green"
        st_dir   = ind.get("supertrend_dir", 0)
        st_val   = ind.get("supertrend_val")
        st_str   = f"{st_val:.4f}" if st_val else "--"
        dir_str  = "↑ Bullish" if st_dir == 1 else ("↓ Bearish" if st_dir == -1 else "--")
        adx_val  = ind.get("adx")
        adx_str  = f"{adx_val:.0f}" if adx_val else "--"
        tbl.add_row(
            Text(sym, style="bold green" if sym in ac_syms else "cyan"),
            f"{price:.5f}" if price else "loading…",
            st_str,
            dir_str,
            f"{ind['atr']:.5f}" if ind["atr"] else "--",
            adx_str,
            Text("✓", style="bold green") if ind["ready"] else Text("…", style="dim"),
            Text(cd_str, style=cd_style),
        )
    return Panel(tbl, title="[bold blue]📊 Symbol Monitor  (Supertrend Edition)", border_style="blue")


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
        conf    = info.get("details", {}).get("ml_confidence")
        conf_s  = f"  ML:{conf*100:.0f}%" if conf is not None else ""
        direction = info.get("direction", "UP")

        if barrier and price and ep:
            span   = abs(float(barrier) - ep)
            moved  = abs(price - ep)
            tp_pct = min(100, int(moved / span * 100)) if span else 0
        else:
            tp_pct = 0

        elapsed   = (now - entry).total_seconds()
        time_pct  = max(0, 100 - int(elapsed / (DURATION * 60) * 100))
        left_secs = max(0, int((expires - now).total_seconds()))
        m, s = divmod(left_secs, 60)
        tp_col   = "green" if tp_pct >= 70 else ("yellow" if tp_pct >= 40 else "red")
        time_col = "green" if time_pct >= 50 else ("yellow" if time_pct >= 20 else "red")

        content.append(f"  {sym:<7}", style="bold cyan")
        content.append(f"  #{cid}  {direction}  score={score}/100{conf_s}  ", style="dim")
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
            total_pnl, win_count, loss_count, consecutive_losses, daily_trades, paused,
        )
        peak, mdd = peak_equity, max_drawdown
    cur_dd = max(0.0, peak - pnl)
    total  = wc + lc
    wr     = wc / total * 100 if total else 0
    dur    = datetime.now(timezone.utc) - session_start
    h, r   = divmod(int(dur.total_seconds()), 3600)
    mi     = r // 60
    col    = "green" if pnl >= 0 else "red"
    risk_used = (
        min(100, int(abs(min(0, pnl)) / abs(DAILY_LOSS_LIMIT) * 100))
        if DAILY_LOSS_LIMIT < 0 else 0
    )
    risk_col = "red" if risk_used >= 80 else ("yellow" if risk_used >= 50 else "green")
    b_sym, b_st, w_sym, w_st = _best_worst_session()
    mkt_s = _get_session_name()

    t = Text()
    t.append("  Market Session: ", style="dim"); t.append(f"{SESSION_EMOJIS.get(mkt_s,'')} {mkt_s}\n", style="bold magenta")
    t.append("  Session P&L  : ", style="dim"); t.append(f"${pnl:+.2f}\n", style=f"bold {col}")
    t.append("  Wins / Losses: ", style="dim")
    t.append(f"{wc}", style="bold green"); t.append(" / ")
    t.append(f"{lc}\n", style="bold red")
    t.append("  Win Rate     : ", style="dim")
    t.append(f"{wr:.1f}%\n", style="bold yellow" if wr >= 50 else "bold red")
    t.append("  Consec Loss  : ", style="dim")
    t.append(f"{cl}/{MAX_CONSECUTIVE_LOSSES}\n", style="bold red" if cl > 0 else "white")
    t.append("  Day Trades   : ", style="dim"); t.append(f"{dt}\n", style="white")
    t.append("  Session Up   : ", style="dim"); t.append(f"{h}h {mi}m\n", style="white")
    t.append("  Drawdown     : ", style="dim")
    t.append(f"-${cur_dd:.2f}", style="bold red" if cur_dd > 0 else "white")
    t.append("  (max ", style="dim"); t.append(f"-${mdd:.2f}", style="bold red"); t.append(")\n", style="dim")
    if b_sym:
        b_wr = b_st["wins"] / (b_st["wins"] + b_st["losses"]) * 100 if (b_st["wins"] + b_st["losses"]) else 0
        w_wr = w_st["wins"] / (w_st["wins"] + w_st["losses"]) * 100 if (w_st["wins"] + w_st["losses"]) else 0
        t.append("\n  🏅 Best  : ", style="dim"); t.append(f"{b_sym}", style="bold green")
        t.append(f"  {'+' if b_st['pnl'] >= 0 else ''}${b_st['pnl']:.2f}  ({b_wr:.0f}%WR)\n", style="green")
        t.append("  💔 Worst : ", style="dim"); t.append(f"{w_sym}", style="bold red")
        t.append(f"  {'+' if w_st['pnl'] >= 0 else ''}${w_st['pnl']:.2f}  ({w_wr:.0f}%WR)\n", style="red")
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
        Align.center(Text(_make_footer_text(), style="dim")),
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
#  SCAN STATUS  (5-minute digest)
# ══════════════════════════════════════════════════════════════════════
def _score_sparkline_loop():
    time.sleep(3600)
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
            mkt_s = _get_session_name(now)
            lines = [
                f"📡 <b>SCAN STATUS</b>  {now.strftime('%H:%M')} UTC  "
                f"{SESSION_EMOJIS.get(mkt_s,'')} {mkt_s}\n━━━━━━━━━━━━━━━━━━━━"
            ]

            # ── Single lock acquisition: snapshot all shared state ──────────
            with _lock:
                ind_snap     = {s: dict(indicators[s]) for s in SYMBOLS}
                cc_snap      = {s: dict(current_candle[s]) if current_candle.get(s) else None
                                for s in SYMBOLS}
                cd_snap      = dict(cooldown_until)
                psd          = paused
                ppnl         = total_pnl
                ohlcv_lens   = {s: len(ohlcv[s]) for s in SYMBOLS}
                locked_syms  = set(locked_symbols.keys())   # snapshot for lock-free read below
            # ────────────────────────────────────────────────────────────────

            for sym in SYMBOLS:
                ind   = ind_snap[sym]
                cc    = cc_snap.get(sym)
                cd    = cd_snap.get(sym, datetime.min.replace(tzinfo=timezone.utc))

                if not ind.get("ready"):
                    lines.append(f"⏳ <code>{sym:<10}</code>  loading… ({ohlcv_lens.get(sym,0)}/500 candles)")
                    continue

                if cc:
                    row_c = {"Close": float(cc.get("close", 0)),
                             "Open":  float(cc.get("open", 0)),
                             "High":  float(cc.get("high", 0)),
                             "Low":   float(cc.get("low", 0))}
                    score, direction, det = score_signal(sym, row_c)
                else:
                    score, direction, det = 0, "UP", {}

                with _lock:
                    if sym in symbol_score_history:
                        symbol_score_history[sym].append(score)
                    else:
                        symbol_score_history[sym] = deque([score], maxlen=20)

                heat   = "🔥🔥" if score >= SCORE_THRESHOLD else "🔥" if score >= SCORE_THRESHOLD - 20 else "  "
                locked = " 🔒" if sym in locked_syms else ""
                blocked = ""
                if score >= SCORE_THRESHOLD:
                    if psd:
                        blocked = " ⏸paused"
                    elif now < cd:
                        left = max(0, int((cd - now).total_seconds() // 60))
                        blocked = f" ⏱{left}m cd"
                    elif ppnl <= DAILY_LOSS_LIMIT:
                        blocked = " 🚫floor"

                st_dir_v = det.get("ema_fast_sl", 0)
                dir_icon = "↑" if st_dir_v > 0 else ("↓" if st_dir_v < 0 else "→")
                t  = det.get("trend", 0)
                eq = det.get("entry_quality", 0)
                v  = det.get("volatility", 0)
                m  = det.get("momentum", 0)
                bar = f"T{t} E{eq} V{v} M{m}"
                lines.append(
                    f"{heat}<code>{sym:<10}</code>  <b>{score:>3}/100</b> {direction} {dir_icon} "
                    f"{bar}{blocked}{locked}"
                )

            lines.append("")
            lines.append(_best_worst_line())
            _send_tg("\n".join(lines))
        except Exception as e:
            logger.error(f"scan_status_loop: {e}")
        time.sleep(300)


# ══════════════════════════════════════════════════════════════════════
#  MIDNIGHT BREAKDOWN  (00:00 UTC daily)
# ══════════════════════════════════════════════════════════════════════
def _midnight_breakdown_loop():
    """At every UTC midnight, send a full day breakdown by market session."""
    while True:
        try:
            now  = datetime.now(timezone.utc)
            # Calculate seconds until next midnight UTC
            tomorrow = (now + timedelta(days=1)).replace(
                hour=0, minute=0, second=5, microsecond=0
            )
            wait = (tomorrow - now).total_seconds()
            time.sleep(max(1, wait))

            # Build midnight report.
            # IMPORTANT: snapshot stats FIRST, before rolling resets them for the new day.
            # 'now' was captured before the sleep so its date IS the just-ended day.
            today = now.strftime("%Y-%m-%d")

            with _lock:
                ms_snap = {k: dict(v) for k, v in market_session_stats.items()}
                log     = list(daily_session_log)

            _roll_market_session_stats_if_needed()

            total_day_pnl = sum(v["pnl"] for v in ms_snap.values())
            total_day_trades = sum(v["trades"] for v in ms_snap.values())
            total_day_wins   = sum(v["wins"]   for v in ms_snap.values())
            day_wr = total_day_wins / total_day_trades * 100 if total_day_trades else 0

            # Best session today
            traded_sessions = {k: v for k, v in ms_snap.items() if v["trades"] > 0}
            if traded_sessions:
                best_s  = max(traded_sessions, key=lambda k: traded_sessions[k]["pnl"])
                worst_s = min(traded_sessions, key=lambda k: traded_sessions[k]["pnl"])
            else:
                best_s = worst_s = None

            lines = [
                f"🌙 <b>MIDNIGHT BREAKDOWN — {today}</b>",
                "━━━━━━━━━━━━━━━━━━━━",
                f"Day P&L    : <b>{'+' if total_day_pnl >= 0 else ''}${total_day_pnl:.2f}</b>",
                f"Day Trades : {total_day_trades}  ({total_day_wins}W  {day_wr:.0f}%WR)",
                f"Sessions   : {len(log)} TP/SL hits",
                "",
                "<b>By Market Session</b>",
            ]

            for name in ["Asian", "London", "New York", "Midnight"]:
                s    = ms_snap.get(name, {"wins": 0, "losses": 0, "pnl": 0.0, "trades": 0})
                tot  = s["wins"] + s["losses"]
                wr   = s["wins"] / tot * 100 if tot else 0
                sign = "+" if s["pnl"] >= 0 else ""
                tag  = "  🏆 BEST" if name == best_s else ("  💔 WORST" if name == worst_s else "")
                lines.append(
                    f"  {SESSION_EMOJIS.get(name,'')} <b>{name}</b>{tag}\n"
                    f"    {s['trades']} trades  {s['wins']}W/{s['losses']}L "
                    f"({wr:.0f}%WR)  <b>{sign}${s['pnl']:.2f}</b>"
                )

            if best_s:
                lines += [
                    "",
                    f"🏆 <b>Best Session Today</b>: {SESSION_EMOJIS.get(best_s,'')} {best_s}  "
                    f"+${ms_snap[best_s]['pnl']:.2f}",
                ]

            lines += [
                "",
                "━━━━━━━━━━━━━━━━━━━━",
                "<i>New day has started — session stats reset.</i>",
            ]
            _send_tg("\n".join(lines), reply_markup=_main_kb())
            logger.info(f"Midnight breakdown sent for {today}")
        except Exception as e:
            logger.error(f"midnight_breakdown_loop: {e}")
            time.sleep(60)


# ══════════════════════════════════════════════════════════════════════
#  PINNED TELEGRAM DASHBOARD  (auto-updated every 5 min)
# ══════════════════════════════════════════════════════════════════════
def _pinned_dashboard_loop():
    """Send/edit a pinned Telegram message with live scores & P&L every 5 min."""
    global _pinned_msg_id
    time.sleep(90)   # let WS feeds connect first
    while True:
        try:
            if not _tg_loop_ok():
                time.sleep(30)
                continue

            async def _do_update():
                global _pinned_msg_id
                try:
                    now   = datetime.now(timezone.utc)
                    mkt_s = _get_session_name(now)
                    with _lock:
                        pnl        = total_pnl
                        wc_        = win_count
                        lc_        = loss_count
                        is_paused_ = paused
                    tot = wc_ + lc_
                    wr  = wc_ / tot * 100 if tot else 0

                    # Hot signals
                    scored = []
                    for sym in SYMBOLS:
                        with _lock:
                            cc = current_candle.get(sym)
                        if cc:
                            try:
                                row_c = {
                                    "Close": float(cc.get("close", 0)),
                                    "Open":  float(cc.get("open",  0)),
                                    "High":  float(cc.get("high",  0)),
                                    "Low":   float(cc.get("low",   0)),
                                }
                                s, d, _ = score_signal(sym, row_c)
                                scored.append((s, sym, d))
                            except Exception:
                                pass
                    scored.sort(reverse=True)
                    hot = ""
                    for s, sym, d in scored[:5]:
                        m5d   = indicators_m5.get(sym,  {}).get("supertrend_dir", 0)
                        m15d  = indicators_m15.get(sym, {}).get("supertrend_dir", 0)
                        m5ic  = "↑" if m5d  == 1 else ("↓" if m5d  == -1 else "→")
                        m15ic = "↑" if m15d == 1 else ("↓" if m15d == -1 else "→")
                        heat  = "🔥🔥" if s >= SCORE_THRESHOLD else ("🔥" if s >= SCORE_THRESHOLD - 15 else "  ")
                        hot  += f"  {heat}<code>{sym:<10}</code> {s:>3}/100 {d} (M5:{m5ic} M15:{m15ic})\n"
                    if not hot:
                        hot = "  — no hot signals —\n"

                    with ml_lock:
                        ml_m_ = ml_model; ml_t_ = ml_trained_on
                    ml_s = f"✅ trained on {ml_t_}" if ml_m_ else "⏳ warming up"

                    text = (
                        f"📌 <b>LIVE DASHBOARD</b>  ·  {now.strftime('%H:%M UTC')}\n"
                        f"{'⏸ PAUSED' if is_paused_ else '▶ RUNNING'}  ·  "
                        f"{SESSION_EMOJIS.get(mkt_s,'')} {mkt_s}\n"
                        f"━━━━━━━━━━━━━━━━━━━━\n"
                        f"💵 P&L  : <b>{'+' if pnl >= 0 else ''}${pnl:.2f}</b>  "
                        f"({tot} trades  {wc_}W/{lc_}L  {wr:.0f}%)\n"
                        f"🤖 ML   : {ml_s}\n"
                        f"🎯 Gate : score≥{SCORE_THRESHOLD}  |  ML≥{ML_CONFIDENCE_MIN*100:.0f}%  "
                        f"|  ATR×{ATR_BARRIER_MULT}\n"
                        f"━━━━━━━━━━━━━━━━━━━━\n"
                        f"<b>🔥 Hot Signals:</b>\n{hot}"
                        f"<i>↻ updates every 5 min</i>"
                    )

                    async def _pin_msg(msg_id):
                        try:
                            await telegram_app.bot.pin_chat_message(
                                chat_id=TELEGRAM_CHAT_ID,
                                message_id=msg_id,
                                disable_notification=True,
                            )
                        except Exception:
                            pass   # pin may fail in groups without admin rights — that's OK

                    if _pinned_msg_id:
                        try:
                            await telegram_app.bot.edit_message_text(
                                chat_id=TELEGRAM_CHAT_ID,
                                message_id=_pinned_msg_id,
                                text=text,
                                parse_mode="HTML",
                            )
                        except Exception:
                            # Message too old or deleted — send fresh and re-pin
                            m = await telegram_app.bot.send_message(
                                chat_id=TELEGRAM_CHAT_ID, text=text, parse_mode="HTML"
                            )
                            _pinned_msg_id = m.message_id
                            await _pin_msg(_pinned_msg_id)
                    else:
                        m = await telegram_app.bot.send_message(
                            chat_id=TELEGRAM_CHAT_ID, text=text, parse_mode="HTML"
                        )
                        _pinned_msg_id = m.message_id
                        await _pin_msg(_pinned_msg_id)

                except Exception as e:
                    logger.error(f"pinned_dashboard update: {e}")

            asyncio.run_coroutine_threadsafe(_do_update(), _tg_loop)
        except Exception as e:
            logger.error(f"pinned_dashboard_loop: {e}")
        time.sleep(300)   # every 5 minutes


# ══════════════════════════════════════════════════════════════════════
#  SESSION-CHANGE MONITOR  (auto end-of-session report)
# ══════════════════════════════════════════════════════════════════════
def _session_change_monitor_loop():
    """Detect market-session boundaries and send a full end-of-session report."""
    global _current_session_name
    time.sleep(120)   # let indicators & WS settle first
    _current_session_name = _get_session_name(datetime.now(timezone.utc))
    while True:
        try:
            time.sleep(60)
            now     = datetime.now(timezone.utc)
            new_ses = _get_session_name(now)
            if _current_session_name and new_ses != _current_session_name:
                # Snapshot stats BEFORE updating _current_session_name
                with _lock:
                    snap = {
                        "reason":         "SESSION_END",
                        "market_session": _current_session_name,
                        "pnl":            total_pnl,
                        "wins":           win_count,
                        "losses":         loss_count,
                        "peak":           peak_equity,
                        "max_dd":         max_drawdown,
                        "duration":       now - session_start,
                        "symbols":        {s: dict(v) for s, v in session_symbol_stats.items()},
                    }
                logger.info(f"Session boundary: {_current_session_name} → {new_ses}")
                _send_tg(_session_summary_text(snap))
                _current_session_name = new_ses
            else:
                _current_session_name = new_ses
        except Exception as e:
            logger.error(f"session_change_monitor_loop: {e}")


# ══════════════════════════════════════════════════════════════════════
#  TELEGRAM STARTUP
# ══════════════════════════════════════════════════════════════════════
def _start_telegram():
    async def _run():
        global telegram_app, _tg_loop
        # Clear readiness in case this is a restart
        _tg_ready.clear()
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
        app.add_handler(CommandHandler("export",  cmd_export))
        app.add_handler(CommandHandler("backup",  cmd_backup))
        # CSV upload: catch ALL document messages and check filename inside the
        # handler — avoids MIME-type mismatches (mobile sends text/plain,
        # desktop sends text/csv, some clients send application/octet-stream).
        app.add_handler(
            MessageHandler(filters.Document.ALL, cmd_upload_csv)
        )
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

        _log("📱 Telegram bot v3 started  (PTB " +
             __import__("telegram").__version__ + ")")
        async with app:
            await app.start()
            await app.updater.start_polling(
                allowed_updates=["message", "callback_query"],
                drop_pending_updates=True,   # discard stale updates from downtime
            )
            # Signal readiness only after polling is confirmed up — any
            # thread calling _send_tg will now find a stable, open loop.
            _tg_ready.set()
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
        # Fast /ping path — minimal JSON for UptimeRobot / health checks
        if self.path in ("/ping", "/ping/"):
            body = b'{"ping":"pong","status":"ok"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except Exception:
                pass
            return

        # Full status JSON for /  /health  /status  (and any other path)
        with _lock:
            pnl, wc, lc, is_paused = total_pnl, win_count, loss_count, paused
            peak, mdd = peak_equity, max_drawdown
        uptime = datetime.now(timezone.utc) - session_start
        body = json.dumps({
            "status": "ok",
            "version": "3.0",
            "state": "paused" if is_paused else "running",
            "market_session": _get_session_name(),
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
        _log(
            f"🩺 Health server on :{port}  —  "
            f"UptimeRobot URL: https://<your-render-url>/ping"
        )
        server.serve_forever()
    except Exception as e:
        logger.error(f"Health server failed on :{port}: {e}")


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    is_tty = sys.stdout.isatty()
    console.print(Panel(
        Align.center(Text(
            f"\n  DERIV SNIPER BOT  v3.0  –  Supertrend Edition\n"
            f"  Loading history for {len(SYMBOLS)} symbols…\n",
            style="bold cyan",
        )),
        border_style="blue", box=box.DOUBLE_EDGE,
    ))

    # Initialise symbol state
    for sym in SYMBOLS:
        _init_symbol(sym)

    # ── Health server starts immediately (no async loop dependency)
    _watch_thread(_start_health_server,         name="HealthServer")

    # Load candle history (batched, parallel within batch).
    # Telegram is started AFTER history so the async event loop is fully
    # stable before any thread calls _send_tg (avoids "Event loop is closed"
    # RuntimeError that occurs when MLBootstrap fires _send_tg too early).
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

    console.print("[green]✓ History phase done.[/green]")

    # Load existing ML model
    _ml_load()

    # ── Start Telegram NOW — history is loaded so the loop is stable before
    # MLBootstrap (or any other thread) calls _send_tg.
    _watch_thread(_start_telegram,              name="Telegram")

    # Block until the Telegram polling loop confirms it is fully up.
    # _tg_ready is set inside _run() only after app.start() + start_polling()
    # succeed, so any subsequent _send_tg call is guaranteed a live loop.
    _tg_ready.wait(timeout=30)

    # Bootstrap ML from candle history (runs in background thread)
    threading.Thread(
        target=_ml_bootstrap_from_history,
        daemon=True, name="MLBootstrap"
    ).start()

    # ── Indicator computation executor: serialises heavy pandas work across
    # all 15 symbols so they don't all spike CPU simultaneously at candle-close.
    global _indicator_executor
    _indicator_executor = ThreadPoolExecutor(
        max_workers=2, thread_name_prefix="IndComp"
    )

    # Start remaining core threads
    _watch_thread(_db_writer,                   name="DBWriter")
    for sym in SYMBOLS:
        _watch_thread(_ws_thread, args=(sym,),  name=f"WS-{sym}")
    # Terminal dashboard only makes sense in a real TTY (Render/headless = skip)
    if is_tty:
        _watch_thread(_terminal_loop,           name="TermRefresh")
    _watch_thread(_scan_status_loop,            name="ScanStatus")
    _watch_thread(_score_sparkline_loop,        name="ScoreSparkline")
    _watch_thread(_pending_trade_timeout_loop,  name="PendingTimeout")
    _watch_thread(_stale_contract_watchdog,     name="StaleWatchdog")
    _watch_thread(_midnight_breakdown_loop,     name="MidnightBreakdown")
    _watch_thread(_pinned_dashboard_loop,       name="PinnedDashboard")
    _watch_thread(_session_change_monitor_loop, name="SessionMonitor")

    # Mark bot as fully ready — commands that need DB/indicators are now safe
    global _bot_ready
    _bot_ready = True
    _send_tg("✅ <b>Bot fully ready</b> — all history loaded, WS feeds live, trading active.")

    console.print("[green]✓ All threads started. Bot is live.[/green]")

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

