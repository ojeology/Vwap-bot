#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║        DERIV RISE/FALL BOT v4  –  Professional Edition               ║
║  Market Structure · Candlestick Patterns · Supertrend · Fast ML      ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import json, time, sqlite3, threading, queue, logging, asyncio, os, sys, pickle, io, csv as _csv, html as _html
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from collections import deque
from typing import Optional

import math
import numpy as np
import pandas as pd

# PostgreSQL — prefers NEON_DATABASE_URL (user-supplied Neon connection string),
# falls back to Replit-managed DATABASE_URL, then SQLite.
DATABASE_URL = os.environ.get("NEON_DATABASE_URL") or os.environ.get("DATABASE_URL")
USE_PG = bool(DATABASE_URL)
if USE_PG:
    try:
        import psycopg2
        import psycopg2.extras
    except ImportError:
        # psycopg2 not installed — fall back to SQLite and warn
        import logging as _lg
        _lg.getLogger("sniper").warning(
            "NEON_DATABASE_URL is set but psycopg2-binary is not installed. "
            "Falling back to SQLite. Run: pip install psycopg2-binary"
        )
        DATABASE_URL = None
        USE_PG = False
import websocket
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score

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
SYNTH_STEP          = ["stpRNG"]                                      # Step Index — Rise/Fall
SYNTH_JUMP          = ["JD10", "JD25", "JD50", "JD75", "JD100"]      # Jump 10/25/50/75/100

SYMBOLS        = (SYNTH_VOLATILITY + SYNTH_VOLATILITY_1S + SYNTH_RANGE_BREAK
                  + SYNTH_STEP + SYNTH_JUMP)
ALL_RF_SYMBOLS = SYMBOLS   # all Rise/Fall tradeable symbols

# ── Risk parameters ───────────────────────────────────────────────────
STAKE                  = 1.0          # $1 per trade
TARGET_PROFIT          = 0.70         # display only
PROFIT_MIN             = 0.60         # reject proposal if payout profit ratio < this (Rise/Fall)
# No upper bound — Rise/Fall accepts any payout above the minimum
DURATION               = 5            # 5-minute Rise/Fall contracts
CONTRACT_TYPE_RISE     = "CALL"       # Rise contract (UP direction)
CONTRACT_TYPE_FALL     = "PUT"        # Fall contract (DOWN direction)
COOLDOWN_MINUTES       = 20
# ── Second entry (re-entry on same symbol after a win) ────────────────
ALLOW_SECOND_ENTRY     = True     # enable controlled re-entry after a win
SECOND_ENTRY_COOLDOWN  = 5        # shortened cooldown (min) after a win
SECOND_ENTRY_ML_MIN    = 0.90     # stricter ML gate for re-entry (90%)
SECOND_ENTRY_WINDOW    = 15       # window (min) to wait for re-entry signal after win
MAX_CONSECUTIVE_LOSSES = 3
PAUSE_MINUTES          = 30

# ── Martingale (3-step: base → 2× → 4× stake on consecutive losses) ──
MARTINGALE_ENABLED    = True    # double stake on each consecutive loss up to MAX_STEPS
MARTINGALE_MULTIPLIER = 2.0     # stake multiplier per loss step
MARTINGALE_MAX_STEPS  = 1       # levels: 0=1× base, 1=2× base → then reset (1 recovery step only)
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
SCORE_THRESHOLD        = 75           # B-grade minimum (95=A+, 90=A, 85=B+, 75=B)
HEARTBEAT_INTERVAL_SEC = 3600      # richer hourly heartbeat

# ── Strategy version ────────────────────────────────────────────────────
# Bump this any time DURATION / SCORE_THRESHOLD / ML_CONFIDENCE_MIN /
# ATR_BARRIER_MULT (or the underlying strategy logic) changes, so reports
# can be compared apples-to-apples across config revisions over time.
STRATEGY_VERSION = "V4.0-RF"  # Rise/Fall edition

# ── Confirmation gate: require at least one entry signal before executing ──
# Checks: candle pattern aligned, MACD confirmed, ROC aligned, or wick rejection.
# Prevents "perfect trend, early entry" trades that miss the actual turn.
CONFIRMATION_GATE_ENABLED = True

# ── Market sessions (UTC hours) ───────────────────────────────────────
# Maps session name → (start_hour_utc, end_hour_utc).
# Midnight wraps across 00:00, handled in _get_session_name().
MARKET_SESSIONS = {
    "Midnight":       (22, 2),
    "Early Asian":    (2,  6),
    "Late Asian":     (6,  10),
    "London":         (10, 13),
    "Early New York": (13, 18),
    "Late New York":  (18, 22),
}
SESSION_EMOJIS = {
    "Midnight":       "🌙",
    "Early Asian":    "🌏",
    "Late Asian":     "🌏",
    "London":         "🇬🇧",
    "Early New York": "🗽",
    "Late New York":  "🌆",
}
# Broad session group mapping (for DB backward-compat display)
SESSION_GROUP = {
    "Midnight":       "Midnight",
    "Early Asian":    "Asian",
    "Late Asian":     "Asian",
    "London":         "London",
    "Early New York": "New York",
    "Late New York":  "New York",
}
# UTC time-range labels shown in session breakdown reports
SESSION_TIMES = {
    "Midnight":       "22:00–02:00 UTC",
    "Early Asian":    "02:00–06:00 UTC",
    "Late Asian":     "06:00–10:00 UTC",
    "London":         "10:00–13:00 UTC",
    "Early New York": "13:00–18:00 UTC",
    "Late New York":  "18:00–22:00 UTC",
}

# ── ML filter ────────────────────────────────────────────────────────
MODEL_PATH        = "ml_model.pkl"
ML_MIN_TRADES     = 50
ML_RETRAIN_EVERY  = 50           # retrain every 50 trades (continuous rolling)
ML_CONFIDENCE_MIN = 0.75              # require ≥75% confidence (adjustable via Telegram)
# Rich 16-feature set used when signal_features table has ≥ ML_MIN_TRADES rows
ML_FEATURE_COLS = [
    "score",          # total signal score [0-100]
    "adx_val",        # ADX trend strength
    "rsi_val",        # RSI oscillator
    "stochrsi_k_val", # StochRSI K line
    "ms_pts",         # market structure points
    "sd_pts",         # supply/demand zone points
    "et_pts",         # entry timing points
    "mom_pts",        # momentum points
    "cc_pts",         # candle confirmation points
    "sd_dist",        # zone proximity (ATR multiples)
    "atr_val",        # ATR volatility
    "ema_aligned",    # EMA50/200 alignment (0/1 int)
    "ms_bos",         # break of structure (0/1 int)
    "macd_bullish",   # MACD direction (0/1 int)
    "roc_val",        # rate of change
    "wick_atr_ratio", # candle wick size in ATR
]
# Legacy 7-feature set for trades-table fallback (column names in `trades` table)
_ML_LEGACY_COLS = [
    "score", "wick_atr_ratio", "atr", "atr_ma",
    "ema_fast_slope", "ema_slow_slope", "ema_distance",
]
# Engineered on top of the raw DB columns at training time (not stored):
# cyclical time-of-day + ordinal market-session position, so the model can
# learn session-dependent edge without a one-hot blowing up on small data.
_ML_SESSION_ORDER = {name: i for i, name in enumerate(MARKET_SESSIONS.keys())}
_ML_ENGINEERED_COLS = ["hour_sin", "hour_cos", "session_ord"]
# Below this many rows, a stacked/calibrated ensemble has too few samples
# per cross-val fold to be trustworthy — fall back to a single regularized
# GradientBoosting model instead of forcing "exotic" onto noise.
ML_STACKING_MIN_TRADES = 60

# ── Per-class ML: separate model per volatility family ───────────────
ML_SYMBOL_CLASSES = {
    "standard_vol": ["R_100", "R_75", "R_50", "R_25", "R_10"],
    "tick_vol":     ["1HZ100V", "1HZ90V", "1HZ75V", "1HZ50V",
                     "1HZ30V", "1HZ25V", "1HZ15V", "1HZ10V"],
    "range_break":  ["RDBULL", "RDBEAR"],
    "step":         ["stpRNG"],
    "jump":         ["JD10", "JD25", "JD50", "JD75", "JD100"],
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
ws_registry: dict = {}   # symbol -> live WebSocketApp, used to re-query contracts before assuming a loss
# symbol -> {"details":..., "requested_at":...} — set while we ask Deriv's
# portfolio for a contract that may have opened despite a lost buy ack, so
# we never silently abandon a real, un-tracked open position.
_portfolio_checks: dict = {}
total_pnl       = 0.0
daily_total_pnl = 0.0   # Cumulates all day; resets at UTC midnight only (session resets don't touch it)
peak_equity = 0.0
max_drawdown = 0.0
win_count = loss_count = daily_trades = consecutive_losses = 0
paused = False
pause_until = datetime.min.replace(tzinfo=timezone.utc)
session_start = datetime.now(timezone.utc)
signal_log: deque[str] = deque(maxlen=20)
session_symbol_stats: dict = {}   # symbol → {wins, losses, pnl}

# Second-entry tracking ───────────────────────────────────────────────
# Maps symbol → expiry datetime while a re-entry window is open after a win
second_entry_eligible: dict = {}   # symbol → window_expiry datetime

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

# Signal funnel (scanned / executed / rejection reasons) — resets daily ──
daily_funnel: dict = {"scanned": 0, "executed": 0, "rejections": {}}
_funnel_date: str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

telegram_app = None
_tg_loop = None
_tg_ready = threading.Event()   # set only after app.start() + polling confirmed up
_auto_resume_active = False
_bot_ready = False          # set True after history + DB + WS threads are up
_pinned_msg_id: Optional[int] = None   # ID of the auto-pinned dashboard message
_current_session_name: str = ""        # for session-change auto-reports
_test_trade_sem = threading.Semaphore(1)
_test_trade_active: dict = {}

# Martingale tracking — symbol → current loss step (0=base, 1=2×, 2=4×)
martingale_level: dict = {}

# ML state ─────────────────────────────────────────────────────────────
ml_model = None          # global combined model (all symbols)
ml_trained_on = 0
ml_total_trades = 0
ml_lock = threading.Lock()
ml_training_active = False
ml_models_per_class: dict = {cls: None for cls in ML_SYMBOL_CLASSES}   # per-class models
ml_trained_per_class: dict = {cls: 0   for cls in ML_SYMBOL_CLASSES}
# Live in-memory ML confidence performance tracker (actual ML probability, not score proxy)
# bucket → {wins, losses, pnl}  — reset only on restart, not on session reset
_ml_conf_live_stats: dict = {}

# Live in-memory signal score performance tracker (the raw score gate used to trigger)
# Same bucket shape as above. Lets us see whether score bands (75-84, 85-89, etc.)
# translate into real P&L independent of the ML confidence value.
_ml_score_live_stats: dict = {}

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
    """Return the detailed market session name (Early/Late sub-splits) for a UTC datetime."""
    if dt is None:
        dt = datetime.now(timezone.utc)
    h = dt.hour
    if h >= 22 or h < 2:
        return "Midnight"
    elif h < 6:
        return "Early Asian"
    elif h < 10:
        return "Late Asian"
    elif h < 13:
        return "London"
    elif h < 18:
        return "Early New York"
    else:
        return "Late New York"


def _get_session_group(session_name: str) -> str:
    """Return the broad session group for a sub-session (e.g. Early Asian → Asian)."""
    return SESSION_GROUP.get(session_name, session_name)


def _next_session_start(dt: Optional[datetime] = None) -> datetime:
    """Return the UTC datetime when the NEXT market sub-session begins."""
    if dt is None:
        dt = datetime.now(timezone.utc)
    current = _get_session_name(dt)
    # Each sub-session ends at this UTC hour
    next_start_hour = {
        "Midnight": 2, "Early Asian": 6, "Late Asian": 10,
        "London": 13, "Early New York": 18, "Late New York": 22,
    }
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
        else:
            # Ensure any new sub-session key exists (e.g. after config update)
            for name in MARKET_SESSIONS:
                market_session_stats.setdefault(
                    name, {"wins": 0, "losses": 0, "pnl": 0.0, "trades": 0}
                )


def _roll_daily_funnel_if_needed_locked():
    """Clear the signal funnel counters if UTC date has rolled over.
    MUST be called while already holding _lock — the roll-check and the
    counter mutation that follows must be one atomic critical section, or a
    date rollover landing between them can wipe/misattribute an event."""
    global daily_funnel, _funnel_date
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if today != _funnel_date:
        daily_funnel = {"scanned": 0, "executed": 0, "rejections": {}}
        _funnel_date = today


def _record_scan():
    """Count one signal that reached the entry-decision funnel (score + momentum confirmed)."""
    with _lock:
        _roll_daily_funnel_if_needed_locked()
        daily_funnel["scanned"] += 1


def _record_execution():
    """Count one signal that actually got bought (passed all gates + broker confirmed)."""
    with _lock:
        _roll_daily_funnel_if_needed_locked()
        daily_funnel["executed"] += 1


def _record_funnel_rejection(reason_key: str):
    """Count one signal rejected at the decision-funnel stage, bucketed by reason."""
    with _lock:
        _roll_daily_funnel_if_needed_locked()
        daily_funnel["rejections"][reason_key] = daily_funnel["rejections"].get(reason_key, 0) + 1


def _funnel_snapshot() -> dict:
    with _lock:
        _roll_daily_funnel_if_needed_locked()
        return {"scanned": daily_funnel["scanned"], "executed": daily_funnel["executed"],
                "rejections": dict(daily_funnel["rejections"])}


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
    """Return ML win-probability (0.0–1.0). Supports rich 16-feature and legacy 7-feature models."""
    cls = _get_symbol_class(symbol) if symbol else "standard_vol"
    with ml_lock:
        model = ml_models_per_class.get(cls) or ml_model
    if model is None:
        return 1.0
    try:
        n_feats = getattr(model, "n_features_in_", len(ML_FEATURE_COLS))
        if n_feats == len(ML_FEATURE_COLS):
            # Rich 16-feature vector (trained on signal_features table)
            feats = [[
                float(details.get("total_score", 0) or 0),
                float(details.get("adx", 0) or 0),
                float(details.get("rsi", 0) or 0),
                float(details.get("stochrsi_k", 0) or 0),
                float(details.get("market_struct_pts", 0) or 0),
                float(details.get("sd_zone_pts", 0) or 0),
                float(details.get("entry_quality", 0) or 0),
                float(details.get("momentum", 0) or 0),
                float(details.get("candle_pattern_pts", 0) or 0),
                float(details.get("sd_zone_dist", 99.0) or 99.0),
                float(details.get("atr", 0) or 0),
                float(int(bool(details.get("ema_aligned", False)))),
                float(int(bool(details.get("market_struct_bos", False)))),
                float(int(bool(details.get("macd_bullish", False)))),
                float(details.get("roc", 0) or 0),
                float(details.get("wick_atr_ratio", 0) or 0),
            ]]
        else:
            # Legacy 7-feature vector (trained on trades table)
            feats = [[
                float(details.get("total_score", 0) or 0),
                float(details.get("wick_atr_ratio", 0) or 0),
                float(details.get("atr", 0) or 0),
                float(details.get("atr_ma", 0) or 0),
                float(details.get("ema_fast_sl", 0) or 0),
                float(details.get("ema_slow_sl", 0) or 0),
                float(details.get("ema_distance", 0) or 0),
            ]]
        proba   = model.predict_proba(feats)[0]
        classes = list(model.classes_)
        win_idx = classes.index(1) if 1 in classes else len(classes) - 1
        return float(proba[win_idx])
    except Exception as e:
        logger.error(f"ML confidence failed, allowing trade: {e}")
        return 1.0


def _ml_should_trade(details: dict, symbol: str = "", min_conf: float = None) -> bool:
    """Filter trade by ML confidence. Stores 'ml_confidence' in details for display.
    Pass min_conf to override the global gate (e.g. higher bar for second entries)."""
    conf = _ml_get_confidence(details, symbol)
    details["ml_confidence"] = conf
    cls = _get_symbol_class(symbol) if symbol else "standard_vol"
    with ml_lock:
        has_model = (ml_models_per_class.get(cls) or ml_model) is not None
    if not has_model:
        return True   # observe-only until first model
    gate = min_conf if min_conf is not None else ML_CONFIDENCE_MIN
    return conf >= gate


def _ml_export_csv(total: int):
    try:
        cols = [
            "id", "timestamp", "symbol", "direction", "barrier",
            "stake", "payout", "profit", "win", "score",
            "wick_atr_ratio", "atr", "atr_ma",
            "ema_fast_slope", "ema_slow_slope", "ema_distance",
            "market_session",
        ]
        # Use _db_fetch for PostgreSQL + SQLite compatibility
        rows = _db_fetch(f"SELECT {', '.join(cols)} FROM trades ORDER BY id")
        if not rows:
            # Fallback: try without market_session (older schema)
            cols = cols[:-1]
            rows = _db_fetch(f"SELECT {', '.join(cols)} FROM trades ORDER BY id")
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


def _ml_engineer(raw_vals: list, timestamp: str, session: str) -> list:
    """Extend the raw DB feature vector with cyclical time-of-day + ordinal
    market-session position, computed at train/predict time (not stored),
    so the model can pick up session-dependent edge without a one-hot
    exploding the feature space on a small dataset."""
    hour = 12.0
    if timestamp:
        try:
            hour = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00")).hour
        except (ValueError, TypeError):
            pass
    hour_sin = math.sin(2 * math.pi * hour / 24.0)
    hour_cos = math.cos(2 * math.pi * hour / 24.0)
    session_ord = float(_ML_SESSION_ORDER.get(session, -1))
    return list(raw_vals) + [hour_sin, hour_cos, session_ord]


def _ml_build_matrix(rows: list, n_raw: int) -> tuple:
    """rows: (symbol, *raw_feature_vals, timestamp, session, win). Returns (X, y)
    with engineered columns appended."""
    X, y = [], []
    for r in rows:
        raw = [float(r[1 + i]) if r[1 + i] is not None else 0.0 for i in range(n_raw)]
        ts, sess, win = r[1 + n_raw], r[2 + n_raw], r[3 + n_raw]
        X.append(_ml_engineer(raw, ts, sess))
        y.append(win)
    return X, y


def _make_stacking_clf(random_state: int = 42) -> StackingClassifier:
    """The 'exotic' ensemble: 3 diverse base learners (boosting, bagging,
    extra-randomized trees) feeding a logistic-regression meta-learner that
    learns how to blend them, wrapped in probability calibration so the
    output is a genuine, trustworthy confidence — not just a raw vote."""
    base = [
        ("gb", GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=5, random_state=random_state)),
        ("rf", RandomForestClassifier(
            n_estimators=200, max_depth=6, class_weight="balanced",
            random_state=random_state, n_jobs=1)),
        ("et", ExtraTreesClassifier(
            n_estimators=200, max_depth=6, class_weight="balanced",
            random_state=random_state, n_jobs=1)),
    ]
    stack = StackingClassifier(
        estimators=base,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=3, stack_method="predict_proba", n_jobs=1, passthrough=False,
    )
    return CalibratedClassifierCV(stack, method="isotonic", cv=3)


def _make_gb_clf(random_state: int = 42) -> GradientBoostingClassifier:
    """Lightweight single-model fallback used when there isn't enough data
    for the full stacking ensemble.  Regularised to avoid over-fitting on
    small datasets; returns a plain GradientBoostingClassifier so that
    feature_importances_ is directly accessible after fitting."""
    return GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=5, random_state=random_state,
    )


def _walk_forward_eval(build_fn, X: list, y: list, weights) -> Optional[dict]:
    """Time-ordered (no shuffling) holdout: train on the first 80% chronologically,
    score on the last 20% never seen in training — an honest read on whether the
    model generalizes forward in time, not just fits the past."""
    n = len(X)
    split = int(n * 0.8)
    if split < 10 or (n - split) < 5:
        return None
    Xtr, Xte = X[:split], X[split:]
    ytr, yte = y[:split], y[split:]
    if len(set(ytr)) < 2 or len(set(yte)) < 2:
        return None
    try:
        m = build_fn()
        m.fit(Xtr, ytr, sample_weight=weights[:split]) if weights is not None else m.fit(Xtr, ytr)
        proba = m.predict_proba(Xte)[:, 1]
        preds = (proba >= 0.5).astype(int)
        return {
            "acc":   accuracy_score(yte, preds),
            "auc":   roc_auc_score(yte, proba),
            "brier": brier_score_loss(yte, proba),
            "n_test": len(yte),
        }
    except Exception as e:
        logger.warning(f"ML walk-forward eval failed: {e}")
        return None


def _ml_train():
    """Train ML models.
    Global model: a calibrated stacking ensemble (GradientBoosting + RandomForest +
    ExtraTrees -> logistic meta-learner -> isotonic calibration) once there's enough
    data per cross-val fold; otherwise a single regularized GradientBoosting so a
    handful of trades can't be over-fit into false confidence.
    Per-class: RandomForest (fast, good on small per-symbol-class datasets).
    Prefers signal_features table (16 rich + 3 engineered features); falls back to
    the trades table (7 legacy features) if signal_features is too sparse.
    """
    global ml_model, ml_trained_on, ml_training_active, ml_models_per_class, ml_trained_per_class
    try:
        # ── Choose training data source ──────────────────────────────────────
        try:
            sf_rows = _db_fetch(
                f"SELECT symbol, {', '.join(ML_FEATURE_COLS)}, timestamp, session, win "
                f"FROM signal_features WHERE win IN (0, 1) ORDER BY id"
            )
        except Exception:
            sf_rows = []

        if len(sf_rows) >= ML_MIN_TRADES:
            rows_sym     = sf_rows
            feature_cols = ML_FEATURE_COLS
            feat_src     = f"signal_features ({len(ML_FEATURE_COLS)}+{len(_ML_ENGINEERED_COLS)} features)"
        else:
            try:
                rows_sym = _db_fetch(
                    f"SELECT symbol, {', '.join(_ML_LEGACY_COLS)}, timestamp, market_session, win "
                    f"FROM trades WHERE win IN (0, 1) ORDER BY id"
                )
            except Exception as e:
                logger.error(f"ML training query failed: {e}")
                _send_tg(f"🤖 <b>ML Training Error</b>\n<code>{e}</code>\nWill retry on next trade.")
                return
            feature_cols = _ML_LEGACY_COLS
            feat_src     = f"trades ({len(_ML_LEGACY_COLS)}+{len(_ML_ENGINEERED_COLS)} features, legacy)"

        n_raw = len(feature_cols)
        total = len(rows_sym)

        _send_tg(
            f"🤖 <b>ML RETRAINING</b> — {feat_src}\n"
            f"Training on <b>{total}</b> trades…\n"
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

        X, y = _ml_build_matrix(rows_sym, n_raw)
        all_feature_cols = feature_cols + _ML_ENGINEERED_COLS

        if len(set(y)) < 2:
            _send_tg("🤖 <b>ML RETRAINING</b> — skipped\nNeed both wins AND losses in history.")
            return

        # ── Adaptive recency weights (exponential decay, half-life = 100 trades) ─
        half_life = 100.0
        weights   = np.exp(np.linspace(-total / half_life, 0.0, total))
        use_stack = total >= ML_STACKING_MIN_TRADES

        try:
            # ── Honest forward-looking holdout BEFORE fitting on everything ──
            eval_res = _walk_forward_eval(
                (_make_stacking_clf if use_stack else _make_gb_clf), X, y, weights,
            )

            # ── Global model: exotic stack once data supports it, else a single
            #    regularized GradientBoosting to avoid over-fitting on noise ──
            if use_stack:
                clf = _make_stacking_clf()
                clf.fit(X, y)   # CalibratedClassifierCV doesn't accept sample_weight cleanly through Stacking
                arch_label = "Stacked (GB+RF+ExtraTrees → LogReg) + isotonic calibration"
                imp_line = "n/a (stacked ensemble — see per-class importances below)"
            else:
                clf = _make_gb_clf()
                clf.fit(X, y, sample_weight=weights)
                arch_label = "GradientBoosting (single model — building toward stacking at " \
                             f"{ML_STACKING_MIN_TRADES} trades)"
                imp = sorted(zip(all_feature_cols, clf.feature_importances_),
                             key=lambda x: x[1], reverse=True)
                # Full ranked importance table sent to Telegram
                _imp_full = "\n".join(
                    [f"  {i+1:2d}. {n:<20s} {v*100:5.1f}%" for i, (n, v) in enumerate(imp)]
                )
                threading.Thread(
                    target=_send_tg,
                    args=(f"📊 <b>ML Feature Importance</b>\n<pre>{_imp_full}</pre>",),
                    daemon=True,
                ).start()
                imp_line = "  ".join([f"{n} {v*100:.0f}%" for n, v in imp[:4]])

            with ml_lock:
                ml_model     = clf
                ml_trained_on = total

            eval_line = ""
            if eval_res:
                eval_line = (
                    f"Holdout (last {eval_res['n_test']}, unseen in training): "
                    f"Acc {eval_res['acc']*100:.0f}%  ·  AUC {eval_res['auc']:.2f}  ·  "
                    f"Brier {eval_res['brier']:.3f}\n"
                )

            # ── Per-class models: RandomForest (fast, handles small datasets) ──
            new_cls_models  = {}
            new_cls_trained = {}
            cls_lines       = []
            for cls, syms in ML_SYMBOL_CLASSES.items():
                cls_rows = [r for r in rows_sym if r[0] in syms]
                if len(cls_rows) < 20:
                    cls_lines.append(f"  {cls}: only {len(cls_rows)} trades — skipped")
                    continue
                Xc, yc = _ml_build_matrix(cls_rows, n_raw)
                if len(set(yc)) < 2:
                    cls_lines.append(f"  {cls}: need both outcomes — skipped")
                    continue
                n_c = len(cls_rows)
                w_c = np.exp(np.linspace(-n_c / half_life, 0.0, n_c))
                clf_c = RandomForestClassifier(
                    n_estimators=150, max_depth=6, random_state=42,
                    class_weight="balanced", n_jobs=1,
                )
                clf_c.fit(Xc, yc, sample_weight=w_c)
                new_cls_models[cls]  = clf_c
                new_cls_trained[cls] = n_c
                wr_c = sum(yc) / n_c * 100
                cls_lines.append(f"  {cls}: {n_c} trades  {wr_c:.0f}%WR ✅")

            with ml_lock:
                ml_models_per_class.update(new_cls_models)
                ml_trained_per_class.update(new_cls_trained)

            _ml_save()
            _send_tg(
                f"🤖 <b>ML RETRAINING COMPLETE</b> ✅\n"
                f"<code>[{_ml_progress_bar(1.0)}] 100%</code>\n"
                f"Source: {feat_src}  |  <b>{total}</b> trades\n"
                f"Global: {arch_label}\n"
                f"Gate ≥<b>{ML_CONFIDENCE_MIN*100:.0f}%</b>\n"
                f"{eval_line}"
                f"Top features: {imp_line}\n"
                f"Per-class (RandomForest):\n" + "\n".join(cls_lines)
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
    # Use max() so an instant +1 increment from on_contract_update is never
    # overwritten by a stale DB count flushed milliseconds later.
    with ml_lock:
        ml_total_trades = max(ml_total_trades, total_trades)
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


def _ml_retrain_guard_loop():
    """Independent guard that forces a retrain every ML_RETRAIN_EVERY trades.
    Catches cases where the DB-writer path might be delayed or skipped."""
    while True:
        try:
            time.sleep(60)
            with ml_lock:
                trained, total, training = ml_trained_on, ml_total_trades, ml_training_active
            if training:
                continue
            if ml_model is None:
                # Model not ready yet — let DB writer bootstrap it first
                continue
            if total - trained >= ML_RETRAIN_EVERY:
                logger.info(f"ML retrain guard triggered: {total} trades, last trained on {trained}")
                _ml_maybe_retrain(total)
        except Exception as e:
            logger.error(f"ML retrain guard error: {e}")


def _ml_bootstrap_from_history():
    """
    Build initial ML training data from historical candle data.
    Simulates Rise/Fall outcomes: did the close AFTER DURATION candles
    close higher (CALL/UP) or lower (PUT/DOWN) than the entry close?
    This lets the ML filter work from the very first real trade.
    """
    global ml_model, ml_trained_on

    with ml_lock:
        if ml_model is not None:
            _send_tg(
                f"🤖 ML model already loaded (trained on {ml_trained_on} samples) — skipping bootstrap."
            )
            return

    _send_tg("🤖 <b>ML Bootstrap</b> — building training data from candle history (Rise/Fall)…")

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

        st_dir  = float(ind.get("supertrend_dir", 0))
        adx_v   = float(ind.get("adx") or 0)
        atr_ma  = float(ind.get("atr_ma") or atr)
        st_dist_f = abs(float(ind.get("supertrend_val") or 0) - float(df["Close"].iloc[-1])) / atr

        check_from = max(30, len(df) - 400)
        check_to   = len(df) - DURATION - 1
        if check_to <= check_from:
            continue

        for i in range(check_from, check_to):
            candle       = df.iloc[i]
            future_close = float(df.iloc[i + DURATION]["Close"])
            close        = float(candle["Close"])

            direction = "UP" if st_dir >= 0 else "DOWN"
            # Rise/Fall win: close higher for CALL, close lower for PUT
            if direction == "UP":
                won = future_close > close
            else:
                won = future_close < close

            hi   = float(candle["High"])
            lo   = float(candle["Low"])
            wick = (hi - lo) / (atr or 1)

            all_X.append([
                float(SCORE_THRESHOLD), wick, float(atr), atr_ma,
                st_dir, adx_v, st_dist_f,
            ])
            all_y.append(1 if won else 0)

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


def _ml_conf_bucket(conf: float) -> str:
    """Return the display bucket label for a raw ML probability 0.0–1.0."""
    pct = conf * 100
    if pct >= 95:   return "95-100"
    if pct >= 90:   return "90-94"
    if pct >= 85:   return "85-89"
    if pct >= 75:   return "75-84"
    return "<75"


def _ml_score_bucket(score: float) -> str:
    """Return the display bucket label for a raw signal score 0.0–100.0."""
    if score >= 95:   return "95-100"
    if score >= 90:   return "90-94"
    if score >= 85:   return "85-89"
    if score >= 75:   return "75-84"
    return "<75"


def _ml_confidence_perf_text() -> str:
    """ML Confidence + Signal Score performance — live in-memory + DB lifetime."""
    bucket_order = ["95-100", "90-94", "85-89", "75-84", "<75"]
    bar_w = 8

    def _render_section(title: str, subtitle: str, stats: dict, total: int) -> list:
        out = [
            f"<b>{title}</b>  ({total} trades)",
            f"<i>{subtitle}</i>",
            "",
        ]
        any_data = False
        for bucket in bucket_order:
            blabel = _html.escape(bucket)
            bst = stats.get(bucket)
            if bst and (bst["wins"] + bst["losses"]) > 0:
                cnt  = bst["wins"] + bst["losses"]
                wr   = bst["wins"] / cnt * 100
                sign = "+" if bst["pnl"] >= 0 else ""
                bar_f = max(0, min(bar_w, int(wr / 100 * bar_w)))
                bar   = "█" * bar_f + "░" * (bar_w - bar_f)
                out.append(
                    f"  <b>{blabel:>6}%</b>  {cnt:>3} trades  "
                    f"{bst['wins']}W/{bst['losses']}L  ({wr:.0f}%WR)  [{bar}]  "
                    f"<b>{sign}${bst['pnl']:.2f}</b>"
                )
                any_data = True
            else:
                out.append(f"  <b>{blabel:>6}%</b>  — no trades yet —")
        if not any_data:
            out.append("<i>No data in this section yet.</i>")
        return out

    # ── Section 1: Live ML confidence (actual model probability) ────────────
    with ml_lock:
        conf_snap = {k: dict(v) for k, v in _ml_conf_live_stats.items()}
    conf_total = sum(v["wins"] + v["losses"] for v in conf_snap.values())

    # ── Section 2: Live signal score (raw trigger score) ─────────────────────
    with ml_lock:
        score_snap = {k: dict(v) for k, v in _ml_score_live_stats.items()}
    score_total = sum(v["wins"] + v["losses"] for v in score_snap.values())

    # ── Section 3: Per-class model status ────────────────────────────────────
    with ml_lock:
        cls_status = {k: (v is not None) for k, v in ml_models_per_class.items()}
        cls_trained = dict(ml_trained_per_class)

    cls_lines = []
    for cls in ML_SYMBOL_CLASSES:
        ready = "✅" if cls_status.get(cls) else "⏳"
        cls_lines.append(f"  {ready} {cls:<14} {cls_trained.get(cls, 0)} trades")

    # ── Section 4: DB lifetime — score proxy ─────────────────────────────────
    rows = get_ml_confidence_buckets()
    bucket_data = {r[0]: r for r in rows} if rows else {}
    db_total = sum((r[1] or 0) for r in rows) if rows else 0

    lines = [
        "🤖 <b>ML Confidence Performance</b>",
        "━━━━━━━━━━━━━━━━━━━━",
    ]
    lines += _render_section(
        "Live Session — actual ML probability",
        "Real confidence from RandomForest (per-class where available)",
        conf_snap, conf_total,
    )
    lines += ["", "━━━━━━━━━━━━━━━━━━━━"]
    lines += _render_section(
        "Live Session — signal score",
        "Raw score gate that triggered the trade (independent of ML)",
        score_snap, score_total,
    )
    lines += [
        "",
        "━━━━━━━━━━━━━━━━━━━━",
        "<b>Per-Class ML Models</b>",
    ] + cls_lines + [
        "",
        "━━━━━━━━━━━━━━━━━━━━",
        f"<b>All-Time DB — signal score proxy</b>  ({db_total} trades)",
        "<i>Higher score = stronger signal; buckets mirror the score gate</i>",
        "",
    ]

    any_db = False
    for bucket in bucket_order:
        blabel = _html.escape(bucket)
        r = bucket_data.get(bucket)
        if r:
            _, cnt, wins, pnl = r[0], r[1], r[2], r[3]
            wins = wins or 0; pnl = pnl or 0.0; cnt = cnt or 0
            wr   = wins / cnt * 100 if cnt else 0
            sign = "+" if pnl >= 0 else ""
            bar_f = max(0, min(bar_w, int(wr / 100 * bar_w)))
            bar   = "█" * bar_f + "░" * (bar_w - bar_f)
            lines.append(
                f"  <b>{blabel:>6}%</b>  {cnt:>3} trades  "
                f"{wins}W ({wr:.0f}%WR)  [{bar}]  "
                f"<b>{sign}${pnl:.2f}</b>"
            )
            any_db = True
        else:
            lines.append(f"  <b>{blabel:>6}%</b>  — no trades yet —")

    if not any_db:
        lines.append("<i>No trades in DB yet.</i>")

    # Current session totals for context
    with _lock:
        wc, lc, pnl_s = win_count, loss_count, total_pnl
    tot = wc + lc
    wr_s = wc / tot * 100 if tot else 0
    lines += [
        "",
        "━━━━━━━━━━━━━━━━━━━━",
        f"Session: {tot} trades  {wc}W/{lc}L  {wr_s:.0f}%WR  "
        f"P&amp;L: {'+' if pnl_s >= 0 else ''}${pnl_s:.2f}",
        f"<i>ML gate: ≥{ML_CONFIDENCE_MIN*100:.0f}%  "
        f"(2nd entry: ≥{SECOND_ENTRY_ML_MIN*100:.0f}%)</i>",
    ]
    return "\n".join(lines)


def _ml_progress_text() -> str:
    with ml_lock:
        trained_on, model, training, total = ml_trained_on, ml_model, ml_training_active, ml_total_trades
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
#  PATTERN DISCOVERY  (fires every 100 settled trades)
# ══════════════════════════════════════════════════════════════════════
def _pattern_discovery():
    """Analyse signal_features for top/worst performing combinations and send to Telegram."""
    try:
        rows = _db_fetch(
            "SELECT ms_type, sd_zone, adx_val, candle_pattern, session, "
            "ms_bos, sd_fresh, score, ml_confidence, win, profit "
            "FROM signal_features WHERE win IN (0, 1) ORDER BY id"
        )
        if len(rows) < 50:
            return

        def adx_bucket(v):
            v = v or 0
            if v >= 30: return "ADX>30"
            if v >= 20: return "ADX20-30"
            if v >= 15: return "ADX15-20"
            return "ADX<15"

        def conf_bucket(v):
            if v is None: return "NoML"
            pct = v * 100
            if pct >= 85: return "ML≥85%"
            if pct >= 75: return "ML75-85%"
            return "ML<75%"

        combo_stats: dict = {}
        for r in rows:
            ms_type, sd_zone, adx_v, pat, sess, ms_bos, sd_fresh, score, conf, win, profit = r
            adx_b  = adx_bucket(adx_v)
            conf_b = conf_bucket(conf)
            key = f"{pat or 'none'} + {sd_zone or 'none'} + {adx_b} + {conf_b}"
            s   = combo_stats.setdefault(key, {"wins": 0, "total": 0, "pnl": 0.0})
            s["total"] += 1
            s["wins"]  += int(win)
            s["pnl"]   += float(profit or 0)

        # Only show combos with ≥10 trades (statistical significance)
        qualified = [(k, v) for k, v in combo_stats.items() if v["total"] >= 10]
        top  = sorted(qualified, key=lambda x: x[1]["wins"] / x[1]["total"], reverse=True)[:8]
        worst = sorted(qualified, key=lambda x: x[1]["wins"] / x[1]["total"])[:3]

        n_total  = len(rows)
        wr_all   = sum(r[-2] for r in rows) / n_total * 100 if n_total else 0
        pnl_all  = sum(float(r[-1] or 0) for r in rows)

        lines = [
            f"🔬 <b>PATTERN DISCOVERY</b>  ({n_total} trades  {wr_all:.1f}%WR  "
            f"{'+'if pnl_all>=0 else ''}${pnl_all:.2f})",
            "━━━━━━━━━━━━━━━━━━━━",
            "<b>🏆 Top Combinations:</b>",
        ]
        for key, v in top:
            wr  = v["wins"] / v["total"] * 100
            sgn = "+" if v["pnl"] >= 0 else ""
            lines.append(
                f"  <b>{wr:.0f}%WR</b> ({v['wins']}/{v['total']})  {sgn}${v['pnl']:.2f}\n"
                f"  ↳ {key}"
            )

        if worst:
            lines += ["", "<b>⚠️ Worst Combinations:</b>"]
            for key, v in worst:
                wr  = v["wins"] / v["total"] * 100
                sgn = "+" if v["pnl"] >= 0 else ""
                lines.append(
                    f"  <b>{wr:.0f}%WR</b> ({v['wins']}/{v['total']})  {sgn}${v['pnl']:.2f}\n"
                    f"  ↳ {key}"
                )

        lines.append("\n<i>Combinations need ≥10 trades. Tune signal gates accordingly.</i>")
        _send_tg("\n".join(lines))
        logger.info(f"Pattern discovery sent: {len(qualified)} combos analysed")
    except Exception as e:
        logger.error(f"_pattern_discovery error: {e}")


# ══════════════════════════════════════════════════════════════════════
#  PERFORMANCE ANALYTICS
# ══════════════════════════════════════════════════════════════════════
def _performance_analytics_text() -> str:
    """Win-rate breakdown by score, ADX, pattern, session, S&D distance, ML confidence."""
    try:
        rows = _db_fetch(
            "SELECT score, adx_val, candle_pattern, session, "
            "sd_dist, ml_confidence, win, profit "
            "FROM signal_features WHERE win IN (0, 1)"
        )
    except Exception as e:
        return f"⚠️ Analytics query failed: <code>{e}</code>"

    if len(rows) < 10:
        try:
            n_tr = _db_fetch("SELECT COUNT(*) FROM trades WHERE win IN (0,1)")
            n    = int(n_tr[0][0]) if n_tr else 0
        except Exception:
            n = 0
        return (
            f"📊 <b>Performance Analytics</b>\n"
            f"Signal features DB has no data yet ({n} trades in trades table).\n"
            f"Analytics available after first {ML_MIN_TRADES} trades are logged."
        )

    def bucket_analysis(rows, key_fn):
        buckets: dict = {}
        for r in rows:
            k = key_fn(r)
            s = buckets.setdefault(k, {"wins": 0, "total": 0, "pnl": 0.0})
            s["total"] += 1
            s["wins"]  += int(r[-2])
            s["pnl"]   += float(r[-1] or 0)
        out = []
        for k in sorted(buckets.keys()):
            v  = buckets[k]
            if v["total"] == 0:
                continue
            wr  = v["wins"] / v["total"] * 100
            sgn = "+" if v["pnl"] >= 0 else ""
            bar = "█" * min(10, int(wr / 10)) + "░" * max(0, 10 - int(wr / 10))
            out.append(f"  <b>{k:<14}</b>  {v['total']:>3}t  {wr:.0f}%WR [{bar}]  {sgn}${v['pnl']:.2f}")
        return out or ["  — no data —"]

    n     = len(rows)
    wr_all = sum(r[-2] for r in rows) / n * 100 if n else 0

    score_fn = lambda r: f"Score {int((r[0] or 0)//5*5)}-{int((r[0] or 0)//5*5+4)}"
    adx_fn   = lambda r: ("ADX>30" if (r[1] or 0)>=30 else "ADX20-30" if (r[1] or 0)>=20 else "ADX<20")
    pat_fn   = lambda r: (r[2] or "none")[:18]
    sess_fn  = lambda r: r[3] or "unknown"
    dist_fn  = lambda r: ("Dist<0.8ATR" if (r[4] or 99)<0.8 else "Dist0.8-1.5" if (r[4] or 99)<1.5 else "Dist>1.5ATR")
    conf_fn  = lambda r: ("ML≥85%" if (r[5] or 0)*100>=85 else "ML75-85%" if (r[5] or 0)*100>=75 else "ML<75%" if r[5] else "NoML")

    lines = [
        "📊 <b>Performance Analytics</b>",
        "━━━━━━━━━━━━━━━━━━━━",
        f"Total: <b>{n}</b> trades  |  Overall WR: <b>{wr_all:.1f}%</b>",
        "", "<b>By Score Range</b>",
    ] + bucket_analysis(rows, score_fn)

    lines += ["", "<b>By ADX</b>"] + bucket_analysis(rows, adx_fn)
    lines += ["", "<b>By Candle Pattern</b>"] + bucket_analysis(rows, pat_fn)
    lines += ["", "<b>By Session</b>"] + bucket_analysis(rows, sess_fn)
    lines += ["", "<b>By S&amp;D Zone Distance</b>"] + bucket_analysis(rows, dist_fn)
    lines += ["", "<b>By ML Confidence</b>"] + bucket_analysis(rows, conf_fn)
    return "\n".join(lines)


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
        # Sober Trading Book filters (core gates)
        "market_structure":    "sideways",  # bullish / bearish / sideways
        "market_struct_bos":   False,       # Break of Structure detected
        "market_struct_choch": False,       # Change of Character detected
        "market_struct_hh":    False,       # Higher High
        "market_struct_hl":    False,       # Higher Low
        "market_struct_lh":    False,       # Lower High
        "market_struct_ll":    False,       # Lower Low
        "sd_zone":             "none",      # demand / supply / none
        "sd_zone_dist":        99.0,        # distance to zone in ATR units
        "sd_zone_fresh":       False,       # zone untouched since formation
        "candle_pattern":      "none",      # pattern name
        "candle_pattern_bias": "neutral",   # bullish / bearish / neutral
        # Extra indicators for display
        "ema_fast":            None,        # EMA 50
        "roc":                 None,        # 10-period Rate of Change (%)
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

# ── Signal-features table (rich 39-column per-trade signal log) ──────
_CREATE_SF_SQLITE = """
    CREATE TABLE IF NOT EXISTS signal_features (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp        TEXT,     symbol           TEXT,
        direction        TEXT,     expiry_min        INTEGER,
        score            REAL,     grade            TEXT,
        ml_confidence    REAL,
        ms_pts           REAL,     ms_type          TEXT,
        ms_bos           INTEGER,  ms_choch         INTEGER,
        ms_hh            INTEGER,  ms_hl            INTEGER,
        ms_lh            INTEGER,  ms_ll            INTEGER,
        ema_fast         REAL,     ema_slow         REAL,
        ema_aligned      INTEGER,
        adx_val          REAL,     rsi_val          REAL,
        stochrsi_k_val   REAL,
        et_pts           REAL,
        mom_pts          REAL,     macd_bullish     INTEGER,
        macd_hist_rising INTEGER,  roc_val          REAL,
        sd_zone          TEXT,     sd_dist          REAL,
        sd_fresh         INTEGER,  sd_pts           REAL,
        cc_pts           REAL,     candle_pattern   TEXT,
        candle_bias      TEXT,
        atr_val          REAL,     atr_ma_val       REAL,
        wick_atr_ratio   REAL,
        session          TEXT,
        win              INTEGER,
        profit           REAL
    )
"""

_CREATE_SF_PG = """
    CREATE TABLE IF NOT EXISTS signal_features (
        id               SERIAL PRIMARY KEY,
        timestamp        TEXT,     symbol           TEXT,
        direction        TEXT,     expiry_min        INTEGER,
        score            REAL,     grade            TEXT,
        ml_confidence    REAL,
        ms_pts           REAL,     ms_type          TEXT,
        ms_bos           INTEGER,  ms_choch         INTEGER,
        ms_hh            INTEGER,  ms_hl            INTEGER,
        ms_lh            INTEGER,  ms_ll            INTEGER,
        ema_fast         REAL,     ema_slow         REAL,
        ema_aligned      INTEGER,
        adx_val          REAL,     rsi_val          REAL,
        stochrsi_k_val   REAL,
        et_pts           REAL,
        mom_pts          REAL,     macd_bullish     INTEGER,
        macd_hist_rising INTEGER,  roc_val          REAL,
        sd_zone          TEXT,     sd_dist          REAL,
        sd_fresh         INTEGER,  sd_pts           REAL,
        cc_pts           REAL,     candle_pattern   TEXT,
        candle_bias      TEXT,
        atr_val          REAL,     atr_ma_val       REAL,
        wick_atr_ratio   REAL,
        session          TEXT,
        win              INTEGER,
        profit           REAL
    )
"""

_SF_COLS = (
    "timestamp,symbol,direction,expiry_min,score,grade,ml_confidence,"
    "ms_pts,ms_type,ms_bos,ms_choch,ms_hh,ms_hl,ms_lh,ms_ll,"
    "ema_fast,ema_slow,ema_aligned,adx_val,rsi_val,stochrsi_k_val,"
    "et_pts,mom_pts,macd_bullish,macd_hist_rising,roc_val,"
    "sd_zone,sd_dist,sd_fresh,sd_pts,cc_pts,candle_pattern,candle_bias,"
    "atr_val,atr_ma_val,wick_atr_ratio,session,win,profit"
)
_SF_N = 39   # number of columns in signal_features (excluding id)


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


def _pg_safe_row(row):
    """Convert numpy scalar types (np.float64, np.int64, np.bool_, etc.) to native
    Python types before handing a row to psycopg2. Needed because psycopg2's numpy
    adapter can render e.g. np.float64(0.92) as literal unquoted SQL text
    ('np.float64(0.92)'), which Postgres then misparses as a schema reference."""
    safe = []
    for v in row:
        if isinstance(v, np.generic):
            v = v.item()
        safe.append(v)
    return tuple(safe)


def _db_writer():
    if USE_PG:
        conn = psycopg2.connect(DATABASE_URL)
        conn.autocommit = True
        cur  = conn.cursor()
        cur.execute(_CREATE_TABLE_PG)
        # Add market_session column if missing (older PG schema)
        cur.execute("""
            DO $$ BEGIN
                ALTER TABLE trades ADD COLUMN market_session TEXT;
            EXCEPTION WHEN duplicate_column THEN NULL;
            END $$;
        """)
        def _write(item):
            cur.execute(
                f"INSERT INTO trades ({_INSERT_COLS}) "
                f"VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                _pg_safe_row(item),
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
        conn.execute(_CREATE_SF_SQLITE)   # signal_features table
        conn.commit()

    # ── PostgreSQL path: write immediately (autocommit) ──────────────────
    if USE_PG:
        cur.execute(_CREATE_SF_PG)   # ensure signal_features table exists
        while True:
            item = db_queue.get()
            if item is None:
                break
            # Signal-features dict item
            if isinstance(item, dict) and item.get("type") == "sf":
                try:
                    cur.execute(
                        f"INSERT INTO signal_features ({_SF_COLS}) "
                        f"VALUES ({','.join(['%s'] * _SF_N)})",
                        _pg_safe_row(item["data"]),
                    )
                    cur.execute("SELECT COUNT(*) FROM signal_features WHERE win IN (0,1)")
                    n_sf = cur.fetchone()[0]
                    if n_sf > 0 and n_sf % 100 == 0:
                        threading.Thread(target=_pattern_discovery, daemon=True, name="PatDisc").start()
                except Exception as e:
                    logger.error(f"SF write error (PG): {e}")
                continue
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

        # Signal-features dict item — write immediately, no batching
        if isinstance(item, dict) and item.get("type") == "sf":
            try:
                conn.execute(
                    f"INSERT INTO signal_features ({_SF_COLS}) "
                    f"VALUES ({','.join(['?'] * _SF_N)})",
                    item["data"],
                )
                conn.commit()
                n_sf = conn.execute(
                    "SELECT COUNT(*) FROM signal_features WHERE win IN (0,1)"
                ).fetchone()[0]
                if n_sf > 0 and n_sf % 100 == 0:
                    threading.Thread(target=_pattern_discovery, daemon=True, name="PatDisc").start()
            except Exception as e:
                logger.error(f"SF write error (SQLite): {e}")
            continue

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
    # win=-1 rows are VOID/unresolved contracts (never confirmed win or loss) —
    # excluded from the count so win-rate isn't diluted by unresolved trades.
    rows = _db_fetch(
        "SELECT SUM(CASE WHEN win IN (0,1) THEN 1 ELSE 0 END), SUM(profit), "
        "SUM(CASE WHEN win=1 THEN 1 ELSE 0 END), "
        "SUM(CASE WHEN win=0 THEN 1 ELSE 0 END) FROM trades"
    )
    return rows[0] if rows else (0, 0, 0, 0)


def get_alltime_symbol_stats(limit=10):
    return _db_fetch(
        "SELECT symbol, SUM(CASE WHEN win IN (0,1) THEN 1 ELSE 0 END), "
        "SUM(CASE WHEN win=1 THEN 1 ELSE 0 END), SUM(profit) "
        "FROM trades WHERE win IN (0,1) GROUP BY symbol ORDER BY SUM(profit) DESC LIMIT ?", (limit,)
    )


def get_alltime_daily_stats(limit=7):
    # win=-1 rows are VOID/unresolved contracts — excluded so day totals and
    # win-rate aren't diluted by trades that never got a confirmed result.
    return _db_fetch(
        "SELECT date(timestamp) as day, SUM(CASE WHEN win IN (0,1) THEN 1 ELSE 0 END), "
        "SUM(CASE WHEN win=1 THEN 1 ELSE 0 END), SUM(profit) "
        "FROM trades WHERE win IN (0,1) GROUP BY day ORDER BY day DESC LIMIT ?", (limit,)
    )


def get_7day_full_breakdown():
    """Return per-day, per-session breakdown for the last 7 days."""
    if USE_PG:
        sql = (
            "SELECT DATE(timestamp), market_session, COUNT(*), "
            "SUM(CASE WHEN win=1 THEN 1 ELSE 0 END), SUM(profit) "
            "FROM trades "
            "WHERE DATE(timestamp) >= CURRENT_DATE - INTERVAL '7 days' AND win IN (0,1) "
            "GROUP BY DATE(timestamp), market_session "
            "ORDER BY DATE(timestamp) DESC, SUM(profit) DESC"
        )
    else:
        sql = (
            "SELECT date(timestamp), market_session, COUNT(*), "
            "SUM(CASE WHEN win=1 THEN 1 ELSE 0 END), SUM(profit) "
            "FROM trades "
            "WHERE date(timestamp) >= date('now', '-7 days') AND win IN (0,1) "
            "GROUP BY date(timestamp), market_session "
            "ORDER BY date(timestamp) DESC, SUM(profit) DESC"
        )
    return _db_fetch(sql)


def get_session_alltime_stats():
    """Return per-market-session lifetime stats."""
    return _db_fetch(
        "SELECT market_session, COUNT(*), "
        "SUM(CASE WHEN win=1 THEN 1 ELSE 0 END), SUM(profit) "
        "FROM trades WHERE market_session IS NOT NULL AND win IN (0,1) "
        "GROUP BY market_session ORDER BY SUM(profit) DESC"
    )


def get_ml_confidence_buckets():
    """Return P&L and win-rate grouped by ML confidence buckets (from score column proxy)."""
    # Confidence isn't stored directly in DB, but we can approximate using score
    # and return lifetime performance segmented by score ranges.
    # Buckets: 95-100, 90-94, 85-89, 75-84, <75
    rows = _db_fetch(
        "SELECT "
        "  CASE "
        "    WHEN score >= 95 THEN '95-100' "
        "    WHEN score >= 90 THEN '90-94' "
        "    WHEN score >= 85 THEN '85-89' "
        "    WHEN score >= 75 THEN '75-84' "
        "    ELSE '<75' "
        "  END as bucket, "
        "  COUNT(*) as trades, "
        "  SUM(CASE WHEN win=1 THEN 1 ELSE 0 END) as wins, "
        "  SUM(profit) as pnl "
        "FROM trades WHERE win IN (0,1) "
        "GROUP BY bucket ORDER BY MIN(score) DESC"
    )
    return rows


def _compute_advanced_stats(rows) -> dict:
    """
    Shared analytics engine — takes an ordered list of (profit, win, symbol,
    market_session) rows (oldest→newest) and derives profit-factor style
    trading metrics that a single SQL aggregate can't express cleanly
    (streaks, equity-curve-based drawdown/recovery factor, etc).
    """
    if not rows:
        return None
    profits = [float(r[0] or 0.0) for r in rows]
    wins    = [int(r[1]) for r in rows]

    gross_profit = sum(p for p in profits if p > 0)
    gross_loss   = sum(p for p in profits if p < 0)   # negative number
    net_pnl      = sum(profits)
    total        = len(profits)
    win_profits  = [p for p, w in zip(profits, wins) if w == 1]
    loss_profits = [p for p, w in zip(profits, wins) if w == 0]
    n_wins, n_losses = len(win_profits), len(loss_profits)

    avg_win   = (sum(win_profits) / n_wins) if n_wins else 0.0
    avg_loss  = (sum(loss_profits) / n_losses) if n_losses else 0.0
    largest_win  = max(win_profits) if win_profits else 0.0
    largest_loss = min(loss_profits) if loss_profits else 0.0
    profit_factor = (gross_profit / abs(gross_loss)) if gross_loss != 0 else float("inf")
    expectancy    = net_pnl / total if total else 0.0

    # Equity curve → peak / max drawdown / recovery factor
    equity = 0.0
    peak   = 0.0
    max_dd = 0.0
    for p in profits:
        equity += p
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    recovery_factor = (net_pnl / max_dd) if max_dd > 0 else float("inf")

    # Longest win / loss streaks
    longest_win_streak = longest_loss_streak = cur_win = cur_loss = 0
    for w in wins:
        if w == 1:
            cur_win += 1; cur_loss = 0
            longest_win_streak = max(longest_win_streak, cur_win)
        else:
            cur_loss += 1; cur_win = 0
            longest_loss_streak = max(longest_loss_streak, cur_loss)

    # Best/worst symbol and market session (if present in the rows)
    by_symbol: dict = {}
    by_session: dict = {}
    for r in rows:
        profit = float(r[0] or 0.0)
        if len(r) > 2 and r[2]:
            by_symbol.setdefault(r[2], 0.0)
            by_symbol[r[2]] += profit
        if len(r) > 3 and r[3]:
            by_session.setdefault(r[3], 0.0)
            by_session[r[3]] += profit
    best_symbol  = max(by_symbol, key=by_symbol.get) if by_symbol else None
    worst_symbol = min(by_symbol, key=by_symbol.get) if by_symbol else None
    best_session  = max(by_session, key=by_session.get) if by_session else None
    worst_session = min(by_session, key=by_session.get) if by_session else None

    return {
        "total": total, "wins": n_wins, "losses": n_losses,
        "net_pnl": net_pnl, "gross_profit": gross_profit, "gross_loss": gross_loss,
        "avg_win": avg_win, "avg_loss": avg_loss,
        "largest_win": largest_win, "largest_loss": largest_loss,
        "profit_factor": profit_factor, "expectancy": expectancy,
        "max_drawdown": max_dd, "peak_equity": peak, "recovery_factor": recovery_factor,
        "longest_win_streak": longest_win_streak, "longest_loss_streak": longest_loss_streak,
        "best_symbol": best_symbol, "worst_symbol": worst_symbol,
        "best_symbol_pnl": by_symbol.get(best_symbol, 0.0) if best_symbol else 0.0,
        "worst_symbol_pnl": by_symbol.get(worst_symbol, 0.0) if worst_symbol else 0.0,
        "best_session": best_session, "worst_session": worst_session,
        "best_session_pnl": by_session.get(best_session, 0.0) if best_session else 0.0,
        "worst_session_pnl": by_session.get(worst_session, 0.0) if worst_session else 0.0,
    }


def get_alltime_advanced_stats() -> Optional[dict]:
    """All-time trading metrics derived from every confirmed trade in the DB."""
    rows = _db_fetch(
        "SELECT profit, win, symbol, market_session FROM trades "
        "WHERE win IN (0,1) ORDER BY id ASC"
    )
    return _compute_advanced_stats(rows)


def get_today_advanced_stats() -> Optional[dict]:
    """Today's (UTC) trading metrics derived from confirmed trades in the DB."""
    if USE_PG:
        sql = ("SELECT profit, win, symbol, market_session FROM trades "
               "WHERE DATE(timestamp) = CURRENT_DATE AND win IN (0,1) ORDER BY id ASC")
    else:
        sql = ("SELECT profit, win, symbol, market_session FROM trades "
               "WHERE date(timestamp) = date('now') AND win IN (0,1) ORDER BY id ASC")
    rows = _db_fetch(sql)
    return _compute_advanced_stats(rows)


def _strategy_header() -> str:
    """Config fingerprint shown at the top of every analytical report so
    results can be compared apples-to-apples across strategy revisions."""
    return (
        f"🧬 <b>Strategy {STRATEGY_VERSION}</b>  ·  Expiry {DURATION}m  ·  "
        f"Score ≥{SCORE_THRESHOLD}  ·  ML ≥{ML_CONFIDENCE_MIN*100:.0f}%  ·  "
        f"Rise/Fall  ·  Martingale {'ON' if MARTINGALE_ENABLED else 'OFF'}\n"
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
    # EMA Fast (50) for trend display
    df["EMA_FAST"] = df["Close"].ewm(span=50, adjust=False).mean()
    # Rate of Change (10-period) for momentum display
    df["ROC"] = df["Close"].pct_change(10) * 100

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
        ind["ema_fast"]         = df["EMA_FAST"].iloc[-1]   # EMA 50
        roc_val = df["ROC"].iloc[-1]
        ind["roc"]              = round(float(roc_val), 4) if pd.notna(roc_val) else None
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

        # ── Sober Trading Book: Market Structure ─────────────────────────
        ms = _detect_market_structure(df)
        ind["market_structure"]    = ms["type"]
        ind["market_struct_bos"]   = ms["bos"]
        ind["market_struct_choch"] = ms["choch"]
        ind["market_struct_hh"]    = ms.get("hh", False)
        ind["market_struct_hl"]    = ms.get("hl", False)
        ind["market_struct_lh"]    = ms.get("lh", False)
        ind["market_struct_ll"]    = ms.get("ll", False)

        # ── Sober Trading Book: Supply & Demand Zones ─────────────────
        atr_for_sd = float(ind.get("atr") or 1.0)
        sdz = _detect_sd_zones(df, atr_for_sd)
        ind["sd_zone"]       = sdz["nearest_zone"]
        ind["sd_zone_dist"]  = sdz["distance_atr"]
        ind["sd_zone_fresh"] = sdz["zone_fresh"]

        # ── Sober Trading Book: Candlestick Pattern ───────────────────
        pat = _detect_candle_pattern(df)
        ind["candle_pattern"]      = pat["name"]
        ind["candle_pattern_bias"] = pat["bias"]

        ind["ready"] = True
    return True


# ══════════════════════════════════════════════════════════════════════
#  MARKET STRUCTURE DETECTION  (Sober Trading Book)
# ══════════════════════════════════════════════════════════════════════
def _detect_market_structure(df: pd.DataFrame) -> dict:
    """
    Identify swing highs/lows and classify market structure.
    Sober Trading Book: HH+HL = bullish, LH+LL = bearish, EH+EL = sideways.
    Also detects Break of Structure (BOS) and Change of Character (CHoCH).
    """
    try:
        if len(df) < 20:
            return {"type": "sideways", "bos": False, "choch": False}
        n = min(len(df), 80)
        highs  = df["High"].values[-n:].astype(float)
        lows   = df["Low"].values[-n:].astype(float)
        closes = df["Close"].values[-n:].astype(float)

        lb = 5   # swing pivot lookback
        swing_highs, swing_lows = [], []
        for i in range(lb, len(highs) - lb):
            if all(highs[i] > highs[i - j] for j in range(1, lb + 1)) and \
               all(highs[i] > highs[i + j] for j in range(1, lb + 1)):
                swing_highs.append((i, float(highs[i])))
            if all(lows[i] < lows[i - j] for j in range(1, lb + 1)) and \
               all(lows[i] < lows[i + j] for j in range(1, lb + 1)):
                swing_lows.append((i, float(lows[i])))

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return {"type": "sideways", "bos": False, "choch": False}

        sh1, sh2 = swing_highs[-2], swing_highs[-1]   # older → newer
        sl1, sl2 = swing_lows[-2],  swing_lows[-1]

        hh = sh2[1] > sh1[1]   # higher high
        hl = sl2[1] > sl1[1]   # higher low
        lh = sh2[1] < sh1[1]   # lower high
        ll = sl2[1] < sl1[1]   # lower low

        if hh and hl:
            structure = "bullish"
        elif lh and ll:
            structure = "bearish"
        else:
            structure = "sideways"

        # Break of Structure: price breaks the most recent swing extreme
        last_close = float(closes[-1])
        bos = bool(
            (structure == "bullish" and last_close > sh2[1]) or
            (structure == "bearish" and last_close < sl2[1])
        )

        # Change of Character: existing trend forms opposing extreme
        choch = bool(
            (structure == "bullish" and ll) or
            (structure == "bearish" and hh)
        )

        return {"type": structure, "bos": bos, "choch": choch,
                "hh": bool(hh), "hl": bool(hl), "lh": bool(lh), "ll": bool(ll)}
    except Exception:
        return {"type": "sideways", "bos": False, "choch": False,
                "hh": False, "hl": False, "lh": False, "ll": False}


# ══════════════════════════════════════════════════════════════════════
#  SUPPLY & DEMAND ZONE DETECTION  (Sober Trading Book)
# ══════════════════════════════════════════════════════════════════════
def _detect_sd_zones(df: pd.DataFrame, atr: float) -> dict:
    """
    Detect Supply & Demand zones:
      Drop-Base-Rally → Demand zone (bullish reversal)
      Rally-Base-Drop → Supply zone (bearish reversal)
      Rally-Base-Rally → Demand continuation
      Drop-Base-Drop   → Supply continuation
    Returns nearest zone type + ATR-normalised distance from current price.
    """
    try:
        if len(df) < 30 or atr <= 0:
            return {"nearest_zone": "none", "distance_atr": 99.0, "zone_fresh": False}

        n = min(len(df), 120)
        opens  = df["Open"].values[-n:].astype(float)
        highs  = df["High"].values[-n:].astype(float)
        lows   = df["Low"].values[-n:].astype(float)
        closes = df["Close"].values[-n:].astype(float)
        current_price = float(closes[-1])

        demand_zones = []
        supply_zones = []

        for i in range(4, n - 4):
            body = abs(closes[i] - opens[i])
            if body < 1.2 * atr:   # must be a significant candle
                continue
            bullish_move = closes[i] > opens[i]

            # Identify the base (consolidation) before this big candle
            base_end = i - 1
            base_start = base_end
            while base_start > 0:
                b = abs(closes[base_start] - opens[base_start])
                if b > 0.8 * atr:
                    break
                base_start -= 1
            if base_end <= base_start:
                base_start = max(0, i - 3)   # fallback: last 3 bars

            if bullish_move:
                zone_top    = float(np.max(closes[base_start:base_end + 1]))
                zone_bottom = float(np.min(lows[base_start:base_end + 1]))
                demand_zones.append({
                    "top": zone_top, "bottom": zone_bottom,
                    "bar": i, "fresh": True,
                })
            else:
                zone_top    = float(np.max(highs[base_start:base_end + 1]))
                zone_bottom = float(np.min(closes[base_start:base_end + 1]))
                supply_zones.append({
                    "top": zone_top, "bottom": zone_bottom,
                    "bar": i, "fresh": True,
                })

        # Mark stale: zone was tested (price passed through) after formation
        for z in demand_zones:
            for j in range(z["bar"] + 1, n):
                if lows[j] < z["bottom"]:
                    z["fresh"] = False; break
        for z in supply_zones:
            for j in range(z["bar"] + 1, n):
                if highs[j] > z["top"]:
                    z["fresh"] = False; break

        # Find nearest fresh zone above/below current price
        best_demand = None; best_demand_dist = float("inf")
        best_supply = None; best_supply_dist = float("inf")

        for z in demand_zones:
            if not z["fresh"]: continue
            if current_price > z["bottom"]:   # price above demand zone
                dist = (current_price - z["top"]) / atr
                if 0 <= dist < best_demand_dist:
                    best_demand_dist = dist; best_demand = z

        for z in supply_zones:
            if not z["fresh"]: continue
            if current_price < z["top"]:      # price below supply zone
                dist = (z["bottom"] - current_price) / atr
                if 0 <= dist < best_supply_dist:
                    best_supply_dist = dist; best_supply = z

        if best_demand is not None and (best_supply is None or best_demand_dist <= best_supply_dist):
            return {"nearest_zone": "demand", "distance_atr": round(best_demand_dist, 2), "zone_fresh": True}
        elif best_supply is not None:
            return {"nearest_zone": "supply", "distance_atr": round(best_supply_dist, 2), "zone_fresh": True}
        return {"nearest_zone": "none", "distance_atr": 99.0, "zone_fresh": False}
    except Exception:
        return {"nearest_zone": "none", "distance_atr": 99.0, "zone_fresh": False}


# ══════════════════════════════════════════════════════════════════════
#  CANDLESTICK PATTERN DETECTION  (Sober Trading Book)
# ══════════════════════════════════════════════════════════════════════
def _detect_candle_pattern(df: pd.DataFrame) -> dict:
    """
    Detect Sober Trading Book candlestick patterns from the last 3 candles.
    Bullish: Hammer, Inverted Hammer, Bullish Engulfing, Piercing Line,
             Morning Star, Three White Soldiers.
    Bearish: Hanging Man, Shooting Star, Bearish Engulfing, Evening Star,
             Three Black Crows.
    Neutral: Doji, Spinning Top.
    Returns {"name": str, "bias": "bullish"/"bearish"/"neutral"}.
    """
    try:
        if len(df) < 3:
            return {"name": "none", "bias": "neutral"}
        rows   = df.tail(3)
        opens  = rows["Open"].values.astype(float)
        closes = rows["Close"].values.astype(float)
        highs  = rows["High"].values.astype(float)
        lows   = rows["Low"].values.astype(float)

        o1, o2, o3 = opens[0],  opens[1],  opens[2]
        c1, c2, c3 = closes[0], closes[1], closes[2]
        h1, h2, h3 = highs[0],  highs[1],  highs[2]
        l1, l2, l3 = lows[0],   lows[1],   lows[2]

        def body(o, c):     return abs(c - o)
        def upper_wick(o, c, h): return float(h - max(o, c))
        def lower_wick(o, c, l): return float(min(o, c) - l)
        def rng(h, l):      return float(h - l) if h > l else 1e-10
        def is_bull(o, c):  return c > o
        def is_bear(o, c):  return c < o

        b3 = body(o3, c3); r3 = rng(h3, l3)
        uw3 = upper_wick(o3, c3, h3)
        lw3 = lower_wick(o3, c3, l3)
        b2 = body(o2, c2);  b1 = body(o1, c1)

        downtrend = is_bear(o1, c1) or is_bear(o2, c2)
        uptrend   = is_bull(o1, c1) or is_bull(o2, c2)

        # ── Hammer (bullish reversal after downtrend) ────────────────
        if downtrend and b3 > 0 and lw3 >= 2.0 * b3 and uw3 < 0.3 * r3 and b3 < 0.4 * r3:
            return {"name": "Hammer 🔨", "bias": "bullish"}

        # ── Inverted Hammer (bullish reversal) ───────────────────────
        if downtrend and b3 > 0 and uw3 >= 2.0 * b3 and lw3 < 0.3 * r3 and b3 < 0.4 * r3 and is_bull(o3, c3):
            return {"name": "Inverted Hammer", "bias": "bullish"}

        # ── Shooting Star (bearish reversal after uptrend) ───────────
        if uptrend and b3 > 0 and uw3 >= 2.0 * b3 and lw3 < 0.3 * r3 and b3 < 0.4 * r3:
            return {"name": "Shooting Star ⭐", "bias": "bearish"}

        # ── Hanging Man (bearish reversal after uptrend) ─────────────
        if uptrend and b3 > 0 and lw3 >= 2.0 * b3 and uw3 < 0.3 * r3 and b3 < 0.4 * r3 and is_bear(o3, c3):
            return {"name": "Hanging Man", "bias": "bearish"}

        # ── Bullish Engulfing ────────────────────────────────────────
        if is_bear(o2, c2) and is_bull(o3, c3) and o3 <= c2 and c3 >= o2 and b3 > b2:
            return {"name": "Bullish Engulfing 📈", "bias": "bullish"}

        # ── Bearish Engulfing ────────────────────────────────────────
        if is_bull(o2, c2) and is_bear(o3, c3) and o3 >= c2 and c3 <= o2 and b3 > b2:
            return {"name": "Bearish Engulfing 📉", "bias": "bearish"}

        # ── Piercing Line (bullish) ──────────────────────────────────
        if is_bear(o2, c2) and is_bull(o3, c3) and o3 < c2 and c3 > (o2 + c2) / 2 and c3 < o2:
            return {"name": "Piercing Line", "bias": "bullish"}

        # ── Morning Star (3-candle bullish reversal) ─────────────────
        if is_bear(o1, c1) and b2 < 0.4 * b1 and is_bull(o3, c3) and c3 > (o1 + c1) / 2:
            return {"name": "Morning Star 🌟", "bias": "bullish"}

        # ── Evening Star (3-candle bearish reversal) ─────────────────
        if is_bull(o1, c1) and b2 < 0.4 * b1 and is_bear(o3, c3) and c3 < (o1 + c1) / 2:
            return {"name": "Evening Star 🌆", "bias": "bearish"}

        # ── Three White Soldiers (bullish continuation) ──────────────
        if (is_bull(o1,c1) and is_bull(o2,c2) and is_bull(o3,c3)
                and c2 > c1 and c3 > c2 and o2 > o1 and o3 > o2):
            return {"name": "Three White Soldiers 🪖", "bias": "bullish"}

        # ── Three Black Crows (bearish continuation) ──────────────────
        if (is_bear(o1,c1) and is_bear(o2,c2) and is_bear(o3,c3)
                and c2 < c1 and c3 < c2 and o2 < o1 and o3 < o2):
            return {"name": "Three Black Crows 🐦", "bias": "bearish"}

        # ── Doji (indecision) ────────────────────────────────────────
        if b3 < 0.05 * r3:
            return {"name": "Doji ✚", "bias": "neutral"}

        # ── Spinning Top (indecision) ────────────────────────────────
        if b3 < 0.35 * r3 and uw3 > 0.2 * r3 and lw3 > 0.2 * r3:
            return {"name": "Spinning Top ⬆️⬇️", "bias": "neutral"}

        return {"name": "none", "bias": "neutral"}
    except Exception:
        return {"name": "none", "bias": "neutral"}


# ══════════════════════════════════════════════════════════════════════
#  SIGNAL SCORING  (Supertrend-based)
# ══════════════════════════════════════════════════════════════════════
def score_signal(symbol: str, candle: dict) -> tuple:
    """
    Calibrated 100-point scoring system.

    Component breakdown:
      1. Market Structure   — 25 pts  (HH+HL/LH+LL, BOS, CHoCH)
      2. Trend Strength     — 20 pts  (EMA50/200, slope, ADX)
      3. Supply/Demand Zone — 20 pts  (freshness, proximity, wick, retests)
      4. Entry Timing       — 15 pts  (pullback depth, wick, RSI/StochRSI)
      5. Momentum           — 10 pts  (MACD, ROC, ATR expansion)
      6. Candle Confirm     — 10 pts  (engulfing, pin-bar, body quality)

    Grades: A+ ≥ 95 · A ≥ 90 · B+ ≥ 85 · B ≥ 75 · NO TRADE < 75
    """
    with _lock:
        ind = dict(indicators[symbol])
    score, details = 0, {}

    price    = float(candle["Close"])
    ema_slow = ind.get("ema_slow") or price       # EMA200 (long-term trend)
    ema_fast = ind.get("ema_fast") or price       # EMA50  (medium trend)
    st_dir   = ind.get("supertrend_dir", 0)
    st_val   = ind.get("supertrend_val")
    atr      = max(ind.get("atr") or 1.0, 1e-9)  # guard division by zero

    # ── Direction: Supertrend primary, EMA200 fallback ────────────────
    if st_dir != 0:
        direction = "UP" if st_dir == 1 else "DOWN"
    else:
        direction = "UP" if price > ema_slow else "DOWN"

    # ══════════════════════════════════════════════════════════════════
    # 1. MARKET STRUCTURE  (max 25 pts, floor 0)
    # ══════════════════════════════════════════════════════════════════
    ms_type  = ind.get("market_structure", "sideways")
    ms_bos   = ind.get("market_struct_bos", False)
    ms_choch = ind.get("market_struct_choch", False)
    ms_hh    = ind.get("market_struct_hh", False)
    ms_hl    = ind.get("market_struct_hl", False)
    ms_lh    = ind.get("market_struct_lh", False)
    ms_ll    = ind.get("market_struct_ll", False)

    ms_pts = 0
    # Full swing structure: HH+HL for UP, LH+LL for DOWN → 20 pts
    if   direction == "UP"   and ms_hh and ms_hl: ms_pts += 20
    elif direction == "DOWN" and ms_lh and ms_ll: ms_pts += 20
    elif direction == "UP"   and (ms_hh or ms_hl): ms_pts += 10   # partial
    elif direction == "DOWN" and (ms_lh or ms_ll): ms_pts += 10
    # BOS in direction → +5
    if ms_bos and (
        (direction == "UP"   and ms_type == "bullish") or
        (direction == "DOWN" and ms_type == "bearish")
    ):
        ms_pts += 5
    # CHoCH against direction → bearish reversal signal against our trade → −10
    if ms_choch and not (
        (direction == "UP"   and ms_type == "bullish") or
        (direction == "DOWN" and ms_type == "bearish")
    ):
        ms_pts -= 10
    ms_pts = max(0, ms_pts)
    score  += ms_pts

    details.update({
        "market_structure":    ms_type,
        "market_struct_pts":   ms_pts,
        "market_struct_bos":   ms_bos,
        "market_struct_choch": ms_choch,
        "market_struct_hh":    ms_hh,
        "market_struct_hl":    ms_hl,
        "market_struct_lh":    ms_lh,
        "market_struct_ll":    ms_ll,
    })

    # ══════════════════════════════════════════════════════════════════
    # 2. TREND STRENGTH  (max 20, can go negative)
    # ══════════════════════════════════════════════════════════════════
    adx_val  = float(ind.get("adx") or 0.0)
    di_bull  = bool(ind.get("di_bullish", False))

    tr_pts = 0
    # EMA50 vs EMA200 alignment → +8
    ema_aligned = (
        (direction == "UP"   and ema_fast > ema_slow) or
        (direction == "DOWN" and ema_fast < ema_slow)
    )
    if ema_aligned: tr_pts += 8

    # EMA slope proxy: Supertrend aligns with EMA alignment → +4
    st_aligned = (
        (direction == "UP"   and st_dir == 1) or
        (direction == "DOWN" and st_dir == -1)
    )
    if st_aligned: tr_pts += 4

    # ADX trend strength — banded scoring (very high ADX can mean exhaustion)
    if   adx_val >= 45: tr_pts += 6    # very strong but possibly extended
    elif adx_val >= 35: tr_pts += 8    # sweet spot — strong, not exhausted
    elif adx_val >= 25: tr_pts += 5    # moderate trend
    elif adx_val >= 20: tr_pts += 3    # mild trend forming
    elif adx_val >= 15: tr_pts += 0    # borderline — no bonus, no penalty
    else:               tr_pts -= 5    # weak/choppy — penalise

    score += tr_pts
    details.update({
        "trend":       tr_pts,
        "adx":         adx_val,
        "ema_aligned": ema_aligned,
        "di_bullish":  di_bull,
    })

    # ══════════════════════════════════════════════════════════════════
    # 3. SUPPLY & DEMAND ZONE  (max 20 pts)
    # ══════════════════════════════════════════════════════════════════
    sd_zone   = ind.get("sd_zone", "none") or "none"
    sd_dist   = float(ind.get("sd_zone_dist") or 99.0)
    sd_fresh  = bool(ind.get("sd_zone_fresh", False))
    sd_tests  = int(ind.get("sd_zone_tests") or 0)
    uw        = float(ind.get("upper_wick_atr") or 0.0)
    lw        = float(ind.get("lower_wick_atr") or 0.0)

    sd_zone_agree = (
        (direction == "UP"   and sd_zone == "demand") or
        (direction == "DOWN" and sd_zone == "supply")
    )
    sd_pts = 0
    if sd_zone != "none" and sd_zone_agree:
        # Freshness tiers — zone loses strength with each retest
        if sd_fresh:
            sd_pts += 12                              # untouched: maximum strength
        elif sd_tests <= 1:
            sd_pts += 4                               # first retest: still valid
        elif sd_tests == 2:
            sd_pts += 1                               # second retest: weakening
        else:
            sd_pts -= 2                               # third+ retest: over-tested
        if   sd_dist <= 0.5:  sd_pts += 5            # price AT zone
        elif sd_dist <= 1.5:  sd_pts += 3            # approaching
        elif sd_dist <= 3.0:  sd_pts += 1            # within range
        # Strong rejection wick at zone → confirms reaction
        if (direction == "UP" and lw >= 0.5) or (direction == "DOWN" and uw >= 0.5):
            sd_pts += 7

    score += sd_pts
    details.update({
        "sd_zone":       sd_zone,
        "sd_zone_dist":  round(sd_dist, 2),
        "sd_zone_fresh": sd_fresh,
        "sd_zone_pts":   sd_pts,
        "sd_zone_agree": sd_zone_agree,
    })

    # ══════════════════════════════════════════════════════════════════
    # 4. ENTRY TIMING  (max 15 pts)
    # ══════════════════════════════════════════════════════════════════
    extension = abs(price - (st_val or price)) / atr
    rsi       = ind.get("rsi")
    srsi_k    = ind.get("stochrsi_k")
    body_r    = float(ind.get("body_ratio") or 0.0)

    et_pts = 0
    # Pullback depth: price pulling back to ST/EMA rather than over-extended
    if   extension <= 1.0: et_pts += 5
    elif extension <= 2.0: et_pts += 2
    # > 2.0: over-extended entry → 0 pts

    # Rejection wick (buyers/sellers defending zone)
    wick_pts = 0
    if direction == "UP":
        if   lw >= 0.5:   wick_pts = 5
        elif lw >= 0.25:  wick_pts = 2
    else:
        if   uw >= 0.5:   wick_pts = 5
        elif uw >= 0.25:  wick_pts = 2
    et_pts += wick_pts

    # RSI / StochRSI zone score (5 pts max, partial for neutral zones)
    osc_pts = 0
    if srsi_k is not None:
        if direction == "UP":
            if   srsi_k < 25: osc_pts = 5   # oversold — prime
            elif srsi_k < 45: osc_pts = 3   # neutral-bearish — ok
            elif srsi_k < 65: osc_pts = 1   # mid — caution
            # ≥65: overbought for UP → 0
        else:
            if   srsi_k > 75: osc_pts = 5   # overbought — prime
            elif srsi_k > 55: osc_pts = 3   # neutral-bullish — ok
            elif srsi_k > 35: osc_pts = 1   # mid — caution
            # ≤35: oversold for DOWN → 0
    elif rsi is not None:
        if direction == "UP":
            if 35 <= rsi <= 60:                            osc_pts = 5
            elif 25 <= rsi < 35 or 60 < rsi <= 72:        osc_pts = 3
        else:
            if 40 <= rsi <= 65:                            osc_pts = 5
            elif 28 <= rsi < 40 or 65 < rsi <= 75:        osc_pts = 3
    et_pts += osc_pts

    score += et_pts
    details.update({
        "entry_quality": et_pts,
        "extension_atr": round(extension, 2),
        "wick_pts":      wick_pts,
        "rsi":           round(rsi,    1) if rsi    is not None else None,
        "stochrsi_k":    round(srsi_k, 1) if srsi_k is not None else None,
        "body_ratio":    round(body_r,  2),
    })

    # ══════════════════════════════════════════════════════════════════
    # 5. MOMENTUM  (max 10 pts)
    # ══════════════════════════════════════════════════════════════════
    macd_bullish    = bool(ind.get("macd_bullish", False))
    macd_hist_rising = bool(ind.get("macd_hist_rising", False))
    roc              = ind.get("roc")

    mom_pts = 0
    # MACD: direction + histogram momentum
    macd_ok = (
        (direction == "UP"   and macd_bullish     and macd_hist_rising) or
        (direction == "DOWN" and not macd_bullish and macd_hist_rising)
    )
    if macd_ok: mom_pts += 5

    # ROC direction
    roc_ok = (
        (direction == "UP"   and roc is not None and roc > 0) or
        (direction == "DOWN" and roc is not None and roc < 0)
    )
    if roc_ok: mom_pts += 3

    # Volume expansion proxy: ATR expanding (Deriv synthetics have no real volume)
    if ind.get("atr_rising") and (ind.get("atr") or 0) > (ind.get("atr_ma") or 0):
        mom_pts += 2

    score += mom_pts
    details.update({
        "momentum":          mom_pts,
        "volatility":        mom_pts,   # alias for signal-card compat
        "macd_bullish":      macd_bullish,
        "macd_hist_rising":  macd_hist_rising,
        "roc":               roc,
    })

    # ══════════════════════════════════════════════════════════════════
    # 6. CANDLE CONFIRMATION  (max 10, can be negative)
    # ══════════════════════════════════════════════════════════════════
    pat_name = ind.get("candle_pattern", "none") or "none"
    pat_bias = ind.get("candle_pattern_bias", "neutral") or "neutral"
    cc_pts   = 0
    pat_n    = pat_name.lower()

    pat_aligned = (
        (direction == "UP"   and pat_bias == "bullish") or
        (direction == "DOWN" and pat_bias == "bearish")
    )
    pat_against = (
        (direction == "UP"   and pat_bias == "bearish") or
        (direction == "DOWN" and pat_bias == "bullish")
    )

    if pat_aligned:
        if   "engulfing"                               in pat_n: cc_pts += 5
        elif "pin" in pat_n or "hammer" in pat_n or \
             "star"  in pat_n or "shoot" in pat_n:               cc_pts += 4
        else:                                                     cc_pts += 3  # strong close

    # Spinning top / doji AGAINST trade direction → −5 (indecision against us)
    if pat_against and ("spinning" in pat_n or "doji" in pat_n):
        cc_pts -= 5

    # Weak body weakens confirmation
    if body_r < 0.20:
        cc_pts = min(cc_pts, 2)

    score += cc_pts
    details.update({
        "candle_pattern":      pat_name,
        "candle_pattern_bias": pat_bias,
        "candle_pattern_pts":  cc_pts,
    })

    # ══════════════════════════════════════════════════════════════════
    # SOBER GATE: hard structure filter
    # Structure must confirm direction OR a CHoCH reversal must be present.
    # Sideways → 20% penalty; directly contradicting → cap below threshold.
    # ══════════════════════════════════════════════════════════════════
    structure_ok = (
        (direction == "UP"   and ms_type == "bullish") or
        (direction == "DOWN" and ms_type == "bearish") or
        ms_choch
    )
    if not structure_ok:
        if ms_type == "sideways":
            score = int(score * 0.80)
            details["sober_gate"] = "sideways_penalty"
        else:
            score = min(score, SCORE_THRESHOLD - 1)
            details["sober_gate"] = "structure_mismatch_blocked"
    else:
        details["sober_gate"] = "passed"

    # Clamp [0, 100]
    score = max(0, min(score, 100))

    # ── Confluence summary (used by ML feature vector) ────────────────
    confluence_count  = int(ema_aligned) + int(macd_ok) + int(roc_ok)
    confluence        = tr_pts + mom_pts
    bb_pos = ind.get("bb_position")

    # ── DB-compat column remapping ────────────────────────────────────
    st_dir_f  = float(st_dir)
    st_dist_f = abs(price - (st_val or price)) / atr
    wick_atr  = (float(candle.get("High", price)) - float(candle.get("Low", price))) / atr

    details.update({
        "total_score":           score,
        "atr":                   float(ind.get("atr")) if ind.get("atr") is not None else None,
        "atr_ma":                float(ind.get("atr_ma")) if ind.get("atr_ma") is not None else None,
        "adx":                   adx_val,
        "macd_hist":             ind.get("macd_hist"),
        "bb_position":           round(bb_pos, 2) if bb_pos is not None else None,
        "confluence":            confluence,
        "confluence_count":      confluence_count,
        "confluence_gate_passed": confluence_count >= 2,
        "ema200_agree":          ema_aligned,
        "ema_fast_val":          float(ema_fast) if ema_fast is not None else None,
        "roc":                   roc,
        "price":                 price,
        "ema_slow_val":          float(ema_slow) if ema_slow is not None else None,
        # DB column remapping (legacy ML feature names)
        "ema_fast_sl":           st_dir_f,   # Supertrend direction
        "ema_slow_sl":           adx_val,    # ADX value
        "ema_distance":          st_dist_f,  # ST distance / ATR
        "wick_atr_ratio":        float(round(wick_atr, 3)),
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
                ms_t   = details.get("market_structure", "sideways")
                ms_ico = "↑" if ms_t == "bullish" else "↓" if ms_t == "bearish" else "→"
                pat    = details.get("candle_pattern", "none")
                sd_z   = details.get("sd_zone", "none")
                sd_d   = details.get("sd_zone_dist", 99.0)
                sd_s   = f"{sd_z} {sd_d:.1f}×ATR" if sd_z != "none" else "none"
                bos_s  = "  BOS✅" if details.get("market_struct_bos") else ""
                _send_tg(
                    f"🔒 <b>LOCKED</b> — <code>{symbol}</code>  {'📈' if direction=='UP' else '📉'} {direction}\n"
                    f"Score <b>{score}/100</b>  |  "
                    f"Entry {details.get('entry_quality',0)}/38 "
                    f"(ext {details.get('extension_atr',0):.2f}×ATR)\n"
                    f"MktStr: {ms_ico}{ms_t}{bos_s}  |  S&amp;D: {sd_s}\n"
                    f"Pattern: {pat}  |  "
                    f"{('⏱ Cooldown ' + str(cooldown_left) + 'm remaining' if cooldown_left else '✅ Armed — waiting for tick momentum')}"
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
def _get_contract_type(direction: str) -> str:
    """Return Deriv contract type for Rise/Fall based on direction."""
    return CONTRACT_TYPE_RISE if direction == "UP" else CONTRACT_TYPE_FALL


def _compute_barrier(symbol: str, direction: str = "UP", barrier_mult: float = None) -> str:
    """Not used for Rise/Fall — kept as stub for legacy compatibility."""
    return ""


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
                document=file_bytes,
                filename=filename,
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


def _send_rejection(symbol: str, direction: str, score: int, reason: str,
                     details: dict = None):
    """Send a detailed trade-rejected card to Telegram immediately and write to log."""
    _log(f"❌ {symbol} {direction} rejected [{score}/100] — {reason}")
    now_str  = datetime.now(timezone.utc).strftime("%H:%M UTC")
    mkt_sess = _get_session_name()
    sess_ico = SESSION_EMOJIS.get(mkt_sess, "")

    # Build a compact score breakdown from details if available
    breakdown = ""
    if details:
        t   = details.get("trend", 0)
        eq  = details.get("entry_quality", 0)
        v   = details.get("volatility", 0)
        m   = details.get("momentum", 0)
        cf  = details.get("confluence", 0)
        ms  = details.get("market_struct_pts", 0)
        sdp = details.get("sd_zone_pts", 0)
        pp  = details.get("candle_pattern_pts", 0)
        pat = details.get("candle_pattern", "none")
        ms_type = details.get("market_structure", "sideways")
        sd_zone = details.get("sd_zone", "none")
        sd_dist = details.get("sd_zone_dist", 99.0)
        conf = details.get("ml_confidence")
        conf_s = f"  ML: {conf*100:.0f}%\n" if conf is not None else ""
        ms_icon = "↑" if ms_type == "bullish" else "↓" if ms_type == "bearish" else "→"
        sd_s = f"{sd_zone} {sd_dist:.1f}×ATR" if sd_zone != "none" else "none"
        breakdown = (
            f"  Trend {t}/30 · Entry {eq}/30 · Vol {v}/15\n"
            f"  Mom {m}/10 · Conf {cf}/25 · MktStr {ms}/10\n"
            f"  S&amp;D {sdp}/8 ({sd_s}) · Pattern {pp}/7\n"
            f"  Structure: {ms_icon}{ms_type}  |  Pattern: {pat}\n"
            f"{conf_s}"
        )

    _send_tg(
        f"🚫 <b>REJECTED</b>  <code>{symbol}</code>  {'📈' if direction=='UP' else '📉'} {direction}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🕐 {now_str}  {sess_ico} {mkt_sess}\n"
        f"🎯 Score  : <b>{score}/100</b>\n"
        f"❌ Reason : <i>{_html.escape(reason)}</i>\n"
        f"{breakdown}"
        f"━━━━━━━━━━━━━━━━━━━━"
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


def _trade_grade(score: int, conf: Optional[float]) -> tuple:
    """Return (grade_str, stars_str) based on score and ML confidence."""
    combined = score * 0.7 + (conf * 100 if conf else 75) * 0.3
    if combined >= 96:   return "A+", "⭐⭐⭐⭐⭐"
    elif combined >= 91: return "A",  "⭐⭐⭐⭐"
    elif combined >= 85: return "B+", "⭐⭐⭐"
    elif combined >= 80: return "B",  "⭐⭐"
    else:                return "C",  "⭐"


def _adx_label(adx: float) -> str:
    if adx >= 40:   return "Very Strong"
    elif adx >= 25: return "Strong"
    elif adx >= 20: return "Moderate"
    else:           return "Weak"


def _ml_top_reasons(direction: str, details: dict) -> str:
    """Build ML top-reasons checklist from signal details."""
    checks = []
    ms_type  = details.get("market_structure", "sideways")
    ms_bos   = details.get("market_struct_bos", False)
    sd_agree = details.get("sd_zone_agree", False)
    sd_zone  = details.get("sd_zone", "none")
    macd_b   = details.get("macd_bullish", False)
    macd_r   = details.get("macd_hist_rising", False)
    momentum = details.get("momentum", 0)
    trend    = details.get("trend", 0)
    pat_name = details.get("candle_pattern", "none")
    pat_bias = details.get("candle_pattern_bias", "neutral")
    ema200   = details.get("ema200_agree", False)
    roc      = details.get("roc")
    adx_val  = details.get("adx") or 0

    struct_confirms = (
        (direction == "UP" and ms_type == "bullish") or
        (direction == "DOWN" and ms_type == "bearish")
    )
    pat_confirms = (
        (direction == "UP" and pat_bias == "bullish") or
        (direction == "DOWN" and pat_bias == "bearish")
    )
    macd_confirms = (
        (direction == "UP" and macd_b and macd_r) or
        (direction == "DOWN" and not macd_b and macd_r)
    )

    if struct_confirms:
        struct_lbl = "Bullish" if direction == "UP" else "Bearish"
        bos_tag = " + BOS" if ms_bos else ""
        checks.append(f"  ✔ {struct_lbl} market structure{bos_tag}")
    else:
        checks.append(f"  ✖ Structure not confirmed")

    if sd_agree and sd_zone != "none":
        zone_lbl = "Demand rejection" if direction == "UP" else "Supply rejection"
        checks.append(f"  ✔ {zone_lbl} zone")
    else:
        checks.append(f"  ✖ No aligned S&amp;D zone")

    if trend >= 25:
        checks.append(f"  ✔ Trend continuation (ST aligned)")
    else:
        checks.append(f"  ✖ Trend weak")

    if macd_confirms:
        checks.append(f"  ✔ MACD momentum aligned")
    else:
        checks.append(f"  ✖ MACD not confirmed")

    if pat_confirms:
        checks.append(f"  ✔ {pat_name}")
    elif pat_name != "none":
        checks.append(f"  ✖ {pat_name} (opposing)")
    else:
        checks.append(f"  ✖ No candlestick confirmation")

    if ema200 and adx_val >= 25:
        checks.append(f"  ✔ EMA200 + ADX {adx_val:.0f} trending")

    return "\n".join(checks)


def _compute_risk(score: int, conf: Optional[float], atr: Optional[float]) -> tuple:
    """Return (continuation_pct, reversal_pct, expected_atr) for risk section."""
    base = score * 0.65 + (conf * 100 if conf else 75) * 0.35
    cont_pct = min(95, max(50, base))
    rev_pct  = 100 - cont_pct
    atr_val  = atr or 1.0
    # Expected move: stronger signals project bigger moves
    move_atr = round(1.0 + (score - 80) / 40, 1) if score >= 80 else 1.0
    move_atr = max(0.8, min(2.5, move_atr))
    return round(cont_pct, 1), round(rev_pct, 1), move_atr


def _signal_card(sym: str, score: int, direction: str, details: dict) -> str:
    with _lock:
        wc, lc, pnl = win_count, loss_count, total_pnl
    total = wc + lc
    wr    = wc / total * 100 if total else 0
    pnl_str = f"{'+'if pnl>=0 else ''}${pnl:.2f}"
    session_line = f"#{total + 1}  |  {wc}W/{lc}L  {wr:.0f}%WR  |  P&L {pnl_str}"
    mkt_session  = _get_session_name()

    conf    = details.get("ml_confidence")
    grade, stars = _trade_grade(score, conf)

    # ── Score components ────────────────────────────────────────────────
    t   = details.get("trend", 0)
    eq  = details.get("entry_quality", 0)
    v   = details.get("volatility", 0)
    m   = details.get("momentum", 0)
    cf  = details.get("confluence", 0)
    ms  = details.get("market_struct_pts", 0)
    sdp = details.get("sd_zone_pts", 0)
    pp  = details.get("candle_pattern_pts", 0)
    raw_sum = t + eq + v + m + cf + ms + sdp + pp

    # ── Market Structure ────────────────────────────────────────────────
    ms_type  = details.get("market_structure", "sideways")
    ms_bos   = details.get("market_struct_bos", False)
    ms_choch = details.get("market_struct_choch", False)
    ms_hh    = details.get("market_struct_hh", False)
    ms_hl    = details.get("market_struct_hl", False)
    ms_lh    = details.get("market_struct_lh", False)
    ms_ll    = details.get("market_struct_ll", False)
    ms_str   = "↑ Bullish" if ms_type == "bullish" else "↓ Bearish" if ms_type == "bearish" else "→ Sideways"
    sober_gate = details.get("sober_gate", "passed")

    # ── Supply & Demand ─────────────────────────────────────────────────
    sd_zone  = details.get("sd_zone", "none")
    sd_dist  = details.get("sd_zone_dist", 99.0)
    sd_fresh = details.get("sd_zone_fresh", False)
    sd_agree = details.get("sd_zone_agree", False)

    # ── Trend ───────────────────────────────────────────────────────────
    st_dir   = details.get("ema_fast_sl", 0)
    adx_val  = float(details.get("ema_slow_sl") or details.get("adx") or 0)
    ema50    = details.get("ema_fast_val")
    ema200   = details.get("ema_slow_val")
    price    = details.get("price")
    ema_rel  = ""
    if ema50 is not None and ema200 is not None:
        if ema50 > ema200:
            ema_rel = "EMA50 &gt; EMA200 ✅" if direction == "UP" else "EMA50 &gt; EMA200 ⚠️"
        else:
            ema_rel = "EMA50 &lt; EMA200 ✅" if direction == "DOWN" else "EMA50 &lt; EMA200 ⚠️"
    adx_lbl = _adx_label(adx_val)
    slope_icon = "↑" if st_dir > 0 else "↓"

    # ── Entry ───────────────────────────────────────────────────────────
    rsi      = details.get("rsi")
    srsi_k   = details.get("stochrsi_k")
    ext_atr  = details.get("extension_atr", 0)
    uw       = (details.get("atr") or 0) and details.get("entry_quality", 0) > 0   # proxy
    pullback = ext_atr <= 1.5
    wick_pts = details.get("wick_pts", 0)
    rejection_ok = wick_pts >= 4
    rsi_disp = f"{srsi_k:.0f}" if srsi_k is not None else (f"{rsi:.0f}" if rsi else "—")

    # ── Momentum ────────────────────────────────────────────────────────
    macd_bull = details.get("macd_bullish", False)
    macd_rise = details.get("macd_hist_rising", False)
    roc_val   = details.get("roc")
    if direction == "UP":
        macd_ok = macd_bull and macd_rise
        roc_ok  = roc_val is not None and roc_val > 0
    else:
        macd_ok = not macd_bull and macd_rise
        roc_ok  = roc_val is not None and roc_val < 0

    # ── Risk ────────────────────────────────────────────────────────────
    cont_pct, rev_pct, exp_move = _compute_risk(score, conf, details.get("atr"))

    # ── ML reasons ──────────────────────────────────────────────────────
    ml_reasons = _ml_top_reasons(direction, details)

    # ── Header ──────────────────────────────────────────────────────────
    dir_icon = "🟢 📈 RISE" if direction == "UP" else "🔴 📉 FALL"
    gate_warn = "" if sober_gate == "passed" else "  ⚠️ <i>sideways mkt</i>" if "sideways" in sober_gate else ""

    lines = [
        f"{dir_icon}  <code>{sym}</code>",
        f"━━━━━━━━━━━━━━━━━━━━━━━━",
        f"🏆 Grade: <b>{grade}</b>  {stars}",
        f"📊 Score: <b>{score}/100</b>  [{_score_bar_str(score)}]  (raw sum: {raw_sum}){gate_warn}",
        f"━━ Market Structure ━━",
        f"  BOS {'✅' if ms_bos else '✗'}   CHoCH {'✅' if ms_choch else '✗'}",
        f"  HH {'✅' if ms_hh else '✗'}  HL {'✅' if ms_hl else '✗'}  "
        f"LH {'✅' if ms_lh else '✗'}  LL {'✅' if ms_ll else '✗'}",
        f"  Strength: <b>{ms_str}</b>",
        f"━━ Supply &amp; Demand ━━",
    ]

    if sd_zone != "none":
        fresh_icon = "✅" if sd_fresh else "⚠️ stale"
        agree_icon = "✅" if sd_agree else "⚠️ opposing"
        lines += [
            f"  {sd_zone.capitalize()} Zone  {agree_icon}",
            f"  Fresh: {fresh_icon}   Distance: {sd_dist:.2f} ATR",
        ]
    else:
        lines.append("  No zone detected")

    lines += [
        f"━━ Trend ━━",
    ]
    if ema_rel:
        lines.append(f"  {ema_rel}")
    lines += [
        f"  Slope {slope_icon}   ADX {adx_val:.0f} ({adx_lbl})",
        f"━━ Entry ━━",
        f"  Pullback {'✅' if pullback else '✗'}   Rejection {'✅' if rejection_ok else '✗'}",
        f"  StochRSI/RSI: {rsi_disp}   Entry Quality: {eq}/38",
        f"━━ Momentum ━━",
        f"  MACD {'✅ Bullish' if direction=='UP' and macd_ok else '✅ Bearish' if direction=='DOWN' and macd_ok else '✗ Weak'}"
        f"   ROC {'✅ +' if roc_ok and roc_val and roc_val>0 else '✅ -' if roc_ok else '✗'}",
        f"━━ ML Analysis ━━",
        f"🤖 Confidence: <b>{_conf_str(conf)}</b>",
        ml_reasons,
        f"━━ Risk ━━",
        f"  Continuation : <b>{cont_pct:.0f}%</b>   Reversal : {rev_pct:.0f}%",
        f"  Expected Move: {exp_move:.1f} ATR",
        f"━━━━━━━━━━━━━━━━━━━━━━━━",
        f"💵 Stake: ${STAKE:.2f}  →  win ~${STAKE*1.87:.2f}",
        f"⏱ {DURATION} min  |  {SESSION_EMOJIS.get(mkt_session,'')} {mkt_session}",
        f"📊 {session_line}",
    ]
    return "\n".join(lines)


def _make_footer_text() -> str:
    return (
        f"  Stake ${STAKE}  ·  Session TP ${DAILY_PROFIT_TARGET}  ·  Session SL ${DAILY_LOSS_LIMIT}  ·  "
        f"Duration {DURATION}min  ·  Cooldown {COOLDOWN_MINUTES}min  ·  "
        f"Min Score ≥{SCORE_THRESHOLD}  ·  {_ml_progress_text()}"
    )


def _result_card(sym: str, profit: float, win: bool, details: dict,
                 _wc: int = None, _lc: int = None, _pnl: float = None, _cl: int = None) -> str:
    """Build a trade-result Telegram card.
    Pass _wc/_lc/_pnl/_cl snapshots captured *before* any session reset so the
    card always shows the stats that include this trade.
    """
    try:
        if _wc is None:
            with _lock:
                _wc, _lc, _pnl, _cl = win_count, loss_count, total_pnl, consecutive_losses
        # Safety: ensure numeric types
        _wc   = int(_wc  or 0)
        _lc   = int(_lc  or 0)
        _pnl  = float(_pnl or 0.0)
        _cl   = int(_cl  or 0)
        profit = float(profit or 0.0)

        total  = _wc + _lc
        wr     = _wc / total * 100 if total else 0
        mkt_sess = _get_session_name()
        score    = int(details.get("total_score", 0) or 0)
        direction = details.get("direction", "")
        conf    = details.get("ml_confidence")
        grade, _ = _trade_grade(score, conf)
        mlevel  = int(details.get("martingale_level", 0) or 0)
        stake   = float(details.get("stake_used", STAKE) or STAKE)

        dir_icon = "📈" if direction == "UP" else "📉"
        pnl_sign = "+" if profit > 0 else ""

        # ── Header ────────────────────────────────────────────────────
        if win:
            hdr_emoji = "🏆"
            result_lbl = "WIN"
        else:
            hdr_emoji = "💀"
            result_lbl = "LOSS"

        # ── Streak display ────────────────────────────────────────────
        if win:
            streak_str = "🟢 Streak cleared"
        elif _cl >= 3:
            streak_str = "🔴 " * min(_cl, 5) + f"  ({_cl} in a row)"
        elif _cl > 0:
            streak_str = "🔴 " * _cl + f"  ({_cl})"
        else:
            streak_str = "🟢 None"

        # ── Entry context ──────────────────────────────────────────────
        ms_type  = details.get("market_structure", "")
        sd_zone  = details.get("sd_zone", "none")
        pat_name = details.get("candle_pattern", "none")
        has_conf = details.get("has_confirmation")
        conf_tag = "  ✅ confirmed" if has_conf else ("  ⚠️ no confirmation" if has_conf is False else "")

        entry_parts = []
        if ms_type and ms_type != "sideways":
            entry_parts.append(f"{ms_type.capitalize()} structure")
        if sd_zone != "none":
            entry_parts.append(f"{sd_zone} zone")
        if pat_name not in ("none", "", None):
            entry_parts.append(pat_name)
        entry_str = "  ·  ".join(entry_parts) if entry_parts else "—"

        martin_line = (f"\n  Martingale step {mlevel}  ·  Stake ${stake:.2f}" if mlevel > 0 else "")

        ml_line = (f"\n🤖 ML  : <b>{_conf_str(conf)}</b>  ·  Grade <b>{grade}</b>{conf_tag}"
                   if conf is not None else f"\n🏅 Grade: <b>{grade}</b>{conf_tag}")

        pnl_sess_sign = "+" if _pnl >= 0 else ""

        return (
            f"{hdr_emoji} <b>{result_lbl}</b>  ·  <code>{sym}</code>  {dir_icon} {direction}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💵 Result : <b>{pnl_sign}${profit:.2f}</b>  ·  Score <b>{score}/100</b>"
            f"{ml_line}{martin_line}\n"
            f"━━ Entry Context ━━\n"
            f"  {entry_str}\n"
            f"━━ Session ━━\n"
            f"  P&amp;L  : <b>{pnl_sess_sign}${_pnl:.2f}</b>  ·  {_wc}W/{_lc}L  {wr:.0f}%WR\n"
            f"  Streak: {streak_str}\n"
            f"  Market: {SESSION_EMOJIS.get(mkt_sess,'')} {mkt_sess}\n"
        )
    except Exception as _e:
        logger.error(f"_result_card exception for {sym}: {_e}")
        p_sign = "+" if (profit or 0) > 0 else ""
        _dir = details.get("direction", "") if details else ""
        return (
            f"{'🏆 WIN' if win else '💀 LOSS'}  <code>{sym}</code>  {_dir}\n"
            f"Result: <b>{p_sign}${float(profit or 0):.2f}</b>  Score: {details.get('total_score','?') if details else '?'}/100\n"
            f"Session: {int(_wc or 0)}W/{int(_lc or 0)}L\n"
        )


def _run_backtest():
    """
    Historical backtest using existing OHLCV data.
    For each symbol, simulates Rise/Fall outcomes over the last 200 candles
    and reports per-symbol win rates. Also kicks off ML bootstrap if needed.
    """
    _send_tg(
        "🔬 <b>Backtest running…</b>\n"
        "Simulating Rise/Fall outcomes on recent history.\n"
        "<i>Results will arrive in a few seconds.</i>"
    )

    results   = {}
    bar_width = 8

    for sym in SYMBOLS:
        with _lock:
            if sym not in ohlcv or len(ohlcv[sym]) < 30:
                continue
            df  = ohlcv[sym].copy()
            ind = dict(indicators.get(sym, {}))

        if not ind.get("ready"):
            continue

        st_dir = ind.get("supertrend_dir", 0)
        if st_dir == 0:
            continue
        direction = "UP" if st_dir == 1 else "DOWN"

        check_from = max(10, len(df) - 200)
        check_to   = len(df) - DURATION - 1
        if check_to <= check_from:
            continue

        wins = losses = 0
        for i in range(check_from, check_to):
            entry_close  = float(df.iloc[i]["Close"])
            future_close = float(df.iloc[i + DURATION]["Close"])
            won = (future_close > entry_close if direction == "UP"
                   else future_close < entry_close)
            if won: wins += 1
            else:   losses += 1

        total = wins + losses
        if total < 5:
            continue

        wr = wins / total * 100
        adx_v = float(ind.get("adx") or 0)
        results[sym] = {
            "dir": direction, "wins": wins, "losses": losses,
            "wr": wr, "adx": adx_v,
        }

    if not results:
        _send_tg(
            "🔬 <b>Backtest</b>: Not enough historical data yet.\n"
            "<i>Wait for the history loader to finish, then try again.</i>"
        )
        return

    sorted_r    = sorted(results.items(), key=lambda x: -x[1]["wr"])
    total_wins  = sum(v["wins"]   for v in results.values())
    total_loss  = sum(v["losses"] for v in results.values())
    total_trades = total_wins + total_loss
    overall_wr   = total_wins / total_trades * 100 if total_trades else 0
    breakeven_wr = 100 / (1 + PROFIT_MIN)   # e.g. 62.5% at 60% payout

    profitable = sum(1 for v in results.values() if v["wr"] >= breakeven_wr)

    lines = [
        "🔬 <b>Backtest Report</b>",
        "━━━━━━━━━━━━━━━━━━━━━━━━",
        f"  Symbols tested  : <b>{len(results)}</b>",
        f"  Candles/symbol  : last 200  (1-min candles)",
        f"  Outcome window  : <b>{DURATION} candles ahead</b>",
        f"  Payout          : {PROFIT_MIN*100:.0f}%  (min)",
        "",
        "<b>📊 Overall</b>",
        f"  Simulated trades: <b>{total_trades}</b>  ({total_wins}W / {total_loss}L)",
        f"  Simulated WR    : <b>{overall_wr:.1f}%</b>",
        f"  Breakeven WR    : <b>{breakeven_wr:.1f}%</b>  (at {PROFIT_MIN*100:.0f}% payout)",
        f"  Profitable syms : <b>{profitable}/{len(results)}</b>  above breakeven",
        "",
        "<b>📈 Per Symbol  (sorted by WR)</b>",
        "<code>Symbol    Dir  W    L   WR    Bar      ADX</code>",
        "<code>──────────────────────────────────────────</code>",
    ]

    for sym, r in sorted_r:
        bar_f = max(0, min(bar_width, int(r["wr"] / 100 * bar_width)))
        bar   = "█" * bar_f + "░" * (bar_width - bar_f)
        status = "✅" if r["wr"] >= breakeven_wr else ("⚠️" if r["wr"] >= 50 else "❌")
        dir_c  = "↑" if r["dir"] == "UP" else "↓"
        adx_s  = f"{r['adx']:.0f}" if r["adx"] else "—"
        lines.append(
            f"{status} <code>{sym:<9}{dir_c}  "
            f"{r['wins']:>3}W/{r['losses']:<3}L  "
            f"{r['wr']:>4.0f}%  [{bar}]  {adx_s:>3}</code>"
        )

    lines += [
        "",
        f"<i>✅ above {breakeven_wr:.0f}%  ⚠️ above 50%  ❌ below 50%</i>",
        "",
        "<b>🤖 ML Model</b>",
    ]

    with ml_lock:
        _ml_ready  = ml_model is not None
        _ml_trades = ml_trained_on

    if _ml_ready:
        lines.append(
            f"  Status  : ✅ Trained on <b>{_ml_trades}</b> samples\n"
            f"  Gate    : ≥{ML_CONFIDENCE_MIN*100:.0f}% confidence required"
        )
    else:
        lines.append(
            f"  Status  : ⏳ Not yet trained ({ml_total_trades}/{ML_MIN_TRADES} real trades)\n"
            f"  Bootstrapping from historical data now…"
        )

    _send_tg("\n".join(lines))

    # Kick off ML bootstrap if model isn't ready yet
    with ml_lock:
        needs_boot = ml_model is None
    if needs_boot:
        try:
            MLBootstrap()
        except Exception as _be:
            logger.error(f"Backtest MLBootstrap failed: {_be}")


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
def request_proposal(ws, symbol: str, details: dict, direction: str, barrier_mult: float = None):
    # ── Clear stale timing fields from previous trade ─────────────────
    # Re-entry details are copied from the settling contract's details dict,
    # which carries buy_sent_at / proposal_id from that older trade.
    # The pending-timeout loop checks buy_sent_at and fires "ack missing"
    # immediately if it sees a timestamp that is 60+ s old — killing the
    # step-2 (and any higher-step) martingale re-entry before Deriv even
    # responds to the new proposal.  Clear them here so every call to
    # request_proposal starts with a clean timing slate.
    details.pop("buy_sent_at",  None)
    details.pop("proposal_id",  None)
    # ── Martingale: stake doubles on each consecutive loss ────────────
    if MARTINGALE_ENABLED:
        with _lock:
            mlevel = martingale_level.get(symbol, 0)
        m_stake = round(STAKE * (MARTINGALE_MULTIPLIER ** mlevel), 2)
    else:
        mlevel  = 0
        m_stake = STAKE
    details["direction"]              = direction
    details["proposal_time"]          = datetime.now(timezone.utc)
    details["martingale_level"]       = mlevel
    details["stake_used"]             = m_stake
    details["_is_martingale_reentry"] = mlevel > 0
    with _lock:
        pending_signals[symbol] = details
    contract_type = _get_contract_type(direction)
    if mlevel > 0:
        _log(f"📈 Martingale {symbol}: step {mlevel}  stake ${m_stake:.2f}  "
             f"(base ${STAKE:.2f} × {MARTINGALE_MULTIPLIER**mlevel:.0f}×)")
    ws.send(json.dumps({
        "proposal": 1,
        "amount": m_stake,
        "basis": "stake",
        "contract_type": contract_type,
        "currency": "USD",
        "duration": DURATION,
        "duration_unit": "m",
        "symbol": symbol,
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
        m_stake = pending_signals[symbol].get("stake_used", STAKE)  # martingale stake
    if not pid:
        return

    # ── Payout quality gate — check profit RATIO (not dollar amount) ──
    # Rise/Fall payout ratio is typically 70–95% depending on symbol/time.
    try:
        offered_payout = float(prop.get("payout", 0))
        # profit ratio = (total_payout - stake) / stake
        offered_ratio = round((offered_payout - m_stake) / m_stake, 4) if m_stake > 0 else 0.0
    except (TypeError, ValueError):
        offered_ratio = 0.0

    if offered_ratio < PROFIT_MIN:
        with _lock:
            details = pending_signals.pop(symbol, None)
        is_martin_reentry = (details or {}).get("_is_martingale_reentry", False)
        if not is_martin_reentry:
            _release_trade_slot(symbol)
        else:
            # Martingale never reserved a slot — just reset the level cleanly
            with _lock:
                martingale_level[symbol] = 0
        _record_funnel_rejection("Payout below minimum")
        direction = (details or {}).get("direction", "UP")
        _send_tg(
            f"💸 <b>Payout rejected</b> — <code>{symbol}</code> ({direction})\n"
            f"Offered ratio: <b>{offered_ratio*100:.1f}%</b>  "
            f"(min {PROFIT_MIN*100:.0f}%) — skipping."
        )
        _log(f"💸 {symbol} proposal rejected — profit ratio {offered_ratio*100:.1f}% below "
             f"minimum {PROFIT_MIN*100:.0f}%")
        return

    ws.send(json.dumps({"buy": pid, "price": m_stake}))
    with _lock:
        if symbol in pending_signals:
            pending_signals[symbol]["buy_sent_at"] = datetime.now(timezone.utc)


def _adopt_orphan_contract(symbol: str, direction: str, details: dict, c: dict):
    """A buy ack never arrived, but Deriv's portfolio shows a real open
    contract matching this symbol/direction — adopt it into active_contracts
    so it gets tracked to settlement normally, instead of silently
    abandoning a real position with money on it."""
    cid = c.get("contract_id")
    if not cid or cid in active_contracts:
        return False
    with _lock:
        active_contracts[cid] = {
            "symbol":      symbol,
            "direction":   direction,
            "barrier":     None,
            "stake":       c.get("buy_price", details.get("stake_used", STAKE)),
            "payout":      c.get("payout"),
            "entry_time":  datetime.now(timezone.utc),
            "entry_price": last_price.get(symbol, 0),
            "details":     details,
            "settled":     False,
        }
    ws = ws_registry.get(symbol)
    if ws is not None:
        try:
            ws.send(json.dumps({
                "proposal_open_contract": 1,
                "contract_id": int(cid),
                "subscribe": 1,
            }))
        except Exception as e:
            logger.warning(f"{symbol} adopt-orphan subscribe failed: {e}")
    _record_execution()
    _send_tg(
        f"🔎 <b>Recovered untracked contract</b> — <code>{symbol}</code>\n"
        f"Buy ack was lost, but Deriv's portfolio confirms contract #{cid} "
        f"is really open. Now tracking it to settlement."
    )
    _log(f"🔎 {symbol} adopted orphan contract cid={cid} (buy ack was lost)")
    return True


def on_portfolio(ws, msg: dict, symbol: str):
    """Response to our own {'portfolio': 1} safety check, fired when a buy
    ack goes missing — confirms whether a contract actually opened before we
    give up and reset/release, so we never abandon a real open position."""
    check = _portfolio_checks.pop(symbol, None)
    if check is None:
        return
    details   = check["details"]
    direction = details.get("direction", "UP")
    want_type = _get_contract_type(direction)
    contracts = (msg.get("portfolio", {}) or {}).get("contracts", []) or []
    since = check["requested_at"] - timedelta(seconds=30)
    for c in contracts:
        if c.get("symbol") != symbol or c.get("contract_type") != want_type:
            continue
        with _lock:
            already_tracked = c.get("contract_id") in active_contracts
        if already_tracked:
            continue
        purchase_time = c.get("purchase_time") or c.get("date_start")
        try:
            pt = datetime.fromtimestamp(int(purchase_time), tz=timezone.utc) if purchase_time else None
        except (TypeError, ValueError):
            pt = None
        if pt is not None and pt < since:
            continue  # too old — not the contract we're looking for
        if _adopt_orphan_contract(symbol, direction, details, c):
            unconfirmed_buys.pop(symbol, None)
            return
    # No matching open contract found — genuinely never executed. Let the
    # timeout loop's normal cleanup (reset martingale / release slot) proceed.
    with _lock:
        _portfolio_checks.pop(symbol, None)


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
    _record_execution()
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
    global total_pnl, daily_total_pnl, win_count, loss_count, consecutive_losses, paused, pause_until
    global daily_trades, _auto_resume_active, peak_equity, max_drawdown, session_start
    global session_symbol_stats, daily_session_log, _daily_session_log_date
    global market_session_stats, _market_session_date, ml_total_trades

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

        # ── Settlement detection: require a definitive outcome ──────────
        # is_expired can fire before Deriv has calculated profit/status
        # (mid-settlement). Only settle when we have a confirmed outcome.
        status = contract.get("status", "")
        profit_raw = contract.get("profit")
        is_settled = (
            status in ("won", "lost", "void")
            or bool(contract.get("is_sold"))
            or (bool(contract.get("is_expired")) and (
                status in ("won", "lost", "void")
                or (profit_raw is not None and float(profit_raw) != 0)
            ))
        )
        if not is_settled:
            return

        info["settled"] = True
        is_void = status == "void"

        # Extract profit robustly
        profit = float(contract.get("profit") or 0)
        if not is_void and profit == 0:
            if status == "won":
                profit = float(info.get("payout", STAKE) or STAKE) - float(info.get("stake", STAKE) or STAKE)
            elif status == "lost":
                profit = -float(info.get("stake", STAKE) or STAKE)

        win = profit > 0 or status == "won"
        d   = info.get("details", {})
        mkt_session = _get_session_name(info.get("entry_time"))

        # ── VOID / unresolved contracts: never guessed as a win or a loss.
        # Just remove from tracking and report — stats stay untouched.
        if is_void:
            with_active = cid in active_contracts
            if with_active:
                del active_contracts[cid]
            db_queue.put((
                datetime.now(timezone.utc).isoformat(), symbol, info["direction"],
                info["barrier"], info["stake"], info["payout"], 0.0, -1,
                d.get("total_score", 0), d.get("wick_atr_ratio", 0),
                d.get("atr", 0), d.get("atr_ma", 0),
                d.get("ema_fast_sl", 0), d.get("ema_slow_sl", 0), d.get("ema_distance", 0),
                mkt_session,
            ))
            _log(f"VOID  {symbol}  contract {cid} unresolved — excluded from win/loss stats")
            return

        # ── Session P&L counters ──────────────────────────────────────
        if win:
            win_count += 1
            consecutive_losses = 0
            # Second-entry: shorten cooldown so a re-entry can fire within 5 min
            if ALLOW_SECOND_ENTRY:
                cooldown_until[symbol] = datetime.now(timezone.utc) + timedelta(
                    minutes=SECOND_ENTRY_COOLDOWN
                )
                second_entry_eligible[symbol] = datetime.now(timezone.utc) + timedelta(
                    minutes=SECOND_ENTRY_WINDOW
                )
                _log(f"🔁 {symbol} WIN → second-entry window open for {SECOND_ENTRY_WINDOW} min "
                     f"(cooldown ↓{SECOND_ENTRY_COOLDOWN} min, ML gate ≥{SECOND_ENTRY_ML_MIN*100:.0f}%)")
        else:
            loss_count += 1
            consecutive_losses += 1
            # Clear any open second-entry window on a loss
            second_entry_eligible.pop(symbol, None)

        # ── Martingale level update + re-entry flag ──────────────────
        do_martingale_reentry = False
        reentry_direction     = info["direction"]
        reentry_details       = dict(d)   # copy signal details for re-entry proposal

        if MARTINGALE_ENABLED:
            cur_ml = martingale_level.get(symbol, 0)
            if win:
                if cur_ml > 0:
                    _log(f"↩️  Martingale {symbol}: WIN at step {cur_ml} — "
                         f"resetting to base ${STAKE:.2f}")
                martingale_level[symbol] = 0
                # win cooldown already handled by ALLOW_SECOND_ENTRY above
            elif cur_ml < MARTINGALE_MAX_STEPS:
                # Smart gate: only re-enter if the original signal was high quality.
                # Blindly doubling on a weak setup is the most dangerous part of
                # martingale — require A-grade score AND strong ML confidence.
                _m_score = float(d.get("total_score", 0) or 0)
                _m_conf  = float(d.get("ml_confidence") or 0.0)
                if _m_score >= 90 and _m_conf >= 0.90:
                    new_ml = cur_ml + 1
                    next_s = round(STAKE * (MARTINGALE_MULTIPLIER ** new_ml), 2)
                    martingale_level[symbol] = new_ml
                    do_martingale_reentry   = True   # re-entry queued
                    _log(f"📈 Martingale {symbol}: LOSS → step {new_ml}  "
                         f"stake ${next_s:.2f}  "
                         f"(smart gate ✅ score={_m_score:.0f} ML={_m_conf*100:.0f}%)")
                else:
                    martingale_level[symbol] = 0
                    _log(f"↩️  Martingale {symbol}: LOSS — smart gate blocked re-entry "
                         f"(score={_m_score:.0f} <90 or ML={_m_conf*100:.0f}% <90%) "
                         f"→ reset to base ${STAKE:.2f}")
            else:
                # All steps exhausted — reset and impose standard cooldown
                _log(f"↩️  Martingale {symbol}: all {MARTINGALE_MAX_STEPS} steps exhausted — "
                     f"resetting, applying {COOLDOWN_MINUTES}-min cooldown")
                martingale_level[symbol] = 0
                cooldown_until[symbol]   = (datetime.now(timezone.utc)
                                            + timedelta(minutes=COOLDOWN_MINUTES))

        total_pnl       += profit
        daily_total_pnl += profit   # persists across session resets; resets at UTC midnight
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
        # ── Snapshot stats BEFORE any session reset so _result_card shows the
        # correct cumulative figures that include this trade.
        rc_wc  = win_count
        rc_lc  = loss_count
        rc_pnl = total_pnl
        rc_cl  = consecutive_losses

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
                "peak": peak_equity,
                "max_dd": max_drawdown,
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
        # Capture state AFTER all triggers so snapshots are accurate.
        # MUST be inside the lock and AFTER trigger_consec / session_reset_trigger
        # are evaluated — reading session_reset_trigger before it is assigned
        # would cause UnboundLocalError (Python treats assigned locals as local
        # throughout the whole function).
        _paused_snap     = paused
        _sess_reset_snap = session_reset_trigger

    # Instantly bump ML trade counter so status display stays accurate.
    # The DB batch may not flush for ~2 s; this eliminates the display lag.
    with ml_lock:
        ml_total_trades += 1
        # Update live ML confidence performance tracker (actual model probability, not score)
        _conf_val = d.get("ml_confidence")
        if _conf_val is not None:
            _bkt = _ml_conf_bucket(_conf_val)
            _bst = _ml_conf_live_stats.setdefault(_bkt, {"wins": 0, "losses": 0, "pnl": 0.0})
            if win: _bst["wins"]   += 1
            else:   _bst["losses"] += 1
            _bst["pnl"] += profit
        # Update live signal-score tracker so we can see whether raw score bands
        # (independent of ML confidence) actually make money.
        _score_val = d.get("total_score", 0)
        _sbkt = _ml_score_bucket(_score_val)
        _sbst = _ml_score_live_stats.setdefault(_sbkt, {"wins": 0, "losses": 0, "pnl": 0.0})
        if win: _sbst["wins"]   += 1
        else:   _sbst["losses"] += 1
        _sbst["pnl"] += profit

    # DB write — includes market_session
    db_queue.put((
        datetime.now(timezone.utc).isoformat(), symbol, info["direction"],
        info["barrier"], info["stake"], info["payout"], profit, int(win),
        d.get("total_score", 0), d.get("wick_atr_ratio", 0),
        d.get("atr", 0), d.get("atr_ma", 0),
        d.get("ema_fast_sl", 0), d.get("ema_slow_sl", 0), d.get("ema_distance", 0),
        mkt_session,
    ))

    # Signal-features write (rich 39-column log for advanced ML + analytics)
    _sf_grade, _ = _trade_grade(d.get("total_score", 0), d.get("ml_confidence"))
    db_queue.put({"type": "sf", "data": (
        datetime.now(timezone.utc).isoformat(),
        symbol, info["direction"], DURATION,
        float(d.get("total_score", 0) or 0),
        _sf_grade,
        d.get("ml_confidence"),
        float(d.get("market_struct_pts", 0) or 0),
        d.get("market_structure", "sideways"),
        int(bool(d.get("market_struct_bos"))),
        int(bool(d.get("market_struct_choch"))),
        int(bool(d.get("market_struct_hh"))),
        int(bool(d.get("market_struct_hl"))),
        int(bool(d.get("market_struct_lh"))),
        int(bool(d.get("market_struct_ll"))),
        float(d.get("ema_fast_val", 0) or 0),
        float(d.get("ema_slow_val", 0) or 0),
        int(bool(d.get("ema_aligned"))),
        float(d.get("adx", 0) or 0),
        float(d.get("rsi", 0) or 0) if d.get("rsi") is not None else None,
        float(d.get("stochrsi_k", 0) or 0) if d.get("stochrsi_k") is not None else None,
        float(d.get("entry_quality", 0) or 0),
        float(d.get("momentum", 0) or 0),
        int(bool(d.get("macd_bullish"))),
        int(bool(d.get("macd_hist_rising"))),
        float(d.get("roc", 0) or 0) if d.get("roc") is not None else None,
        d.get("sd_zone", "none"),
        float(d.get("sd_zone_dist", 99.0) or 99.0),
        int(bool(d.get("sd_zone_fresh"))),
        float(d.get("sd_zone_pts", 0) or 0),
        float(d.get("candle_pattern_pts", 0) or 0),
        d.get("candle_pattern", "none"),
        d.get("candle_pattern_bias", "neutral"),
        float(d.get("atr", 0) or 0),
        float(d.get("atr_ma", 0) or 0),
        float(d.get("wick_atr_ratio", 0) or 0),
        mkt_session,
        int(win),
        float(profit),
    )})

    # Pass pre-reset stat snapshot so card always shows correct cumulative figures
    try:
        result_msg = _result_card(symbol, profit, win, d,
                                  _wc=rc_wc, _lc=rc_lc, _pnl=rc_pnl, _cl=rc_cl)
    except Exception as _rce:
        logger.error(f"_result_card build error for {symbol}: {_rce}")
        p_sign = "+" if profit > 0 else ""
        result_msg = (
            f"{'🏆 WIN' if win else '💀 LOSS'}  <code>{symbol}</code>  "
            f"{d.get('direction','')}\n"
            f"Result: <b>{p_sign}${profit:.2f}</b>  Score: {d.get('total_score','?')}/100\n"
            f"Session: {rc_wc}W/{rc_lc}L\n"
        )
    logger.info(f"Sending result card for {symbol} win={win} profit={profit:.2f}")
    _send_tg(result_msg)
    _log(f"{'WIN' if win else 'LOSS'}  {symbol}  ${profit:+.2f}  total=${pnl_snap:+.2f}  "
         f"session={mkt_session}")

    # ── Martingale instant re-entry (fires immediately, no signal gates) ──
    if do_martingale_reentry and not _paused_snap and not _sess_reset_snap:
        _ml_new   = martingale_level.get(symbol, 0)
        _ml_stake = round(STAKE * (MARTINGALE_MULTIPLIER ** _ml_new), 2)
        _dir_lbl  = "🟢 Rise (UP)" if reentry_direction == "UP" else "🔴 Fall (DOWN)"
        # Always use the live ws from the registry — the on_contract_update ws
        # may be the same, but ws_registry ensures we have a fresh connection.
        _reentry_ws = ws_registry.get(symbol) or ws
        if _reentry_ws is None:
            _log(f"⚠  Martingale re-entry for {symbol} skipped — no live WS")
            with _lock:
                martingale_level[symbol] = 0
        else:
            _send_tg(
                f"⚡ <b>Martingale Re-entry — Step {_ml_new}/{MARTINGALE_MAX_STEPS}</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"Symbol    : <code>{symbol}</code>\n"
                f"Direction : {_dir_lbl}  <i>(same as losing trade)</i>\n"
                f"Stake     : <b>${_ml_stake:.2f}</b>  "
                f"(×{int(MARTINGALE_MULTIPLIER**_ml_new)} base ${STAKE:.2f})\n"
                f"<i>Bypassing signal gates — instant recovery entry.</i>"
            )
            request_proposal(_reentry_ws, symbol, reentry_details, reentry_direction)
            _log(f"⚡ Martingale re-entry {symbol} {reentry_direction}  "
                 f"step={_ml_new}  stake=${_ml_stake:.2f}")
    elif do_martingale_reentry and (_paused_snap or _sess_reset_snap):
        # Bot paused or session ended — suppress re-entry, reset level cleanly
        with _lock:
            martingale_level[symbol] = 0
        reason = "session reset" if _sess_reset_snap else "bot paused"
        _log(f"↩️  Martingale {symbol}: re-entry suppressed ({reason}) — level reset to 0")

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
                f"▶ Bot is back to trading. Let's go! 💪"
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
    """
    Recover contracts whose real settlement message never arrived (missed
    proposal_open_contract update — common when several contracts settle
    around the same time and the per-symbol WS socket drops a message).

    Two-stage recovery:
      1. At expiry + 5 min grace: actively RE-QUERY the contract's real
         status from Deriv (one-shot, non-subscribed request) instead of
         guessing. If Deriv answers, the normal on_contract_update() path
         settles it correctly (win or loss) — this is the fix for "wins
         getting counted as losses" when multiple trades are open at once.
      2. Only if the re-query itself gets no answer within an additional
         grace window (contract truly unreachable / API issue) do we fall
         back to a synthetic settlement, and we now record it as an
         UNKNOWN/void result (profit 0, not auto-counted as a loss) so it
         doesn't silently corrupt the win/loss stats — the user is told to
         verify manually instead.
    """
    while True:
        time.sleep(30)
        try:
            now = datetime.now(timezone.utc)
            grace = timedelta(minutes=DURATION + 5)
            requery_grace = timedelta(minutes=DURATION + 10)

            with _lock:
                need_requery = [
                    (cid, dict(info))
                    for cid, info in active_contracts.items()
                    if not info.get("settled")
                    and not info.get("requeried")
                    and (now - info.get("entry_time", now)) > grace
                ]
                still_stale = [
                    (cid, dict(info))
                    for cid, info in active_contracts.items()
                    if not info.get("settled")
                    and info.get("requeried")
                    and (now - info.get("entry_time", now)) > requery_grace
                ]

            # Stage 1: ask Deriv for the real status before assuming anything
            for cid, info in need_requery:
                sym = info["symbol"]
                ws  = ws_registry.get(sym)
                if ws is None:
                    logger.warning(f"⏰ {sym} contract {cid} stale but no live WS to re-query — will retry")
                    continue
                try:
                    ws.send(json.dumps({
                        "proposal_open_contract": 1,
                        "contract_id": int(cid),
                    }))
                    logger.info(f"🔎 Re-queried stale contract {cid} ({sym}) for real result")
                    # Only mark as requeried once the send actually succeeded —
                    # otherwise leave it eligible for retry on the next tick
                    # instead of silently aging into an assumed VOID.
                    with _lock:
                        if cid in active_contracts:
                            active_contracts[cid]["requeried"] = True
                except Exception as e:
                    logger.warning(f"⏰ {sym} contract {cid} re-query send failed: {e} — will retry")

            # Stage 2: still nothing after the re-query grace — mark as
            # unresolved (void), never auto-guessed as a loss.
            for cid, info in still_stale:
                sym    = info["symbol"]
                logger.warning(f"⏰ Contract {cid} ({sym}) unresolved after re-query — marking VOID")
                _send_tg(
                    f"⚠️ <b>UNRESOLVED CONTRACT</b> – {sym}\n"
                    f"Contract #{cid} had no result after {DURATION+10} min, "
                    f"even after re-querying Deriv.\n"
                    f"Recorded as VOID (not counted as win/loss).\n"
                    f"<i>Check your Deriv account to confirm the real outcome.</i>"
                )
                synthetic_msg = {
                    "proposal_open_contract": {
                        "contract_id": cid,
                        "is_expired": True,
                        "profit": 0,
                        "status": "void",
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
    ws_registry[symbol] = ws
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
                            # For Rise/Fall the entry fires immediately on tick-momentum
                            # confirmation — the M5 + ML gates below are the quality control.
                            with _lock:
                                locked_symbols.pop(symbol, None)

                            _record_scan()

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
                                _record_funnel_rejection("M5 Supertrend disagreement")
                                _send_rejection(symbol, direction, score,
                                                f"M5 Supertrend disagrees "
                                                f"({'↑' if m5_dir==1 else '↓'} on M5 vs {direction} on M1)",
                                                details)
                            elif (m15_ready and m15_dir != 0
                                    and ((direction == "UP" and m15_dir != 1)
                                         or (direction == "DOWN" and m15_dir != -1))):
                                _record_funnel_rejection("M15 Supertrend disagreement")
                                _send_rejection(symbol, direction, score,
                                                f"M15 Supertrend disagrees "
                                                f"({'↑' if m15_dir==1 else '↓'} on M15 vs {direction} on M1)",
                                                details)
                            else:
                                # ── ML gate: second-entry requires higher confidence ────
                                now_t = datetime.now(timezone.utc)
                                # Expire stale second-entry windows first (under lock for safety)
                                with _lock:
                                    _se_expiry = second_entry_eligible.get(symbol)
                                    if _se_expiry is not None and now_t >= _se_expiry:
                                        second_entry_eligible.pop(symbol, None)
                                        _se_expiry = None
                                _is_second = _se_expiry is not None  # expiry already validated
                                _ml_gate   = SECOND_ENTRY_ML_MIN if _is_second else None
                                _gate_lbl  = (f"{SECOND_ENTRY_ML_MIN*100:.0f}% (2nd entry)"
                                              if _is_second else f"{ML_CONFIDENCE_MIN*100:.0f}%")
                                if not _ml_should_trade(details, symbol, min_conf=_ml_gate):
                                    conf = details.get("ml_confidence", 0)
                                    _record_funnel_rejection("ML confidence below gate")
                                    _send_rejection(symbol, direction, score,
                                                    f"ML confidence {conf*100:.1f}% below {_gate_lbl} gate "
                                                    f"[{_get_symbol_class(symbol)}]",
                                                    details)
                                elif _is_second:
                                    if _reserve_trade_slot(symbol, now_t):
                                        # Success — consume the second-entry window now
                                        with _lock:
                                            second_entry_eligible.pop(symbol, None)
                                        request_proposal(ws, symbol, details, direction)
                                        _log(f"🔁 {symbol} {direction} SECOND-ENTRY  score={score}/100  "
                                             f"conf={details.get('ml_confidence',1.0)*100:.0f}%  "
                                             f"class={_get_symbol_class(symbol)}")
                                        _send_tg(
                                            f"🔁 <b>SECOND ENTRY</b> — <code>{symbol}</code> {direction}\n"
                                            f"Score: <b>{score}/100</b>  ML: {details.get('ml_confidence',1.0)*100:.0f}%\n"
                                            f"<i>Re-entry after win — trend still confirmed.</i>"
                                        )
                                    else:
                                        # Gate blocked this tick — window stays open, retry next tick
                                        _record_funnel_rejection("Risk gate closed (paused/cooldown/daily limit)")
                                        _send_rejection(symbol, direction, score,
                                                        "Risk gate closed for 2nd entry (paused / daily limit)",
                                                        details)
                                elif _reserve_trade_slot(symbol, now_t):
                                    # ── Confirmation gate ────────────────────────────────────
                                    # Require at least one momentum confirmation before firing.
                                    # Prevents "perfect structure, early entry" trades that
                                    # have no actual seller/buyer returning to the zone yet.
                                    _has_conf = True   # default: pass if gate disabled
                                    if CONFIRMATION_GATE_ENABLED:
                                        _pat = details.get("candle_pattern", "none")
                                        _pat_bias = details.get("candle_pattern_bias", "neutral")
                                        _pat_ok = (
                                            _pat not in ("none", "", None) and
                                            ((direction == "UP"   and _pat_bias == "bullish") or
                                             (direction == "DOWN" and _pat_bias == "bearish"))
                                        )
                                        _macd_b = details.get("macd_bullish", False)
                                        _macd_r = details.get("macd_hist_rising", False)
                                        _macd_ok = (
                                            (direction == "UP"   and _macd_b and _macd_r) or
                                            (direction == "DOWN" and not _macd_b and _macd_r)
                                        )
                                        _roc = details.get("roc")
                                        _roc_ok = (
                                            (direction == "UP"   and _roc is not None and _roc > 0) or
                                            (direction == "DOWN" and _roc is not None and _roc < 0)
                                        )
                                        _wick_ok = details.get("wick_pts", 0) >= 4
                                        _has_conf = _pat_ok or _macd_ok or _roc_ok or _wick_ok
                                        details["has_confirmation"] = _has_conf

                                    if not _has_conf:
                                        _release_trade_slot(symbol)
                                        _record_funnel_rejection("No entry confirmation (MACD/ROC/pattern/wick)")
                                        _log(f"⏳ {symbol} {direction} score={score}/100 — no confirmation yet, holding")
                                    else:
                                        request_proposal(ws, symbol, details, direction)
                                        _log(f"🎯 {symbol} {direction} TICK-ENTRY  score={score}/100  "
                                             f"momentum={len(tick_history.get(symbol,[]))}ticks  "
                                             f"conf={details.get('ml_confidence',1.0)*100:.0f}%  "
                                             f"class={_get_symbol_class(symbol)}")
                                else:
                                    _record_funnel_rejection("Risk gate closed (paused/cooldown/daily limit)")
                                    _send_rejection(symbol, direction, score,
                                                    "Risk gate closed (paused / cooldown / daily limit)",
                                                    details)

                with _lock:
                    current_candle[symbol] = dict(c)

        elif mtype == "proposal":
            on_proposal(ws, msg, symbol)
        elif mtype == "buy":
            on_buy(ws, msg, symbol)
        elif mtype == "proposal_open_contract":
            on_contract_update(ws, msg, symbol)
        elif mtype == "portfolio":
            on_portfolio(ws, msg, symbol)
        elif mtype == "error":
            err = msg.get("error", {})
            err_msg = err.get("message", err)
            logger.error(f"{symbol} API error: {err_msg}")
            with _lock:
                det = pending_signals.pop(symbol, None)
                ub  = unconfirmed_buys.pop(symbol, None)
                if ub and det is None:
                    det = ub.get("details")
            if det is not None:
                is_reentry = _cleanup_failed_entry(symbol, det)
                _send_tg(
                    f"⚠️ <b>Trade request failed</b> — <code>{symbol}</code>\n"
                    f"Deriv API error: {err_msg}\n"
                    + ("Martingale step reset to base — no trade was opened."
                       if is_reentry else "Slot released.")
                )
        elif mtype not in (None, "authorize", "ping"):
            # Diagnostic net: log any message type we don't explicitly route,
            # so a repeat of the "buy sent, ack never arrives" incident shows
            # exactly what (if anything) Deriv actually sent back, instead of
            # leaving us guessing between a dropped request and a misrouted
            # response. "authorize"/"ping" are expected, already-handled-
            # elsewhere connection housekeeping — never worth logging, and
            # authorize's payload includes account balance/email, so it must
            # not be dumped into the log on every (re)connect.
            logger.warning(f"{symbol} unhandled msg_type={mtype}: {msg}")

    except json.JSONDecodeError as e:
        logger.error(f"{symbol} bad JSON: {e}")
    except Exception as e:
        logger.exception(f"{symbol} on_message exception: {e}")


def _on_error(ws, error):
    logger.error(f"WS error: {error}")


def _cleanup_failed_entry(symbol: str, det: dict):
    """Common cleanup for any proposal/buy that never completed (timeout or
    explicit API error). Martingale re-entries never reserve a daily-trade
    slot (see request_proposal / on_contract_update comments), so releasing
    one for them would wrongly decrement daily_trades and wipe the symbol's
    cooldown. They also need martingale_level reset — otherwise a technical
    failure (no trade ever opened) silently escalates the NEXT real signal's
    stake as if it were continuing a real loss streak.
    """
    is_reentry = bool((det or {}).get("_is_martingale_reentry"))
    if is_reentry:
        martingale_level[symbol] = 0
    else:
        _release_trade_slot(symbol)
    return is_reentry


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
                        _cleanup_failed_entry(symbol, det)
                        _log(f"⏰ {symbol} pending proposal timed out — slot released")
                for symbol in list(unconfirmed_buys.keys()):
                    entry = unconfirmed_buys[symbol]
                    if now < entry["expires_at"]:
                        continue
                    # Before giving up, confirm with Deriv's own portfolio
                    # that no contract actually opened — a lost ack does not
                    # mean a lost trade; the buy may have gone through with
                    # only the confirmation dropped. Give that check ~15s.
                    if not entry.get("portfolio_checked_at"):
                        ws2 = ws_registry.get(symbol)
                        if ws2 is not None:
                            try:
                                ws2.send(json.dumps({"portfolio": 1}))
                                _portfolio_checks[symbol] = {
                                    "details": entry.get("details"),
                                    "requested_at": now,
                                }
                                entry["portfolio_checked_at"] = now
                                entry["expires_at"] = now + timedelta(seconds=15)
                                _log(f"🔎 {symbol} checking Deriv portfolio before declaring "
                                     f"trade unconfirmed")
                                continue
                            except Exception as e:
                                logger.warning(f"{symbol} portfolio safety check failed: {e}")
                        # No live WS to check with — fall through and finalize as before.
                    det = unconfirmed_buys.pop(symbol, None)
                    _portfolio_checks.pop(symbol, None)
                    is_reentry = _cleanup_failed_entry(symbol, (det or {}).get("details"))
                    _send_tg(
                        f"⚠️ <b>UNCONFIRMED TRADE EXPIRED</b> — {symbol}\n"
                        f"Buy order sent but no ack within 3 min "
                        f"(confirmed via Deriv portfolio — no contract opened).\n"
                        + (f"Martingale step reset to base — no trade was opened.\n"
                           if is_reentry else
                           f"Slot released — check your Deriv account.\n")
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
        # Drop the stale socket so the watchdog doesn't try to send on a dead
        # connection while we're reconnecting.
        if ws_registry.get(symbol) is ws_app:
            ws_registry.pop(symbol, None)
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
    pause_lbl = "▶ Resume Bot" if is_paused else "⏸ Pause Bot"
    return InlineKeyboardMarkup([
        # ── Live Status ──────────────────────────────────────────────────
        [InlineKeyboardButton("📊 Bot Status · Trades",     callback_data="status"),
         InlineKeyboardButton("💰 P&L · Win Rate",          callback_data="pnl")],
        # ── Trade History ────────────────────────────────────────────────
        [InlineKeyboardButton("📜 Recent Trades",           callback_data="history"),
         InlineKeyboardButton("📋 Signal Log",              callback_data="signals")],
        # ── Session Reporting ─────────────────────────────────────────────
        [InlineKeyboardButton("📄 This Session Report",     callback_data="session_report"),
         InlineKeyboardButton("🏆 All-Time Stats",          callback_data="alltime")],
        [InlineKeyboardButton("📅 Today's Session Log",     callback_data="daily_history"),
         InlineKeyboardButton("🏅 Best / Worst Symbol",     callback_data="best_worst")],
        # ── Analysis ─────────────────────────────────────────────────────
        [InlineKeyboardButton("📈 Signal Score Charts",     callback_data="score_sparklines"),
         InlineKeyboardButton("🌏 Sessions Breakdown",      callback_data="market_sessions")],
        [InlineKeyboardButton("📆 7-Day P&L",               callback_data="seven_day_pnl"),
         InlineKeyboardButton("🤖 ML Confidence Perf",      callback_data="ml_conf_perf")],
        [InlineKeyboardButton("🔬 Run Backtest",            callback_data="backtest"),
         InlineKeyboardButton("📊 Export Trades CSV",       callback_data="export_csv_quick")],
        [InlineKeyboardButton("📉 Performance Analytics",   callback_data="analytics"),
         InlineKeyboardButton("🧬 Pattern Discovery",       callback_data="patterns")],
        # ── Control ──────────────────────────────────────────────────────
        [InlineKeyboardButton(pause_lbl,                    callback_data="toggle_pause"),
         InlineKeyboardButton("⏭ Skip a Symbol",           callback_data="skip_menu")],
        [InlineKeyboardButton("⚙ Settings & Config",       callback_data="settings"),
         InlineKeyboardButton("🧪 Fire Test Trade",         callback_data="test_menu")],
        # ── Payout quick adjust ────────────────────────────────────────────
        [InlineKeyboardButton("➖ Lower Payout",           callback_data="payout_down"),
         InlineKeyboardButton("➕ Raise Payout",          callback_data="payout_up")],
        # ── Tools ────────────────────────────────────────────────────────
        [InlineKeyboardButton("🔄 Refresh Dashboard",       callback_data="refresh"),
         InlineKeyboardButton("📦 Backup · Restore",        callback_data="backup")],
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
        [InlineKeyboardButton("🪜 Step Index",             callback_data="tg_step")],
        [InlineKeyboardButton("⚡ Jump Indices (JD*)",    callback_data="tg_jump")],
        [InlineKeyboardButton("🔙 Back",                  callback_data="main_menu")],
    ])


def _test_sym_kb(group: str) -> InlineKeyboardMarkup:
    groups = {
        "tg_vol":   SYNTH_VOLATILITY,
        "tg_vol1s": SYNTH_VOLATILITY_1S,
        "tg_rdb":   SYNTH_RANGE_BREAK,
        "tg_step":  SYNTH_STEP,
        "tg_jump":  SYNTH_JUMP,
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
        pnl       = total_pnl
        wc, lc    = win_count, loss_count
        cl        = consecutive_losses
        peak      = peak_equity
        mdd       = max_drawdown
    mkt_sess = _get_session_name(now)
    tot      = wc + lc
    wr       = wc / tot * 100 if tot else 0
    cur_dd   = max(0.0, peak - pnl)
    mdd_pct  = (mdd / peak * 100) if peak > 0 else 0.0
    state_emoji = "⏸" if is_paused else "▶"

    lines = [
        f"📊 <b>DERIV SNIPER BOT — STATUS</b>  <i>{now.strftime('%H:%M:%S UTC')}</i>",
        "━━━━━━━━━━━━━━━━━━━━",
        f"{state_emoji} State : <b>{'PAUSED' if is_paused else 'RUNNING'}</b>",
        f"🌐 Time  : <b>Deriv Server Time (UTC)</b>",
        f"🌍 Market : {SESSION_EMOJIS.get(mkt_sess,'')} <b>{mkt_sess}</b>",
        "",
        "<b>💰 P&amp;L — Current Session</b>",
        f"  Total P&amp;L : <b>{'+' if pnl >= 0 else ''}${pnl:.2f}</b>",
        f"  Trades    : {tot}  ({wc}W / {lc}L)  •  Win Rate : <b>{wr:.0f}%</b>",
        f"  Drawdown  : ${cur_dd:.2f}  •  Max DD : ${mdd:.2f} ({mdd_pct:.1f}%)",
        f"  Consec. Losses : {cl} / {MAX_CONSECUTIVE_LOSSES}",
        "",
        "<b>⚙ Engine</b>",
        f"  Day trades : {trades}  •  Active : {len(contracts)}  •  Cooldowns : {len(cds)}",
        f"  {_ml_progress_text()}",
        "",
        _best_worst_line(),
    ]
    if contracts:
        lines.append("<b>🎯 Active Positions</b>")
        for cid, c in contracts.items():
            exp  = c["entry_time"] + timedelta(minutes=DURATION)
            left = max(0, int((exp - now).total_seconds()))
            m, s = divmod(left, 60)
            lines.append(f"  • <code>{c['symbol']:<10}</code> #{cid} {c['direction']:<4} {m:02d}:{s:02d} left")
    if cds:
        lines.append("<b>🚫 Cooldowns</b>")
        for sym, t in cds.items():
            lines.append(f"  • <code>{sym:<10}</code> {int((t - now).total_seconds() // 60)}m left")
    lines.append("━━━━━━━━━━━━━━━━━━━━")
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
    adv   = get_alltime_advanced_stats()

    lines = [
        "🏆 <b>ALL-TIME SCOREBOARD</b>",
        "━━━━━━━━━━━━━━━━━━━━",
        _strategy_header(),
        f"💵 <b>Total P&L</b>  : {'+' if ap >= 0 else ''}${ap:.2f}",
        f"📊 <b>Trades</b>     : {at}  ({aw}W / {al}L)",
        f"🎯 <b>Win Rate</b>   : {at_wr:.1f}%",
        f"📈 <b>Avg/Trade</b>  : {'+' if avg >= 0 else ''}${avg:.2f}",
    ]

    if adv:
        pf_str = "∞" if adv["profit_factor"] == float("inf") else f"{adv['profit_factor']:.2f}"
        rf_str = "∞" if adv["recovery_factor"] == float("inf") else f"{adv['recovery_factor']:.2f}"
        lines += [
            "",
            "<b>Trade Quality</b>",
            f"  Gross Profit    : +${adv['gross_profit']:.2f}",
            f"  Gross Loss      : -${abs(adv['gross_loss']):.2f}",
            f"  Profit Factor   : {pf_str}",
            f"  Avg Win         : +${adv['avg_win']:.2f}",
            f"  Avg Loss        : -${abs(adv['avg_loss']):.2f}",
            f"  Largest Win     : +${adv['largest_win']:.2f}",
            f"  Largest Loss    : -${abs(adv['largest_loss']):.2f}",
            f"  Expectancy/Trade: {'+' if adv['expectancy']>=0 else ''}${adv['expectancy']:.2f}",
            "",
            "<b>Drawdown & Recovery</b>",
            f"  Peak Equity     : ${adv['peak_equity']:.2f}",
            f"  Max Drawdown    : -${adv['max_drawdown']:.2f}",
            f"  Recovery Factor : {rf_str}",
            "",
            "<b>Streaks</b>",
            f"  Longest Win  Streak: {adv['longest_win_streak']}",
            f"  Longest Loss Streak: {adv['longest_loss_streak']}",
        ]
        if adv["best_symbol"]:
            lines += [
                "",
                "<b>Best / Worst Symbol (all-time)</b>",
                f"  🏅 {adv['best_symbol']}  {'+' if adv['best_symbol_pnl']>=0 else ''}${adv['best_symbol_pnl']:.2f}",
                f"  💔 {adv['worst_symbol']}  {'+' if adv['worst_symbol_pnl']>=0 else ''}${adv['worst_symbol_pnl']:.2f}",
            ]
        if adv["best_session"]:
            lines += [
                "",
                "<b>Best / Worst Market Session (all-time)</b>",
                f"  🏅 {adv['best_session']}  {'+' if adv['best_session_pnl']>=0 else ''}${adv['best_session_pnl']:.2f}",
                f"  💔 {adv['worst_session']}  {'+' if adv['worst_session_pnl']>=0 else ''}${adv['worst_session_pnl']:.2f}",
            ]

    lines += ["", "<b>By Symbol (all-time)</b>"]
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
        icon = "🏆" if win == 1 else ("💀" if win == 0 else "⚪")
        lines.append(
            f"{icon}  {ts[11:16]}  <code>{sym:<7}</code> {direction} "
            f"{sign}${profit:.2f}  score={score:.0f}"
        )
    return "\n".join(lines)


def _signals_text():
    now_str = datetime.now(timezone.utc).strftime("%H:%M UTC")
    with _lock:
        raw_lines = list(signal_log)

    out = [f"📋 <b>Signal Log</b>  [{now_str}]\n━━━━━━━━━━━━━━━━━━━━"]

    if raw_lines:
        # HTML-escape each line so '<' in rejection reasons doesn't break parse_mode=HTML
        escaped = [_html.escape(ln) for ln in raw_lines[:15]]
        out.append("<b>Live Session Activity</b>")
        out += escaped
    else:
        out.append("<i>No signals in memory this session.</i>")

    # Always append recent trades from DB for context
    recent = get_recent_trades(8)
    if recent:
        out += ["", "━━━━━━━━━━━━━━━━━━━━", "<b>Recent Trades (DB)</b>"]
        for ts, sym, direction, profit, win_flag, score in recent:
            sign = "+" if profit > 0 else ""
            icon = "🏆" if win_flag == 1 else ("💀" if win_flag == 0 else "⚪")
            out.append(
                f"{icon} {ts[11:16]}  <code>{sym:<7}</code> {direction} "
                f"{sign}${profit:.2f}  score={score:.0f}"
            )
    else:
        out += ["", "<i>No trades in DB yet.</i>"]

    return "\n".join(out)


def _funnel_report_lines(funnel: dict) -> list:
    """Render the signal funnel (scanned/executed/rejected + top reasons)."""
    scanned  = funnel["scanned"]
    executed = funnel["executed"]
    rejects  = funnel["rejections"]
    total_rejected = sum(rejects.values())
    lines = [
        "<b>Signal Funnel (today)</b>",
        f"  Scanned  : {scanned}",
        f"  Executed : {executed}",
        f"  Rejected : {total_rejected}",
    ]
    if rejects:
        lines.append("  Top rejection reasons:")
        for reason, cnt in sorted(rejects.items(), key=lambda kv: kv[1], reverse=True)[:5]:
            lines.append(f"    • {reason}: {cnt}")
    return lines


def _daily_market_session_lines() -> list:
    """Compact per-market-session (Asian/London/NY, fixed UTC windows) mini
    breakdown for today — these windows are fixed clock ranges and never
    overlap, unlike the TP/SL-triggered 'rounds' listed below them, which
    can start and end mid-session. Kept separate so the two concepts don't
    get conflated in the report."""
    _roll_market_session_stats_if_needed()
    with _lock:
        ms_snap = {k: dict(v) for k, v in market_session_stats.items()}
    broad_groups = {
        "Midnight": ["Midnight"], "Asian": ["Early Asian", "Late Asian"],
        "London": ["London"], "New York": ["Early New York", "Late New York"],
    }
    out = ["<b>By Market Session (today, fixed UTC windows)</b>"]
    any_data = False
    for broad, subs in broad_groups.items():
        w = l = 0; pnl = 0.0
        for sub in subs:
            s = ms_snap.get(sub, {"wins": 0, "losses": 0, "pnl": 0.0})
            w += s["wins"]; l += s["losses"]; pnl += s["pnl"]
        tot = w + l
        if tot == 0:
            continue
        any_data = True
        wr = w / tot * 100 if tot else 0
        sign = "+" if pnl >= 0 else ""
        tag = "🏅" if pnl > 0 else ("💔" if pnl < 0 else "➖")
        out.append(f"  {SESSION_EMOJIS.get(broad,'')} {broad:<9} {tag} {sign}${pnl:.2f}  ({w}W/{l}L {wr:.0f}%WR)")
    if not any_data:
        out.append("  — no trades yet today —")
    out.append("<i>Full breakdown incl. all-time: 🌏 Sessions Breakdown button.</i>")
    return out


def _daily_history_text() -> str:
    global daily_session_log, _daily_session_log_date
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with _lock:
        if today != _daily_session_log_date:
            daily_session_log    = []
            _daily_session_log_date = today
        log = list(daily_session_log)

    funnel = _funnel_snapshot()
    funnel_lines = _funnel_report_lines(funnel)

    # Always source the headline figure from the dedicated daily accumulator
    # (persists across TP/SL round resets, only zeroed at UTC midnight) —
    # NOT from summing closed rounds, which used to be blank/wrong whenever
    # no TP/SL round had closed yet today even though real trades happened.
    with _lock:
        live_pnl, live_peak, live_mdd = total_pnl, peak_equity, max_drawdown
        live_trades, live_wins, live_losses = daily_trades, win_count, loss_count
        day_pnl_total = daily_total_pnl

    if not log:
        day_wr = live_wins / (live_wins + live_losses) * 100 if (live_wins + live_losses) else 0
        lines = [
            "📅 <b>Daily Session History</b>", "━━━━━━━━━━━━━━━━━━━━",
            _strategy_header(),
            f"Date: {today}",
            f"Day P&L    : <b>{'+' if day_pnl_total >= 0 else ''}${day_pnl_total:.2f}</b>",
            f"Trades     : {live_trades}   Wins: {live_wins}   Losses: {live_losses}   WR: {day_wr:.1f}%",
            f"📉 Drawdown : -${live_mdd:.2f}",
            "",
            "— No TP or SL round has closed yet today —",
            "",
            *_daily_market_session_lines(),
            "",
            *funnel_lines,
            "",
            "<i>Each time the bot hits its daily TP or SL it resets and a new round begins below. Resets at midnight UTC.</i>",
        ]
        return "\n".join(lines)

    tp_hits = sum(1 for e in log if e["reason"] == "TP")
    sl_hits = sum(1 for e in log if e["reason"] == "SL")
    total_day_trades = sum(e["trades"] for e in log) + live_trades
    total_day_wins   = sum(e["wins"]   for e in log) + live_wins
    total_day_losses = sum(e["losses"] for e in log) + live_losses
    total_possible   = total_day_wins + total_day_losses
    day_wr = total_day_wins / total_possible * 100 if total_possible else 0
    day_avg_trade  = day_pnl_total / total_day_trades if total_day_trades else 0.0
    day_max_dd     = max((e.get("max_dd", 0.0) for e in log), default=0.0)
    day_max_dd_sess = max(log, key=lambda e: e.get("max_dd", 0.0)) if log else None
    day_max_dd = max(day_max_dd, live_mdd)

    adv_today = get_today_advanced_stats()

    lines = [
        "📅 <b>Daily Session History</b>",
        "━━━━━━━━━━━━━━━━━━━━",
        _strategy_header(),
        f"Date   : {today}",
        f"🎯 TP hits: <b>{tp_hits}</b>   🛑 SL hits: <b>{sl_hits}</b>",
        f"Day P&L    : <b>{'+' if day_pnl_total >= 0 else ''}${day_pnl_total:.2f}</b>",
        f"Total Trades: {total_day_trades}  <i>(incl. current round)</i>",
        f"Total Wins  : {total_day_wins}",
        f"Total Losses: {total_day_losses}",
        f"Win Rate    : {day_wr:.1f}%",
        f"Avg/Trade   : {'+' if day_avg_trade >= 0 else ''}${day_avg_trade:.2f}",
        f"📉 Max Drawdown: -${day_max_dd:.2f}"
        + (f"  (in {day_max_dd_sess['market_session']} session)"
           if day_max_dd_sess and day_max_dd == day_max_dd_sess.get("max_dd", 0.0) and day_max_dd > 0
           else ("  (current session)" if day_max_dd > 0 and day_max_dd == live_mdd else "")),
    ]

    if adv_today:
        lines += [
            "",
            "<b>Today's Trade Quality (DB)</b>",
            f"  Gross Profit : +${adv_today['gross_profit']:.2f}",
            f"  Gross Loss   : -${abs(adv_today['gross_loss']):.2f}",
            f"  Longest Win  Streak: {adv_today['longest_win_streak']}",
            f"  Longest Loss Streak: {adv_today['longest_loss_streak']}",
        ]
        if adv_today["best_symbol"]:
            lines.append(
                f"  🏅 Best Symbol : {adv_today['best_symbol']}  "
                f"{'+' if adv_today['best_symbol_pnl']>=0 else ''}${adv_today['best_symbol_pnl']:.2f}"
            )
            lines.append(
                f"  💔 Worst Symbol: {adv_today['worst_symbol']}  "
                f"{'+' if adv_today['worst_symbol_pnl']>=0 else ''}${adv_today['worst_symbol_pnl']:.2f}"
            )
        if adv_today["best_session"]:
            lines.append(
                f"  🏅 Best Session : {adv_today['best_session']}  "
                f"{'+' if adv_today['best_session_pnl']>=0 else ''}${adv_today['best_session_pnl']:.2f}"
            )
            lines.append(
                f"  💔 Worst Session: {adv_today['worst_session']}  "
                f"{'+' if adv_today['worst_session_pnl']>=0 else ''}${adv_today['worst_session_pnl']:.2f}"
            )

    lines += ["", *_daily_market_session_lines(), "", *funnel_lines, "━━━━━━━━━━━━━━━━━━━━",
              "<b>TP/SL Rounds today</b>  <i>(each starts on the previous reset, ends on the next "
              "TP/SL hit — a round can span more than one market session above)</i>"]

    for i, e in enumerate(log, 1):
        icon  = "🎯" if e["reason"] == "TP" else "🛑"
        sign  = "+" if e["pnl"] >= 0 else ""
        h, mr = divmod(e["duration_min"], 60)
        dur_str  = f"{h}h {mr}m" if h else f"{mr}m"
        ms_name  = e.get("market_session", "—")
        e_wr  = e["wins"] / (e["wins"] + e["losses"]) * 100 if (e["wins"] + e["losses"]) else 0
        entry = (
            f"{icon} <b>Round {i}</b>  [{e['time']}]  {dur_str}  "
            f"ended in {SESSION_EMOJIS.get(ms_name,'')} {ms_name}\n"
            f"   P&L: <b>{sign}${e['pnl']:.2f}</b>  ·  "
            f"Trades: {e['trades']} ({e['wins']}W/{e['losses']}L  {e_wr:.0f}%WR)\n"
            f"   Peak: ${e.get('peak', 0.0):.2f}  ·  Max DD: -${e.get('max_dd', 0.0):.2f}"
        )
        if e["best_sym"]:
            b_sign = "+" if e["best_pnl"] >= 0 else ""
            w_sign = "+" if e["worst_pnl"] >= 0 else ""
            entry += (
                f"\n   🏅 {e['best_sym']} {b_sign}${e['best_pnl']:.2f}  "
                f"· 💔 {e['worst_sym']} {w_sign}${e['worst_pnl']:.2f}"
            )
        lines.append(entry)
        lines.append("")   # blank separator so back-to-back sessions don't run together

    if live_trades or live_wins or live_losses:
        live_wr = live_wins / (live_wins + live_losses) * 100 if (live_wins + live_losses) else 0
        lines.append(
            f"▶ <b>Current round (still running)</b>\n"
            f"   P&L: <b>{'+' if live_pnl >= 0 else ''}${live_pnl:.2f}</b>  ·  "
            f"Trades: {live_trades} ({live_wins}W/{live_losses}L  {live_wr:.0f}%WR)\n"
            f"   Peak: ${live_peak:.2f}  ·  Max DD: -${live_mdd:.2f}"
        )
        lines.append("")

    lines += ["━━━━━━━━━━━━━━━━━━━━", f"<i>Resets at midnight UTC. Today: {today}</i>"]
    return "\n".join(lines)


def _market_sessions_text() -> str:
    """Current-day per-market-session breakdown with UTC time ranges."""
    _roll_market_session_stats_if_needed()
    now = datetime.now(timezone.utc)
    current = _get_session_name(now)
    with _lock:
        ms_snap = {k: dict(v) for k, v in market_session_stats.items()}

    # All-time session stats from DB
    at_rows = {row[0]: row for row in get_session_alltime_stats()}

    # Next session countdown
    next_dt   = _next_session_start(now)
    secs_left = max(0, int((next_dt - now).total_seconds()))
    hh_l, rem = divmod(secs_left, 3600)
    mm_l = rem // 60
    next_in_str = f"{hh_l}h {mm_l}m" if hh_l else f"{mm_l}m"

    lines = [
        "🌏 <b>Market Session Performance</b>",
        "━━━━━━━━━━━━━━━━━━━━",
        f"Now  : {SESSION_EMOJIS.get(current,'')} <b>{current}</b>  "
        f"({now.strftime('%H:%M UTC')})  <i>{SESSION_TIMES.get(current,'')}</i>",
        f"Next : starts in <b>{next_in_str}</b>  ({next_dt.strftime('%H:%M UTC')})",
        "",
        "<b>Today (in-memory)</b>",
    ]

    # ── Sub-session breakdown (today) ──────────────────────────────────
    broad_groups = {
        "Midnight":   ["Midnight"],
        "Asian":      ["Early Asian", "Late Asian"],
        "London":     ["London"],
        "New York":   ["Early New York", "Late New York"],
    }
    for broad, subs in broad_groups.items():
        btot_w = btot_l = btot_pnl = 0
        for sub in subs:
            s = ms_snap.get(sub, {"wins": 0, "losses": 0, "pnl": 0.0, "trades": 0})
            btot_w += s["wins"]; btot_l += s["losses"]; btot_pnl += s["pnl"]
        btot  = btot_w + btot_l
        bwr   = btot_w / btot * 100 if btot else 0
        bsign = "+" if btot_pnl >= 0 else ""
        broad_em = SESSION_EMOJIS.get(broad, "")
        lines.append(
            f"\n{broad_em} <b>{broad}</b>  "
            f"[{btot} trades  {btot_w}W/{btot_l}L  {bwr:.0f}%WR  {bsign}${btot_pnl:.2f}]"
        )
        for sub in subs:
            s   = ms_snap.get(sub, {"wins": 0, "losses": 0, "pnl": 0.0, "trades": 0})
            tot = s["wins"] + s["losses"]
            wr  = s["wins"] / tot * 100 if tot else 0
            sign = "+" if s["pnl"] >= 0 else ""
            live_tag  = " ◄ LIVE" if sub == current else ""
            sub_em    = SESSION_EMOJIS.get(sub, "")
            time_tag  = SESSION_TIMES.get(sub, "")
            lines.append(
                f"  {sub_em} <b>{sub:<16}</b>  <i>{time_tag}</i>{live_tag}\n"
                f"    {tot} trades  {s['wins']}W/{s['losses']}L ({wr:.0f}%WR)  "
                f"P&amp;L: <b>{sign}${s['pnl']:.2f}</b>"
            )

    lines += ["", "<b>All-Time (DB)</b>"]
    if at_rows:
        for name in ["Midnight", "Early Asian", "Late Asian",
                     "London", "Early New York", "Late New York"]:
            row = at_rows.get(name)
            if row:
                _, cnt, wins, pnl = row
                wins = wins or 0; pnl = pnl or 0.0
                wr   = wins / cnt * 100 if cnt else 0
                sign = "+" if pnl >= 0 else ""
                t_range = SESSION_TIMES.get(name, "")
                lines.append(
                    f"  {SESSION_EMOJIS.get(name,'')} {name:<18}  <i>{t_range}</i>\n"
                    f"    {cnt} trades  {wins}W ({wr:.0f}%)  {sign}${pnl:.2f}"
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
    se_status = "✅ ENABLED" if ALLOW_SECOND_ENTRY else "❌ OFF"
    return (
        f"⚙ <b>Settings &amp; Config</b>\n━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>Risk</b>\n"
        f"  Stake          : <b>${STAKE}</b>  per trade\n"
        f"  Duration       : <b>{DURATION} min</b>  contract expiry\n"
        f"  Cooldown       : <b>{COOLDOWN_MINUTES} min</b>  between trades\n"
        f"  Session TP     : <b>${DAILY_PROFIT_TARGET:.2f}</b>\n"
        f"  Session SL     : <b>${DAILY_LOSS_LIMIT:.2f}</b>\n"
        f"  Max Consec Los : <b>{MAX_CONSECUTIVE_LOSSES}</b>  then pause {PAUSE_MINUTES} min\n"
        f"\n<b>Signal Gates</b>\n"
        f"  Min Score      : <b>{SCORE_THRESHOLD}/100</b>  (Supertrend + ADX + confluences)\n"
        f"  ML Gate        : <b>≥{ML_CONFIDENCE_MIN*100:.0f}%</b>  confidence required\n"
        f"  ML Min Trades  : <b>{ML_MIN_TRADES}</b>  before model activates\n"
        f"  ML Retrain     : every <b>{ML_RETRAIN_EVERY}</b> trades\n"
        f"\n<b>Second Entry</b>  {se_status}\n"
        f"  ML Gate (2nd)  : <b>≥{SECOND_ENTRY_ML_MIN*100:.0f}%</b>  stricter re-entry bar\n"
        f"  Cooldown (2nd) : <b>{SECOND_ENTRY_COOLDOWN} min</b>  after win\n"
        f"  Window         : <b>{SECOND_ENTRY_WINDOW} min</b>  re-entry window\n"
        f"\n<b>Martingale</b>\n"
        f"  Status   : {'✅ ENABLED' if MARTINGALE_ENABLED else '❌ OFF'}\n"
        f"  Steps    : base (${STAKE:.2f}) → ×{MARTINGALE_MULTIPLIER:.0f} → ×{MARTINGALE_MULTIPLIER**2:.0f}  "
        f"(max {MARTINGALE_MAX_STEPS} recovery steps)\n"
        f"\n<b>Engine</b>\n"
        f"  Supertrend     : period={SUPERTREND_PERIOD}  mult={SUPERTREND_ATR_MULT}\n"
        f"  Contract       : Rise/Fall  (Deriv API: CALL=Rise / PUT=Fall)\n"
        f"\n{_ml_progress_text()}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<i>Tap ⚙ Adjust to change any value live. Or: /set &lt;param&gt; &lt;value&gt;</i>\n"
        f"<i>E.g. /set ml_gate 85  ·  /set score 91  ·  /set stake 5</i>"
    )


def _settings_adj_kb() -> InlineKeyboardMarkup:
    """Inline +/- keyboard — auto-generated from _ADJ_PARAMS so it never drifts."""
    import sys as _sys
    _mod = _sys.modules[__name__]

    # Friendly labels for each key (order matters for display)
    _LABELS = [
        ("stake",      "Stake $"),
        ("duration",   "Duration"),
        ("score",      "Score Gate"),
        ("ml_gate",    "ML Gate"),
        ("ml2_gate",   "ML 2nd Gate"),
        ("cooldown",   "Cooldown"),
        ("tp",         "Session TP"),
        ("sl",         "Session SL"),
        ("max_loss",   "Max C.Loss"),
        ("m_mult",     "M Multiplier"),
        ("m_steps",    "M Max Steps"),
        ("retrain",    "ML Retrain"),
        ("ml_min",     "ML Min Trades"),
        ("pause_min",  "Pause Min"),
        ("second_cd",  "2nd Cooldown"),
        ("second_win", "2nd Window"),
    ]

    def _row(key: str, label: str) -> list:
        cfg   = _ADJ_PARAMS.get(key)
        cur   = getattr(_mod, cfg[0]) if cfg else "?"
        val_s = cfg[5](cur) if cfg else str(cur)
        return [
            InlineKeyboardButton(f"−",         callback_data=f"adj_{key}_down"),
            InlineKeyboardButton(f"{label}: {val_s}", callback_data="noop"),
            InlineKeyboardButton(f"+",         callback_data=f"adj_{key}_up"),
        ]

    rows = [[InlineKeyboardButton("⚙ ADJUST LIVE PARAMETERS", callback_data="noop")]]
    rows += [_row(k, lbl) for k, lbl in _LABELS if k in _ADJ_PARAMS]
    rows.append([
        InlineKeyboardButton("🔙 Settings", callback_data="settings"),
        InlineKeyboardButton("🏠 Menu",     callback_data="main_menu"),
    ])
    return InlineKeyboardMarkup(rows)


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
        f"Type     : Rise/Fall\n"
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

        contract_type = _get_contract_type(direction)
        tg(
            f"🧪 <b>Test Trade</b>  –  ✅ Authorised\n"
            f"Account  : <code>{account}</code>\n"
            f"Spot now : {spot_now}\n"
            f"Direction: {'🟢 Rise (UP)' if direction == 'UP' else '🔴 Fall (DOWN)'}\n"
            f"Type     : {contract_type}  ({('Rise' if direction == 'UP' else 'Fall')})\n"
            f"<i>Step 2/4 – Requesting proposal…</i>"
        )
        # ── Rise/Fall proposal — no barrier needed ────────────────────
        ws.send(json.dumps({
            "proposal": 1, "amount": STAKE, "basis": "stake",
            "contract_type": contract_type, "currency": "USD",
            "duration": DURATION, "duration_unit": "m",
            "symbol": symbol,
        }))
        prop_msg = recv_typed("proposal", timeout=10)
        if not prop_msg or "error" in prop_msg:
            err = (prop_msg or {}).get("error", {}).get("message", "timeout")
            tg(f"🧪 <b>Test Trade FAILED</b>\n❌ Proposal error: <code>{err}</code>")
            return
        prop   = prop_msg["proposal"]
        pid    = prop["id"]
        ask    = prop.get("ask_price", STAKE)
        payout = prop.get("payout", 0)
        try:
            payout_f       = float(payout)
            offered_ratio  = round((payout_f - STAKE) / STAKE, 4) if STAKE > 0 else 0.0
        except (TypeError, ValueError):
            offered_ratio = 0.0

        if not (offered_ratio >= PROFIT_MIN):
            tg(
                f"🧪 <b>Test Trade</b>  –  💸 Payout gate\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"Profit ratio: <b>{offered_ratio*100:.1f}%</b>\n"
                f"Minimum required: {PROFIT_MIN*100:.0f}%\n"
                f"<i>Proceeding anyway (test mode bypasses payout gate).</i>"
            )

        tg(
            f"🧪 <b>Test Trade</b>  –  ✅ Proposal OK\n"
            f"Proposal ID : <code>{pid}</code>\n"
            f"Ask Price   : ${ask}\n"
            f"Payout      : ${payout}  (profit ratio <b>{offered_ratio*100:.1f}%</b>)\n"
            f"<i>Step 3/4 – Buying contract…</i>"
        )
        ws.send(json.dumps({"buy": pid, "price": STAKE}))   # test always uses base stake
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
            f"<i>Test trade itself does NOT affect session stats.</i>"
        )
        if not win:
            _fire_test_martingale_reentry(symbol, direction, tg)
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


def _fire_test_martingale_reentry(symbol: str, direction: str, tg):
    """A test trade lost — drive the REAL production martingale re-entry
    path (request_proposal on the live per-symbol WS, same as a genuine
    losing signal would) instead of the test harness's isolated one-shot
    socket. This is the only way to actually exercise/validate the
    re-entry pipeline on demand — unlike the base test trade, this places
    a real contract at real (multiplied) stake and DOES flow into session
    P&L / win-loss counts / DB history like any other martingale re-entry,
    because it settles through the same on_contract_update() path.
    """
    if not MARTINGALE_ENABLED:
        tg("🧪 <i>Martingale is disabled (MARTINGALE_ENABLED=False) — skipping re-entry test.</i>")
        return
    with _lock:
        busy = (symbol in pending_signals or symbol in unconfirmed_buys
                or any(c["symbol"] == symbol for c in active_contracts.values()))
    if busy:
        tg(f"🧪 <b>Martingale re-entry test skipped</b> — <code>{symbol}</code> already has "
           f"a live/pending trade from the main engine. Try again once it's clear.")
        return
    ws2 = ws_registry.get(symbol)
    if ws2 is None:
        tg(f"🧪 <b>Martingale re-entry test skipped</b> — no live WS for <code>{symbol}</code>.")
        return
    with _lock:
        new_ml = min(martingale_level.get(symbol, 0) + 1, MARTINGALE_MAX_STEPS)
        martingale_level[symbol] = new_ml
    m_stake = round(STAKE * (MARTINGALE_MULTIPLIER ** new_ml), 2)
    tg(
        f"⚡ <b>Test Martingale Re-entry — Step {new_ml}/{MARTINGALE_MAX_STEPS}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Symbol    : <code>{symbol}</code>\n"
        f"Direction : {'🟢 Rise (UP)' if direction == 'UP' else '🔴 Fall (DOWN)'}  <i>(same as losing test trade)</i>\n"
        f"Stake     : <b>${m_stake:.2f}</b>  (×{int(MARTINGALE_MULTIPLIER**new_ml)} base ${STAKE:.2f})\n"
        f"<i>Real contract via live engine — this WILL count toward session P&amp;L and history "
        f"(unlike the base test trade above). Watch for the normal WIN/LOSS result card.</i>"
    )
    request_proposal(ws2, symbol, {}, direction)
    _log(f"🧪⚡ Test-triggered martingale re-entry {symbol} {direction} step={new_ml} stake=${m_stake:.2f}")


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

async def cmd_analytics(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Win-rate breakdown by score, ADX, pattern, session, S&D distance, ML confidence."""
    if not _bot_ready:
        await update.message.reply_text(_NOT_READY_MSG, parse_mode="HTML"); return
    await update.message.reply_text("⏳ Building analytics…", parse_mode="HTML")
    text = _performance_analytics_text()
    # Telegram message limit is 4096 chars; split if needed
    for i in range(0, len(text), 4000):
        await update.message.reply_text(text[i:i+4000], reply_markup=_main_kb(), parse_mode="HTML")

async def cmd_patterns(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Run pattern discovery on demand."""
    if not _bot_ready:
        await update.message.reply_text(_NOT_READY_MSG, parse_mode="HTML"); return
    await update.message.reply_text("🔬 Running pattern discovery…", parse_mode="HTML")
    threading.Thread(target=_pattern_discovery, daemon=True, name="PatDiscCmd").start()

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
            document=csv_bytes,
            filename=f"trades_{ts}.csv",
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


async def cmd_set(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """
    /set <param> <value>  — change any live config parameter on the fly.

    Examples:
      /set ml_gate 85       → ML gate ≥85%
      /set ml2_gate 90      → Second-entry ML gate ≥90%
      /set score 91         → Signal score threshold 91/100
      /set stake 5          → $5 per trade
      /set duration 10      → 10-min contracts
      /set cooldown 15      → 15 min between trades
      /set tp 8             → Session take-profit $8
      /set sl 15            → Session stop-loss $15
      /set max_loss 3       → Max 3 consecutive losses before pause
      /set retrain 30       → Retrain ML every 30 trades
      /set ml_min 80        → ML activates after 80 trades
      /set pause_min 20     → Pause 20 min after consec-loss trigger
      /set second_cd 5      → 2nd-entry cooldown 5 min after win
      /set second_win 15    → 2nd-entry window open 15 min after win
    """
    args = (ctx.args or [])
    if len(args) < 2:
        import sys as _sys_help
        _mod = _sys_help.modules[__name__]
        params_help = "\n".join(
            f"  <code>/set {k:<12}</code>  now: <b>{_ADJ_PARAMS[k][5](getattr(_mod, _ADJ_PARAMS[k][0]))}</b>"
            for k in _ADJ_PARAMS
        )
        await update.message.reply_text(
            f"⚙ <b>/set — Live Parameter Editor</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Usage: <code>/set &lt;param&gt; &lt;value&gt;</code>\n\n"
            f"<b>Current values</b>:\n{params_help}\n\n"
            f"<i>Changes take effect immediately with no restart needed.</i>",
            parse_mode="HTML",
        )
        return

    key, raw = args[0].lower(), args[1]
    ok, msg = _apply_param(key, raw)
    emoji = "✅" if ok else "❌"
    await update.message.reply_text(
        f"{emoji} {msg}\n\n{_settings_text()}" if ok else f"{emoji} {msg}",
        parse_mode="HTML",
        reply_markup=_settings_adj_kb() if ok else None,
    )
    if ok:
        _log(f"⚙ /set {key} {raw} → applied")


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
# ── Shared parameter map for /set command and adj_ inline buttons ──────
_ADJ_PARAMS = {
    # key          : (global_var_name,          min,     max,   step,  is_pct,  display_fn)
    "ml_gate"      : ("ML_CONFIDENCE_MIN",       0.60,    0.99,  0.01,  True,   lambda v: f"{v*100:.0f}%"),
    "ml2_gate"     : ("SECOND_ENTRY_ML_MIN",     0.75,    0.99,  0.01,  True,   lambda v: f"{v*100:.0f}%"),
    "score"        : ("SCORE_THRESHOLD",         70,      99,    1,     False,  lambda v: f"{v}/100"),
    "stake"        : ("STAKE",                   0.35,    500,   0.5,   False,  lambda v: f"${v:.2f}"),
    "duration"     : ("DURATION",                1,       60,    1,     False,  lambda v: f"{v} min"),
    "cooldown"     : ("COOLDOWN_MINUTES",        1,       120,   5,     False,  lambda v: f"{v} min"),
    "tp"           : ("DAILY_PROFIT_TARGET",     1,       9999,  1,     False,  lambda v: f"${v:.2f}"),
    "sl"           : ("DAILY_LOSS_LIMIT",        -9999,  -1,    -1,    False,  lambda v: f"${v:.2f}"),
    "max_loss"     : ("MAX_CONSECUTIVE_LOSSES",  1,       20,    1,     False,  lambda v: f"{int(v)}"),
    "retrain"      : ("ML_RETRAIN_EVERY",        5,       5000,  10,    False,  lambda v: f"{int(v)} trades"),
    "ml_min"       : ("ML_MIN_TRADES",           20,      5000,  10,    False,  lambda v: f"{int(v)} trades"),
    "pause_min"    : ("PAUSE_MINUTES",           1,       1440,  5,     False,  lambda v: f"{int(v)} min"),
    "second_cd"    : ("SECOND_ENTRY_COOLDOWN",   1,       60,    1,     False,  lambda v: f"{int(v)} min"),
    "second_win"   : ("SECOND_ENTRY_WINDOW",     2,       120,   5,     False,  lambda v: f"{int(v)} min"),
    # ── Martingale ─────────────────────────────────────────────────────────────
    "m_mult"       : ("MARTINGALE_MULTIPLIER",   1.2,     5.0,   0.1,   False,  lambda v: f"×{v:.1f}"),
    "m_steps"      : ("MARTINGALE_MAX_STEPS",    1,       5,     1,     False,  lambda v: f"{int(v)} steps"),
}


def _apply_param(key: str, raw_val) -> tuple[bool, str]:
    """Apply a parameter change. Returns (ok, message)."""
    import sys as _sys
    global ML_CONFIDENCE_MIN, SECOND_ENTRY_ML_MIN, SCORE_THRESHOLD, STAKE
    global DURATION, COOLDOWN_MINUTES, DAILY_PROFIT_TARGET, DAILY_LOSS_LIMIT
    global MAX_CONSECUTIVE_LOSSES, ML_RETRAIN_EVERY, ML_MIN_TRADES, PAUSE_MINUTES
    global SECOND_ENTRY_COOLDOWN, SECOND_ENTRY_WINDOW
    global MARTINGALE_MULTIPLIER, MARTINGALE_MAX_STEPS

    if key not in _ADJ_PARAMS:
        return False, f"Unknown param '{key}'. Valid: {', '.join(_ADJ_PARAMS)}"

    var_name, lo, hi, _, is_pct, fmt = _ADJ_PARAMS[key]
    try:
        val = float(raw_val)
        if is_pct and val > 1:        # user typed 85 instead of 0.85
            val = val / 100.0
        if key == "sl":               # user types 15 → stored as -15
            val = -abs(val)
    except ValueError:
        return False, f"Value must be a number. Got: {raw_val!r}"

    val = max(lo, min(hi, val))

    # Map key → global variable assignment
    _MAP = {
        "ml_gate":    lambda v: setattr(_sys.modules[__name__], "ML_CONFIDENCE_MIN",   v),
        "ml2_gate":   lambda v: setattr(_sys.modules[__name__], "SECOND_ENTRY_ML_MIN", v),
        "score":      lambda v: setattr(_sys.modules[__name__], "SCORE_THRESHOLD",     int(v)),
        "stake":      lambda v: setattr(_sys.modules[__name__], "STAKE",               v),
        "duration":   lambda v: setattr(_sys.modules[__name__], "DURATION",            int(v)),
        "cooldown":   lambda v: setattr(_sys.modules[__name__], "COOLDOWN_MINUTES",    int(v)),
        "tp":         lambda v: setattr(_sys.modules[__name__], "DAILY_PROFIT_TARGET", v),
        "sl":         lambda v: setattr(_sys.modules[__name__], "DAILY_LOSS_LIMIT",    -abs(v)),
        "max_loss":   lambda v: setattr(_sys.modules[__name__], "MAX_CONSECUTIVE_LOSSES", int(v)),
        "retrain":    lambda v: setattr(_sys.modules[__name__], "ML_RETRAIN_EVERY",    int(v)),
        "ml_min":     lambda v: setattr(_sys.modules[__name__], "ML_MIN_TRADES",       int(v)),
        "pause_min":  lambda v: setattr(_sys.modules[__name__], "PAUSE_MINUTES",       int(v)),
        "second_cd":  lambda v: setattr(_sys.modules[__name__], "SECOND_ENTRY_COOLDOWN",   int(v)),
        "second_win": lambda v: setattr(_sys.modules[__name__], "SECOND_ENTRY_WINDOW",     int(v)),
        "m_mult":     lambda v: setattr(_sys.modules[__name__], "MARTINGALE_MULTIPLIER",   v),
        "m_steps":    lambda v: setattr(_sys.modules[__name__], "MARTINGALE_MAX_STEPS",    int(v)),
    }
    _MAP[key](val)

    # Re-read after set for accurate display
    var_name2, lo2, hi2, _, is_pct2, fmt2 = _ADJ_PARAMS[key]
    new_val = getattr(_sys.modules[__name__], var_name2)
    return True, f"✅ <b>{var_name2}</b> → <b>{fmt2(new_val)}</b>"


async def _handle_adj_callback(q, d: str):
    """Handle adj_<key>_up / adj_<key>_down inline button presses."""
    import sys as _sys
    parts = d.split("_")            # adj, <key>, up/down
    if len(parts) < 3:
        await q.answer("Bad callback")
        return
    direction = parts[-1]          # "up" or "down"
    key = "_".join(parts[1:-1])    # everything between adj_ and _up/_down

    if key not in _ADJ_PARAMS:
        await q.answer("Unknown parameter")
        return

    var_name, lo, hi, step, is_pct, fmt = _ADJ_PARAMS[key]
    cur_val = getattr(_sys.modules[__name__], var_name)
    delta = step if direction == "up" else -step
    new_val = max(lo, min(hi, cur_val + delta))
    ok, msg = _apply_param(key, new_val * 100 if is_pct else new_val)
    await q.answer(msg[:200])
    await q.edit_message_text(_settings_text(), reply_markup=_settings_adj_kb(), parse_mode="HTML")
    _log(f"⚙ {var_name} adjusted to {getattr(_sys.modules[__name__], var_name)} via Telegram button")


async def btn_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    global paused, pause_until, consecutive_losses, PROFIT_MIN
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
        elif d == "ml_conf_perf":
            await q.edit_message_text(_ml_confidence_perf_text(), reply_markup=_main_kb(), parse_mode="HTML")
        elif d == "analytics":
            await q.edit_message_text(
                "⏳ Building analytics…",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back", callback_data="main_menu")]]),
                parse_mode="HTML",
            )
            text = _performance_analytics_text()
            for i in range(0, len(text), 4000):
                chunk = text[i:i+4000]
                kb    = _main_kb() if i + 4000 >= len(text) else None
                await q.message.reply_text(chunk, reply_markup=kb, parse_mode="HTML")
        elif d == "patterns":
            await q.edit_message_text(
                "🔬 Running pattern discovery…",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back", callback_data="main_menu")]]),
                parse_mode="HTML",
            )
            threading.Thread(target=_pattern_discovery, daemon=True, name="PatDiscBtn").start()
        elif d == "export_csv_quick":
            await q.edit_message_text(
                "⏳ Generating CSV export…",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back", callback_data="main_menu")]]),
                parse_mode="HTML",
            )
            # Reuse existing export logic via a thread
            async def _do_export_quick():
                try:
                    rows = _db_fetch("SELECT * FROM trades ORDER BY id")
                    if USE_PG:
                        cols = ["id","timestamp","symbol","direction","barrier","stake","payout",
                                "profit","win","score","wick_atr_ratio","atr","atr_ma",
                                "ema_fast_slope","ema_slow_slope","ema_distance","market_session"]
                    else:
                        import sqlite3 as _sl
                        conn2 = _sl.connect("trades.db")
                        desc2 = conn2.execute("PRAGMA table_info(trades)").fetchall()
                        conn2.close()
                        cols = [d2[1] for d2 in desc2]
                    buf = io.StringIO()
                    w   = _csv.writer(buf)
                    w.writerow(cols)
                    w.writerows(rows)
                    csv_bytes = buf.getvalue().encode("utf-8")
                    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
                    await q.message.reply_document(
                        document=csv_bytes,
                        filename=f"trades_{ts}.csv",
                        caption=f"📊 <b>Trades Export</b> — {len(rows)} trades",
                        parse_mode="HTML",
                    )
                except Exception as exc:
                    await q.message.reply_text(f"❌ Export failed: <code>{exc}</code>", parse_mode="HTML")
            asyncio.ensure_future(_do_export_quick())
        elif d == "payout_up":
            delta = 0.05
            new_min = round(min(0.95, PROFIT_MIN + delta), 2)
            PROFIT_MIN = new_min
            await q.edit_message_text(
                f"💸 <b>Minimum payout raised</b>\n"
                f"Min ratio : <b>{PROFIT_MIN*100:.0f}%</b>  (no upper limit)\n"
                f"<i>Higher minimum = only takes the best payouts.</i>",
                reply_markup=_main_kb(), parse_mode="HTML",
            )
            _log(f"💸 Min payout raised to {PROFIT_MIN*100:.0f}%")
        elif d == "payout_down":
            delta = 0.05
            new_min = round(max(0.10, PROFIT_MIN - delta), 2)
            PROFIT_MIN = new_min
            await q.edit_message_text(
                f"💸 <b>Minimum payout lowered</b>\n"
                f"Min ratio : <b>{PROFIT_MIN*100:.0f}%</b>  (no upper limit)\n"
                f"<i>Lower minimum = accepts more fills.</i>",
                reply_markup=_main_kb(), parse_mode="HTML",
            )
            _log(f"💸 Min payout lowered to {PROFIT_MIN*100:.0f}%")
        elif d == "noop" or d == "settings_adj":
            await q.answer()   # acknowledge silently — spacer / no-op buttons
        elif d == "settings":
            await q.edit_message_text(
                _settings_text(),
                reply_markup=_settings_adj_kb(),
                parse_mode="HTML",
            )
        elif d.startswith("adj_"):
            await _handle_adj_callback(q, d)
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
        elif d in ("tg_vol", "tg_vol1s", "tg_rdb", "tg_step", "tg_jump"):
            labels = {
                "tg_vol":   "📊 Volatility (R_*)",
                "tg_vol1s": "⚡ Volatility 1s (1HZ*)",
                "tg_rdb":   "📉 Range Break",
                "tg_step":  "🪜 Step Index",
                "tg_jump":  "⚡ Jump Indices",
            }
            await q.edit_message_text(
                f"🧪 <b>Test Trade  –  {labels[d]}</b>\n━━━━━━━━━━━━━━━━━━━━\nSelect a symbol:",
                reply_markup=_test_sym_kb(d), parse_mode="HTML",
            )
        elif d.startswith("test_sym_"):
            sym = d[9:]
            if sym not in ALL_RF_SYMBOLS:
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
        elif d == "backtest":
            await q.edit_message_text(
                "🔬 <b>Backtest</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "Simulating Rise/Fall outcomes on historical candles…\n"
                "<i>Results will arrive as a new message in a few seconds.</i>",
                reply_markup=_main_kb(), parse_mode="HTML",
            )
            threading.Thread(target=_run_backtest, daemon=True, name="Backtest").start()
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
        pct_warm = min(100, int(ml_total / ML_MIN_TRADES * 100)) if ML_MIN_TRADES else 100
        bar_warm = _ml_progress_bar(ml_total / ML_MIN_TRADES if ML_MIN_TRADES else 1.0, 10)
        ml_line  = f"🔄 Warming up [{bar_warm}] {ml_total}/{ML_MIN_TRADES} ({pct_warm}%)"
    else:
        nxt = max(0, ML_RETRAIN_EVERY - (ml_total - ml_t))
        bar_retrain = _ml_progress_bar(1.0 - nxt / ML_RETRAIN_EVERY if ML_RETRAIN_EVERY else 1.0, 10)
        ml_line = f"✅ Active [{bar_retrain}] trained={ml_t}  retrain in {nxt} trades"

    # ML confidence bracket performance (live session, from _ml_conf_live_stats)
    with _lock:
        conf_snap = dict(_ml_conf_live_stats)
    ml_conf_lines = ""
    if conf_snap:
        ml_conf_lines = "<b>  ML Confidence Performance</b>\n"
        for bucket in sorted(conf_snap.keys(), reverse=True):
            s = conf_snap[bucket]
            t_b = s.get("wins", 0) + s.get("losses", 0)
            wr_b = s.get("wins", 0) / t_b * 100 if t_b else 0
            pnl_b = s.get("pnl", 0.0)
            sign = "+" if pnl_b >= 0 else ""
            ml_conf_lines += (
                f"  {bucket}% : {t_b} trades · {wr_b:.0f}% WR · {sign}${pnl_b:.2f}\n"
            )
    else:
        ml_conf_lines = "  — no ML-filtered trades this session —\n"

    # Next session countdown
    next_dt   = _next_session_start(now)
    secs_left = max(0, int((next_dt - now).total_seconds()))
    hh_l, rem = divmod(secs_left, 3600)
    mm_l      = rem // 60
    nxt_countdown = f"{hh_l}h {mm_l}m" if hh_l else f"{mm_l}m"
    next_sess = _get_session_name(next_dt)
    next_line = (f"  ⏭ Next: {SESSION_EMOJIS.get(next_sess,'')} {next_sess} "
                 f"in {nxt_countdown}  ({next_dt.strftime('%H:%M UTC')})\n")

    # Top 5 hottest symbols right now (with Sober Book extras)
    scored = []
    for sym in SYMBOLS:
        with _lock:
            cc = current_candle.get(sym)
        if cc:
            try:
                row_c = {"Close": float(cc.get("close", 0)), "Open": float(cc.get("open", 0)),
                         "High":  float(cc.get("high",  0)), "Low":  float(cc.get("low",  0))}
                s, d, det = score_signal(sym, row_c)
                scored.append((s, sym, d, det))
            except Exception:
                pass
    scored.sort(reverse=True)
    hot_lines = ""
    for s, sym, d, det in scored[:5]:
        heat  = "🔥🔥" if s >= SCORE_THRESHOLD else "🔥" if s >= SCORE_THRESHOLD - 15 else "  "
        m5d   = indicators_m5.get(sym,  {}).get("supertrend_dir", 0)
        m15d  = indicators_m15.get(sym, {}).get("supertrend_dir", 0)
        m5ic  = ("↑" if m5d  == 1 else "↓" if m5d  == -1 else "→") + "M5"
        m15ic = ("↑" if m15d == 1 else "↓" if m15d == -1 else "→") + "M15"
        ms_t  = det.get("market_structure", "?")[:4]
        pat   = det.get("candle_pattern", "—").split()[0]   # first word only
        hot_lines += (
            f"  {heat}<code>{sym:<10}</code> {s:>3}/100 {d} {m5ic} {m15ic}\n"
            f"       MktStr:{ms_t}  Pat:{pat}\n"
        )
    if not hot_lines:
        hot_lines = "  — no data yet —\n"

    # Compute next retrain based on real trade count (already capped at startup)
    with ml_lock:
        _hb_t, _hb_trn = ml_total_trades, ml_trained_on
    _retrain_nxt = max(0, ML_RETRAIN_EVERY - max(0, _hb_t - _hb_trn))

    # Per-session breakdown today
    _roll_market_session_stats_if_needed()
    with _lock:
        all_sess = {k: dict(v) for k, v in market_session_stats.items()}
    sess_detail = ""
    for sess_n in ["Midnight", "Early Asian", "Late Asian", "London", "Early New York", "Late New York"]:
        sv = all_sess.get(sess_n, {})
        tr = sv.get("trades", 0)
        if tr == 0:
            continue
        sw = sv.get("wins", 0); sl = sv.get("losses", 0); sp = sv.get("pnl", 0.0)
        swr = sw / tr * 100 if tr else 0
        is_cur = "◀" if sess_n == mkt_s else ""
        sess_detail += (
            f"  {SESSION_EMOJIS.get(sess_n,'')} {sess_n:<16} {tr}tr  {sw}W/{sl}L "
            f"{swr:.0f}%  {'+' if sp >= 0 else ''}${sp:.2f} {is_cur}\n"
        )
    if not sess_detail:
        sess_detail = "  — no session trades yet today —\n"

    msg = (
        f"❤️ <b>Hourly Heartbeat</b>  –  {now.strftime('%H:%M UTC')}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"State   : {'⏸ PAUSED' if is_paused else '▶ RUNNING'}\n"
        f"Market  : {SESSION_EMOJIS.get(mkt_s,'')} {mkt_s}  ·  Active: {ac}\n"
        f"{next_line}"
        f"\n"
        f"<b>💰 P&amp;L</b>\n"
        f"  {'+' if pnl >= 0 else ''}${pnl:.2f}  ({total} trades  {wc}W/{lc}L  {wr:.0f}%WR)\n"
        f"  Drawdown: -${cur_dd:.2f}  (max -${mdd:.2f})\n"
        f"  Streak  : {'🔴×' + str(cl) if cl else '🟢 None'}\n"
        f"\n"
        f"{session_line}"
        f"<b>📅 Today by Session</b>\n{sess_detail}"
        f"\n"
        f"<b>🤖 ML Engine</b>\n"
        f"  {ml_line}\n"
        f"{ml_conf_lines}"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>🔥 Top Symbols</b>\n{hot_lines}"
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
    dpnl     = daily_total_pnl
    dpnl_col = "green" if dpnl >= 0 else "red"
    t.append("  Market Session: ", style="dim"); t.append(f"{SESSION_EMOJIS.get(mkt_s,'')} {mkt_s}\n", style="bold magenta")
    t.append("  Daily P&L    : ", style="dim"); t.append(f"${dpnl:+.2f}", style=f"bold {dpnl_col}"); t.append("  (all sessions)\n", style="dim")
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
    global daily_total_pnl
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

            daily_total_pnl = 0.0   # reset daily accumulator for the new day
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

            for name in ["Midnight", "Early Asian", "Late Asian", "London", "Early New York", "Late New York"]:
                s    = ms_snap.get(name, {"wins": 0, "losses": 0, "pnl": 0.0, "trades": 0})
                tot  = s["wins"] + s["losses"]
                if tot == 0:
                    continue
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

                    # Rich one-glance summary
                    db_sum = get_db_summary()
                    at, ap, aw, al = db_sum if db_sum[0] else (0, 0, 0, 0)
                    at_wr = aw / at * 100 if at else 0
                    with _lock:
                        active_n = len(active_contracts)
                        dd       = max(0.0, peak_equity - pnl)
                        mdd      = max_drawdown
                    next_dt   = _next_session_start(now)
                    secs_left = max(0, int((next_dt - now).total_seconds()))
                    hh_l, rem = divmod(secs_left, 3600)
                    mm_l = rem // 60
                    next_in = f"{hh_l}h {mm_l}m" if hh_l else f"{mm_l}m"

                    # ── Build a professional, information-dense dashboard ───────────────
                    # Average win/loss and profit factor from the in-memory session
                    with _lock:
                        wc_l, lc_l = win_count, loss_count
                    tot_l = wc_l + lc_l
                    wr_l  = wc_l / tot_l * 100 if tot_l else 0
                    # Next ML retrain count (already capped at real DB count on startup)
                    with ml_lock:
                        _t_now, _trn = ml_total_trades, ml_trained_on
                    _retrain_nxt = max(0, ML_RETRAIN_EVERY - max(0, _t_now - _trn))

                    text = (
                        f"📌 <b>LIVE DASHBOARD</b>  ·  Deriv Server Time  <b>{now.strftime('%H:%M:%S UTC')}</b>\n"
                        f"{'⏸ PAUSED' if is_paused_ else '▶ RUNNING'}  ·  "
                        f"{SESSION_EMOJIS.get(mkt_s,'')} <b>{mkt_s}</b>  ·  "
                        f"Active: <b>{active_n}</b>  ·  Day: <b>{daily_trades}</b>\n"
                        "━━━━━━━━━━━━━━━━━━━━\n"
                        "<b>💰 Current Session</b>\n"
                        f"  P&amp;L        : <b>{'+' if pnl >= 0 else ''}${pnl:.2f}</b>\n"
                        f"  Trades     : {tot_l}  ({wc_l}W / {lc_l}L)  ·  WR <b>{wr_l:.0f}%</b>\n"
                        f"  Drawdown   : ${dd:.2f}  ·  Max DD ${max(0.0, mdd):.2f}\n"
                        "\n"
                        "<b>🏦 All-Time Database</b>\n"
                        f"  {at} trades  {aw}W/{at-aw}L  ({at_wr:.0f}%WR)  "
                        f"P&amp;L {'+' if ap >= 0 else ''}${ap:.2f}\n"
                        "\n"
                        "<b>⚙ Engine</b>\n"
                        f"  ML    : {ml_s}  ·  retrain in <b>{_retrain_nxt}</b> trades\n"
                        f"  Gate  : score≥{SCORE_THRESHOLD}  ·  ML≥{ML_CONFIDENCE_MIN*100:.0f}%\n"
                        f"  Payout  : min {PROFIT_MIN*100:.0f}% profit ratio (no upper cap)\n"
                        "━━━━━━━━━━━━━━━━━━━━\n"
                        f"<b>🔥 Hot Signals</b>\n{hot}"
                        f"🕐 Next session: <b>{next_dt.strftime('%H:%M UTC')}</b> in {next_in} · "
                        f"<i>↻ every 5 min</i>"
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
    # Re-read the token at call time so a late-injected env var is picked up
    # after the first attempt, and so config changes don't need a full restart.
    token = os.environ.get("TG_BOT_TOKEN") or TELEGRAM_BOT_TOKEN
    if not token:
        logger.warning(
            "TG_BOT_TOKEN is not set — Telegram disabled. "
            "Set the secret and restart the bot to enable it."
        )
        # Sleep a long time so the watch-thread does not spam-retry every 5 s.
        time.sleep(300)
        return

    async def _run():
        global telegram_app, _tg_loop
        # Clear readiness in case this is a restart
        _tg_ready.clear()
        _tg_loop = asyncio.get_running_loop()
        try:
            app = Application.builder().token(token).build()
        except Exception as _tok_err:
            logger.error(
                f"Telegram Application build failed ({_tok_err}). "
                "Check TG_BOT_TOKEN — sleeping 300 s before retry."
            )
            await asyncio.sleep(300)
            return
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
        app.add_handler(CommandHandler("set",       cmd_set))
        app.add_handler(CommandHandler("analytics", cmd_analytics))
        app.add_handler(CommandHandler("patterns",  cmd_patterns))
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

    # ── Sync ml_total_trades with the real DB row count so the "retrain in N"
    # counter starts from the correct baseline instead of 0 after a restart.
    # ml_trained_on is loaded from the pickle (e.g. 100) but ml_total_trades
    # would otherwise be 0, making since=max(0,0-100)=0 → "retrain in 50".
    # Also cap ml_trained_on to the real DB count: bootstrap models are trained
    # on historical candle simulations, which is a much larger number than real
    # trades. Without this cap the counter can stay stuck at 50 forever.
    global ml_total_trades, ml_trained_on  # must declare global here — in main(), not module scope
    try:
        _db_cnt_rows = _db_fetch("SELECT COUNT(*) FROM trades WHERE win IN (0,1)")
        _db_count = int(_db_cnt_rows[0][0]) if _db_cnt_rows and _db_cnt_rows[0][0] else 0
        with ml_lock:
            ml_total_trades = max(ml_total_trades, _db_count)
            # If the pickle came from a bootstrap model, trained_on can be huge.
            # Pull it back to the real trade count so the counter counts real trades.
            if ml_trained_on > ml_total_trades:
                ml_trained_on = ml_total_trades
                logger.info(f"Capped ml_trained_on to real DB count {ml_total_trades}")
        logger.info(f"ml_total_trades initialised to {ml_total_trades} from DB "
                    f"(ml_trained_on={ml_trained_on})")
    except Exception as _e:
        logger.warning(f"Failed to init ml_total_trades from DB: {_e}")

    # ── Start Telegram NOW — history is loaded so the loop is stable before
    # MLBootstrap (or any other thread) calls _send_tg.
    _watch_thread(_start_telegram,              name="Telegram")

    # Block until the Telegram polling loop confirms it is fully up.
    # _tg_ready is set inside _run() only after app.start() + start_polling()
    # succeed, so any subsequent _send_tg call is guaranteed a live loop.
    _tg_ready.wait(timeout=30)

    # ML bootstrap skipped — avoids CPU spike across 21 symbols at startup.
    # The model trains automatically once ML_MIN_TRADES real trades are settled.
    logger.info("ML bootstrap skipped — accumulating real trades for first model.")

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
    # Guard thread: makes sure ML retrains even if DB-writer path is delayed
    _watch_thread(_ml_retrain_guard_loop,       name="MLRetrainGuard")

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

