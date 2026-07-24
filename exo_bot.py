#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║        DERIV TOUCH BOT v1  –  Professional Edition                   ║
║  Supertrend · Market Sessions · Fast ML · 93% Confidence Gate        ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import json, time, sqlite3, threading, queue, logging, asyncio, os, sys, pickle, io, csv as _csv, html as _html
import math
import random
import hashlib
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from collections import deque
from typing import Optional

import numpy as np
import pandas as pd
import warnings as _warnings_module
# Suppress repeated pandas/numpy RuntimeWarnings from ATR/ADX edge-case divisions
# (NaN and Inf are handled explicitly before any ML feature matrix is built)
_warnings_module.filterwarnings("ignore", message="Mean of empty slice",       category=RuntimeWarning)
_warnings_module.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
_warnings_module.filterwarnings("ignore", message="Degrees of freedom",        category=RuntimeWarning)
_warnings_module.filterwarnings("ignore", message="invalid value encountered", category=RuntimeWarning)
_warnings_module.filterwarnings("ignore", message="divide by zero encountered",category=RuntimeWarning)

# PostgreSQL support (optional – falls back to SQLite when neither URL is set)
# Prefers NEON_DATABASE_URL (same convention as sniper_bot.py) so this bot's
# data lands in the same managed Neon Postgres instance, not a local placeholder DB.
DATABASE_URL = os.environ.get("NEON_DATABASE_URL") or os.environ.get("DATABASE_URL")
USE_PG = bool(DATABASE_URL)
if USE_PG:
    try:
        import psycopg2
        import psycopg2.extras
    except ImportError:
        # psycopg2 not installed — fall back to SQLite and warn
        import logging as _lg
        _lg.getLogger("touch").warning(
            "DATABASE_URL is set but psycopg2-binary is not installed. "
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

# ── Dynamic Gemini / Gemma model discovery (Google AI Studio) ────────────────
_GEMINI_MODEL_FALLBACK  = "gemma-3-27b-it"   # last-resort hardcoded fallback
_GEMINI_MODEL_CACHE_TTL = 4 * 3600           # re-discover every 4 hours
_gemini_model_cache: dict = {}               # {"model": str, "expires": float}

try:
    from google import genai as _genai
    _GEMINI_KEY    = os.environ.get("GEMINI_API_KEY", "")
    _GEMINI_CLIENT = _genai.Client(api_key=_GEMINI_KEY) if _GEMINI_KEY else None
except Exception as _gem_err:
    _GEMINI_CLIENT = None


def _discover_gemini_model(force: bool = False) -> str:
    """Query Google AI to list available models; pick best Gemma → best Flash → fallback.
    Thread-safe. Cached for _GEMINI_MODEL_CACHE_TTL seconds to minimise API calls.
    Never crashes — always returns a usable model string."""
    global _gemini_model_cache
    now = time.time()
    if not force and _gemini_model_cache.get("expires", 0) > now:
        return _gemini_model_cache["model"]

    if _GEMINI_CLIENT is None:
        _gemini_model_cache = {"model": _GEMINI_MODEL_FALLBACK, "expires": now + 300}
        return _GEMINI_MODEL_FALLBACK

    try:
        models_iter = _GEMINI_CLIENT.models.list()
        available = []
        for m in models_iter:
            name = getattr(m, "name", "") or ""
            # supported_actions (newer SDK) or supported_generation_methods (older)
            methods = getattr(m, "supported_actions", None) or \
                      getattr(m, "supported_generation_methods", None) or []
            method_strs = " ".join(str(x) for x in methods)
            if "generateContent" in method_strs or not methods:
                short = name.removeprefix("models/")
                if short:
                    available.append(short)
        logger.info(f"Gemini model discovery: {len(available)} generateContent models found")

        # Priority 1 — best Gemma model (sort descending → largest version first)
        gemma_models = sorted([m for m in available if "gemma" in m.lower()], reverse=True)
        if gemma_models:
            chosen = gemma_models[0]
            logger.info(f"AI model selected (Gemma): {chosen}")
            _gemini_model_cache = {"model": chosen, "expires": now + _GEMINI_MODEL_CACHE_TTL}
            return chosen

        # Priority 2 — best Gemini Flash model
        flash_models = sorted([m for m in available if "flash" in m.lower()], reverse=True)
        if flash_models:
            chosen = flash_models[0]
            logger.info(f"AI model selected (Flash fallback): {chosen}")
            _gemini_model_cache = {"model": chosen, "expires": now + _GEMINI_MODEL_CACHE_TTL}
            return chosen

        # Priority 3 — any available model
        if available:
            chosen = available[0]
            logger.info(f"AI model selected (first available): {chosen}")
            _gemini_model_cache = {"model": chosen, "expires": now + _GEMINI_MODEL_CACHE_TTL}
            return chosen

    except Exception as e:
        logger.warning(f"Gemini model discovery failed: {e}")

    # Fallback: reuse cached entry if present, else hardcoded constant
    cached = _gemini_model_cache.get("model")
    if cached:
        return cached
    _gemini_model_cache = {"model": _GEMINI_MODEL_FALLBACK, "expires": now + 300}
    return _GEMINI_MODEL_FALLBACK

# ══════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════
DERIV_APP_TOKEN    = os.environ.get("DERIV_TOKEN",    "")
DERIV_APP_ID       = os.environ.get("DERIV_APP_ID",   "1089")
TELEGRAM_BOT_TOKEN = os.environ.get("TG_BOT_TOKEN",   "8908331931:AAHg-50KK8DLcYH1d2O7i9tIhredM-TIHnI")
TELEGRAM_CHAT_ID   = os.environ.get("TG_CHAT_ID",     "6400145232")

if not DERIV_APP_TOKEN:
    print("⚠   DERIV_TOKEN not set — bot cannot authenticate to Deriv. Set the DERIV_TOKEN secret.")
if not os.environ.get("TG_BOT_TOKEN"):
    print("⚠   TG_BOT_TOKEN not set — using hardcoded fallback. Set env var for production.")

DERIV_ACCOUNT_ID = os.environ.get("DERIV_ACCOUNT_ID", "")
DERIV_USE_NEW_API = os.environ.get("DERIV_USE_NEW_API", "").lower() in ("1", "true", "yes")

def _is_new_deriv_api():
    """Detect whether to use the new Deriv Options API (OTP-based WS) or the legacy v3 API."""
    if DERIV_USE_NEW_API:
        return True
    # New API app IDs are alphanumeric; legacy app IDs are numeric.
    return bool(DERIV_APP_ID) and not DERIV_APP_ID.isdigit()

_deriv_options_accounts_cache = None

def _deriv_rest_api_call(method, path, body=None, max_retries=5):
    """Make an authenticated request to the new Deriv REST API.
    Retries on 429 or 5xx with exponential backoff and jitter to survive
    Cloudflare rate-limits and transient Render startup bursts."""
    url = f"https://api.derivws.com{path}"
    headers = {
        "Accept": "application/json",
        "Deriv-App-ID": DERIV_APP_ID,
        "Authorization": f"Bearer {DERIV_APP_TOKEN}",
    }
    data = json.dumps(body).encode("utf-8") if body is not None else None
    if data is not None:
        headers["Content-Type"] = "application/json"

    # Auth/account endpoints are stricter; start backoff at 30s for 429s.
    base_delay = 30 if ("accounts" in path or "otp" in path) else 5

    for attempt in range(max_retries + 1):
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            try:
                err = json.loads(err_body)
            except Exception:
                err = {"message": err_body}

            is_rate_limit = e.code == 429
            is_server_error = 500 <= e.code < 600
            if attempt < max_retries and (is_rate_limit or is_server_error):
                retry_after = 0
                if isinstance(err, dict):
                    retry_after = err.get("retry_after", 0)
                delay = max(base_delay, retry_after) * (2 ** attempt) + random.uniform(0, 2)
                if is_rate_limit:
                    logger.warning(
                        f"Deriv REST API rate-limited ({method} {path}); "
                        f"waiting {delay:.1f}s before retry {attempt + 1}/{max_retries}"
                    )
                else:
                    logger.warning(
                        f"Deriv REST API server error {e.code} ({method} {path}); "
                        f"retrying in {delay:.1f}s"
                    )
                time.sleep(delay)
                continue

            raise RuntimeError(f"Deriv REST API {method} {path} failed: {e.code} {err}")

def _deriv_options_accounts():
    """Return the list of Options trading accounts for the authenticated user."""
    global _deriv_options_accounts_cache
    if _deriv_options_accounts_cache is None:
        r = _deriv_rest_api_call("GET", "/trading/v1/options/accounts")
        _deriv_options_accounts_cache = r.get("data", [])
    return _deriv_options_accounts_cache

def _deriv_account_cache_key():
    """Return a stable key for the current app_id/token pair."""
    return hashlib.sha256(f"{DERIV_APP_ID}:{DERIV_APP_TOKEN}".encode("utf-8")).hexdigest()


def _deriv_load_cached_account_id():
    """Load a previously resolved account ID from disk (7-day TTL)."""
    try:
        with open(".deriv_account_id_cache.json", "r") as f:
            cache = json.load(f)
        entry = cache.get(_deriv_account_cache_key())
        if not entry:
            return None
        ts = datetime.fromisoformat(entry.get("ts", "1970-01-01T00:00:00+00:00"))
        if datetime.now(timezone.utc) - ts > timedelta(days=7):
            return None
        return entry.get("account_id")
    except Exception:
        return None


def _deriv_save_cached_account_id(account_id):
    """Persist a resolved account ID so restarts don't hammer the accounts endpoint."""
    try:
        cache = {}
        try:
            with open(".deriv_account_id_cache.json", "r") as f:
                cache = json.load(f)
        except Exception:
            pass
        cache[_deriv_account_cache_key()] = {
            "account_id": account_id,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        with open(".deriv_account_id_cache.json", "w") as f:
            json.dump(cache, f)
    except Exception as e:
        logger.warning(f"Could not cache Deriv account ID: {e}")


def _deriv_options_account_id():
    """Return the account ID to use for the WebSocket connection.
    Prefer the DERIV_ACCOUNT_ID env var, then a cached account ID, then the
    accounts endpoint (demo first, then any account)."""
    if DERIV_ACCOUNT_ID:
        return DERIV_ACCOUNT_ID
    cached = _deriv_load_cached_account_id()
    if cached:
        logger.info("Using cached Deriv account ID; skipping accounts endpoint lookup.")
        return cached
    accounts = _deriv_options_accounts()
    if not accounts:
        raise RuntimeError("No Options trading accounts found for this token/app_id.")
    # Prefer demo for safety when not explicitly configured.
    for a in accounts:
        if a.get("account_type") == "demo":
            account_id = a.get("account_id")
            _deriv_save_cached_account_id(account_id)
            return account_id
    account_id = accounts[0].get("account_id")
    _deriv_save_cached_account_id(account_id)
    return account_id

def deriv_ws_url():
    """Return the WebSocket URL to use for Deriv connections."""
    if _is_new_deriv_api():
        account_id = _deriv_options_account_id()
        r = _deriv_rest_api_call("POST", f"/trading/v1/options/accounts/{account_id}/otp")
        url = r.get("data", {}).get("url")
        if not url:
            raise RuntimeError(f"Failed to get Options WebSocket URL: {r}")
        return url
    return f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"

def deriv_send_auth(ws):
    """Send the authorize message on legacy v3 connections. New API uses OTP in URL."""
    if not _is_new_deriv_api():
        ws.send(json.dumps({"authorize": DERIV_APP_TOKEN}))

def deriv_proposal_payload(amount, basis, contract_type, currency, duration, duration_unit, symbol, barrier):
    """Build a proposal request payload compatible with legacy or new Deriv API."""
    payload = {
        "proposal": 1,
        "amount": amount,
        "basis": basis,
        "contract_type": contract_type,
        "currency": currency,
        "duration": duration,
        "duration_unit": duration_unit,
        "barrier": barrier,
    }
    symbol_field = "underlying_symbol" if _is_new_deriv_api() else "symbol"
    payload[symbol_field] = symbol
    return payload

def deriv_float(value, default=0.0):
    """Parse a value that may be a number or string (new API returns strings for some fields)."""
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default

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
# ── Second entry (re-entry on same symbol after a win) ────────────────
ALLOW_SECOND_ENTRY     = True     # enable controlled re-entry after a win
SECOND_ENTRY_COOLDOWN  = 5        # shortened cooldown (min) after a win
SECOND_ENTRY_ML_MIN    = 0.90     # stricter ML gate for re-entry (90%)
SECOND_ENTRY_WINDOW    = 15       # window (min) to wait for re-entry signal after win
MAX_CONSECUTIVE_LOSSES = 3
PAUSE_MINUTES          = 30
DAILY_LOSS_LIMIT       = -10.0        # session SL
DAILY_PROFIT_TARGET    = 5.0          # session TP
MAX_DAILY_TRADES       = 9999

# ── Indicator parameters ──────────────────────────────────────────────
EMA_SLOW               = 200          # EMA200 — long-term trend
EMA_FAST               = 50           # EMA50  — medium trend (Sober Book trend component)
ROC_PERIOD             = 10
ATR_PERIOD             = 14
ATR_MA_PERIOD          = 30
RSI_PERIOD             = 14
SUPERTREND_PERIOD      = 10
SUPERTREND_ATR_MULT    = 3.0
SCORE_THRESHOLD        = 75           # minimum score to trade (lowered from 93 per user request)
HEARTBEAT_INTERVAL_SEC = 3600      # richer hourly heartbeat

# ── Strategy version ────────────────────────────────────────────────────
# Bump this any time DURATION / SCORE_THRESHOLD / ML_CONFIDENCE_MIN /
# ATR_BARRIER_MULT (or the underlying strategy logic) changes, so reports
# can be compared apples-to-apples across config revisions over time.
STRATEGY_VERSION = "V2.6"

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
# Ordinal position of each session, for the ML "session_ord" engineered feature.
_ML_SESSION_ORDER = {name: i for i, name in enumerate(MARKET_SESSIONS.keys())}
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

# ── V3.0 Session Profiles ─────────────────────────────────────────────────
# Each session has its own configurable ML gate, 2nd-entry gate, TP, SL,
# max trades, cooldown, and mode.  Persisted to JSON so no code changes
# are ever needed — edit via /setsession command.
SESSION_PROFILES_PATH = "touch_session_profiles.json"
_SESSION_PROFILE_DEFAULTS: dict = {
    "Midnight":       {"ml_gate": 0.75, "ml2_gate": 0.90, "tp":  5.0, "sl":  -8.0, "max_trades":  6, "cooldown": 20, "mode": "Normal"},
    "Early Asian":    {"ml_gate": 0.80, "ml2_gate": 0.90, "tp":  5.0, "sl":  -8.0, "max_trades":  8, "cooldown": 20, "mode": "Normal"},
    "Late Asian":     {"ml_gate": 0.75, "ml2_gate": 0.90, "tp":  6.0, "sl":  -8.0, "max_trades":  8, "cooldown": 20, "mode": "Normal"},
    "London":         {"ml_gate": 0.80, "ml2_gate": 0.92, "tp":  5.0, "sl":  -8.0, "max_trades":  8, "cooldown": 20, "mode": "Normal"},
    "Early New York": {"ml_gate": 0.85, "ml2_gate": 0.92, "tp":  4.0, "sl":  -6.0, "max_trades":  6, "cooldown": 25, "mode": "Defensive"},
    "Late New York":  {"ml_gate": 0.70, "ml2_gate": 0.90, "tp": 10.0, "sl": -10.0, "max_trades": 10, "cooldown": 15, "mode": "Aggressive"},
}
SESSION_PROFILES: dict = {}   # populated at startup by _load_session_profiles()

# ── V3.0 Rolling trade statistics (in-memory, rebuilt each restart) ──────
from collections import deque as _deque
_ROLLING_MAXLEN         = 100   # per-session / per-symbol sliding window size
_rolling_session_deque: dict = {}   # session → deque[{win, profit, symbol}]
_rolling_symbol_deque:  dict = {}   # symbol  → deque[{win, profit, session}]
_prev_retrain_metrics:  dict = {}   # last walk-forward snapshot for before/after comparison

# ── ML filter ────────────────────────────────────────────────────────
MODEL_PATH        = "touch_ml_model.pkl"   # separate from sniper_bot.py's ml_model.pkl — different contract type/features
ML_MIN_TRADES     = 100
ML_RETRAIN_EVERY  = 100          # retrain every 100 new trades (V3.0: was 50, reduced spam)
ML_CONFIDENCE_MIN = 0.60              # require ≥60% confidence (lowered from 75%; adjustable via Telegram)
ML_STACKING_MIN_TRADES = 200     # below this, use a single regularized GB model;
                                 # at/above it, switch to the stacked ensemble (needs
                                 # more data to avoid over-fitting 3 base learners)
ML_FEATURE_COLS   = [
    "score", "wick_atr_ratio", "atr", "atr_ma",
    "ema_fast_slope", "ema_slow_slope", "ema_distance",
    "bb_width",                  # V2.6: Bollinger bandwidth — regime proxy
    # V2.7: oscillator + momentum — all stored in DB, normalised at train/predict time
    "rsi", "stochrsi_k", "roc", "body_ratio", "bb_position", "adx",
    # V2.9: context features — previously computed but discarded; now persisted so
    # ML can learn from session/asset health, each Sober Book component, and MTF state.
    "session_health", "asset_health",
    "sober_structure_pts", "sober_trend_pts", "sober_zone_pts",
    "sober_timing_pts", "sober_momentum_pts", "sober_candle_pts",
    "mtf_agreement",
]
# Engineered columns appended at train/predict time only (never stored) — see
# _ml_engineer(). Cyclical time-of-day lets the model learn session-dependent
# edge; regime_enc and atr_expansion encode market condition without one-hot
# blowing up the feature space on a small dataset.
_ML_ENGINEERED_COLS = ["hour_sin", "hour_cos", "session_ord", "regime_enc", "atr_expansion"]

# ── V2.6 adaptive gate & health ───────────────────────────────────────
ML_CONFIDENCE_MIN_FLOOR = 0.55   # gate never drops below this
ML_CONFIDENCE_MAX_CAP   = 0.70   # gate never rises above this
ASSET_BLACKLIST_WR          = 0.30   # WR below this triggers auto-blacklist
ASSET_BLACKLIST_MIN_TRADES  = 20     # minimum trades required to trigger
ASSET_BLACKLIST_HOURS       = 24     # hours to stay blacklisted
HEALTH_CACHE_TTL            = 300    # seconds between health-cache refreshes

# Regime encoding (ordinal, used in ML feature vector)
REGIME_ORDINAL = {
    "Strong Trend": 4.0,
    "Expansion":    3.0,
    "Weak Trend":   2.0,
    "Choppy":       1.0,
    "Compression":  0.0,
}

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
    filename="touch.log", level=logging.INFO,
    format="%(asctime)s [%(threadName)s] %(levelname)s %(message)s",
)
logger = logging.getLogger("touch")

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
# Awaiting a {"portfolio": 1} response to confirm/deny an orphaned buy whose
# ack never arrived — symbol -> {"details":..., "requested_at":...}
_portfolio_checks: dict = {}
ws_registry: dict = {}   # symbol -> live WebSocketApp, used to re-query contracts before assuming a loss
total_pnl = 0.0
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

# V2.6: Health, blacklist & feature history ───────────────────────────
_asset_health:               dict  = {}   # symbol  → float WR (0–1) last 30 trades
_session_health:             dict  = {}   # session → float WR (0–1) last 30 trades
_blacklisted_assets:         dict  = {}   # symbol  → UTC expiry datetime
_feature_importance_history: list  = []   # [{ts, importances}] across all retrains
_health_last_refresh:        float = 0.0  # epoch of last health cache update
_ml_last_retrain_tg_time:    float = 0.0  # epoch of last "retrain started" TG message
ML_RETRAIN_COOLDOWN_SECS:    int   = 300   # minimum seconds between completed retrains
_ml_last_retrain_time:       float = 0.0   # epoch of last completed retrain (cooldown)
_ml_optimal_threshold:       float = 0.60  # auto-tuned decision threshold from walk-forward eval
_ml_importance_weights:      dict  = {}   # feature_name → normalized importance (updated after retrain)
_ml_component_weights:       dict  = {}   # sober component → ML-tuned score multiplier (0.70–1.30)
ML_PER_SYMBOL_MIN_TRADES:    int   = 50   # min trades for a dedicated per-symbol model
ml_models_per_symbol:        dict  = {}   # symbol → model (checked before per-class + global)
ml_trained_per_symbol:       dict  = {}   # symbol → trade count at last per-symbol train

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


# ── V3.0 Session profile helpers ──────────────────────────────────────────

def _load_session_profiles() -> None:
    """Load per-session parameters from JSON; fall back to built-in defaults."""
    global SESSION_PROFILES
    import copy
    SESSION_PROFILES = copy.deepcopy(_SESSION_PROFILE_DEFAULTS)
    if os.path.exists(SESSION_PROFILES_PATH):
        try:
            with open(SESSION_PROFILES_PATH) as _f:
                _saved = json.load(_f)
            for _sess, _vals in _saved.items():
                if _sess in SESSION_PROFILES:
                    SESSION_PROFILES[_sess].update(_vals)
            logger.info(f"Session profiles loaded from {SESSION_PROFILES_PATH}")
        except Exception as _e:
            logger.warning(f"Could not load session profiles: {_e}")


def _save_session_profiles() -> None:
    """Persist current session profiles to JSON."""
    try:
        with open(SESSION_PROFILES_PATH, "w") as _f:
            json.dump(SESSION_PROFILES, _f, indent=2)
    except Exception as _e:
        logger.warning(f"Could not save session profiles: {_e}")


def _get_session_profile(session: str = None) -> dict:
    """Return the live profile dict for a session (falls back to defaults)."""
    if session is None:
        session = _get_session_name()
    src = SESSION_PROFILES if SESSION_PROFILES else _SESSION_PROFILE_DEFAULTS
    return src.get(session, _SESSION_PROFILE_DEFAULTS.get(session, {}))


# ── V3.0 Rolling statistics helpers ───────────────────────────────────────

def _rolling_push(session: str, symbol: str, win: bool, profit: float) -> None:
    """Record a settled trade into the in-memory rolling windows."""
    rec = {"win": win, "profit": profit, "symbol": symbol, "session": session}
    _rolling_session_deque.setdefault(session, _deque(maxlen=_ROLLING_MAXLEN)).append(rec)
    _rolling_symbol_deque.setdefault(symbol,  _deque(maxlen=_ROLLING_MAXLEN)).append(rec)


def _rolling_stats_for(records, n: int = None) -> dict:
    """WR, profit, profit-factor, avg, max-drawdown for last n records (or all)."""
    recs = list(records)[-n:] if n else list(records)
    if not recs:
        return {"n": 0, "wr": None, "profit": 0.0, "pf": None, "avg": 0.0, "dd": 0.0}
    total   = len(recs)
    wins    = sum(1 for r in recs if r["win"])
    pos_sum = sum(r["profit"] for r in recs if r["profit"] > 0)
    neg_sum = abs(sum(r["profit"] for r in recs if r["profit"] < 0))
    cum, peak, dd = [0.0], 0.0, 0.0
    for r in recs:
        cum.append(cum[-1] + r["profit"])
    for v in cum[1:]:
        if v > peak:
            peak = v
        dd = max(dd, peak - v)
    return {
        "n":      total,
        "wr":     wins / total,
        "profit": round(sum(r["profit"] for r in recs), 2),
        "pf":     round(pos_sum / max(neg_sum, 0.001), 2),
        "avg":    round(sum(r["profit"] for r in recs) / total, 2),
        "dd":     round(dd, 2),
    }


def _ml_generate_recommendations(eval_res: dict) -> str:
    """Build a suggest-only recommendations message — never auto-applied."""
    lines = [
        "📋 <b>Session Recommendations</b>",
        "<i>No changes applied — manual approval only.</i>",
    ]
    if not eval_res:
        lines.append("  Not enough data for recommendations.")
        return "\n".join(lines)
    opt_t    = eval_res.get("opt_threshold", ML_CONFIDENCE_MIN)
    cur_gate = ML_CONFIDENCE_MIN
    if abs(opt_t - cur_gate) >= 0.05:
        word = "raise" if opt_t > cur_gate else "lower"
        lines.append(
            f"  🎯 ML Gate: current <b>{cur_gate*100:.0f}%</b> → suggested "
            f"<b>{opt_t*100:.0f}%</b>  ({word} — walk-forward optimum)\n"
            f"     → <code>/set ml_gate {opt_t*100:.0f}</code>"
        )
    for sess_name, drec in _rolling_session_deque.items():
        stats = _rolling_stats_for(drec, n=50)
        if stats["n"] < 20:
            continue
        prof   = _get_session_profile(sess_name)
        cur_tp = prof.get("tp", DAILY_PROFIT_TARGET)
        cur_sl = prof.get("sl", DAILY_LOSS_LIMIT)
        if stats["wr"] and stats["wr"] > 0.75 and stats["avg"] > 0.40:
            sug_tp = round(cur_tp * 1.15, 1)
            lines.append(
                f"  📈 <b>{sess_name} TP:</b> ${cur_tp:.1f} → <b>${sug_tp:.1f}</b> "
                f"(WR {stats['wr']*100:.0f}%, avg +${stats['avg']:.2f})\n"
                f"     → <code>/setsession \"{sess_name}\" tp {sug_tp}</code>"
            )
        elif stats["wr"] and stats["wr"] < 0.55 and stats["avg"] < -0.20:
            sug_sl = round(cur_sl * 0.80, 1)
            lines.append(
                f"  📉 <b>{sess_name} SL:</b> ${cur_sl:.1f} → <b>${sug_sl:.1f}</b> "
                f"(WR {stats['wr']*100:.0f}%, avg ${stats['avg']:.2f})\n"
                f"     → <code>/setsession \"{sess_name}\" sl {sug_sl}</code>"
            )
    if len(lines) == 2:
        lines.append("  ✅ All settings look appropriate for current performance.")
    return "\n".join(lines)


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
    global ml_models_per_symbol, ml_trained_per_symbol
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                payload = pickle.load(f)
            if isinstance(payload, dict):
                ml_model = payload.get("model")
                ml_trained_on = payload.get("trained_on", 0)
                ml_models_per_class  = payload.get("per_class_models",  ml_models_per_class)
                ml_trained_per_class = payload.get("per_class_trained", ml_trained_per_class)
                ml_models_per_symbol  = payload.get("per_symbol_models",  ml_models_per_symbol)
                ml_trained_per_symbol = payload.get("per_symbol_trained", ml_trained_per_symbol)
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
                "per_symbol_models":  ml_models_per_symbol,
                "per_symbol_trained": ml_trained_per_symbol,
            }, f)
    except Exception as e:
        logger.error(f"Failed to save ML model: {e}")


def _ml_engineer(raw_vals: list, timestamp: str, session: str, regime: str = "Choppy") -> list:
    """Extend the raw DB feature vector with engineered features computed at
    train/predict time (never stored in DB):
      hour_sin / hour_cos — cyclical time-of-day (session edge without one-hot)
      session_ord         — ordinal market-session position
      regime_enc          — ordinal regime encoding (V2.6)
      atr_expansion       — ATR / ATR_MA ratio (V2.6)
    """
    hour = 12.0
    if timestamp:
        try:
            hour = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00")).hour
        except (ValueError, TypeError):
            pass
    hour_sin    = math.sin(2 * math.pi * hour / 24.0)
    hour_cos    = math.cos(2 * math.pi * hour / 24.0)
    session_ord = float(_ML_SESSION_ORDER.get(session, -1))
    regime_enc  = REGIME_ORDINAL.get(regime, 1.0)
    # ATR expansion: raw_vals[2]=atr, raw_vals[3]=atr_ma (ML_FEATURE_COLS order)
    try:
        def _safe_float(v, default=0.0):
            if v is None:
                return default
            try:
                f = float(v)
                return default if not np.isfinite(f) else f
            except (TypeError, ValueError):
                return default
        atr_v         = _safe_float(raw_vals[2] if len(raw_vals) > 2 else None, 0.0)
        atr_ma_v      = _safe_float(raw_vals[3] if len(raw_vals) > 3 else None, 0.0)
        atr_expansion = atr_v / max(atr_ma_v, 1e-9) if atr_ma_v > 0 else 1.0
        if not np.isfinite(atr_expansion):
            atr_expansion = 1.0
    except (IndexError, TypeError, ValueError):
        atr_expansion = 1.0
    # Replace any NaN in raw_vals with 0 before appending engineered features
    clean_raw = [0.0 if (v is None or (isinstance(v, float) and not np.isfinite(v))) else v
                 for v in raw_vals]
    return clean_raw + [hour_sin, hour_cos, session_ord, regime_enc, atr_expansion]


def _ml_build_matrix(rows: list, n_raw: int) -> tuple:
    """rows: (symbol, *raw_feature_vals, timestamp, session, regime, win).
    V2.6: parses regime for _ml_engineer; cleans NaN/inf before returning.
    V2.8: replaces inf with NaN first, then fills column-wise with medians
    instead of zero — avoids systematic bias from clipping and suppresses
    pandas/numpy RuntimeWarnings from ATR/ADX edge-case divisions."""
    X, y = [], []
    for r in rows:
        raw = []
        for i in range(n_raw):
            v = r[1 + i]
            if v is None:
                raw.append(np.nan)
            else:
                try:
                    fv = float(v)
                    raw.append(np.nan if not np.isfinite(fv) else fv)
                except (ValueError, TypeError):
                    raw.append(np.nan)
        ts         = r[1 + n_raw]
        sess       = r[2 + n_raw]
        regime_str = str(r[3 + n_raw]) if r[3 + n_raw] else "Choppy"
        win        = r[4 + n_raw]
        X.append(_ml_engineer(raw, ts, sess, regime_str))
        y.append(win)
    if X:
        X_arr = np.array(X, dtype=float)
        # Replace ±inf with NaN so they participate in median computation
        X_arr = np.where(np.isinf(X_arr), np.nan, X_arr)
        # Fill NaN column-wise with column medians (neutral, data-driven value)
        col_medians = np.nanmedian(X_arr, axis=0)
        col_medians = np.nan_to_num(col_medians, nan=0.0)
        nan_mask = np.isnan(X_arr)
        X_arr[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])
        X = X_arr.tolist()
    return X, y


def _make_stacking_clf(random_state: int = 42) -> StackingClassifier:
    """The 'exotic' ensemble: 3 diverse base learners (boosting, bagging,
    extra-randomized trees) feeding a logistic-regression meta-learner that
    learns how to blend them, wrapped in probability calibration so the
    output is a genuine, trustworthy confidence — not just a raw vote.
    Ported from sniper_bot.py, used once enough trades exist (see
    ML_STACKING_MIN_TRADES)."""
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
    for the full stacking ensemble. Regularised to avoid over-fitting on
    small datasets; returns a plain GradientBoostingClassifier so that
    feature_importances_ is directly accessible after fitting."""
    return GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=5, random_state=random_state,
    )


def _walk_forward_eval(build_fn, X: list, y: list, weights) -> Optional[dict]:
    """Time-ordered (no shuffling) holdout: train on the first 80% chronologically,
    score on the last 20% never seen in training.
    V2.8: ROC-AUC, PR-AUC, MCC, Balanced Accuracy, ECE calibration error, and
    auto-searched optimal profit threshold (replaces fixed 60% heuristic)."""
    from sklearn.metrics import (
        f1_score, recall_score, roc_auc_score, brier_score_loss, confusion_matrix,
        average_precision_score, matthews_corrcoef, balanced_accuracy_score,
    )
    import warnings as _w
    n = len(X)
    split = int(n * 0.8)
    if split < 10 or (n - split) < 5:
        return None
    Xtr, Xte = X[:split], X[split:]
    ytr, yte = y[:split], y[split:]
    if len(set(ytr)) < 2 or len(set(yte)) < 2:
        return None
    try:
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            Xtr_arr = np.nan_to_num(np.array(Xtr, dtype=float), nan=0.0, posinf=10.0, neginf=-10.0)
            Xte_arr = np.nan_to_num(np.array(Xte, dtype=float), nan=0.0, posinf=10.0, neginf=-10.0)
            m = build_fn()
            if weights is not None:
                try:
                    m.fit(Xtr_arr, ytr, sample_weight=weights[:split])
                except TypeError:
                    m.fit(Xtr_arr, ytr)
            else:
                m.fit(Xtr_arr, ytr)

        classes = list(m.classes_)
        win_idx = classes.index(1) if 1 in classes else len(classes) - 1
        proba   = m.predict_proba(Xte_arr)[:, win_idx]
        yte_arr = np.array(yte, dtype=int)

        # ── Auto-search optimal threshold: maximise expected profit ────────
        # Sweep thresholds 0.40–0.90; for each, count net wins (TP − FP).
        # Requires at least 3 trades to be meaningful.
        best_thresh, best_ep = 0.60, -999.0
        for thresh in np.arange(0.40, 0.91, 0.01):
            preds_t = (proba >= thresh).astype(int)
            taken   = int(preds_t.sum())
            if taken < 3:
                continue
            tp_t = int(((preds_t == 1) & (yte_arr == 1)).sum())
            fp_t = int(((preds_t == 1) & (yte_arr == 0)).sum())
            ep   = tp_t - fp_t   # net wins = expected profit at equal stakes
            if ep > best_ep or (ep == best_ep and thresh < best_thresh):
                best_ep, best_thresh = ep, float(thresh)

        preds      = (proba >= best_thresh).astype(int).tolist()
        acc        = sum(1 for p, t in zip(preds, yte) if p == t) / len(yte)
        said_trade = [i for i, p in enumerate(preds) if p == 1]
        precision  = (sum(1 for i in said_trade if yte[i] == 1) / len(said_trade)
                      if said_trade else 0.0)
        recall      = recall_score(yte, preds, zero_division=0)
        f1          = f1_score(yte, preds, zero_division=0)
        baseline_wr = float(sum(yte) / len(yte))

        try:
            roc_auc = roc_auc_score(yte, proba)
        except ValueError:
            roc_auc = 0.5
        try:
            pr_auc = average_precision_score(yte, proba)
        except ValueError:
            pr_auc = float("nan")
        brier = brier_score_loss(yte, proba)
        try:
            mcc = float(matthews_corrcoef(yte, preds))
        except Exception:
            mcc = 0.0
        try:
            bal_acc = float(balanced_accuracy_score(yte, preds))
        except Exception:
            bal_acc = float("nan")

        # Expected Calibration Error — 10-bin uniform
        try:
            from sklearn.calibration import calibration_curve
            frac_pos, mean_pred = calibration_curve(yte, proba, n_bins=10, strategy="uniform")
            ece = float(np.mean(np.abs(frac_pos - mean_pred)))
        except Exception:
            ece = float("nan")

        try:
            cm = confusion_matrix(yte, preds, labels=[0, 1])
            tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
        except Exception:
            tn = fp = fn = tp = 0

        return {
            "n_test": len(yte), "acc": acc, "precision": precision,
            "recall": recall, "f1": f1, "roc_auc": roc_auc, "pr_auc": pr_auc,
            "brier": brier, "ece": ece, "mcc": mcc, "bal_acc": bal_acc,
            "baseline_wr": baseline_wr, "opt_threshold": best_thresh,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        }
    except Exception as e:
        logger.error(f"walk-forward eval failed: {e}")
        return None


def _ml_get_confidence(details: dict, symbol: str = "") -> float:
    """Return ML win-probability (0.0–1.0) using the best available model for this symbol.
    V2.6: 8 raw features (+ bb_width); passes regime to engineer."""
    cls = _get_symbol_class(symbol) if symbol else "standard_vol"
    with ml_lock:
        # V2.9: per-symbol model checked first; per-class next; global fallback
        model = ml_models_per_symbol.get(symbol) or ml_models_per_class.get(cls) or ml_model
    if model is None:
        return 1.0
    try:
        now     = datetime.now(timezone.utc)
        session = _get_session_name(now)
        raw = [
            details.get("total_score", 0),
            details.get("wick_atr_ratio", 0),
            details.get("atr", 0) or 0,
            details.get("atr_ma", 0) or 0,
            details.get("ema_fast_sl", 0) or 0,
            details.get("ema_slow_sl", 0) or 0,
            details.get("ema_distance", 0) or 0,
            details.get("bb_width", 0) or 0,
            # V2.7: oscillators normalised so all raw vals sit in ~[0,1] range
            float(details.get("rsi") or 50.0) / 100.0,
            float(details.get("stochrsi_k") or 50.0) / 100.0,
            float(details.get("roc") or 0.0),
            float(details.get("body_ratio") or 0.0),
            float(details.get("bb_position") or 0.5),
            float(details.get("adx") or 0.0) / 100.0,
            # V2.9: context features — normalised to match DB storage convention
            float(details.get("session_health") or _get_session_health(session)),
            float(details.get("asset_health")   or _get_asset_health(symbol)),
            float(details.get("sober_structure_pts") or 0.0) / 25.0,
            float(details.get("sober_trend_pts")     or 0.0) / 20.0,
            float(details.get("sober_zone_pts")      or 0.0) / 20.0,
            float(details.get("sober_timing_pts")    or 0.0) / 15.0,
            float(details.get("sober_momentum_pts")  or 0.0) / 10.0,
            float(details.get("sober_candle_pts")    or 0.0) / 10.0,
            float(details.get("mtf_agreement")       or 0.0) / 2.0,
        ]
        regime  = details.get("regime", "Choppy")
        feats   = [_ml_engineer(raw, now.isoformat(), session, regime)]
        feats   = np.nan_to_num(np.array(feats, dtype=float), nan=0.0, posinf=10.0, neginf=-10.0).tolist()
        proba   = model.predict_proba(feats)[0]
        classes = list(model.classes_)
        win_idx = classes.index(1) if 1 in classes else len(classes) - 1
        return float(proba[win_idx])
    except Exception as e:
        logger.error(f"ML confidence failed, allowing trade: {e}")
        return 1.0


def _ml_should_trade(details: dict, symbol: str = "", min_conf: float = None) -> bool:
    """Filter trade by ML confidence. Stores 'ml_confidence' in details for display.
    V2.6: Uses adaptive gate based on regime + health.
    V2.9: Captures session/asset health into details BEFORE ML inference so the model
    sees the same values that get written to the DB training row."""
    # Capture health at decision time — must precede _ml_get_confidence so the
    # model sees real health values during inference AND they persist in the DB row.
    _now_dt  = datetime.now(timezone.utc)
    _session = _get_session_name(_now_dt)
    details.setdefault("session_health", _get_session_health(_session))
    details.setdefault("asset_health",   _get_asset_health(symbol))
    conf = _ml_get_confidence(details, symbol)
    details["ml_confidence"] = conf
    cls = _get_symbol_class(symbol) if symbol else "standard_vol"
    with ml_lock:
        has_model = (ml_models_per_symbol.get(symbol) or
                     ml_models_per_class.get(cls) or ml_model) is not None
    if not has_model:
        return True   # observe-only until first model
    if min_conf is not None:
        gate = min_conf   # explicit override (e.g. second-entry higher bar)
    else:
        now     = datetime.now(timezone.utc)
        session = _get_session_name(now)
        regime  = details.get("regime", "Choppy")
        gate    = _adaptive_ml_gate(symbol, session, regime)
    details["ml_gate_used"] = gate
    # V3.0 decision tree — record ML gate outcome in details for gate trace
    details.setdefault("gate_trace", [])
    details["gate_trace"].append((
        "ML",
        "PASS" if conf >= gate else "FAIL",
        f"{conf*100:.1f}% vs {gate*100:.0f}% gate ({_get_session_name()} profile)",
    ))
    # Send explanation only for very high (≥90%) or very low (<45%) confidence
    if conf >= 0.90 or conf < 0.45:
        try:
            now     = datetime.now(timezone.utc)
            session = _get_session_name(now)
            expl    = _ml_confidence_explanation(details, conf, symbol, session)
            _send_tg(f"🤖 <b>ML Confidence Note</b> — <code>{symbol}</code>\n{expl}")
        except Exception:
            pass
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
        rows = _db_fetch(f"SELECT {', '.join(cols)} FROM touch_trades ORDER BY id")
        if not rows:
            # Fallback: try without market_session (older schema)
            cols = cols[:-1]
            rows = _db_fetch(f"SELECT {', '.join(cols)} FROM touch_trades ORDER BY id")
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


# ══════════════════════════════════════════════════════════════════════
#  V2.6 — MARKET REGIME DETECTION
# ══════════════════════════════════════════════════════════════════════
def _compute_market_regime(ind: dict) -> str:
    """Classify market into one of 5 regimes using ADX, ATR expansion, BB width.
    Strong Trend / Expansion / Weak Trend / Choppy / Compression."""
    try:
        adx      = float(ind.get("adx") or 0)
        atr      = float(ind.get("atr") or 1)
        atr_ma   = float(ind.get("atr_ma") or 1)
        bb_u     = float(ind.get("bb_upper") or 0)
        bb_l     = float(ind.get("bb_lower") or 0)
        bb_m     = float(ind.get("bb_mid") or 1)
        atr_exp  = atr / max(atr_ma, 1e-9)
        bb_w     = (bb_u - bb_l) / max(bb_m, 1e-9) if bb_m > 0 else 0.0
        if   adx >= 35 and atr_exp >= 1.10:                         return "Strong Trend"
        elif atr_exp >= 1.15:                                        return "Expansion"
        elif adx >= 22 and atr_exp >= 0.95:                         return "Weak Trend"
        elif atr_exp <= 0.88 or (0 < bb_w <= 0.008):               return "Compression"
        else:                                                        return "Choppy"
    except Exception:
        return "Choppy"


# ══════════════════════════════════════════════════════════════════════
#  V2.6 — ASSET & SESSION HEALTH
# ══════════════════════════════════════════════════════════════════════
def _refresh_health_caches():
    """Query DB for rolling WR per asset (last 30 trades) and per session (last 30).
    Stores results in module-level _asset_health and _session_health dicts."""
    global _asset_health, _session_health, _health_last_refresh
    try:
        # ── Per-asset: newest 600 confirmed rows, take first 30 per symbol ──
        rows = _db_fetch(
            "SELECT symbol, win FROM touch_trades WHERE win IN (0,1) ORDER BY id DESC LIMIT 600"
        )
        a_wins, a_total = {}, {}
        for sym, win in rows:
            if a_total.get(sym, 0) >= 30:
                continue
            a_wins[sym]  = a_wins.get(sym, 0) + (1 if win == 1 else 0)
            a_total[sym] = a_total.get(sym, 0) + 1
        _asset_health = {sym: a_wins.get(sym, 0) / max(a_total[sym], 1) for sym in a_total}
        # ── Per-session: newest 300 confirmed rows, take first 30 per session ──
        rows2 = _db_fetch(
            "SELECT market_session, win FROM touch_trades WHERE win IN (0,1) "
            "AND market_session IS NOT NULL ORDER BY id DESC LIMIT 300"
        )
        s_wins, s_total = {}, {}
        for sess, win in rows2:
            if s_total.get(sess, 0) >= 30:
                continue
            s_wins[sess]  = s_wins.get(sess, 0) + (1 if win == 1 else 0)
            s_total[sess] = s_total.get(sess, 0) + 1
        _session_health = {s: s_wins.get(s, 0) / max(s_total[s], 1) for s in s_total}
        _health_last_refresh = time.time()
        logger.debug(f"Health cache refreshed: {len(_asset_health)} assets, {len(_session_health)} sessions")
    except Exception as e:
        logger.error(f"_refresh_health_caches: {e}")


def _maybe_refresh_health():
    """Spawn a background refresh if the health cache is older than HEALTH_CACHE_TTL."""
    if time.time() - _health_last_refresh > HEALTH_CACHE_TTL:
        threading.Thread(target=_refresh_health_caches, daemon=True, name="HealthRefresh").start()


def _get_asset_health(symbol: str) -> float:
    """Return rolling WR for this symbol (0–1). 0.65 = neutral when no data."""
    _maybe_refresh_health()
    return _asset_health.get(symbol, 0.65)


def _get_session_health(session: str) -> float:
    """Return rolling WR for this session (0–1). 0.65 = neutral when no data."""
    return _session_health.get(session, 0.65)


# ══════════════════════════════════════════════════════════════════════
#  V2.6 — AUTO BLACKLIST
# ══════════════════════════════════════════════════════════════════════
def _is_blacklisted(symbol: str) -> bool:
    """Return True if the asset is currently blacklisted and the ban hasn't expired."""
    expiry = _blacklisted_assets.get(symbol)
    if expiry is None:
        return False
    if datetime.now(timezone.utc) >= expiry:
        _blacklisted_assets.pop(symbol, None)
        return False
    return True


def _check_and_update_blacklist(symbol: str):
    """Blacklist an asset for ASSET_BLACKLIST_HOURS if its last-20-trade WR < threshold."""
    try:
        rows = _db_fetch(
            "SELECT win FROM touch_trades WHERE symbol=? AND win IN (0,1) ORDER BY id DESC LIMIT 20",
            (symbol,)
        )
        if len(rows) < ASSET_BLACKLIST_MIN_TRADES:
            return
        wr = sum(1 for (w,) in rows if w == 1) / len(rows)
        if wr < ASSET_BLACKLIST_WR and not _is_blacklisted(symbol):
            expiry = datetime.now(timezone.utc) + timedelta(hours=ASSET_BLACKLIST_HOURS)
            _blacklisted_assets[symbol] = expiry
            logger.warning(f"AUTO-BLACKLIST: {symbol} WR={wr:.0%} in last 20 trades → paused {ASSET_BLACKLIST_HOURS}h")
            _send_tg(
                f"⛔ <b>AUTO-BLACKLIST</b> — <code>{symbol}</code>\n"
                f"Last {ASSET_BLACKLIST_MIN_TRADES} trades WR: <b>{wr:.0%}</b> "
                f"(below {ASSET_BLACKLIST_WR:.0%} threshold)\n"
                f"Asset paused for <b>{ASSET_BLACKLIST_HOURS} hours</b>."
            )
    except Exception as e:
        logger.error(f"_check_and_update_blacklist {symbol}: {e}")


# ══════════════════════════════════════════════════════════════════════
#  V2.6 — ADAPTIVE ML GATE
# ══════════════════════════════════════════════════════════════════════
def _adaptive_ml_gate(symbol: str, session: str, regime: str) -> float:
    """Return the ML confidence gate for this trade.
    V3.0: base = session profile ml_gate (per-session configurable).
    Adaptive ±2-5% nudges for regime, asset health, session health.
    Hard floor = max(ML_CONFIDENCE_MIN_FLOOR, global ML_CONFIDENCE_MIN, 90% of session gate).
    No hard upper cap — the user's session profile gate is the intended ceiling.
    The walk-forward optimum (V2.8) only RAISES the base, never lowers it below
    the user-set gates, so manual /set ml_gate is always respected."""
    prof         = _get_session_profile(session)
    sess_gate    = prof.get("ml_gate", ML_CONFIDENCE_MIN)
    # Walk-forward optimum may raise the base if the data supports a higher gate,
    # but it never overrides ML_CONFIDENCE_MIN or the session profile gate.
    base = max(ML_CONFIDENCE_MIN, sess_gate, _ml_optimal_threshold)
    gate = base
    asset_h   = _get_asset_health(symbol)
    session_h = _get_session_health(session)
    # Regime nudges (small — adaptive gate should not dramatically override base)
    if   regime == "Strong Trend":  gate -= 0.03
    elif regime == "Expansion":     gate -= 0.01
    elif regime == "Choppy":        gate += 0.03
    elif regime == "Compression":   gate += 0.05
    # Asset health
    if   asset_h < 0.40:  gate += 0.05
    elif asset_h > 0.70:  gate -= 0.02
    # Session health
    if   session_h < 0.40:  gate += 0.05
    elif session_h > 0.70:  gate -= 0.02
    # Floor: never drop below the highest user-configured gate
    hard_floor = max(ML_CONFIDENCE_MIN_FLOOR, ML_CONFIDENCE_MIN, sess_gate * 0.90)
    return round(max(hard_floor, min(0.97, gate)), 3)


# ══════════════════════════════════════════════════════════════════════
#  V2.6 — CONFIDENCE EXPLANATION
# ══════════════════════════════════════════════════════════════════════
def _ml_confidence_explanation(details: dict, conf: float, symbol: str, session: str) -> str:
    """Return bullet-point reasons for this confidence reading (shown for extreme values)."""
    regime    = details.get("regime", "Choppy")
    adx       = float(details.get("ema_slow_sl") or details.get("adx") or 0)
    score     = int(details.get("total_score") or 0)
    wick      = float(details.get("wick_atr_ratio") or 0)
    atr       = float(details.get("atr") or 1)
    atr_ma    = float(details.get("atr_ma") or 1)
    atr_exp   = atr / max(atr_ma, 1e-9)
    asset_h   = _get_asset_health(symbol)
    session_h = _get_session_health(session)
    pos, neg  = [], []
    if regime in ("Strong Trend", "Expansion"):  pos.append(f"✅ {regime}")
    if adx >= 30:                                 pos.append(f"✅ ADX {adx:.0f} (strong trend)")
    if score >= 85:                               pos.append(f"✅ Score {score}/100")
    if wick >= 1.5:                               pos.append(f"✅ Wick {wick:.1f}×ATR")
    if atr_exp >= 1.1:                            pos.append(f"✅ ATR expanding {atr_exp:.2f}×")
    if asset_h >= 0.70:                           pos.append(f"✅ {symbol} health {asset_h:.0%} WR")
    if session_h >= 0.70:                         pos.append(f"✅ {session} health {session_h:.0%} WR")
    if regime in ("Choppy", "Compression"):       neg.append(f"⚠️ {regime} regime")
    if adx < 15:                                  neg.append(f"⚠️ ADX {adx:.0f} (weak trend)")
    if score < 80:                                neg.append(f"⚠️ Score {score}/100 (borderline)")
    if atr_exp <= 0.88:                           neg.append(f"⚠️ ATR contracting {atr_exp:.2f}×")
    if asset_h < 0.40:                            neg.append(f"⚠️ {symbol} health {asset_h:.0%} WR")
    if session_h < 0.40:                          neg.append(f"⚠️ {session} health {session_h:.0%} WR")
    bullets = (pos + neg)[:6] or ["Neutral conditions"]
    label   = "🚀 High confidence" if conf >= 0.90 else "⚠️ Low confidence"
    return f"{label} ({conf*100:.0f}%)\n" + "\n".join(bullets)


# ══════════════════════════════════════════════════════════════════════
#  V2.6 — HEALTH DASHBOARD TEXT
# ══════════════════════════════════════════════════════════════════════
def _health_text() -> str:
    _maybe_refresh_health()
    now_utc = datetime.now(timezone.utc)
    bl_live = {sym: exp for sym, exp in _blacklisted_assets.items() if exp > now_utc}
    lines   = ["🏥 <b>BOT HEALTH DASHBOARD</b>", "━━━━━━━━━━━━━━━━━━━━"]
    # Blacklisted
    if bl_live:
        lines.append("\n⛔ <b>Blacklisted Assets</b>")
        for sym, exp in bl_live.items():
            hrs = max(0, int((exp - now_utc).total_seconds() / 3600))
            lines.append(f"  <code>{sym}</code> — {hrs}h remaining")
    # Asset health
    lines.append("\n📊 <b>Asset Health</b>  (last 30 trades per asset)")
    if _asset_health:
        for sym in sorted(_asset_health, key=lambda s: _asset_health[s], reverse=True):
            h   = _asset_health[sym]
            bar = "🟢" if h >= 0.60 else ("🟡" if h >= 0.45 else "🔴")
            bl  = " ⛔" if sym in bl_live else ""
            lines.append(f"  {bar} <code>{sym:<10}</code>  {h:.0%}{bl}")
    else:
        lines.append("  — no data yet (need trades per asset) —")
    # Session health
    lines.append("\n🕐 <b>Session Health</b>  (last 30 trades per session)")
    if _session_health:
        for sess in sorted(_session_health, key=lambda s: _session_health[s], reverse=True):
            h   = _session_health[sess]
            bar = "🟢" if h >= 0.60 else ("🟡" if h >= 0.45 else "🔴")
            lines.append(f"  {bar} {sess:<20}  {h:.0%}")
    else:
        lines.append("  — no data yet —")
    # Current regimes
    lines.append("\n🌡 <b>Current Market Regimes</b>")
    with _lock:
        regime_map = {sym: (indicators.get(sym) or {}).get("regime", "?") for sym in SYMBOLS}
    groups: dict = {}
    for sym, reg in regime_map.items():
        groups.setdefault(reg, []).append(sym)
    icons = {"Strong Trend": "🚀", "Expansion": "📈", "Weak Trend": "📊",
             "Choppy": "🔀", "Compression": "🔵", "?": "❓"}
    for reg in ["Strong Trend", "Expansion", "Weak Trend", "Choppy", "Compression", "?"]:
        syms = groups.get(reg)
        if syms:
            lines.append(f"  {icons.get(reg,'')} <b>{reg}</b>: {', '.join(syms)}")
    # Feature importance history summary
    if _feature_importance_history:
        last = _feature_importance_history[-1]
        lines += ["", "<b>Last ML Feature Importance</b>  " + last.get("ts", "")]
        for col, pct in sorted(last.get("importances", {}).items(), key=lambda x: x[1], reverse=True)[:5]:
            lines.append(f"  {col}: {pct:.0%}")
    lines += ["━━━━━━━━━━━━━━━━━━━━",
              f"<i>Adaptive gate: {ML_CONFIDENCE_MIN_FLOOR:.0%}–{ML_CONFIDENCE_MAX_CAP:.0%}  "
              f"·  Blacklist threshold: {ASSET_BLACKLIST_WR:.0%} WR / {ASSET_BLACKLIST_MIN_TRADES} trades</i>"]
    return "\n".join(lines)


def _ml_train():
    """Train / retrain the ML model (global + per-class). Global model is a
    stacked ensemble (GB+RF+ExtraTrees -> calibrated LogReg) once enough real
    trades exist, else a single regularized GradientBoosting model. Per-class
    models stay RandomForest — those datasets are smaller and RF is cheap and
    robust there. Sets ml_training_active; clears in finally."""
    global ml_model, ml_trained_on, ml_training_active, ml_models_per_class, ml_trained_per_class
    global _ml_last_retrain_tg_time, _feature_importance_history
    global _ml_last_retrain_time, _ml_optimal_threshold
    global _ml_importance_weights, _ml_component_weights
    global ml_models_per_symbol, ml_trained_per_symbol
    global _prev_retrain_metrics
    try:
        # BUG FIX (V3.0): stamp the cooldown immediately so concurrent or rapid-fire
        # calls to _ml_maybe_retrain see the cooldown before the training query finishes,
        # preventing back-to-back retrain storms after a burst of trade settlements.
        _ml_last_retrain_time = time.time()
        _prev_snap = dict(_prev_retrain_metrics)
        n_raw = len(ML_FEATURE_COLS)
        try:
            # timestamp + market_session + regime needed for engineered features.
            # COALESCE(regime,'Choppy') handles rows inserted before V2.6.
            # ORDER BY id keeps rows chronological for recency weighting + walk-forward.
            if USE_PG:
                regime_col = "COALESCE(regime, 'Choppy')"
            else:
                regime_col = "COALESCE(regime, 'Choppy')"
            rows_sym = _db_fetch(
                f"SELECT symbol, {', '.join(ML_FEATURE_COLS)}, timestamp, market_session, "
                f"{regime_col}, win "
                f"FROM touch_trades WHERE win IN (0, 1) ORDER BY id"
            )
        except Exception as e:
            logger.error(f"ML training query failed: {e}")
            _send_tg(f"🤖 <b>ML Training Error</b>\n<code>{e}</code>\nWill retry on next trade.")
            return

        total = len(rows_sym)
        # ── Spam guard: only send "started" TG message if ≥120s since last one ──
        now_ts = time.time()
        if now_ts - _ml_last_retrain_tg_time >= 120:
            _send_tg(
                f"🤖 <b>ML RETRAINING</b> — started\n"
                f"Training on <b>{total}</b> real trades (global + 3 class models)…\n"
                f"<code>[{_ml_progress_bar(0.0)}]   0%</code>"
            )
            _ml_last_retrain_tg_time = now_ts

        if total < ML_MIN_TRADES:
            _send_tg(
                f"🤖 <b>ML RETRAINING</b> — skipped\n"
                f"Need {ML_MIN_TRADES} trades, only {total} recorded.\n"
                f"<code>[{_ml_progress_bar(total / ML_MIN_TRADES)}] "
                f"{int(total / ML_MIN_TRADES * 100)}%</code> (observe-only)"
            )
            return

        X, y = _ml_build_matrix(rows_sym, n_raw)
        all_feature_cols = ML_FEATURE_COLS + _ML_ENGINEERED_COLS

        if len(set(y)) < 2:
            _send_tg("🤖 <b>ML RETRAINING</b> — skipped\nNeed both wins AND losses in history.")
            return

        # ── Sample weights: recency decay × class balance ───────────────────
        # Exponential decay weights (newer = more important) multiplied by
        # sklearn balanced class weights so wins/losses contribute equally
        # regardless of real-world class imbalance in the training set.
        from sklearn.utils.class_weight import compute_sample_weight
        half_life  = 100.0
        recency_w  = np.exp(np.linspace(-total / half_life, 0.0, total))
        balance_w  = compute_sample_weight("balanced", y)
        weights    = recency_w * balance_w
        weights   /= weights.mean()   # normalise to mean ≈ 1 for numerical stability
        use_stack  = total >= ML_STACKING_MIN_TRADES

        # ── Walk-forward validation: chronological holdout, never random ────
        # Random splits leak future information into training (lookahead bias) —
        # rows are already ordered oldest→newest, so we train on the first 80%
        # and score purely on the most recent, unseen 20%, using the SAME
        # architecture the global model below will actually run.
        wf_line = "Walk-forward: not enough data yet"
        eval_res = _walk_forward_eval(
            (_make_stacking_clf if use_stack else _make_gb_clf), X, y, weights,
        )
        if eval_res:
            opt_t = eval_res.get("opt_threshold", 0.60)
            _ml_optimal_threshold = opt_t   # persist; only raises gate, never lowers below user gates
            # Update prev-metrics store for next cycle's before/after comparison
            _prev_retrain_metrics.update({
                "acc":           eval_res.get("acc", 0),
                "pr_auc":        eval_res.get("pr_auc", 0),
                "brier":         eval_res.get("brier", 1),
                "opt_threshold": opt_t,
            })
            ece_val = eval_res.get("ece", float("nan"))
            ece_str = f"{ece_val:.3f}" if not math.isnan(ece_val) else "n/a"
            pr_auc  = eval_res.get("pr_auc", float("nan"))
            pr_str  = f"{pr_auc:.3f}" if not math.isnan(pr_auc) else "n/a"
            bal_acc = eval_res.get("bal_acc", float("nan"))
            bal_str = f"{bal_acc*100:.1f}%" if not math.isnan(bal_acc) else "n/a"
            wf_line = (
                f"Walk-forward ({eval_res['n_test']} held-out trades, unseen):\n"
                f"  Acc {eval_res['acc']*100:.0f}%  ·  Prec {eval_res['precision']*100:.0f}%  "
                f"·  Recall {eval_res['recall']*100:.0f}%  ·  F1 {eval_res['f1']*100:.0f}%\n"
                f"  ROC-AUC {eval_res['roc_auc']:.3f}  ·  PR-AUC {pr_str}  "
                f"·  Brier {eval_res['brier']:.3f}  ·  ECE {ece_str}\n"
                f"  MCC {eval_res['mcc']:.3f}  ·  Bal-Acc {bal_str}  "
                f"·  Baseline WR {eval_res['baseline_wr']*100:.0f}%\n"
                f"  Optimal threshold: {opt_t:.2f}  ·  "
                f"CM → TP={eval_res['tp']} TN={eval_res['tn']} FP={eval_res['fp']} FN={eval_res['fn']}"
            )

        try:
            # ── NaN / inf guard: clean the matrix before fitting to avoid ──────
            # pandas RuntimeWarnings that can bubble up from ATR/ADX divisions
            import warnings as _warnings
            X_arr = np.nan_to_num(np.array(X, dtype=float), nan=0.0, posinf=10.0, neginf=-10.0)
            X     = X_arr.tolist()

            # ── Global model: stacked ensemble once data supports it, else a
            #    single regularized GradientBoosting to avoid over-fitting noise ──
            if use_stack:
                with _warnings.catch_warnings():
                    _warnings.simplefilter("ignore")
                    clf = _make_stacking_clf()
                    clf.fit(X, y)
                arch_label = "Stacked (GB+RF+ExtraTrees → LogReg) + isotonic calibration"
                imp_line = "n/a (stacked ensemble)"
            else:
                with _warnings.catch_warnings():
                    _warnings.simplefilter("ignore")
                    clf = _make_gb_clf()
                    clf.fit(X, y, sample_weight=weights)
                arch_label = f"GradientBoosting (single model — stacking unlocks at {ML_STACKING_MIN_TRADES} trades)"
                imp = sorted(
                    zip(all_feature_cols, clf.feature_importances_),
                    key=lambda x: x[1], reverse=True,
                )
                imp_line = "  ".join([f"{n} {v*100:.0f}%" for n, v in imp[:3]])
                # ── Feature importance history ─────────────────────────────
                _feature_importance_history.append({
                    "ts":          datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
                    "total":       total,
                    "importances": {n: round(v, 4) for n, v in zip(all_feature_cols, clf.feature_importances_)},
                })
                if len(_feature_importance_history) > 50:
                    _feature_importance_history.pop(0)

                # ── Feature importance → Sober Book component weights (V2.9) ───
                # Derive per-component score multipliers from what the GB model
                # found actually predictive. Each Sober Book component is mapped
                # to its most diagnostic ML features; the average importance of
                # those features sets the multiplier (range 0.70–1.30) so scoring
                # adapts to what historically predicts wins without wild swings.
                _ml_importance_weights = {
                    n: round(float(v), 4)
                    for n, v in zip(all_feature_cols, clf.feature_importances_)
                }
                _COMP_FEAT_MAP = {
                    "sober_structure_pts": ["score", "body_ratio"],
                    "sober_trend_pts":     ["ema_fast_slope", "ema_slow_slope",
                                            "ema_distance", "adx"],
                    "sober_zone_pts":      ["wick_atr_ratio", "bb_position"],
                    "sober_timing_pts":    ["rsi", "stochrsi_k", "atr_expansion"],
                    "sober_momentum_pts":  ["roc"],
                    "sober_candle_pts":    ["body_ratio", "bb_width"],
                }
                _baseline_imp = 1.0 / max(len(all_feature_cols), 1)
                _ml_component_weights = {}
                for _comp, _feats in _COMP_FEAT_MAP.items():
                    _avg = float(np.mean([
                        _ml_importance_weights.get(f, 0.0) for f in _feats
                    ]))
                    _mult = _avg / max(_baseline_imp, 1e-9)
                    _ml_component_weights[_comp] = round(
                        max(0.70, min(1.30, _mult)), 3
                    )
                logger.info(f"ML component weights: {_ml_component_weights}")

            with ml_lock:
                ml_model = clf
                ml_trained_on = total

            # ── Per-class models (RandomForest — small datasets, cheap, robust) ──
            new_cls_models   = {}
            new_cls_trained  = {}
            cls_lines = []
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
                    n_estimators=100, max_depth=5, random_state=42,
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

            # ── Per-symbol models (V2.9) ──────────────────────────────────────
            # One dedicated RandomForest per symbol when ≥ML_PER_SYMBOL_MIN_TRADES
            # exists for that symbol. Checked before per-class and global models in
            # _ml_get_confidence / _ml_should_trade — most specific model wins.
            all_syms_in_data = sorted({r[0] for r in rows_sym})
            new_sym_models  = {}
            new_sym_trained = {}
            sym_lines = []
            for _sym in all_syms_in_data:
                sym_rows = [r for r in rows_sym if r[0] == _sym]
                n_s = len(sym_rows)
                if n_s < ML_PER_SYMBOL_MIN_TRADES:
                    sym_lines.append(f"  {_sym}: {n_s}/{ML_PER_SYMBOL_MIN_TRADES} ⏳")
                    continue
                Xs, ys = _ml_build_matrix(sym_rows, n_raw)
                if len(set(ys)) < 2:
                    sym_lines.append(f"  {_sym}: needs both outcomes — skipped")
                    continue
                w_rec_s = np.exp(np.linspace(-n_s / half_life, 0.0, n_s))
                w_bal_s = compute_sample_weight("balanced", ys)
                w_s     = w_rec_s * w_bal_s
                w_s    /= w_s.mean()
                clf_s   = RandomForestClassifier(
                    n_estimators=120, max_depth=6, min_samples_leaf=3,
                    random_state=42, class_weight="balanced", n_jobs=1,
                )
                clf_s.fit(Xs, ys, sample_weight=w_s)
                new_sym_models[_sym]  = clf_s
                new_sym_trained[_sym] = n_s
                wr_s = sum(ys) / n_s * 100
                sym_lines.append(f"  {_sym}: {n_s} trades  {wr_s:.0f}%WR ✅")

            with ml_lock:
                ml_models_per_symbol.update(new_sym_models)
                ml_trained_per_symbol.update(new_sym_trained)

            sym_summary = "\n".join(sym_lines) if sym_lines else "  (need 50+ trades per symbol)"

            _ml_save()
            # V3.0: before/after comparison
            sess_gate_now = _get_session_profile(_get_session_name()).get("ml_gate", ML_CONFIDENCE_MIN)
            if _prev_snap and eval_res:
                _d_acc   = (eval_res.get("acc", 0)   - _prev_snap.get("acc", 0)) * 100
                _d_prauc = eval_res.get("pr_auc", 0) - _prev_snap.get("pr_auc", 0)
                _d_brier = eval_res.get("brier", 1)  - _prev_snap.get("brier", 1)
                cmp_line = (
                    f"Δ vs prev:  Acc {_d_acc:+.0f}%  ·  "
                    f"PR-AUC {_d_prauc:+.3f}  ·  Brier {_d_brier:+.3f}"
                )
            else:
                cmp_line = "Δ vs prev:  first model (no baseline)"

            _send_tg(
                f"🤖 <b>ML RETRAINING COMPLETE</b> ✅\n"
                f"<code>[{_ml_progress_bar(1.0)}] 100%</code>\n"
                f"Global: <b>{total}</b> trades  |  "
                f"Gate ≥<b>{ML_CONFIDENCE_MIN*100:.0f}%</b> global  "
                f"| <b>{sess_gate_now*100:.0f}%</b> {_get_session_name()}\n"
                f"Architecture: {arch_label}\n"
                f"Top features: {imp_line}\n"
                f"Per-class:\n" + "\n".join(cls_lines) +
                f"\nPer-symbol:\n{sym_summary}\n\n"
                f"{wf_line}\n{cmp_line}"
            )
            _ml_export_csv(total)

            # V3.0: suggest-only recommendations — never auto-applied
            try:
                rec_msg = _ml_generate_recommendations(eval_res)
                _send_tg(rec_msg)
            except Exception as _re:
                logger.warning(f"ML recommendation engine error: {_re}")

            # V3.0: Gate Optimizer — replay all signals and report best thresholds
            try:
                _send_tg(_gate_optimizer_text())
            except Exception as _ge:
                logger.warning(f"Gate optimizer error: {_ge}")
        except Exception as e:
            logger.error(f"ML training failed: {e}")
            _send_tg(f"🤖 <b>ML RETRAINING FAILED</b>\n<code>{e}</code>")

    finally:
        with ml_lock:
            ml_training_active = False
        # _ml_last_retrain_time was already stamped at the start of the try block
        # to block concurrent triggers.  Refresh here to measure from completion so
        # the cooldown period runs from when the model was actually ready, not queued.


def _ml_maybe_retrain(total_trades: int):
    global ml_total_trades, ml_training_active
    # Use max() so an instant +1 increment from on_contract_update is never
    # overwritten by a stale DB count flushed milliseconds later.
    with ml_lock:
        ml_total_trades = max(ml_total_trades, total_trades)
    spawn = False
    with ml_lock:
        if not ml_training_active:
            # Cooldown guard: don't retrain again within ML_RETRAIN_COOLDOWN_SECS
            # of a completed retrain to avoid thrashing on back-to-back trade bursts.
            cooldown_ok = (time.time() - _ml_last_retrain_time) >= ML_RETRAIN_COOLDOWN_SECS
            if cooldown_ok:
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
        # Trend (Sober Book)
        "ema_fast": None,   # EMA 50
        "roc":      None,   # Rate of Change (10-period)
        # Sober Trading Book additions
        "market_structure":    "sideways",
        "market_struct_bos":   False,
        "market_struct_choch": False,
        "market_struct_hh":    False,
        "market_struct_hl":    False,
        "market_struct_lh":    False,
        "market_struct_ll":    False,
        "sd_zone":            "none",
        "sd_zone_dist":       99.0,
        "sd_zone_fresh":      False,
        "sd_zone_tests":      0,
        "candle_pattern":      "none",
        "candle_pattern_bias": "neutral",
        "ready": False,
    }


# ══════════════════════════════════════════════════════════════════════
#  DATABASE
# ══════════════════════════════════════════════════════════════════════
db_queue: queue.Queue = queue.Queue()

_CREATE_TABLE_SQLITE = """
    CREATE TABLE IF NOT EXISTS touch_trades (
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
        market_session  TEXT,
        is_backtest     INTEGER DEFAULT 0
    )
"""

_CREATE_TABLE_PG = """
    CREATE TABLE IF NOT EXISTS touch_trades (
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
        market_session  TEXT,
        is_backtest     BOOLEAN DEFAULT FALSE
    )
"""

_INSERT_COLS = (
    "timestamp,symbol,direction,barrier,stake,payout,profit,win,score,"
    "wick_atr_ratio,atr,atr_ma,ema_fast_slope,ema_slow_slope,ema_distance,market_session,"
    "bb_width,regime,"          # V2.6: Bollinger bandwidth + market regime
    "rsi,stochrsi_k,roc,body_ratio,bb_position,adx,"  # V2.7: oscillators + momentum
    # V2.9: context features — health, Sober Book breakdown, MTF agreement
    "session_health,asset_health,"
    "sober_structure_pts,sober_trend_pts,sober_zone_pts,"
    "sober_timing_pts,sober_momentum_pts,sober_candle_pts,"
    "mtf_agreement"
)

# NULL-safe filter for excluding future backtest-tagged rows from live stats.
# touch_trades' own `barrier` column always holds a real numeric barrier price
# (never a tag), so backtest rows must be marked via the dedicated is_backtest
# column instead of overloading `barrier` like sniper_bot.py's `trades` table does.
_LIVE_ONLY_FILTER = "(is_backtest IS NULL OR is_backtest = 0)"

# ── signal_features: rich per-trade feature log for Pattern Discovery /
#    Performance Analytics (mirrors sniper_bot.py's schema+purpose) ─────────
_CREATE_SIGNAL_FEATURES_SQLITE = """
    CREATE TABLE IF NOT EXISTS signal_features (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp       TEXT,
        symbol          TEXT,
        ms_type         TEXT,
        ms_bos          INTEGER,
        sd_zone         TEXT,
        sd_dist         REAL,
        sd_fresh        INTEGER,
        adx_val         REAL,
        candle_pattern  TEXT,
        session         TEXT,
        score           REAL,
        ml_confidence   REAL,
        win             INTEGER,
        profit          REAL
    )
"""

_CREATE_SIGNAL_FEATURES_PG = """
    CREATE TABLE IF NOT EXISTS signal_features (
        id              SERIAL PRIMARY KEY,
        timestamp       TEXT,
        symbol          TEXT,
        ms_type         TEXT,
        ms_bos          BOOLEAN,
        sd_zone         TEXT,
        sd_dist         REAL,
        sd_fresh        BOOLEAN,
        adx_val         REAL,
        candle_pattern  TEXT,
        session         TEXT,
        score           REAL,
        ml_confidence   REAL,
        win             INTEGER,
        profit          REAL
    )
"""

_SIGNAL_FEATURES_INSERT_COLS = (
    "timestamp,symbol,ms_type,ms_bos,sd_zone,sd_dist,sd_fresh,adx_val,"
    "candle_pattern,session,score,ml_confidence,win,profit"
)


def _log_signal_features(symbol: str, details: dict, session: str, win: bool, profit: float):
    """Persist one row of rich per-trade features for Pattern Discovery / Analytics."""
    try:
        row = (
            datetime.now(timezone.utc).isoformat(), symbol,
            details.get("market_structure", "sideways"),
            int(bool(details.get("market_struct_bos", False))),   # INTEGER in Neon
            details.get("sd_zone", "none"),
            float(details.get("sd_zone_dist", 99.0) or 99.0),
            int(bool(details.get("sd_zone_fresh", False))),        # INTEGER in Neon
            float(details.get("adx", 0) or 0),
            details.get("candle_pattern", "none"),
            session,
            float(details.get("total_score", 0) or 0),
            details.get("ml_confidence"),
            int(win),
            float(profit),
        )
        if USE_PG:
            conn = psycopg2.connect(DATABASE_URL)
            cur  = conn.cursor()
            cur.execute(_CREATE_SIGNAL_FEATURES_PG)
            cur.execute(
                f"INSERT INTO signal_features ({_SIGNAL_FEATURES_INSERT_COLS}) "
                f"VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", row,
            )
            conn.commit(); cur.close(); conn.close()
        else:
            conn = sqlite3.connect("touch_trades.db")
            conn.execute(_CREATE_SIGNAL_FEATURES_SQLITE)
            conn.execute(
                f"INSERT INTO signal_features ({_SIGNAL_FEATURES_INSERT_COLS}) "
                f"VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", row,
            )
            conn.commit(); conn.close()
    except Exception as e:
        logger.error(f"_log_signal_features error: {e}")


def _pg_safe(v):
    """Convert numpy / pandas scalars to plain Python types for psycopg2.
    psycopg2 serialises numpy.float64 as 'np.float64(x)' which PostgreSQL
    reads as schema-qualified, raising 'schema np does not exist'."""
    t = type(v)
    m = getattr(t, "__module__", "") or ""
    if m.startswith("numpy") or m.startswith("pandas"):
        if hasattr(v, "item"):          # ndarray scalars have .item()
            return v.item()
        try:
            return float(v)
        except (TypeError, ValueError):
            return v
    return v


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
            conn = sqlite3.connect("touch_trades.db")
            rows = conn.execute(sql, params).fetchall()
            conn.close()
        return rows
    except Exception as e:
        logger.error(f"DB fetch error: {e}")
        return []


_CREATE_ML_STATS_SQLITE = """
    CREATE TABLE IF NOT EXISTS ml_live_stats (
        bucket_type TEXT NOT NULL,
        bucket_key  TEXT NOT NULL,
        wins        INTEGER DEFAULT 0,
        losses      INTEGER DEFAULT 0,
        pnl         REAL DEFAULT 0.0,
        PRIMARY KEY (bucket_type, bucket_key)
    )
"""

_CREATE_ML_STATS_PG = """
    CREATE TABLE IF NOT EXISTS ml_live_stats (
        bucket_type TEXT NOT NULL,
        bucket_key  TEXT NOT NULL,
        wins        INTEGER DEFAULT 0,
        losses      INTEGER DEFAULT 0,
        pnl         REAL DEFAULT 0.0,
        PRIMARY KEY (bucket_type, bucket_key)
    )
"""


def _ml_stats_init_table():
    """Create the ml_live_stats table (idempotent) so ML/score dashboards survive restarts."""
    try:
        if USE_PG:
            conn = psycopg2.connect(DATABASE_URL)
            conn.autocommit = True
            cur = conn.cursor()
            cur.execute(_CREATE_ML_STATS_PG)
            cur.close(); conn.close()
        else:
            conn = sqlite3.connect("touch_trades.db")
            conn.execute(_CREATE_ML_STATS_SQLITE)
            conn.commit(); conn.close()
    except Exception as e:
        logger.error(f"ml_live_stats table init error: {e}")


def _ml_stats_save(bucket_type: str, bucket_key: str, stats: dict):
    """Upsert one bucket's live wins/losses/pnl into the DB (Neon-backed when USE_PG)."""
    try:
        if USE_PG:
            conn = psycopg2.connect(DATABASE_URL)
            conn.autocommit = True
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO ml_live_stats (bucket_type,bucket_key,wins,losses,pnl) "
                "VALUES (%s,%s,%s,%s,%s) "
                "ON CONFLICT (bucket_type,bucket_key) DO UPDATE SET "
                "wins=EXCLUDED.wins, losses=EXCLUDED.losses, pnl=EXCLUDED.pnl",
                (bucket_type, bucket_key, stats["wins"], stats["losses"], stats["pnl"]),
            )
            cur.close(); conn.close()
        else:
            conn = sqlite3.connect("touch_trades.db")
            conn.execute(_CREATE_ML_STATS_SQLITE)
            conn.execute(
                "INSERT INTO ml_live_stats (bucket_type,bucket_key,wins,losses,pnl) "
                "VALUES (?,?,?,?,?) "
                "ON CONFLICT(bucket_type,bucket_key) DO UPDATE SET "
                "wins=excluded.wins, losses=excluded.losses, pnl=excluded.pnl",
                (bucket_type, bucket_key, stats["wins"], stats["losses"], stats["pnl"]),
            )
            conn.commit(); conn.close()
    except Exception as e:
        logger.error(f"ml_live_stats save error: {e}")


def _ml_stats_load_all():
    """Restore _ml_conf_live_stats / _ml_score_live_stats from the DB after a restart."""
    global _ml_conf_live_stats, _ml_score_live_stats
    try:
        _ml_stats_init_table()
        rows = _db_fetch("SELECT bucket_type, bucket_key, wins, losses, pnl FROM ml_live_stats")
        for bucket_type, bucket_key, wins, losses, pnl in rows:
            target = _ml_conf_live_stats if bucket_type == "conf" else _ml_score_live_stats
            target[bucket_key] = {
                "wins": int(wins or 0), "losses": int(losses or 0), "pnl": float(pnl or 0.0),
            }
        logger.info(
            f"ML live stats restored from DB: {len(_ml_conf_live_stats)} conf buckets, "
            f"{len(_ml_score_live_stats)} score buckets"
        )
    except Exception as e:
        logger.error(f"ml_live_stats load error: {e}")


def _db_writer():
    if USE_PG:
        conn = psycopg2.connect(DATABASE_URL)
        conn.autocommit = True
        cur  = conn.cursor()
        cur.execute(_CREATE_TABLE_PG)
        # Add columns missing from older schemas (IF NOT EXISTS is idempotent)
        cur.execute("ALTER TABLE touch_trades ADD COLUMN IF NOT EXISTS market_session TEXT")
        cur.execute("ALTER TABLE touch_trades ADD COLUMN IF NOT EXISTS bb_width REAL")
        cur.execute("ALTER TABLE touch_trades ADD COLUMN IF NOT EXISTS regime TEXT")
        # V2.7: oscillator + momentum columns
        cur.execute("ALTER TABLE touch_trades ADD COLUMN IF NOT EXISTS rsi REAL")
        cur.execute("ALTER TABLE touch_trades ADD COLUMN IF NOT EXISTS stochrsi_k REAL")
        cur.execute("ALTER TABLE touch_trades ADD COLUMN IF NOT EXISTS roc REAL")
        cur.execute("ALTER TABLE touch_trades ADD COLUMN IF NOT EXISTS body_ratio REAL")
        cur.execute("ALTER TABLE touch_trades ADD COLUMN IF NOT EXISTS bb_position REAL")
        cur.execute("ALTER TABLE touch_trades ADD COLUMN IF NOT EXISTS adx REAL")
        # V2.9: context features for ML training
        cur.execute("ALTER TABLE touch_trades ADD COLUMN IF NOT EXISTS session_health REAL")
        cur.execute("ALTER TABLE touch_trades ADD COLUMN IF NOT EXISTS asset_health REAL")
        cur.execute("ALTER TABLE touch_trades ADD COLUMN IF NOT EXISTS sober_structure_pts REAL")
        cur.execute("ALTER TABLE touch_trades ADD COLUMN IF NOT EXISTS sober_trend_pts REAL")
        cur.execute("ALTER TABLE touch_trades ADD COLUMN IF NOT EXISTS sober_zone_pts REAL")
        cur.execute("ALTER TABLE touch_trades ADD COLUMN IF NOT EXISTS sober_timing_pts REAL")
        cur.execute("ALTER TABLE touch_trades ADD COLUMN IF NOT EXISTS sober_momentum_pts REAL")
        cur.execute("ALTER TABLE touch_trades ADD COLUMN IF NOT EXISTS sober_candle_pts REAL")
        cur.execute("ALTER TABLE touch_trades ADD COLUMN IF NOT EXISTS mtf_agreement INTEGER")
        def _write(item):
            cur.execute(
                f"INSERT INTO touch_trades ({_INSERT_COLS}) "
                f"VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s"
                f",%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                item,
            )
            cur.execute("SELECT COUNT(*) FROM touch_trades")
            return cur.fetchone()[0]
    else:
        conn = sqlite3.connect("touch_trades.db")
        conn.execute(_CREATE_TABLE_SQLITE)
        for _col_sql in [
            "ALTER TABLE touch_trades ADD COLUMN market_session TEXT",
            "ALTER TABLE touch_trades ADD COLUMN bb_width REAL",
            "ALTER TABLE touch_trades ADD COLUMN regime TEXT",
            "ALTER TABLE touch_trades ADD COLUMN rsi REAL",
            "ALTER TABLE touch_trades ADD COLUMN stochrsi_k REAL",
            "ALTER TABLE touch_trades ADD COLUMN roc REAL",
            "ALTER TABLE touch_trades ADD COLUMN body_ratio REAL",
            "ALTER TABLE touch_trades ADD COLUMN bb_position REAL",
            "ALTER TABLE touch_trades ADD COLUMN adx REAL",
            # V2.9: context features for ML training
            "ALTER TABLE touch_trades ADD COLUMN session_health REAL",
            "ALTER TABLE touch_trades ADD COLUMN asset_health REAL",
            "ALTER TABLE touch_trades ADD COLUMN sober_structure_pts REAL",
            "ALTER TABLE touch_trades ADD COLUMN sober_trend_pts REAL",
            "ALTER TABLE touch_trades ADD COLUMN sober_zone_pts REAL",
            "ALTER TABLE touch_trades ADD COLUMN sober_timing_pts REAL",
            "ALTER TABLE touch_trades ADD COLUMN sober_momentum_pts REAL",
            "ALTER TABLE touch_trades ADD COLUMN sober_candle_pts REAL",
            "ALTER TABLE touch_trades ADD COLUMN mtf_agreement INTEGER",
        ]:
            try:
                conn.execute(_col_sql)
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
                item = tuple(_pg_safe(v) for v in item)
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
                        f"INSERT INTO touch_trades ({_INSERT_COLS}) "
                        f"VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?"
                        f",?,?,?,?,?,?,?,?,?)",
                        it,
                    )
            _batch.clear()
            _last_flush[0] = time.time()
            return conn.execute("SELECT COUNT(*) FROM touch_trades").fetchone()[0]
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
        "FROM touch_trades ORDER BY id DESC LIMIT ?", (limit,)
    )


def get_db_summary():
    # win=-1 rows are VOID/unresolved contracts (never confirmed win or loss) —
    # excluded from the count so win-rate isn't diluted by unresolved trades.
    rows = _db_fetch(
        "SELECT SUM(CASE WHEN win IN (0,1) THEN 1 ELSE 0 END), SUM(profit), "
        "SUM(CASE WHEN win=1 THEN 1 ELSE 0 END), "
        "SUM(CASE WHEN win=0 THEN 1 ELSE 0 END) FROM touch_trades"
    )
    return rows[0] if rows else (0, 0, 0, 0)


def get_alltime_symbol_stats(limit=10):
    return _db_fetch(
        "SELECT symbol, SUM(CASE WHEN win IN (0,1) THEN 1 ELSE 0 END), "
        "SUM(CASE WHEN win=1 THEN 1 ELSE 0 END), SUM(profit) "
        "FROM touch_trades WHERE win IN (0,1) GROUP BY symbol ORDER BY SUM(profit) DESC LIMIT ?", (limit,)
    )


def get_alltime_daily_stats(limit=7):
    # win=-1 rows are VOID/unresolved contracts — excluded so day totals and
    # win-rate aren't diluted by trades that never got a confirmed result.
    return _db_fetch(
        "SELECT date(timestamp) as day, SUM(CASE WHEN win IN (0,1) THEN 1 ELSE 0 END), "
        "SUM(CASE WHEN win=1 THEN 1 ELSE 0 END), SUM(profit) "
        "FROM touch_trades WHERE win IN (0,1) GROUP BY 1 ORDER BY 1 DESC LIMIT ?", (limit,)
    )


def get_7day_full_breakdown():
    """Return per-day, per-session breakdown for the last 7 days."""
    if USE_PG:
        sql = (
            "SELECT DATE(timestamp), market_session, COUNT(*), "
            "SUM(CASE WHEN win=1 THEN 1 ELSE 0 END), SUM(profit) "
            "FROM touch_trades "
            "WHERE DATE(timestamp) >= CURRENT_DATE - INTERVAL '7 days' AND win IN (0,1) "
            "GROUP BY DATE(timestamp), market_session "
            "ORDER BY DATE(timestamp) DESC, SUM(profit) DESC"
        )
    else:
        sql = (
            "SELECT date(timestamp), market_session, COUNT(*), "
            "SUM(CASE WHEN win=1 THEN 1 ELSE 0 END), SUM(profit) "
            "FROM touch_trades "
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
        "FROM touch_trades WHERE market_session IS NOT NULL AND win IN (0,1) "
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
        "FROM touch_trades WHERE win IN (0,1) "
        "GROUP BY 1 ORDER BY MIN(score) DESC"
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
        "SELECT profit, win, symbol, market_session FROM touch_trades "
        "WHERE win IN (0,1) ORDER BY id ASC"
    )
    return _compute_advanced_stats(rows)


def get_today_advanced_stats() -> Optional[dict]:
    """Today's (UTC) trading metrics derived from confirmed trades in the DB."""
    if USE_PG:
        sql = ("SELECT profit, win, symbol, market_session FROM touch_trades "
               "WHERE DATE(timestamp) = CURRENT_DATE AND win IN (0,1) ORDER BY id ASC")
    else:
        sql = ("SELECT profit, win, symbol, market_session FROM touch_trades "
               "WHERE date(timestamp) = date('now') AND win IN (0,1) ORDER BY id ASC")
    rows = _db_fetch(sql)
    return _compute_advanced_stats(rows)


def _strategy_header() -> str:
    """Config fingerprint shown at the top of every analytical report so
    results can be compared apples-to-apples across strategy revisions."""
    return (
        f"🧬 <b>Strategy {STRATEGY_VERSION}</b>  ·  Expiry {DURATION}m  ·  "
        f"Score ≥{SCORE_THRESHOLD}  ·  ML ≥{ML_CONFIDENCE_MIN*100:.0f}%  ·  "
        f"Barrier {ATR_BARRIER_MULT}×ATR\n"
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

    # EMA Slow (long-term trend / fallback direction)
    df["EMA_SLOW"] = df["Close"].ewm(span=EMA_SLOW, adjust=False).mean()
    # EMA Fast (medium trend, Sober Book trend-strength component)
    df["EMA_FAST"] = df["Close"].ewm(span=EMA_FAST, adjust=False).mean()
    # Rate of Change (momentum display / scoring)
    df["ROC"] = df["Close"].pct_change(ROC_PERIOD) * 100

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
        ind["ema_fast"]         = df["EMA_FAST"].iloc[-1]
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
        # V2.6: Bollinger bandwidth (normalised band width — ML feature + regime input)
        bb_width_raw = df["BB_WIDTH"].iloc[-1]
        ind["bb_width"] = round(float(bb_width_raw), 6) if pd.notna(bb_width_raw) else 0.0

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

        # V2.6: Market regime — summarises ADX / ATR expansion / BB width into
        # one label.  Stored in ind so score_signal and DB writes can use it.
        ind["regime"] = _compute_market_regime(ind)

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

        if downtrend and b3 > 0 and lw3 >= 2.0 * b3 and uw3 < 0.3 * r3 and b3 < 0.4 * r3:
            return {"name": "Hammer 🔨", "bias": "bullish"}
        if downtrend and b3 > 0 and uw3 >= 2.0 * b3 and lw3 < 0.3 * r3 and b3 < 0.4 * r3 and is_bull(o3, c3):
            return {"name": "Inverted Hammer", "bias": "bullish"}
        if uptrend and b3 > 0 and uw3 >= 2.0 * b3 and lw3 < 0.3 * r3 and b3 < 0.4 * r3:
            return {"name": "Shooting Star ⭐", "bias": "bearish"}
        if uptrend and b3 > 0 and lw3 >= 2.0 * b3 and uw3 < 0.3 * r3 and b3 < 0.4 * r3 and is_bear(o3, c3):
            return {"name": "Hanging Man", "bias": "bearish"}
        if is_bear(o2, c2) and is_bull(o3, c3) and o3 <= c2 and c3 >= o2 and b3 > b2:
            return {"name": "Bullish Engulfing 📈", "bias": "bullish"}
        if is_bull(o2, c2) and is_bear(o3, c3) and o3 >= c2 and c3 <= o2 and b3 > b2:
            return {"name": "Bearish Engulfing 📉", "bias": "bearish"}
        if is_bear(o2, c2) and is_bull(o3, c3) and o3 < c2 and c3 > (o2 + c2) / 2 and c3 < o2:
            return {"name": "Piercing Line", "bias": "bullish"}
        if is_bear(o1, c1) and b2 < 0.4 * b1 and is_bull(o3, c3) and c3 > (o1 + c1) / 2:
            return {"name": "Morning Star 🌟", "bias": "bullish"}
        if is_bull(o1, c1) and b2 < 0.4 * b1 and is_bear(o3, c3) and c3 < (o1 + c1) / 2:
            return {"name": "Evening Star 🌆", "bias": "bearish"}
        if (is_bull(o1,c1) and is_bull(o2,c2) and is_bull(o3,c3)
                and c2 > c1 and c3 > c2 and o2 > o1 and o3 > o2):
            return {"name": "Three White Soldiers 🪖", "bias": "bullish"}
        if (is_bear(o1,c1) and is_bear(o2,c2) and is_bear(o3,c3)
                and c2 < c1 and c3 < c2 and o2 < o1 and o3 < o2):
            return {"name": "Three Black Crows 🐦", "bias": "bearish"}
        if b3 < 0.05 * r3:
            return {"name": "Doji ✚", "bias": "neutral"}
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
    Calibrated 100-point scoring system (Sober Trading Book edition).

    Component breakdown:
      1. Market Structure   — 25 pts  (HH+HL/LH+LL, BOS, CHoCH)
      2. Trend Strength     — 20 pts  (EMA50/200, Supertrend, ADX)
      3. Supply/Demand Zone — 20 pts  (freshness, proximity, wick)
      4. Entry Timing       — 15 pts  (pullback depth, wick, RSI/StochRSI)
      5. Momentum           — 10 pts  (MACD, ROC, ATR expansion)
      6. Candle Confirm     — 10 pts  (engulfing, pin-bar, body quality)

    A hard "Sober Gate" penalises/blocks signals where market structure
    contradicts the trade direction (see below).
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
    if   direction == "UP"   and ms_hh and ms_hl: ms_pts += 20
    elif direction == "DOWN" and ms_lh and ms_ll: ms_pts += 20
    elif direction == "UP"   and (ms_hh or ms_hl): ms_pts += 10   # partial
    elif direction == "DOWN" and (ms_lh or ms_ll): ms_pts += 10
    if ms_bos and (
        (direction == "UP"   and ms_type == "bullish") or
        (direction == "DOWN" and ms_type == "bearish")
    ):
        ms_pts += 5
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
    adx_val = float(ind.get("adx") or 0.0)
    di_bull = bool(ind.get("di_bullish", False))

    tr_pts = 0
    ema_aligned = (
        (direction == "UP"   and ema_fast > ema_slow) or
        (direction == "DOWN" and ema_fast < ema_slow)
    )
    if ema_aligned: tr_pts += 8

    st_aligned = (
        (direction == "UP"   and st_dir == 1) or
        (direction == "DOWN" and st_dir == -1)
    )
    if st_aligned: tr_pts += 4

    if   adx_val >= 45: tr_pts += 6
    elif adx_val >= 35: tr_pts += 8
    elif adx_val >= 25: tr_pts += 5
    elif adx_val >= 20: tr_pts += 3
    elif adx_val >= 15: tr_pts += 0
    else:               tr_pts -= 5

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
    sd_zone  = ind.get("sd_zone", "none") or "none"
    sd_dist  = float(ind.get("sd_zone_dist") or 99.0)
    sd_fresh = bool(ind.get("sd_zone_fresh", False))
    sd_tests = int(ind.get("sd_zone_tests") or 0)
    uw       = float(ind.get("upper_wick_atr") or 0.0)
    lw       = float(ind.get("lower_wick_atr") or 0.0)

    sd_zone_agree = (
        (direction == "UP"   and sd_zone == "demand") or
        (direction == "DOWN" and sd_zone == "supply")
    )
    sd_pts = 0
    if sd_zone != "none" and sd_zone_agree:
        if sd_fresh:
            sd_pts += 12
        elif sd_tests <= 1:
            sd_pts += 4
        elif sd_tests == 2:
            sd_pts += 1
        else:
            sd_pts -= 2
        if   sd_dist <= 0.5:  sd_pts += 5
        elif sd_dist <= 1.5:  sd_pts += 3
        elif sd_dist <= 3.0:  sd_pts += 1
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
    if   extension <= 1.0: et_pts += 5
    elif extension <= 2.0: et_pts += 2

    wick_pts = 0
    if direction == "UP":
        if   lw >= 0.5:   wick_pts = 5
        elif lw >= 0.25:  wick_pts = 2
    else:
        if   uw >= 0.5:   wick_pts = 5
        elif uw >= 0.25:  wick_pts = 2
    et_pts += wick_pts

    osc_pts = 0
    if srsi_k is not None:
        if direction == "UP":
            if   srsi_k < 25: osc_pts = 5
            elif srsi_k < 45: osc_pts = 3
            elif srsi_k < 65: osc_pts = 1
        else:
            if   srsi_k > 75: osc_pts = 5
            elif srsi_k > 55: osc_pts = 3
            elif srsi_k > 35: osc_pts = 1
    elif rsi is not None:
        if direction == "UP":
            if 35 <= rsi <= 60:                     osc_pts = 5
            elif 25 <= rsi < 35 or 60 < rsi <= 72:   osc_pts = 3
        else:
            if 40 <= rsi <= 65:                     osc_pts = 5
            elif 28 <= rsi < 40 or 65 < rsi <= 75:   osc_pts = 3
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
    macd_bullish     = bool(ind.get("macd_bullish", False))
    macd_hist_rising = bool(ind.get("macd_hist_rising", False))
    roc              = ind.get("roc")

    mom_pts = 0
    macd_ok = (
        (direction == "UP"   and macd_bullish     and macd_hist_rising) or
        (direction == "DOWN" and not macd_bullish and macd_hist_rising)
    )
    if macd_ok: mom_pts += 5

    roc_ok = (
        (direction == "UP"   and roc is not None and roc > 0) or
        (direction == "DOWN" and roc is not None and roc < 0)
    )
    if roc_ok: mom_pts += 3

    if ind.get("atr_rising") and (ind.get("atr") or 0) > (ind.get("atr_ma") or 0):
        mom_pts += 2

    score += mom_pts
    details.update({
        "momentum":         mom_pts,
        "volatility":       mom_pts,   # alias for signal-card compat
        "macd_bullish":     macd_bullish,
        "macd_hist_rising": macd_hist_rising,
        "roc":              roc,
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
        else:                                                     cc_pts += 3

    if pat_against and ("spinning" in pat_n or "doji" in pat_n):
        cc_pts -= 5

    if body_r < 0.20:
        cc_pts = min(cc_pts, 2)

    score += cc_pts
    details.update({
        "candle_pattern":      pat_name,
        "candle_pattern_bias": pat_bias,
        "candle_pattern_pts":  cc_pts,
    })

    # ── Store per-component Sober Book scores for ML training (V2.9) ──────
    # Raw (unweighted) values are persisted so the model can learn each
    # component's individual edge independent of the total score.
    details.update({
        "sober_structure_pts": ms_pts,
        "sober_trend_pts":     tr_pts,    # may be negative (ADX < 15 → -5)
        "sober_zone_pts":      sd_pts,
        "sober_timing_pts":    et_pts,
        "sober_momentum_pts":  mom_pts,
        "sober_candle_pts":    cc_pts,    # may be negative (pattern mismatch → -5)
    })

    # ── ML-learned component weight feedback (V2.9) ────────────────────────
    # After each retrain _ml_component_weights holds multipliers derived from
    # feature importances — upweights reliable predictors, downweights noisy
    # ones (range 0.70–1.30). No-op until the first GB model is trained.
    if _ml_component_weights:
        def _wsc(raw_pts: float, key: str) -> float:
            return raw_pts * _ml_component_weights.get(key, 1.0)
        score = int(round(
            _wsc(ms_pts,  "sober_structure_pts") +
            _wsc(tr_pts,  "sober_trend_pts")     +
            _wsc(sd_pts,  "sober_zone_pts")      +
            _wsc(et_pts,  "sober_timing_pts")    +
            _wsc(mom_pts, "sober_momentum_pts")  +
            _wsc(cc_pts,  "sober_candle_pts")
        ))
        details["score_ml_weighted"] = True

    # ══════════════════════════════════════════════════════════════════
    # SOBER GATE: hard structure filter
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

    score = max(0, min(score, 100))

    # ── Confluence summary (used by ML feature vector + UI compat) ────
    confluence_count = int(ema_aligned) + int(macd_ok) + int(roc_ok)
    confluence       = tr_pts + mom_pts
    bb_pos = ind.get("bb_position")

    # DB-compat column remapping
    st_dir_f  = float(st_dir)
    st_dist_f = abs(price - (st_val or price)) / atr
    wick_atr  = (float(candle.get("High", price)) - float(candle.get("Low", price))) / atr

    details.update({
        "total_score":            score,
        "atr":                    float(ind.get("atr")) if ind.get("atr") is not None else None,
        "atr_ma":                 float(ind.get("atr_ma")) if ind.get("atr_ma") is not None else None,
        "adx":                    adx_val,
        "macd_hist":              ind.get("macd_hist"),
        "bb_position":            round(bb_pos, 2) if bb_pos is not None else None,
        "bb_width":               float(ind.get("bb_width") or 0),        # V2.6: ML feature
        "regime":                 ind.get("regime", "Choppy"),             # V2.6: regime label
        "confluence":             confluence,
        "confluence_count":       confluence_count,
        "confluence_gate_passed": confluence_count >= 2,
        "ema200_agree":           ema_aligned,
        "roc":                    roc,
        "price":                  price,
        # DB column remapping (legacy ML feature names)
        "ema_fast_sl":            st_dir_f,   # Supertrend direction
        "ema_slow_sl":            adx_val,    # ADX value
        "ema_distance":           st_dist_f,  # ST distance / ATR
        "wick_atr_ratio":         float(round(wick_atr, 3)),
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
            # V3.0 spam fix: suppress LOCKED notification while the symbol is in
            # cooldown — the trade will be rejected immediately anyway, so the
            # alert adds noise without useful information.
            if not already_locked and cooldown_left == 0:
                _send_tg(
                    f"🔒 <b>LOCKED</b> — {symbol}\n"
                    f"Score <b>{score}/100</b> {direction}  |  "
                    f"Entry {details.get('entry_quality',0)}/30 "
                    f"(ext {details.get('extension_atr',0):.2f}×ATR)  |  armed"
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
def _compute_barrier(symbol: str, direction: str = "UP", barrier_mult: float = None) -> str:
    mult = ATR_BARRIER_MULT if barrier_mult is None else barrier_mult
    with _lock:
        atr = (indicators.get(symbol) or {}).get("atr") or 0.0
    if atr <= 0:
        return "+0.20" if direction == "UP" else "-0.20"
    offset = atr * mult
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
    """Send a concise trade-rejected card to Telegram and write to log.
    V3.0: optional `details` dict adds the decision-gate chain to the message."""
    _log(f"❌ {symbol} {direction} rejected — {reason}")
    safe_reason = _html.escape(reason)
    # V3.0 decision tree debug — show PASS/FAIL chain when details carry gate_trace
    chain_str = ""
    if details:
        gt = details.get("gate_trace", [])
        if gt:
            rows = []
            for g_name, g_status, g_note in gt:
                icon = "✅" if g_status == "PASS" else "❌"
                rows.append(f"  {icon} <b>{_html.escape(g_name)}</b>  <i>{_html.escape(g_note)}</i>")
            chain_str = "\n" + "\n".join(rows)
    _send_tg(
        f"🚫 <b>REJECTED</b> — <code>{symbol}</code> {direction}\n"
        f"Score: <b>{score}/100</b>\n"
        f"Reason: <i>{safe_reason}</i>{chain_str}"
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


def _result_card(sym: str, profit: float, win: bool, details: dict,
                 _wc: int = None, _lc: int = None, _pnl: float = None, _cl: int = None) -> str:
    """Build a trade-result Telegram card.
    Pass _wc/_lc/_pnl/_cl snapshots captured *before* any session reset so the
    card always shows the stats that include this trade.
    """
    if _wc is None:
        with _lock:
            _wc, _lc, _pnl, _cl = win_count, loss_count, total_pnl, consecutive_losses
    total  = _wc + _lc
    wr     = _wc / total * 100 if total else 0
    pnl_str = f"+${profit:.2f}" if profit > 0 else f"${profit:.2f}"
    conf    = details.get("ml_confidence")
    conf_line = f"🤖 ML Conf : <b>{_conf_str(conf)}</b>\n" if conf is not None else ""
    mkt_sess = _get_session_name()
    score    = details.get("total_score", 0)
    direction = details.get("direction", "")
    dir_icon  = "📈" if direction == "UP" else "📉"

    if win:
        header = f"🎊 <b>WIN!</b>  <code>{sym}</code>  {dir_icon} {direction}"
    else:
        header = f"💀 <b>LOSS</b>  <code>{sym}</code>  {dir_icon} {direction}"

    streak_str = ("🔴 " * min(_cl, 5)).strip() if _cl else "🟢 none"

    return (
        f"{header}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"💵 <b>Trade P&amp;L  :</b> <b>{pnl_str}</b>\n"
        f"🎯 <b>Score      :</b> {score}/100\n"
        f"{conf_line}"
        f"📊 <b>Session    :</b> {'+' if _pnl >= 0 else ''}${_pnl:.2f}  "
        f"({_wc}W / {_lc}L  {wr:.0f}%WR)\n"
        f"🔥 <b>Streak     :</b> {streak_str}\n"
        f"🕐 <b>Session    :</b> {SESSION_EMOJIS.get(mkt_sess,'')} {mkt_sess}\n"
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
def request_proposal(ws, symbol: str, details: dict, direction: str, barrier_mult: float = None):
    details["direction"]     = direction
    details["proposal_time"] = datetime.now(timezone.utc)
    details.setdefault("barrier_retries", 0)
    details["last_barrier_mult"] = barrier_mult if barrier_mult is not None else ATR_BARRIER_MULT
    with _lock:
        pending_signals[symbol] = details
    ws.send(json.dumps(deriv_proposal_payload(
        amount=STAKE,
        basis="stake",
        contract_type=CONTRACT_TYPE,
        currency="USD",
        duration=DURATION,
        duration_unit="m",
        symbol=symbol,
        barrier=_compute_barrier(symbol, direction, barrier_mult),
    )))


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
    offered_payout = deriv_float(prop.get("payout", 0))
    offered_profit = round(offered_payout - STAKE, 4)

    MAX_BARRIER_RETRIES = 2

    if offered_profit < PROFIT_MIN or offered_profit > PROFIT_MAX:
        with _lock:
            details = pending_signals.get(symbol)
        retries   = (details or {}).get("barrier_retries", 0)
        cur_mult  = (details or {}).get("last_barrier_mult", ATR_BARRIER_MULT)
        direction = (details or {}).get("direction", "UP")

        if details is not None and retries < MAX_BARRIER_RETRIES:
            # ── Adaptive retry: nudge the barrier toward the target band
            # instead of giving up on the first miss. ONETOUCH payout rises
            # as the barrier gets farther away (harder to touch) — so an
            # over-shoot means "move barrier closer", an under-shoot means
            # "move it farther". 20% step per retry, converges within 2 tries
            # for most symbols/volatility regimes.
            step     = 0.80 if offered_profit > PROFIT_MAX else 1.20
            new_mult = max(0.05, round(cur_mult * step, 3))

            with _lock:
                pending_signals.pop(symbol, None)
            _log(f"🔁 {symbol} proposal profit ${offered_profit:.2f} outside band — "
                 f"retry {retries+1}/{MAX_BARRIER_RETRIES} with mult {cur_mult}→{new_mult}")
            # Clear per-attempt fields — otherwise the proposal_id guard in
            # on_proposal() would ignore the retry's response (it thinks a
            # proposal was already accepted for this symbol), and the
            # pending proposal would just time out and release the slot
            # instead of actually retrying.
            details.pop("proposal_id", None)
            details.pop("buy_sent_at", None)
            details["barrier_retries"] = retries + 1
            request_proposal(ws, symbol, details, direction, barrier_mult=new_mult)
            return

        # Out of retries (or no details) — cancel cleanly
        with _lock:
            pending_signals.pop(symbol, None)
        _release_trade_slot(symbol)
        _record_funnel_rejection("Payout outside target band")
        _send_tg(
            f"💸 <b>Payout rejected</b> — <code>{symbol}</code>\n"
            f"Offered profit: <b>${offered_profit:.2f}</b>  "
            f"(target ${PROFIT_MIN:.2f}–${PROFIT_MAX:.2f})\n"
            f"Barrier too {'close' if offered_profit < PROFIT_MIN else 'far'} after "
            f"{MAX_BARRIER_RETRIES} adaptive retries — skipping."
        )
        _log(f"💸 {symbol} proposal rejected — profit ${offered_profit:.2f} outside "
             f"[${PROFIT_MIN}–${PROFIT_MAX}] after {MAX_BARRIER_RETRIES} retries")
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
            "stake":       deriv_float(buy.get("buy_price", STAKE)),
            "payout":      deriv_float(buy.get("payout", 0)),
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
    global total_pnl, win_count, loss_count, consecutive_losses, paused, pause_until
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

        # ── RESULT FIX: accept is_expired OR is_sold OR explicit status ──
        status = contract.get("status", "")
        is_settled = (
            bool(contract.get("is_expired"))
            or bool(contract.get("is_sold"))
            or status in ("won", "lost", "void")
        )
        if not is_settled:
            return

        info["settled"] = True
        is_void = status == "void"

        # Extract profit robustly
        profit = deriv_float(contract.get("profit") or 0)
        if not is_void and profit == 0:
            if status == "won":
                profit = deriv_float(info.get("payout", STAKE) or STAKE) - deriv_float(info.get("stake", STAKE) or STAKE)
            elif status == "lost":
                profit = -deriv_float(info.get("stake", STAKE) or STAKE)

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
                d.get("bb_width", 0), d.get("regime", "Choppy"),
                float(d.get("rsi") or 50.0) / 100.0, float(d.get("stochrsi_k") or 50.0) / 100.0,
                float(d.get("roc") or 0.0), float(d.get("body_ratio") or 0.0),
                float(d.get("bb_position") or 0.5), float(d.get("adx") or 0.0) / 100.0,
                # V2.9: context features
                float(d.get("session_health") or 0.5),
                float(d.get("asset_health") or 0.5),
                float(d.get("sober_structure_pts") or 0.0) / 25.0,
                float(d.get("sober_trend_pts") or 0.0) / 20.0,
                float(d.get("sober_zone_pts") or 0.0) / 20.0,
                float(d.get("sober_timing_pts") or 0.0) / 15.0,
                float(d.get("sober_momentum_pts") or 0.0) / 10.0,
                float(d.get("sober_candle_pts") or 0.0) / 10.0,
                float(d.get("mtf_agreement") or 0.0) / 2.0,
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

        total_pnl   += profit
        peak_equity  = max(peak_equity, total_pnl)
        max_drawdown = max(max_drawdown, peak_equity - total_pnl)

        # Per-symbol session stats
        stats = session_symbol_stats.setdefault(symbol, {"wins": 0, "losses": 0, "pnl": 0.0})
        stats["pnl"] += profit
        if win: stats["wins"]   += 1
        else:   stats["losses"] += 1

        # V3.0: update rolling windows (session + symbol) for recommendations engine
        _rolling_push(mkt_session, symbol, win, profit)

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
            _conf_bkt_snapshot = (_bkt, dict(_bst))
        else:
            _conf_bkt_snapshot = None
        # Update live signal-score tracker so we can see whether raw score bands
        # (independent of ML confidence) actually make money.
        _score_val = d.get("total_score", 0)
        _sbkt = _ml_score_bucket(_score_val)
        _sbst = _ml_score_live_stats.setdefault(_sbkt, {"wins": 0, "losses": 0, "pnl": 0.0})
        if win: _sbst["wins"]   += 1
        else:   _sbst["losses"] += 1
        _sbst["pnl"] += profit
        _score_bkt_snapshot = (_sbkt, dict(_sbst))

    # Persist the updated buckets to Neon (fire-and-forget thread — never block settlement)
    def _persist_ml_stats():
        if _conf_bkt_snapshot:
            _ml_stats_save("conf", _conf_bkt_snapshot[0], _conf_bkt_snapshot[1])
        _ml_stats_save("score", _score_bkt_snapshot[0], _score_bkt_snapshot[1])
    threading.Thread(target=_persist_ml_stats, daemon=True, name="MLStatsSave").start()

    # DB write — V2.6: session/regime; V2.7: oscillators; V2.9: context features
    db_queue.put((
        datetime.now(timezone.utc).isoformat(), symbol, info["direction"],
        info["barrier"], info["stake"], info["payout"], profit, int(win),
        d.get("total_score", 0), d.get("wick_atr_ratio", 0),
        d.get("atr", 0), d.get("atr_ma", 0),
        d.get("ema_fast_sl", 0), d.get("ema_slow_sl", 0), d.get("ema_distance", 0),
        mkt_session,
        d.get("bb_width", 0), d.get("regime", "Choppy"),
        float(d.get("rsi") or 50.0) / 100.0, float(d.get("stochrsi_k") or 50.0) / 100.0,
        float(d.get("roc") or 0.0), float(d.get("body_ratio") or 0.0),
        float(d.get("bb_position") or 0.5), float(d.get("adx") or 0.0) / 100.0,
        # V2.9: context — health captured at signal time, Sober pts, MTF agreement
        float(d.get("session_health") or 0.5),
        float(d.get("asset_health") or 0.5),
        float(d.get("sober_structure_pts") or 0.0) / 25.0,
        float(d.get("sober_trend_pts") or 0.0) / 20.0,
        float(d.get("sober_zone_pts") or 0.0) / 20.0,
        float(d.get("sober_timing_pts") or 0.0) / 15.0,
        float(d.get("sober_momentum_pts") or 0.0) / 10.0,
        float(d.get("sober_candle_pts") or 0.0) / 10.0,
        float(d.get("mtf_agreement") or 0.0) / 2.0,
    ))
    # V2.6: auto-blacklist check after each settled trade (background thread)
    threading.Thread(target=_check_and_update_blacklist, args=(symbol,),
                     daemon=True, name="BlacklistCheck").start()
    _log_signal_features(symbol, d, mkt_session, win, profit)

    # Pass pre-reset stat snapshot so card always shows correct cumulative figures
    _send_tg(_result_card(symbol, profit, win, d,
                          _wc=rc_wc, _lc=rc_lc, _pnl=rc_pnl, _cl=rc_cl))
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
        ws_obj.connect(deriv_ws_url(), timeout=10)
        deriv_send_auth(ws_obj)
        ws_obj.settimeout(8)
        if not _is_new_deriv_api():
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
    deriv_send_auth(ws)
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

                            _record_scan()

                            # ── V2.6: Auto-blacklist gate ─────────────────────────
                            if _is_blacklisted(symbol):
                                _record_funnel_rejection("Asset blacklisted")
                                _send_rejection(symbol, direction, score,
                                                "Asset auto-blacklisted (poor recent WR)")
                                return

                            # ── M5 + M15 multi-timeframe gates ────────────────────
                            m5_ind    = indicators_m5.get(symbol, {})
                            m5_ready  = m5_ind.get("ready", False)
                            m5_dir    = m5_ind.get("supertrend_dir", 0)
                            m15_ind   = indicators_m15.get(symbol, {})
                            m15_ready = m15_ind.get("ready", False)
                            m15_dir   = m15_ind.get("supertrend_dir", 0)
                            # V2.9: count agreeing timeframes (0=neither, 1=M5, 2=both)
                            # Stored in details so it flows to the DB training row.
                            _mtf_agree = 0
                            if m5_ready and m5_dir != 0:
                                if ((direction == "UP"   and m5_dir  == 1) or
                                        (direction == "DOWN" and m5_dir  == -1)):
                                    _mtf_agree += 1
                            if m15_ready and m15_dir != 0:
                                if ((direction == "UP"   and m15_dir == 1) or
                                        (direction == "DOWN" and m15_dir == -1)):
                                    _mtf_agree += 1
                            details["mtf_agreement"] = _mtf_agree
                            if (m5_ready and m5_dir != 0
                                    and ((direction == "UP" and m5_dir != 1)
                                         or (direction == "DOWN" and m5_dir != -1))):
                                _record_funnel_rejection("M5 Supertrend disagreement")
                                _send_rejection(symbol, direction, score,
                                                f"M5 Supertrend disagrees "
                                                f"({'↑' if m5_dir==1 else '↓'} on M5 vs {direction} on M1)")
                            elif (m15_ready and m15_dir != 0
                                    and ((direction == "UP" and m15_dir != 1)
                                         or (direction == "DOWN" and m15_dir != -1))):
                                _record_funnel_rejection("M15 Supertrend disagreement")
                                _send_rejection(symbol, direction, score,
                                                f"M15 Supertrend disagrees "
                                                f"({'↑' if m15_dir==1 else '↓'} on M15 vs {direction} on M1)")
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
                                                    f"ML confidence {conf*100:.1f}% < {_gate_lbl} gate "
                                                    f"[{_get_symbol_class(symbol)}]")
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
                                                        "Risk gate closed for 2nd entry (paused / daily limit)")
                                elif _reserve_trade_slot(symbol, now_t):
                                    request_proposal(ws, symbol, details, direction)
                                    _log(f"🎯 {symbol} {direction} TICK-ENTRY  score={score}/100  "
                                         f"momentum={len(tick_history.get(symbol,[]))}ticks  "
                                         f"conf={details.get('ml_confidence',1.0)*100:.0f}%  "
                                         f"class={_get_symbol_class(symbol)}")
                                else:
                                    _record_funnel_rejection("Risk gate closed (paused/cooldown/daily limit)")
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
                deriv_ws_url(),
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
        [InlineKeyboardButton("📊 Full Analytics",          callback_data="analytics"),
         InlineKeyboardButton("🔍 Pattern Discovery",       callback_data="patterns")],
        [InlineKeyboardButton("🏥 Health · Regimes",        callback_data="health"),
         InlineKeyboardButton("📈 Feature Importance",      callback_data="feat_importance")],
        [InlineKeyboardButton("🎯 Gate Optimizer",           callback_data="gate_optimizer"),
         InlineKeyboardButton("🤖 ML Progress",             callback_data="ml_progress")],
        # ── Control ──────────────────────────────────────────────────────
        [InlineKeyboardButton(pause_lbl,                    callback_data="toggle_pause"),
         InlineKeyboardButton("⏭ Skip a Symbol",           callback_data="skip_menu")],
        [InlineKeyboardButton("⚙ Settings & Config",       callback_data="settings"),
         InlineKeyboardButton("🧪 Fire Test Trade",         callback_data="test_menu")],
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
        f"📊 <b>DERIV TOUCH BOT — STATUS</b>  <i>{now.strftime('%H:%M:%S UTC')}</i>",
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


def _pattern_discovery_text() -> str:
    """
    Bucket settled trades by (market structure, S&D zone, ADX band, ML-confidence
    band) and report which combinations are actually profitable. Needs ≥10 trades
    per combo (and ≥50 total rows) before a combo is reported, to avoid noise.
    """
    try:
        rows = _db_fetch(
            "SELECT ms_type, sd_zone, adx_val, candle_pattern, win, profit "
            "FROM signal_features ORDER BY id"
        )
    except Exception as e:
        return f"⚠️ Pattern discovery unavailable: <code>{e}</code>"

    if not rows:
        return ("🔍 <b>PATTERN DISCOVERY</b>\n━━━━━━━━━━━━━━━━━━━━\n"
                "No signal data logged yet — patterns appear once trades settle.")
    if len(rows) < 50:
        return (f"🔍 <b>PATTERN DISCOVERY</b>\n━━━━━━━━━━━━━━━━━━━━\n"
                f"Only {len(rows)} trades logged — need ≥50 before patterns are reliable.")

    def _adx_band(v):
        v = float(v or 0)
        if v >= 35: return "ADX≥35"
        if v >= 25: return "ADX 25-35"
        if v >= 15: return "ADX 15-25"
        return "ADX&lt;15"

    buckets: dict = {}
    for ms_type, sd_zone, adx_val, candle_pattern, win, profit in rows:
        key = (ms_type or "sideways", sd_zone or "none", _adx_band(adx_val))
        b = buckets.setdefault(key, {"n": 0, "wins": 0, "pnl": 0.0})
        b["n"] += 1
        if win: b["wins"] += 1
        b["pnl"] += float(profit or 0)

    qualified = [(k, v) for k, v in buckets.items() if v["n"] >= 10]
    if not qualified:
        return (f"🔍 <b>PATTERN DISCOVERY</b>\n━━━━━━━━━━━━━━━━━━━━\n"
                f"{len(rows)} trades logged, but no combo has ≥10 samples yet.")

    qualified.sort(key=lambda kv: kv[1]["pnl"] / kv[1]["n"], reverse=True)

    lines = ["🔍 <b>PATTERN DISCOVERY</b>", "━━━━━━━━━━━━━━━━━━━━",
             f"{len(rows)} trades analyzed · showing combos with ≥10 samples\n"]
    for (ms_type, sd_zone, adx_band), v in qualified[:10]:
        wr = v["wins"] / v["n"] * 100
        avg = v["pnl"] / v["n"]
        emoji = "🟢" if avg > 0 else ("🔴" if avg < 0 else "⚪")
        lines.append(
            f"{emoji} <b>{ms_type}</b> / {sd_zone} / {adx_band}\n"
            f"    n={v['n']}  WR={wr:.0f}%  avg={'+' if avg>=0 else ''}${avg:.2f}  "
            f"total={'+' if v['pnl']>=0 else ''}${v['pnl']:.2f}"
        )
    return "\n".join(lines)


def _performance_analytics_text() -> str:
    """Win-rate breakdowns by score band, ADX band, candle pattern, session,
    S&D-zone distance, and ML-confidence band — from signal_features."""
    try:
        rows = _db_fetch(
            "SELECT score, adx_val, candle_pattern, session, sd_dist, "
            "ml_confidence, win, profit FROM signal_features ORDER BY id"
        )
    except Exception as e:
        return f"⚠️ Analytics unavailable: <code>{e}</code>"

    if not rows:
        return ("📈 <b>PERFORMANCE ANALYTICS</b>\n━━━━━━━━━━━━━━━━━━━━\n"
                "No signal data logged yet.")

    def _bucket_stats(rows, keyfn):
        buckets: dict = {}
        for r in rows:
            k = keyfn(r)
            b = buckets.setdefault(k, {"n": 0, "wins": 0, "pnl": 0.0})
            b["n"] += 1
            if r[-2]: b["wins"] += 1
            b["pnl"] += float(r[-1] or 0)
        return buckets

    def _fmt(buckets, order=None):
        keys = order if order else sorted(buckets.keys())
        out = []
        for k in keys:
            if k not in buckets: continue
            v = buckets[k]
            if v["n"] == 0: continue
            wr = v["wins"] / v["n"] * 100
            out.append(f"  {k:<14}: n={v['n']:<4} WR={wr:5.1f}%  pnl={'+' if v['pnl']>=0 else ''}${v['pnl']:.2f}")
        return "\n".join(out) if out else "  (no data)"

    def score_band(r):
        s = r[0] or 0
        if s >= 95: return "A+ (95-100)"
        if s >= 90: return "A (90-94)"
        if s >= 85: return "B+ (85-89)"
        if s >= 75: return "B (75-84)"
        return "C (&lt;75)"

    def adx_band(r):
        a = float(r[1] or 0)
        if a >= 35: return "ADX≥35"
        if a >= 25: return "ADX 25-35"
        if a >= 15: return "ADX 15-25"
        return "ADX&lt;15"

    def conf_band(r):
        c = r[5]
        if c is None: return "no-model"
        c = float(c)
        if c >= 0.90: return "≥90%"
        if c >= 0.80: return "80-90%"
        if c >= 0.75: return "75-80%"
        return "&lt;75%"

    def sd_band(r):
        d = float(r[4] or 99.0)
        if d <= 0.5: return "at zone"
        if d <= 1.5: return "near"
        if d <= 3.0: return "within range"
        return "far/none"

    lines = ["📈 <b>PERFORMANCE ANALYTICS</b>", "━━━━━━━━━━━━━━━━━━━━",
              f"{len(rows)} trades analyzed\n"]

    lines.append("<b>By Score Band</b>")
    lines.append(_fmt(_bucket_stats(rows, score_band),
                       ["A+ (95-100)", "A (90-94)", "B+ (85-89)", "B (75-84)", "C (&lt;75)"]))

    lines.append("\n<b>By ADX Band</b>")
    lines.append(_fmt(_bucket_stats(rows, adx_band),
                       ["ADX≥35", "ADX 25-35", "ADX 15-25", "ADX&lt;15"]))

    lines.append("\n<b>By Candle Pattern</b>")
    pat_buckets = _bucket_stats(rows, lambda r: r[2] or "none")
    top_pats = sorted(pat_buckets.items(), key=lambda kv: kv[1]["n"], reverse=True)[:6]
    lines.append(_fmt(dict(top_pats)))

    lines.append("\n<b>By Market Session</b>")
    lines.append(_fmt(_bucket_stats(rows, lambda r: r[3] or "?")))

    lines.append("\n<b>By S&D Zone Distance</b>")
    lines.append(_fmt(_bucket_stats(rows, sd_band),
                       ["at zone", "near", "within range", "far/none"]))

    lines.append("\n<b>By ML Confidence</b>")
    lines.append(_fmt(_bucket_stats(rows, conf_band), ["≥90%", "80-90%", "75-80%", "&lt;75%", "no-model"]))

    return "\n".join(lines)


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
        "<i>Persists across restarts (stored in Neon PostgreSQL).</i>",
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

    if not log:
        return (
            "📅 <b>Daily Session History</b>\n━━━━━━━━━━━━━━━━━━━━\n"
            + _strategy_header() +
            f"Date: {today}\n\n"
            "— No TP or SL hits yet today. —\n\n"
            + "\n".join(funnel_lines) +
            "\n\n<i>Each time the bot hits its daily TP or SL it resets and this record is updated.</i>"
        )

    tp_hits = sum(1 for e in log if e["reason"] == "TP")
    sl_hits = sum(1 for e in log if e["reason"] == "SL")
    total_day_pnl    = sum(e["pnl"]    for e in log)
    total_day_trades = sum(e["trades"] for e in log)
    total_day_wins   = sum(e["wins"]   for e in log)
    total_day_losses = sum(e["losses"] for e in log)
    total_possible   = total_day_wins + total_day_losses
    day_wr = total_day_wins / total_possible * 100 if total_possible else 0
    day_avg_trade  = total_day_pnl / total_day_trades if total_day_trades else 0.0
    day_max_dd     = max((e.get("max_dd", 0.0) for e in log), default=0.0)
    day_max_dd_sess = max(log, key=lambda e: e.get("max_dd", 0.0)) if log else None

    # Include the still-running current session's live drawdown so "today"
    # reflects the in-progress session too, not just closed TP/SL sessions.
    with _lock:
        live_pnl, live_peak, live_mdd = total_pnl, peak_equity, max_drawdown
        live_trades, live_wins, live_losses = daily_trades, win_count, loss_count
    day_max_dd = max(day_max_dd, live_mdd)

    adv_today = get_today_advanced_stats()

    lines = [
        "📅 <b>Daily Session History</b>",
        "━━━━━━━━━━━━━━━━━━━━",
        _strategy_header(),
        f"Date   : {today}",
        f"🎯 TP hits: <b>{tp_hits}</b>   🛑 SL hits: <b>{sl_hits}</b>",
        f"Day P&L    : <b>{'+' if total_day_pnl >= 0 else ''}${total_day_pnl:.2f}</b>",
        f"Total Trades: {total_day_trades}",
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

    lines += ["", *funnel_lines, "━━━━━━━━━━━━━━━━━━━━"]

    for i, e in enumerate(log, 1):
        icon  = "🎯" if e["reason"] == "TP" else "🛑"
        sign  = "+" if e["pnl"] >= 0 else ""
        h, mr = divmod(e["duration_min"], 60)
        dur_str  = f"{h}h {mr}m" if h else f"{mr}m"
        ms_name  = e.get("market_session", "—")
        e_wr  = e["wins"] / (e["wins"] + e["losses"]) * 100 if (e["wins"] + e["losses"]) else 0
        entry = (
            f"{icon} <b>Session {i}</b>  [{e['time']}]  {dur_str}  "
            f"{SESSION_EMOJIS.get(ms_name,'')} {ms_name}\n"
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
            f"▶ <b>Current session (still running)</b>\n"
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


def _gate_optimizer_text() -> str:
    """
    Replay every row in signal_features (ml_confidence, session, win, profit) and sweep
    gate thresholds 55–90% in 5% steps.  Shows current-vs-recommended comparison,
    a profit stability bar chart, and recommendation confidence stars.
    Pure read — never changes any live setting.
    """
    try:
        rows = _db_fetch(
            "SELECT session, ml_confidence, win, profit "
            "FROM signal_features "
            "WHERE ml_confidence IS NOT NULL AND win IS NOT NULL "
            "ORDER BY id"
        )
    except Exception as e:
        return f"🎯 <b>Gate Optimizer</b>\n\n❌ DB query failed: <code>{e}</code>"

    if not rows:
        return (
            "🎯 <b>Gate Optimizer</b>\n\n"
            "⚠ No signal data yet — trades logged after the next retrain will appear here."
        )

    GATES = [round(g / 100, 2) for g in range(55, 95, 5)]   # 0.55 … 0.90
    BAR_WIDTH = 8   # max █ blocks in stability chart

    def _sweep(subset):
        """Return list of (gate, profit, n, wr) + best_gate, best_profit, best_n."""
        best_profit, best_gate, best_n = float("-inf"), None, 0
        results = []
        for g in GATES:
            filtered = [(w, p) for (c, w, p) in subset if c >= g]
            n = len(filtered)
            if n == 0:
                results.append((g, 0.0, 0, 0.0))
                continue
            profit = sum(p for _, p in filtered)
            wr     = sum(w for w, _ in filtered) / n * 100
            results.append((g, profit, n, wr))
            if profit > best_profit:
                best_profit, best_gate, best_n = profit, g, n
        return results, best_gate, best_profit, best_n

    def _lookup(results, gate):
        """Return the result row for a specific gate value, or None."""
        return next((r for r in results if r[0] == gate), None)

    def _stability(results, best_gate):
        """
        Build a bar chart for the 3 gates centred on best_gate (±1 step each side)
        and return (chart_lines, stability_label, stability_note).
        Stability is defined by how much profit varies in that ±10pp window.
        """
        idx = next((i for i, r in enumerate(results) if r[0] == best_gate), None)
        if idx is None:
            return [], "Unknown", ""
        # window: up to 2 steps either side so user sees context
        window = results[max(0, idx - 2): idx + 3]
        profits  = [r[1] for r in window if r[2] > 0]
        if len(profits) < 2:
            return [], "Insufficient data", ""
        p_max   = max(profits)
        p_range = p_max - min(profits)
        # Relative spread vs best profit
        spread_pct = p_range / max(abs(p_max), 1e-9)
        if spread_pct < 0.05:
            stab_label, stab_note = "Very Stable", "Differences are tiny — any gate in this range is fine."
        elif spread_pct < 0.15:
            stab_label, stab_note = "Stable",      "Small variation — recommendation is reliable."
        elif spread_pct < 0.35:
            stab_label, stab_note = "Moderate",    "Some variation — collect more data before committing."
        else:
            stab_label, stab_note = "Volatile",    "Large swings — wait for more signals before acting."

        chart = []
        for (g, profit, n, wr) in window:
            if n == 0:
                bar = "·" * BAR_WIDTH
            else:
                filled = max(1, round(profit / p_max * BAR_WIDTH)) if profit > 0 else 0
                bar    = "█" * filled + "░" * (BAR_WIDTH - filled)
            star = " ⭐" if g == best_gate else ""
            chart.append(f"  <code>{g*100:.0f}%  {bar}{star}</code>")
        return chart, stab_label, stab_note

    def _rec_confidence(n: int):
        """Return (stars, label, inline_note) based on signal count."""
        if n >= 100: return "★★★★★", "Very High", ""
        if n >= 75:  return "★★★★☆", "High",      ""
        if n >= 50:  return "★★★☆☆", "Moderate",  ""
        if n >= 25:  return "★★☆☆☆", "Low",       " <i>(limited data)</i>"
        return       "★☆☆☆☆", "Very Low",  " <i>(Not enough data yet)</i>"

    def _session_block(sess_name, sess_data, cur_gate_val):
        """Return the formatted lines for one session."""
        s_results, s_best, s_profit, s_n = _sweep(sess_data)
        emoji      = SESSION_EMOJIS.get(sess_name, "")
        prof       = _get_session_profile(sess_name)
        cur_s_gate = prof.get("ml_gate", cur_gate_val)
        n_total    = len(sess_data)
        stars, conf_label, conf_note = _rec_confidence(n_total)

        blk = [f"\n{emoji} <b>{sess_name}</b>"]
        blk.append(f"  {stars} {conf_label}  ·  {n_total} signals analyzed{conf_note}")

        if s_best is None or n_total < 10:
            blk.append("  ★☆☆☆☆ Very Low  — need 10+ signals for any estimate")
            return blk

        # ── Current gate stats ───────────────────────────────────────────
        cur_row  = _lookup(s_results, cur_s_gate)
        best_row = _lookup(s_results, s_best)
        if cur_row:
            c_sign = "+" if cur_row[1] >= 0 else ""
            blk.append(
                f"\n  <b>Current ({cur_s_gate*100:.0f}%)</b>\n"
                f"  Profit: <b>{c_sign}${cur_row[1]:.2f}</b>  "
                f"Trades: {cur_row[2]}  WR: {cur_row[3]:.0f}%"
            )

        # ── Recommended gate stats ────────────────────────────────────────
        if best_row and s_best != cur_s_gate:
            b_sign = "+" if best_row[1] >= 0 else ""
            blk.append(
                f"\n  <b>Recommended ({s_best*100:.0f}%)</b>\n"
                f"  Profit: <b>{b_sign}${best_row[1]:.2f}</b>  "
                f"Trades: {best_row[2]}  WR: {best_row[3]:.0f}%"
            )
            # ── Difference ────────────────────────────────────────────────
            if cur_row:
                d_profit = best_row[1] - cur_row[1]
                d_wr     = best_row[3] - cur_row[3]
                d_trades = best_row[2] - cur_row[2]
                blk.append(
                    f"\n  <b>Difference</b>\n"
                    f"  {'+' if d_profit >= 0 else ''}${d_profit:.2f} profit  "
                    f"  {'+' if d_wr >= 0 else ''}{d_wr:.0f}% WR  "
                    f"  {'+' if d_trades >= 0 else ''}{d_trades} trades"
                )
            diff_s = (s_best - cur_s_gate) * 100
            blk.append(
                f"\n  💡 <i>/setsession \"{sess_name}\" ml_gate {s_best*100:.0f}  "
                f"({diff_s:+.0f}pp)</i>"
            )
        else:
            blk.append(f"\n  ✅ <i>Current gate already optimal for this session.</i>")

        # ── Stability chart ───────────────────────────────────────────────
        chart_lines, stab_label, stab_note = _stability(s_results, s_best)
        if chart_lines:
            blk.append(f"\n  <b>Recommendation Stability</b>")
            blk.extend(chart_lines)
            blk.append(f"  <i>{stab_label} — {stab_note}</i>")

        return blk

    # ── Global sweep ──────────────────────────────────────────────────────
    with ml_lock:
        cur_gate = ML_CONFIDENCE_MIN

    global_rows = [(float(r[1] or 0), int(r[2] or 0), float(r[3] or 0)) for r in rows]
    g_results, g_best_gate, g_best_profit, g_best_n = _sweep(global_rows)

    lines = [
        f"🎯 <b>Gate Optimizer</b>  ({len(rows)} signals replayed)",
        "",
        "<b>── Global ──</b>",
    ]

    g_cur_row  = _lookup(g_results, cur_gate)
    g_best_row = _lookup(g_results, g_best_gate)

    if g_cur_row:
        c_sign = "+" if g_cur_row[1] >= 0 else ""
        lines.append(
            f"\n  <b>Current ({cur_gate*100:.0f}%)</b>\n"
            f"  Profit: <b>{c_sign}${g_cur_row[1]:.2f}</b>  "
            f"Trades: {g_cur_row[2]}  WR: {g_cur_row[3]:.0f}%"
        )

    if g_best_row and g_best_gate != cur_gate:
        b_sign = "+" if g_best_row[1] >= 0 else ""
        lines.append(
            f"\n  <b>Recommended ({g_best_gate*100:.0f}%)</b>\n"
            f"  Profit: <b>{b_sign}${g_best_row[1]:.2f}</b>  "
            f"Trades: {g_best_row[2]}  WR: {g_best_row[3]:.0f}%"
        )
        if g_cur_row:
            gd_p = g_best_row[1] - g_cur_row[1]
            gd_w = g_best_row[3] - g_cur_row[3]
            gd_t = g_best_row[2] - g_cur_row[2]
            lines.append(
                f"\n  <b>Difference</b>\n"
                f"  {'+' if gd_p >= 0 else ''}${gd_p:.2f} profit  "
                f"  {'+' if gd_w >= 0 else ''}{gd_w:.0f}% WR  "
                f"  {'+' if gd_t >= 0 else ''}{gd_t} trades"
            )
        diff = (g_best_gate - cur_gate) * 100
        lines.append(
            f"\n  💡 <i>Suggestion: /set ml_gate {g_best_gate*100:.0f}  "
            f"({diff:+.0f}pp — never auto-applied)</i>"
        )
    elif g_best_gate == cur_gate:
        lines.append(f"\n  ✅ <i>Current global gate ({cur_gate*100:.0f}%) is already optimal.</i>")

    # Global stability chart
    g_chart, g_stab, g_stab_note = _stability(g_results, g_best_gate)
    if g_chart:
        lines.append(f"\n  <b>Recommendation Stability</b>")
        lines.extend(g_chart)
        lines.append(f"  <i>{g_stab} — {g_stab_note}</i>")

    # ── Per-session blocks ────────────────────────────────────────────────
    sessions = {}
    for r in rows:
        sess = r[0] or "Unknown"
        sessions.setdefault(sess, []).append(
            (float(r[1] or 0), int(r[2] or 0), float(r[3] or 0))
        )

    if sessions:
        lines.append("\n<b>── Per Session · Recommendation Confidence ──</b>")
        # Most-confident sessions first
        for sess_name in sorted(sessions.keys(), key=lambda s: len(sessions[s]), reverse=True):
            lines.extend(_session_block(sess_name, sessions[sess_name], cur_gate))

    lines += ["", "<i>Suggestions only — no settings were changed.</i>"]
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
        f"\n<b>Engine</b>\n"
        f"  Supertrend     : period={SUPERTREND_PERIOD}  mult={SUPERTREND_ATR_MULT}\n"
        f"  Barrier        : ATR × {ATR_BARRIER_MULT}\n"
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
    # Trimmed to the params actually worth adjusting live from a phone.
    # ml2_gate / retrain / ml_min / second_cd / second_win are "set once and
    # forget" tuning knobs — still changeable via /set <param> <value> if needed,
    # just not cluttering the quick +/- keyboard.
    _LABELS = [
        ("ml_gate",    "ML Gate"),
        ("score",      "Score Gate"),
        ("stake",      "Stake $"),
        ("duration",   "Duration"),
        ("cooldown",   "Cooldown"),
        ("tp",         "Session TP"),
        ("sl",         "Session SL"),
        ("max_loss",   "Max C.Loss"),
        ("pause_min",  "Pause Min"),
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


def _settings_menu_kb() -> InlineKeyboardMarkup:
    """Top-level Settings submenu — replaces direct jump to adj keyboard."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("⚙ Adjust Live Parameters",  callback_data="settings_adj")],
        [InlineKeyboardButton("🔍 Debug State",             callback_data="debug_state"),
         InlineKeyboardButton("📊 Profiles & Stats",        callback_data="profile_stats")],
        [InlineKeyboardButton("📋 Session Profiles",        callback_data="setsession_list")],
        [InlineKeyboardButton("🎯 Gate Optimizer",          callback_data="gate_optimizer")],
        [InlineKeyboardButton("🏠 Back to Menu",            callback_data="main_menu")],
    ])


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
        logger.info(text)
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
        ws.connect(deriv_ws_url(), timeout=15)
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
        deriv_send_auth(ws)
        if _is_new_deriv_api():
            account = _deriv_options_account_id()
        else:
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
        # ── Payout gate with adaptive barrier retry (same rule as live
        # trading) — nudge the barrier toward the target band instead of
        # giving up on the first miss. ─────────────────────────────────
        MAX_BARRIER_RETRIES = 2
        cur_mult = ATR_BARRIER_MULT
        for attempt in range(MAX_BARRIER_RETRIES + 1):
            ws.send(json.dumps(deriv_proposal_payload(
                amount=STAKE, basis="stake",
                contract_type=CONTRACT_TYPE, currency="USD",
                duration=DURATION, duration_unit="m",
                symbol=symbol,
                barrier=_compute_barrier(symbol, direction, cur_mult),
            )))
            prop_msg = recv_typed("proposal", timeout=10)
            if not prop_msg or "error" in prop_msg:
                err = (prop_msg or {}).get("error", {}).get("message", "timeout")
                tg(f"🧪 <b>Test Trade FAILED</b>\n❌ Proposal error: <code>{err}</code>")
                return
            prop    = prop_msg["proposal"]
            pid     = prop["id"]
            ask     = deriv_float(prop.get("ask_price", STAKE))
            payout  = deriv_float(prop.get("payout", 0))
            offered_profit = round(payout - STAKE, 4)

            if PROFIT_MIN <= offered_profit <= PROFIT_MAX:
                break  # in band — proceed to buy

            if attempt < MAX_BARRIER_RETRIES:
                step     = 0.80 if offered_profit > PROFIT_MAX else 1.20
                new_mult = max(0.05, round(cur_mult * step, 3))
                tg(
                    f"🔁 <b>Test Trade</b> – payout ${offered_profit:.2f} outside band, "
                    f"retrying {attempt+1}/{MAX_BARRIER_RETRIES} "
                    f"(barrier mult {cur_mult}→{new_mult})…"
                )
                cur_mult = new_mult
                continue

            tg(
                f"🧪 <b>Test Trade</b>  –  💸 Payout rejected\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"Offered payout : ${payout}  (profit ${offered_profit:.2f})\n"
                f"Target band    : ${PROFIT_MIN:.2f} – ${PROFIT_MAX:.2f}\n"
                f"Barrier too {'close (easy touch → low payout)' if offered_profit < PROFIT_MIN else 'far (hard touch → high payout)'}.\n"
                f"<i>Still outside band after {MAX_BARRIER_RETRIES} adaptive retries "
                f"(last mult {cur_mult}).</i>"
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
        bought_at = deriv_float(buy.get("buy_price", STAKE))
        paid_out  = deriv_float(buy.get("payout", 0))
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

        profit = deriv_float(contract_data.get("profit", 0))
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
        "🤖 <b>Deriv Touch Bot v1.0</b>\nOne-Touch Options · Sessions · Confidence Gate",
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


# ── V3.0 Commands ─────────────────────────────────────────────────────────

async def cmd_debug(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Full state dump: /debug"""
    if not _bot_ready:
        await update.message.reply_text(_NOT_READY_MSG, parse_mode="HTML"); return
    import html as _h
    now_dt   = datetime.now(timezone.utc)
    session  = _get_session_name(now_dt)
    prof     = _get_session_profile(session)
    regime   = "—"
    with _lock:
        _pnl, _wc, _lc, _cl, _dt, _paused = (
            total_pnl, win_count, loss_count, consecutive_losses, daily_trades, paused
        )
    with ml_lock:
        _gate_global  = ML_CONFIDENCE_MIN
        _gate_session = prof.get("ml_gate", ML_CONFIDENCE_MIN)
        _gate_2nd     = SECOND_ENTRY_ML_MIN
        _opt_t        = _ml_optimal_threshold
        _trained      = ml_trained_on
        _total_ml     = ml_total_trades
        _training     = ml_training_active
        _model_ready  = ml_model is not None
    _until_cd = max(0, int((ML_RETRAIN_EVERY - max(0, _total_ml - _trained))))

    lines = [
        f"🔍 <b>DEBUG STATE</b>  {now_dt.strftime('%H:%M:%S UTC')}",
        "",
        f"<b>Session</b>       {session}  [{prof.get('mode','Normal')}]",
        f"<b>Regime</b>        {regime}",
        "",
        f"<b>ML Gate</b>       Global {_gate_global*100:.0f}%  |  Session {_gate_session*100:.0f}%",
        f"<b>2nd Entry</b>     {_gate_2nd*100:.0f}%",
        f"<b>Opt Threshold</b> {_opt_t*100:.0f}% (walk-forward)",
        f"<b>Model</b>         {'✅ ready' if _model_ready else '⏳ not trained yet'}  "
        f"(trained on {_trained}, current {_total_ml})",
        f"<b>Next retrain</b>  in ~{_until_cd} new trades  "
        f"({'training now' if _training else 'idle'})",
        "",
        f"<b>Session TP</b>    ${prof.get('tp', DAILY_PROFIT_TARGET):.1f}",
        f"<b>Session SL</b>    ${prof.get('sl', DAILY_LOSS_LIMIT):.1f}",
        f"<b>Max Trades</b>    {prof.get('max_trades','unlimited')}",
        f"<b>Cooldown</b>      {prof.get('cooldown', COOLDOWN_MINUTES)} min",
        "",
        f"<b>Session P&L</b>   {'+' if _pnl>=0 else ''}${_pnl:.2f}  "
        f"({_wc}W / {_lc}L  {_wc/max(_wc+_lc,1)*100:.0f}%WR)",
        f"<b>Cons Losses</b>   {_cl}",
        f"<b>Paused</b>        {'⏸ YES' if _paused else '▶ no'}",
        "",
        "<b>Session Profiles loaded:</b>",
    ]
    for sn, sp in SESSION_PROFILES.items():
        emoji = SESSION_EMOJIS.get(sn, "")
        lines.append(
            f"  {emoji} <b>{sn}</b>  ML {sp['ml_gate']*100:.0f}%  "
            f"TP ${sp['tp']}  SL ${sp['sl']}  [{sp['mode']}]"
        )
    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


async def cmd_profile(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Show session and symbol profiles with rolling stats: /profile"""
    if not _bot_ready:
        await update.message.reply_text(_NOT_READY_MSG, parse_mode="HTML"); return
    lines = ["📊 <b>Session Profiles + Rolling Stats</b>", ""]
    for sn, sp in SESSION_PROFILES.items():
        emoji  = SESSION_EMOJIS.get(sn, "")
        drec   = _rolling_session_deque.get(sn)
        st20   = _rolling_stats_for(drec, n=20) if drec else {"n": 0}
        st50   = _rolling_stats_for(drec, n=50) if drec else {"n": 0}
        st_all = _rolling_stats_for(drec)        if drec else {"n": 0}
        def _wr(s):
            return f"{s['wr']*100:.0f}%" if s.get("wr") is not None else "—"
        lines.append(
            f"{emoji} <b>{sn}</b>  [{sp['mode']}]\n"
            f"  Gate {sp['ml_gate']*100:.0f}%  2nd {sp['ml2_gate']*100:.0f}%  "
            f"TP ${sp['tp']}  SL ${sp['sl']}\n"
            f"  Rolling WR:  last 20 <b>{_wr(st20)}</b>  "
            f"last 50 <b>{_wr(st50)}</b>  lifetime <b>{_wr(st_all)}</b>  "
            f"({st_all['n']} trades)\n"
            f"  PF {st50.get('pf') or '—'}  avg ${st50.get('avg', 0):+.2f}"
        )
    lines += ["", "📈 <b>Symbol Rolling Stats</b>  (last 50 this run)", ""]
    sorted_syms = sorted(
        _rolling_symbol_deque.keys(),
        key=lambda s: (_rolling_stats_for(_rolling_symbol_deque[s], 50).get("profit", 0)),
        reverse=True,
    )
    for sym in sorted_syms:
        drec = _rolling_symbol_deque[sym]
        s50  = _rolling_stats_for(drec, 50)
        s20  = _rolling_stats_for(drec, 20)
        if s50["n"] == 0:
            continue
        lines.append(
            f"  <code>{sym:<10}</code>  "
            f"L20 {s20['wr']*100:.0f}%WR  "
            f"L50 {s50['wr']*100:.0f}%WR  "
            f"avg ${s50['avg']:+.2f}  PF {s50['pf']}"
        )
    if not sorted_syms:
        lines.append("  <i>No trades recorded yet this run.</i>")
    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


async def cmd_setsession(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """
    /setsession <session_name> <param> <value>   — update a session profile.
    /setsession list                             — show all session profiles.

    Params: ml_gate  ml2_gate  tp  sl  max_trades  cooldown  mode
    Mode values: Normal  Aggressive  Defensive

    Examples:
      /setsession "Late New York" ml_gate 70
      /setsession "Early New York" tp 8
      /setsession "London" mode Aggressive
    """
    if not _bot_ready:
        await update.message.reply_text(_NOT_READY_MSG, parse_mode="HTML"); return
    args = ctx.args or []
    raw  = " ".join(args).strip()

    if not raw or raw.lower() == "list":
        lines = ["📋 <b>Session Profiles</b>  (edit with /setsession)"]
        for sn, sp in SESSION_PROFILES.items():
            emoji = SESSION_EMOJIS.get(sn, "")
            lines.append(
                f"\n{emoji} <b>{sn}</b>\n"
                f"  ml_gate={sp['ml_gate']*100:.0f}%  ml2_gate={sp['ml2_gate']*100:.0f}%\n"
                f"  tp=${sp['tp']}  sl=${sp['sl']}\n"
                f"  max_trades={sp['max_trades']}  cooldown={sp['cooldown']}m  mode={sp['mode']}"
            )
        await update.message.reply_text("\n".join(lines), parse_mode="HTML")
        return

    # Parse: handle quoted session name then param value
    import shlex
    try:
        parts = shlex.split(raw)
    except ValueError:
        parts = raw.split()
    if len(parts) < 3:
        await update.message.reply_text(
            "Usage: /setsession <i>session_name</i> <i>param</i> <i>value</i>\n"
            "Example: <code>/setsession \"Late New York\" ml_gate 70</code>",
            parse_mode="HTML",
        )
        return

    # Session name may be one or two words — try longest match first
    sess_name = param = val_raw = None
    for split_at in range(len(parts) - 1, 0, -1):
        candidate = " ".join(parts[:split_at])
        if candidate in SESSION_PROFILES:
            sess_name = candidate
            param     = parts[split_at].lower()
            val_raw   = " ".join(parts[split_at + 1:])
            break
    if sess_name is None:
        await update.message.reply_text(
            f"❌ Session not found in: {', '.join(SESSION_PROFILES.keys())}",
            parse_mode="HTML",
        )
        return

    _FLOAT_PARAMS   = {"tp", "sl"}
    _PERCENT_PARAMS = {"ml_gate", "ml2_gate"}
    _INT_PARAMS     = {"max_trades", "cooldown"}
    _STR_PARAMS     = {"mode"}

    prof = SESSION_PROFILES[sess_name]
    try:
        if param in _PERCENT_PARAMS:
            v = float(val_raw) / (100.0 if float(val_raw) > 1.0 else 1.0)
            prof[param] = round(v, 3)
            display = f"{prof[param]*100:.0f}%"
        elif param in _FLOAT_PARAMS:
            prof[param] = round(float(val_raw), 2)
            display = f"${prof[param]}"
        elif param in _INT_PARAMS:
            prof[param] = int(val_raw)
            display = str(prof[param])
        elif param in _STR_PARAMS:
            if val_raw not in ("Normal", "Aggressive", "Defensive"):
                raise ValueError(f"mode must be Normal, Aggressive, or Defensive")
            prof[param] = val_raw
            display = val_raw
        else:
            await update.message.reply_text(
                f"❌ Unknown param <code>{param}</code>. "
                f"Valid: ml_gate ml2_gate tp sl max_trades cooldown mode",
                parse_mode="HTML",
            )
            return
    except ValueError as ve:
        await update.message.reply_text(f"❌ Invalid value: {ve}", parse_mode="HTML")
        return

    _save_session_profiles()
    await update.message.reply_text(
        f"✅ <b>{sess_name}</b>  <code>{param}</code> → <b>{display}</b>\n"
        f"<i>Saved to {SESSION_PROFILES_PATH} — takes effect immediately.</i>",
        parse_mode="HTML",
    )

async def cmd_alltime(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _bot_ready:
        await update.message.reply_text(_NOT_READY_MSG, parse_mode="HTML"); return
    await update.message.reply_text(_alltime_text(), reply_markup=_main_kb(), parse_mode="HTML")

async def cmd_patterns(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _bot_ready:
        await update.message.reply_text(_NOT_READY_MSG, parse_mode="HTML"); return
    await update.message.reply_text(_pattern_discovery_text(), reply_markup=_main_kb(), parse_mode="HTML")

async def cmd_analytics(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _bot_ready:
        await update.message.reply_text(_NOT_READY_MSG, parse_mode="HTML"); return
    await update.message.reply_text(_performance_analytics_text(), reply_markup=_main_kb(), parse_mode="HTML")

async def cmd_export(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Send full trades.db CSV on demand."""
    if not _bot_ready:
        await update.message.reply_text(_NOT_READY_MSG, parse_mode="HTML"); return
    await update.message.reply_text("⏳ Generating CSV export…", parse_mode="HTML")
    try:
        rows = _db_fetch("SELECT * FROM touch_trades ORDER BY id")
        if USE_PG:
            cols = ["id","timestamp","symbol","direction","barrier","stake","payout",
                    "profit","win","score","wick_atr_ratio","atr","atr_ma",
                    "ema_fast_slope","ema_slow_slope","ema_distance","market_session"]
        else:
            conn = sqlite3.connect("touch_trades.db")
            desc = conn.execute("PRAGMA table_info(touch_trades)").fetchall()
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
    elif os.path.exists("touch_trades.db"):
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        with open("touch_trades.db", "rb") as f:
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
            global ml_models_per_symbol, ml_trained_per_symbol
            if isinstance(payload, dict):
                with ml_lock:
                    ml_model              = payload.get("model")
                    ml_trained_on         = payload.get("trained_on", 0)
                    ml_models_per_class   = payload.get("per_class_models",  ml_models_per_class)
                    ml_trained_per_class  = payload.get("per_class_trained", ml_trained_per_class)
                    ml_models_per_symbol  = payload.get("per_symbol_models",  ml_models_per_symbol)
                    ml_trained_per_symbol = payload.get("per_symbol_trained", ml_trained_per_symbol)
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
                    _rc = sqlite3.connect("touch_trades.db")

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
                                "SELECT 1 FROM touch_trades WHERE timestamp=%s AND symbol=%s "
                                "AND direction=%s AND stake=%s LIMIT 1",
                                (ts_val, sym_val, dir_val, stake_val),
                            )
                        else:
                            dup = _rc.execute(
                                "SELECT 1 FROM touch_trades WHERE timestamp=? AND symbol=? "
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
                            f"INSERT INTO touch_trades ({','.join(_db_cols)}) VALUES ({ph})"
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
            total_now = _db_fetch("SELECT COUNT(*) FROM touch_trades")
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
#  GEMMA 3 AI ANALYST  (/ai command)
# ══════════════════════════════════════════════════════════════════════

def _ai_build_summary(cmd: str, arg: str = "") -> str:
    """Query Neon/SQLite and return a plain-text data snapshot for Gemma."""
    cmd = cmd.lower().strip()

    if cmd == "today":
        rows = _db_fetch(
            "SELECT symbol, direction, win, profit, score, market_session, regime "
            "FROM touch_trades WHERE DATE(timestamp)=CURRENT_DATE "
            "AND (is_backtest IS NULL OR is_backtest=0)"
        )
        if not rows:
            return "No trades recorded today yet."
        trades = len(rows)
        wins   = sum(1 for r in rows if r[2] == 1)
        losses = sum(1 for r in rows if r[2] == 0)
        pnl    = sum(float(r[3] or 0) for r in rows)
        wr     = wins / trades * 100 if trades else 0
        avg_sc = sum(float(r[4] or 0) for r in rows) / trades
        sess_stats: dict = {}
        for r in rows:
            s = sess_stats.setdefault(r[5] or "Unknown", {"w": 0, "l": 0, "pnl": 0.0})
            if r[2] == 1: s["w"] += 1
            else: s["l"] += 1
            s["pnl"] += float(r[3] or 0)
        sess_lines = "\n".join(
            f"  {k}: {v['w']}W / {v['l']}L  P&L=${v['pnl']:+.2f}"
            for k, v in sorted(sess_stats.items(), key=lambda x: -x[1]["pnl"])
        )
        return (
            f"TODAY'S TRADING SUMMARY\n"
            f"Trades: {trades}  |  Wins: {wins}  |  Losses: {losses}\n"
            f"Win Rate: {wr:.1f}%  |  Total P&L: ${pnl:+.2f}\n"
            f"Avg Score: {avg_sc:.1f}/100\n"
            f"By Session:\n{sess_lines}"
        )

    elif cmd == "week":
        if USE_PG:
            rows = _db_fetch(
                "SELECT DATE(timestamp) as day, COUNT(*) as t, "
                "SUM(CASE WHEN win=1 THEN 1 ELSE 0 END) as w, SUM(profit) as pnl "
                "FROM touch_trades WHERE timestamp >= NOW() - INTERVAL '7 days' "
                "AND (is_backtest IS NULL OR is_backtest=0) "
                "GROUP BY DATE(timestamp) ORDER BY day DESC"
            )
        else:
            rows = _db_fetch(
                "SELECT DATE(timestamp) as day, COUNT(*) as t, "
                "SUM(CASE WHEN win=1 THEN 1 ELSE 0 END) as w, SUM(profit) as pnl "
                "FROM touch_trades WHERE timestamp >= DATE('now','-7 days') "
                "AND (is_backtest IS NULL OR is_backtest=0) "
                "GROUP BY DATE(timestamp) ORDER BY day DESC"
            )
        if not rows:
            return "No trades in the last 7 days."
        tt = tw = 0; tp = 0.0
        day_lines = []
        for r in rows:
            t, w = int(r[1]), int(r[2] or 0)
            p = float(r[3] or 0)
            day_lines.append(f"  {r[0]}: {t} trades  {w/t*100:.0f}% WR  ${p:+.2f}")
            tt += t; tw += w; tp += p
        return (
            f"7-DAY SUMMARY\n"
            f"Total: {tt} trades  {tw/tt*100:.1f}% WR  ${tp:+.2f} P&L\n"
            f"Daily breakdown:\n" + "\n".join(day_lines)
        )

    elif cmd == "symbol":
        sym = (arg or "R_75").upper()
        rows = _db_fetch(
            "SELECT win, profit, score, market_session, regime, direction "
            "FROM touch_trades WHERE symbol=? "
            "AND (is_backtest IS NULL OR is_backtest=0) ORDER BY id DESC LIMIT 50",
            (sym,)
        )
        if not rows:
            return f"No trades found for {sym}."
        t = len(rows); w = sum(1 for r in rows if r[0] == 1)
        pnl = sum(float(r[1] or 0) for r in rows)
        up  = sum(1 for r in rows if r[5] == "UP")
        reg: dict = {}
        for r in rows:
            s = reg.setdefault(r[4] or "Unknown", {"w":0,"l":0})
            if r[0]==1: s["w"]+=1
            else: s["l"]+=1
        reg_lines = "\n".join(
            f"  {k}: {v['w']}W/{v['l']}L  ({v['w']/(v['w']+v['l'])*100:.0f}% WR)"
            for k, v in reg.items()
        )
        return (
            f"SYMBOL: {sym}  (last {t} trades)\n"
            f"Win Rate: {w/t*100:.1f}%  |  P&L: ${pnl:+.2f}\n"
            f"Direction: {up} UP / {t-up} DOWN\n"
            f"By Regime:\n{reg_lines}"
        )

    elif cmd in ("why-loss", "whyloss"):
        rows = _db_fetch(
            "SELECT symbol, profit, score, market_session, regime, direction, "
            "COALESCE(rsi,0.5)*100, COALESCE(adx,0)*100, COALESCE(bb_position,0.5) "
            "FROM touch_trades WHERE win=0 "
            "AND (is_backtest IS NULL OR is_backtest=0) ORDER BY id DESC LIMIT 20"
        )
        if not rows:
            return "No losses found in DB."
        avg_sc  = sum(float(r[2] or 0) for r in rows) / len(rows)
        avg_rsi = sum(float(r[6] or 50) for r in rows) / len(rows)
        avg_adx = sum(float(r[7] or 0)  for r in rows) / len(rows)
        sess: dict = {}
        for r in rows: sess[r[3] or "?"] = sess.get(r[3] or "?", 0) + 1
        top_5 = "\n".join(
            f"  {r[0]} {r[5]} score={r[2]} sess={r[3]} regime={r[4]} P&L=${float(r[1] or 0):+.2f}"
            for r in rows[:5]
        )
        return (
            f"RECENT LOSSES ANALYSIS (last {len(rows)})\n"
            f"Avg score at loss: {avg_sc:.1f}/100\n"
            f"Avg RSI at loss: {avg_rsi:.1f}  |  Avg ADX: {avg_adx:.1f}\n"
            f"Session distribution: {dict(sorted(sess.items(), key=lambda x:-x[1]))}\n"
            f"Last 5 losses:\n{top_5}"
        )

    elif cmd in ("best-session", "bestsession"):
        rows = _db_fetch(
            "SELECT market_session, COUNT(*) as t, "
            "SUM(CASE WHEN win=1 THEN 1 ELSE 0 END) as w, SUM(profit) as pnl "
            "FROM touch_trades WHERE (is_backtest IS NULL OR is_backtest=0) "
            "GROUP BY market_session ORDER BY pnl DESC"
        )
        if not rows:
            return "No session data yet."
        lines = ["SESSION PERFORMANCE"]
        for r in rows:
            t, w = int(r[1]), int(r[2] or 0)
            lines.append(
                f"  {r[0] or 'Unknown':<18}: {t:3d} trades  {w/t*100:5.1f}% WR  ${float(r[3] or 0):+.2f}"
            )
        return "\n".join(lines)

    elif cmd == "market":
        with _lock:
            snap = {k: dict(v) for k, v in indicators.items()}
        lines = ["CURRENT MARKET CONDITIONS"]
        for sym, ind in sorted(snap.items()):
            if not ind.get("ready"):
                continue
            st = "↑" if ind.get("supertrend_dir", 0)==1 else "↓" if ind.get("supertrend_dir",0)==-1 else "–"
            lines.append(
                f"  {sym:<12} {ind.get('regime','?'):<17} "
                f"ADX={ind.get('adx') or 0:4.0f}  RSI={ind.get('rsi') or 0:4.0f}  "
                f"BB={ind.get('bb_position') or 0:.2f}  ST={st}"
            )
        return "\n".join(lines) if len(lines) > 1 else "Indicators not ready yet."

    elif cmd == "ml":
        cnt = _db_fetch(
            "SELECT COUNT(*) FROM touch_trades WHERE (is_backtest IS NULL OR is_backtest=0)"
        )
        total = int(cnt[0][0]) if cnt else 0
        with ml_lock:
            has_model = bool(ml_model or ml_models_per_class)
        fi = _feature_importance_history[-1] if _feature_importance_history else {}
        fi_lines = ""
        if fi:
            top = sorted(fi.get("importances", {}).items(), key=lambda x: -x[1])[:8]
            fi_lines = "\nFeature importances (last retrain):\n" + "\n".join(
                f"  {k:<20} {v*100:5.1f}%" for k, v in top
            )
        return (
            f"ML MODEL STATUS\n"
            f"DB trades: {total}  |  Model active: {'Yes' if has_model else 'No (<100 trades)'}\n"
            f"Raw features (V2.7): {len(ML_FEATURE_COLS)} stored  +  "
            f"{len(_ML_ENGINEERED_COLS)} engineered  =  "
            f"{len(ML_FEATURE_COLS)+len(_ML_ENGINEERED_COLS)} total\n"
            f"Features: {', '.join(ML_FEATURE_COLS)}\n"
            f"Engineered: {', '.join(_ML_ENGINEERED_COLS)}"
            f"{fi_lines}"
        )

    elif cmd == "improvements":
        rows = _db_fetch(
            "SELECT regime, market_session, "
            "SUM(CASE WHEN win=1 THEN 1 ELSE 0 END) as w, COUNT(*) as t, AVG(profit) as ap "
            "FROM touch_trades WHERE (is_backtest IS NULL OR is_backtest=0) "
            "GROUP BY regime, market_session HAVING COUNT(*) >= 5 "
            "ORDER BY (SUM(CASE WHEN win=1 THEN 1 ELSE 0 END)*1.0/COUNT(*)) ASC LIMIT 10"
        )
        if not rows:
            return "Not enough data yet (need ≥5 trades per regime+session combo)."
        lines = ["WEAKEST REGIME+SESSION COMBOS"]
        for r in rows:
            w, t = int(r[2] or 0), int(r[3])
            lines.append(
                f"  {r[0] or '?'} + {r[1] or '?'}: {t} trades  {w/t*100:.0f}% WR  avg ${float(r[4] or 0):+.2f}"
            )
        return "\n".join(lines)

    else:
        return (
            f"Unknown sub-command '{cmd}'.\n"
            f"Available: today · week · symbol <SYM> · why-loss · best-session · market · ml · improvements"
        )


async def _ai_ask_gemma(summary: str, question: str) -> str:
    """Send data summary + question to Gemma 3; return plain-text analysis."""
    if _GEMINI_CLIENT is None:
        return "⚠️ Gemma unavailable — GEMINI_API_KEY not set or google-genai not installed."
    prompt = (
        "You are a professional trading analyst reviewing a Deriv One-Touch options bot "
        "that trades synthetic indices (R_10, R_25, R_50, R_75, R_100, 1HZ10V–1HZ100V). "
        "The bot uses Supertrend, RSI, StochRSI, ADX, Bollinger Bands, market regime detection, "
        "and a GradientBoosting ML model.\n\n"
        f"DATA:\n{summary}\n\n"
        f"QUESTION: {question}\n\n"
        "Provide a concise, actionable analysis in 4–6 sentences. "
        "Be specific: identify what is working, what is not, and one concrete improvement."
    )
    try:
        selected_model = _discover_gemini_model()
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None,
            lambda: _GEMINI_CLIENT.models.generate_content(model=selected_model, contents=prompt)
        )
        return resp.text or "No response from AI model."
    except Exception as e:
        logger.error(f"AI model error (model={_gemini_model_cache.get('model', '?')}): {e}")
        # Force rediscovery on next call — the model may have been removed
        _gemini_model_cache.clear()
        return f"⚠️ AI error: {e}"


_AI_HELP = (
    "🤖 <b>AI Trading Analyst — Dynamic Model Selection</b>\n\n"
    "<code>/ai today</code>         — Today's performance analysis\n"
    "<code>/ai week</code>          — 7-day trend & momentum\n"
    "<code>/ai symbol R_75</code>   — Deep-dive on a symbol\n"
    "<code>/ai why-loss</code>      — Root-cause of recent losses\n"
    "<code>/ai best-session</code>  — Best/worst trading sessions\n"
    "<code>/ai market</code>        — Live regime overview per symbol\n"
    "<code>/ai ml</code>            — ML model health & features\n"
    "<code>/ai improvements</code>  — Weakest combos to avoid\n"
    "<code>/ai model</code>         — Show active AI model & cache status\n\n"
    "<i>Flow: query DB → summarise data → ask AI model → return analysis\n"
    "Model auto-discovered from Google AI on startup; re-checked every 4 hours.</i>"
)

_AI_QUESTIONS = {
    "today":        "Explain today's performance. What drove results? What should be changed?",
    "week":         "Analyse the 7-day trend. Is performance improving or declining? What pattern stands out?",
    "symbol":       "Is this symbol worth trading? Any session or regime preference? Avoid or lean in?",
    "why-loss":     "What indicator conditions appear most at loss time? What setup should the bot avoid?",
    "whyloss":      "What indicator conditions appear most at loss time? What setup should the bot avoid?",
    "best-session": "Which session is most profitable and why? Should any session be skipped?",
    "bestsession":  "Which session is most profitable and why? Should any session be skipped?",
    "market":       "Based on current regimes and indicators, which symbols look most favourable right now?",
    "ml":           "Is the ML model healthy? Are features balanced? What would improve model accuracy?",
    "improvements": "What are the weakest setups? Which regime+session combos should the bot reduce or skip?",
}

async def cmd_ai(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """AI analyst with dynamic model selection: /ai <sub-command> [arg]"""
    args = ctx.args or []
    if not args:
        await update.message.reply_text(_AI_HELP, parse_mode="HTML")
        return

    sub  = args[0].lower()
    arg2 = args[1] if len(args) > 1 else ""

    # ── /ai model — show currently selected AI model & cache status ───────────
    if sub == "model":
        current = _discover_gemini_model()
        expires = _gemini_model_cache.get("expires", 0)
        expires_in = max(0, int(expires - time.time()))
        h, m = divmod(expires_in // 60, 60)
        status = "✅ Client active" if _GEMINI_CLIENT else "❌ No API key (GEMINI_API_KEY not set)"
        await update.message.reply_text(
            f"🤖 <b>AI Model Status</b>\n\n"
            f"Selected:  <code>{_html.escape(current)}</code>\n"
            f"Client:    {status}\n"
            f"Cache:     expires in {h}h {m}m\n"
            f"Fallback:  <code>{_html.escape(_GEMINI_MODEL_FALLBACK)}</code>\n\n"
            f"<i>Discovery runs on startup and every 4 hours.\n"
            f"Use /ai model to force a status refresh.</i>",
            parse_mode="HTML",
        )
        return

    question = _AI_QUESTIONS.get(sub, f"Analyse the data and answer: {' '.join(args)}")
    if sub == "symbol" and arg2:
        question = f"Is {arg2.upper()} worth trading? Any session or regime preference?"

    current_model = _gemini_model_cache.get("model", _GEMINI_MODEL_FALLBACK)
    msg = await update.message.reply_text(
        f"🤖 Querying data and thinking… (model: <code>{_html.escape(current_model)}</code>)",
        parse_mode="HTML",
    )
    try:
        loop = asyncio.get_event_loop()
        summary  = await loop.run_in_executor(None, _ai_build_summary, sub, arg2)
        analysis = await _ai_ask_gemma(summary, question)
        used_model = _gemini_model_cache.get("model", current_model)
        reply = (
            f"📊 <b>Data summary:</b>\n"
            f"<pre>{_html.escape(summary[:700])}</pre>\n\n"
            f"🤖 <b>{_html.escape(used_model)} says:</b>\n{_html.escape(analysis)}"
        )
        await msg.edit_text(reply, parse_mode="HTML")
    except Exception as e:
        await msg.edit_text(f"❌ Error: <code>{_html.escape(str(e))}</code>", parse_mode="HTML")


# ══════════════════════════════════════════════════════════════════════
#  BACKUP HELPER  (runs in a background thread — safe to call from btn_handler)
# ══════════════════════════════════════════════════════════════════════
def _do_backup(what: str) -> None:
    """Send trades CSV and/or ML model to Telegram.  Runs in a daemon thread."""
    try:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")

        if what in ("backup_csv", "backup_all"):
            rows = _db_fetch("SELECT * FROM touch_trades ORDER BY id")
            if USE_PG:
                cols = ["id","timestamp","symbol","direction","barrier","stake","payout",
                        "profit","win","score","wick_atr_ratio","atr","atr_ma",
                        "ema_fast_slope","ema_slow_slope","ema_distance","market_session"]
            else:
                conn = sqlite3.connect("touch_trades.db")
                desc = conn.execute("PRAGMA table_info(touch_trades)").fetchall()
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
    # key          : (global_var_name,      min,     max,   step,  is_pct,  display_fn)
    "ml_gate"      : ("ML_CONFIDENCE_MIN",  0.60,    0.99,  0.01,  True,   lambda v: f"{v*100:.0f}%"),
    "ml2_gate"     : ("SECOND_ENTRY_ML_MIN",0.75,    0.99,  0.01,  True,   lambda v: f"{v*100:.0f}%"),
    "score"        : ("SCORE_THRESHOLD",    70,      99,    1,     False,  lambda v: f"{v}/100"),
    "stake"        : ("STAKE",              0.35,    500,   0.5,   False,  lambda v: f"${v:.2f}"),
    "duration"     : ("DURATION",           1,       60,    1,     False,  lambda v: f"{v} min"),
    "cooldown"     : ("COOLDOWN_MINUTES",   1,       120,   5,     False,  lambda v: f"{v} min"),
    "tp"           : ("DAILY_PROFIT_TARGET",1,       9999,  1,     False,  lambda v: f"${v:.2f}"),
    "sl"           : ("DAILY_LOSS_LIMIT",   -9999,  -1,    -1,    False,  lambda v: f"${v:.2f}"),
    "max_loss"     : ("MAX_CONSECUTIVE_LOSSES", 1,   20,    1,     False,  lambda v: f"{int(v)}"),
    "retrain"      : ("ML_RETRAIN_EVERY",   5,       5000,  10,    False,  lambda v: f"{int(v)} trades"),
    "ml_min"       : ("ML_MIN_TRADES",      20,      5000,  10,    False,  lambda v: f"{int(v)} trades"),
    "pause_min"    : ("PAUSE_MINUTES",      1,       1440,  5,     False,  lambda v: f"{int(v)} min"),
    "second_cd"    : ("SECOND_ENTRY_COOLDOWN", 1,    60,    1,     False,  lambda v: f"{int(v)} min"),
    "second_win"   : ("SECOND_ENTRY_WINDOW",   2,    120,   5,     False,  lambda v: f"{int(v)} min"),
}


def _apply_param(key: str, raw_val) -> tuple[bool, str]:
    """Apply a parameter change. Returns (ok, message)."""
    import sys as _sys
    global ML_CONFIDENCE_MIN, SECOND_ENTRY_ML_MIN, SCORE_THRESHOLD, STAKE
    global DURATION, COOLDOWN_MINUTES, DAILY_PROFIT_TARGET, DAILY_LOSS_LIMIT
    global MAX_CONSECUTIVE_LOSSES, ML_RETRAIN_EVERY, ML_MIN_TRADES, PAUSE_MINUTES
    global SECOND_ENTRY_COOLDOWN, SECOND_ENTRY_WINDOW

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
        "second_cd":  lambda v: setattr(_sys.modules[__name__], "SECOND_ENTRY_COOLDOWN", int(v)),
        "second_win": lambda v: setattr(_sys.modules[__name__], "SECOND_ENTRY_WINDOW", int(v)),
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
    global paused, pause_until, consecutive_losses, PROFIT_MIN, PROFIT_MAX
    q = update.callback_query
    d = q.data
    logger.info(f"Button pressed: {d!r}")
    try:
        await q.answer()
    except Exception as e:
        logger.debug(f"q.answer() failed (non-fatal): {e}")

    try:
        if d in ("main_menu", "refresh"):
            await q.edit_message_text(
                "🤖 <b>Deriv Touch Bot v1.0</b>  –  Select an option:",
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
            await q.edit_message_text(_performance_analytics_text(), reply_markup=_main_kb(), parse_mode="HTML")
        elif d == "patterns":
            await q.edit_message_text(_pattern_discovery_text(), reply_markup=_main_kb(), parse_mode="HTML")
        elif d == "gate_optimizer":
            await q.edit_message_text(_gate_optimizer_text(), reply_markup=_main_kb(), parse_mode="HTML")
        elif d == "ml_progress":
            with ml_lock:
                trained_on, model, training, total = ml_trained_on, ml_model, ml_training_active, ml_total_trades
            lines = [
                "🤖 <b>ML Training Progress</b>",
                "━━━━━━━━━━━━━━━━━━━━",
                "",
                _ml_progress_text(),
                "",
                f"Total trades logged : <b>{total}</b>",
                f"Min trades to train : <b>{ML_MIN_TRADES}</b>",
                f"Retrain every       : <b>{ML_RETRAIN_EVERY} trades</b>",
                f"Trades since retrain: <b>{max(0, total - (trained_on or 0))}</b>",
                f"Model active        : <b>{'Yes' if model else 'No'}</b>",
                f"Training in progress: <b>{'Yes ⏳' if training else 'No'}</b>",
                f"Confidence gate     : <b>≥{ML_CONFIDENCE_MIN*100:.0f}%</b>",
            ]
            await q.edit_message_text("\n".join(lines), reply_markup=_main_kb(), parse_mode="HTML")
        elif d == "health":
            await q.edit_message_text(_health_text(), reply_markup=_main_kb(), parse_mode="HTML")
        elif d == "feat_importance":
            def _fi_text():
                if not _feature_importance_history:
                    return "🤖 <b>Feature Importance History</b>\n\nNo retrains yet. Need 100+ trades."
                lines = ["🤖 <b>Feature Importance History</b>"]
                for entry in _feature_importance_history[-5:][::-1]:
                    lines.append(f"\n<b>{entry['ts']}</b>  ({entry['total']} trades)")
                    sorted_imp = sorted(entry["importances"].items(), key=lambda x: x[1], reverse=True)
                    for col, pct in sorted_imp[:6]:
                        bar = "█" * int(pct * 20)
                        lines.append(f"  {col:<20} {pct*100:5.1f}% {bar}")
                return "\n".join(lines)
            text = await asyncio.get_event_loop().run_in_executor(None, _fi_text)
            await q.edit_message_text(text, reply_markup=_main_kb(), parse_mode="HTML")
        elif d == "export_csv_quick":
            await q.edit_message_text(
                "⏳ Generating CSV export…",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back", callback_data="main_menu")]]),
                parse_mode="HTML",
            )
            # Reuse existing export logic via a thread
            async def _do_export_quick():
                try:
                    rows = _db_fetch("SELECT * FROM touch_trades ORDER BY id")
                    if USE_PG:
                        cols = ["id","timestamp","symbol","direction","barrier","stake","payout",
                                "profit","win","score","wick_atr_ratio","atr","atr_ma",
                                "ema_fast_slope","ema_slow_slope","ema_distance","market_session"]
                    else:
                        import sqlite3 as _sl
                        conn2 = _sl.connect("touch_trades.db")
                        desc2 = conn2.execute("PRAGMA table_info(touch_trades)").fetchall()
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
            # Widen/raise the target band, but keep min ≤ max and within hard limits
            new_min = round(min(0.85, PROFIT_MIN + delta), 2)
            new_max = round(min(1.20, PROFIT_MAX + delta), 2)
            new_min = min(new_min, new_max)
            PROFIT_MIN, PROFIT_MAX = new_min, new_max
            await q.edit_message_text(
                f"💸 <b>Payout band raised</b>\n"
                f"Target profit : <b>${PROFIT_MIN:.2f} – ${PROFIT_MAX:.2f}</b> per trade\n"
                f"<i>Higher band = harder barriers, usually better payout.</i>",
                reply_markup=_main_kb(), parse_mode="HTML",
            )
            _log(f"💸 Payout band raised to ${PROFIT_MIN:.2f}–${PROFIT_MAX:.2f}")
        elif d == "payout_down":
            delta = 0.05
            new_min = round(max(0.10, PROFIT_MIN - delta), 2)
            new_max = round(max(0.50, PROFIT_MAX - delta), 2)
            new_min = min(new_min, new_max)
            PROFIT_MIN, PROFIT_MAX = new_min, new_max
            await q.edit_message_text(
                f"💸 <b>Payout band lowered</b>\n"
                f"Target profit : <b>${PROFIT_MIN:.2f} – ${PROFIT_MAX:.2f}</b> per trade\n"
                f"<i>Lower band = easier barriers, easier fills.</i>",
                reply_markup=_main_kb(), parse_mode="HTML",
            )
            _log(f"💸 Payout band lowered to ${PROFIT_MIN:.2f}–${PROFIT_MAX:.2f}")
        elif d == "noop":
            await q.answer()   # acknowledge silently — spacer buttons
        elif d == "settings":
            # V3.0: landing page is now a submenu, not the raw adj keyboard
            await q.edit_message_text(
                _settings_text(),
                reply_markup=_settings_menu_kb(),
                parse_mode="HTML",
            )
        elif d == "settings_adj":
            await q.edit_message_text(
                _settings_text(),
                reply_markup=_settings_adj_kb(),
                parse_mode="HTML",
            )
        elif d == "debug_state":
            # Reuse cmd_debug logic but as an inline button
            import sys as _sys2
            _mod2 = _sys2.modules[__name__]
            now_dt   = datetime.now(timezone.utc)
            session  = _get_session_name(now_dt)
            prof     = _get_session_profile(session)
            with _lock:
                _pnl, _wc, _lc, _cl, _dt, _paused_now = (
                    total_pnl, win_count, loss_count,
                    consecutive_losses, daily_trades, paused
                )
            with ml_lock:
                _gate_g  = ML_CONFIDENCE_MIN
                _gate_s  = prof.get("ml_gate", ML_CONFIDENCE_MIN)
                _opt_t   = _ml_optimal_threshold
                _trained = ml_trained_on
                _total   = ml_total_trades
                _busy    = ml_training_active
                _ready   = ml_model is not None
            _until = max(0, ML_RETRAIN_EVERY - max(0, _total - _trained))
            lines = [
                f"🔍 <b>DEBUG STATE</b>  {now_dt.strftime('%H:%M:%S UTC')}",
                "",
                f"<b>Session</b>       {session}  [{prof.get('mode','Normal')}]",
                f"<b>ML Gate</b>       Global {_gate_g*100:.0f}%  |  Session {_gate_s*100:.0f}%",
                f"<b>Opt Threshold</b> {_opt_t*100:.0f}%  (walk-forward, raises only)",
                f"<b>Model</b>         {'✅ ready' if _ready else '⏳ not trained'}  "
                f"(on {_trained}, now {_total}  — retrain in ~{_until} trades)",
                f"<b>Training</b>      {'🔄 in progress' if _busy else 'idle'}",
                "",
                f"<b>TP / SL</b>       ${prof.get('tp', DAILY_PROFIT_TARGET):.1f}"
                f" / ${prof.get('sl', DAILY_LOSS_LIMIT):.1f}",
                f"<b>Cooldown</b>      {prof.get('cooldown', COOLDOWN_MINUTES)} min",
                f"<b>Max Trades</b>    {prof.get('max_trades','unlimited')}",
                "",
                f"<b>Session P&L</b>   {'+'if _pnl>=0 else''}${_pnl:.2f}  "
                f"({_wc}W / {_lc}L  {_wc/max(_wc+_lc,1)*100:.0f}%WR)",
                f"<b>Cons Losses</b>   {_cl}  |  <b>Paused</b> {'⏸ YES' if _paused_now else 'no'}",
            ]
            _dkb = InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Settings", callback_data="settings")]])
            await q.edit_message_text("\n".join(lines), reply_markup=_dkb, parse_mode="HTML")
        elif d == "profile_stats":
            # Inline version of /profile
            lines = ["📊 <b>Profiles &amp; Rolling Stats</b>", ""]
            for sn, sp in SESSION_PROFILES.items():
                emoji  = SESSION_EMOJIS.get(sn, "")
                drec   = _rolling_session_deque.get(sn)
                st20   = _rolling_stats_for(drec, 20) if drec else {"n":0}
                st50   = _rolling_stats_for(drec, 50) if drec else {"n":0}
                st_all = _rolling_stats_for(drec)     if drec else {"n":0}
                def _wr(s): return f"{s['wr']*100:.0f}%" if s.get("wr") is not None else "—"
                lines.append(
                    f"{emoji} <b>{sn}</b>  [{sp['mode']}]\n"
                    f"  Gate {sp['ml_gate']*100:.0f}%  TP ${sp['tp']}  SL ${sp['sl']}\n"
                    f"  WR: L20 <b>{_wr(st20)}</b>  L50 <b>{_wr(st50)}</b>  "
                    f"All <b>{_wr(st_all)}</b>  ({st_all['n']} trades)"
                )
            _pkb = InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Settings", callback_data="settings")]])
            await q.edit_message_text("\n".join(lines), reply_markup=_pkb, parse_mode="HTML")
        elif d == "setsession_list":
            lines = ["📋 <b>Session Profiles</b>  <i>(edit via /setsession)</i>"]
            for sn, sp in SESSION_PROFILES.items():
                emoji = SESSION_EMOJIS.get(sn, "")
                lines.append(
                    f"\n{emoji} <b>{sn}</b>\n"
                    f"  ml_gate={sp['ml_gate']*100:.0f}%  "
                    f"tp=${sp['tp']}  sl=${sp['sl']}  [{sp['mode']}]\n"
                    f"  <i>/setsession \"{sn}\" ml_gate 75</i>"
                )
            _skb = InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Settings", callback_data="settings")]])
            await q.edit_message_text("\n".join(lines), reply_markup=_skb, parse_mode="HTML")
        elif d == "gate_optimizer":
            await q.edit_message_text(
                "⏳ <b>Running Gate Optimizer…</b>\n"
                "<i>Replaying all logged signals, sweeping gates 55–90%.</i>",
                reply_markup=InlineKeyboardMarkup([[
                    InlineKeyboardButton("🔙 Settings", callback_data="settings")
                ]]),
                parse_mode="HTML",
            )
            def _run_gate_opt():
                return _gate_optimizer_text()
            text = await asyncio.get_event_loop().run_in_executor(None, _run_gate_opt)
            _gkb = InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Settings", callback_data="settings")]])
            await q.message.reply_text(text, reply_markup=_gkb, parse_mode="HTML")
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
                "🤖 <b>Deriv Touch Bot v1.0</b>  –  Select an option:",
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

    # Compute next retrain based on real trade count (already capped at startup)
    with ml_lock:
        _hb_t, _hb_trn = ml_total_trades, ml_trained_on
    _retrain_nxt = max(0, ML_RETRAIN_EVERY - max(0, _hb_t - _hb_trn))

    msg = (
        f"❤️ <b>Hourly Heartbeat</b>  ·  {now.strftime('%H:%M UTC')}  ·  Deriv Server Time\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"{'⏸ PAUSED' if is_paused else '▶ RUNNING'}  ·  "
        f"{SESSION_EMOJIS.get(mkt_s,'')} {mkt_s}  ·  Active: {ac}\n"
        "\n"
        "<b>💰 Session P&amp;L</b>\n"
        f"  {'+' if pnl >= 0 else ''}${pnl:.2f}  ({total} trades  {wc}W/{lc}L  {wr:.0f}%WR)\n"
        f"  Drawdown: ${cur_dd:.2f}  ·  Max DD: ${mdd:.2f}\n"
        f"  Streak: {'🔴×' + str(cl) if cl else '🟢 None'}\n"
        "\n"
        f"{session_line}"
        "<b>🤖 ML Engine</b>\n"
        f"  {ml_line}  ·  next retrain in {_retrain_nxt} trades\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
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
    t = Text("  DERIV TOUCH BOT  v1.0   ", style="bold cyan")
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
        stake   = deriv_float(info.get("stake", STAKE))
        payout  = deriv_float(info.get("payout", stake + TARGET_PROFIT))
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
                        f"  Payout: ${PROFIT_MIN:.2f}–${PROFIT_MAX:.2f}  ·  ATR×{ATR_BARRIER_MULT}\n"
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
        app.add_handler(CommandHandler("ai",      cmd_ai))
        app.add_handler(CommandHandler("start",   cmd_start))
        app.add_handler(CommandHandler("status",  cmd_status))
        app.add_handler(CommandHandler("pnl",     cmd_pnl))
        app.add_handler(CommandHandler("pause",   cmd_pause))
        app.add_handler(CommandHandler("resume",  cmd_resume))
        app.add_handler(CommandHandler("history", cmd_history))
        app.add_handler(CommandHandler("session", cmd_session))
        app.add_handler(CommandHandler("alltime", cmd_alltime))
        app.add_handler(CommandHandler("patterns", cmd_patterns))
        app.add_handler(CommandHandler("analytics", cmd_analytics))
        app.add_handler(CommandHandler("export",  cmd_export))
        app.add_handler(CommandHandler("backup",  cmd_backup))
        app.add_handler(CommandHandler("set",        cmd_set))
        app.add_handler(CommandHandler("debug",      cmd_debug))
        app.add_handler(CommandHandler("profile",    cmd_profile))
        app.add_handler(CommandHandler("setsession", cmd_setsession))
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
            f"\n  DERIV TOUCH BOT  v1.0  –  Supertrend Edition\n"
            f"  Loading history for {len(SYMBOLS)} symbols…\n",
            style="bold cyan",
        )),
        border_style="blue", box=box.DOUBLE_EDGE,
    ))

    # V3.0: load per-session profile config (JSON file or built-in defaults)
    _load_session_profiles()

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

    # Restore live ML-confidence / signal-score performance dashboards from Neon
    # so a restart no longer wipes them (previously in-memory only).
    _ml_stats_load_all()

    # ── Sync ml_total_trades with the real DB row count so the "retrain in N"
    # counter starts from the correct baseline instead of 0 after a restart.
    # ml_trained_on is loaded from the pickle (e.g. 100) but ml_total_trades
    # would otherwise be 0, making since=max(0,0-100)=0 → "retrain in 50".
    # Also cap ml_trained_on to the real DB count: bootstrap models are trained
    # on historical candle simulations, which is a much larger number than real
    # trades. Without this cap the counter can stay stuck at 50 forever.
    global ml_total_trades, ml_trained_on  # must declare global here — in main(), not module scope
    try:
        _db_cnt_rows = _db_fetch("SELECT COUNT(*) FROM touch_trades WHERE win IN (0,1)")
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

    # ML bootstrap from candle history is DISABLED — watch-only mode until
    # ML_MIN_TRADES (100) real touch_trades are recorded, then trains automatically.
    # threading.Thread(target=_ml_bootstrap_from_history, daemon=True, name="MLBootstrap").start()

    # ── Indicator computation executor: serialises heavy pandas work across
    # all 15 symbols so they don't all spike CPU simultaneously at candle-close.
    global _indicator_executor
    _indicator_executor = ThreadPoolExecutor(
        max_workers=2, thread_name_prefix="IndComp"
    )

    # Start remaining core threads
    _watch_thread(_db_writer,                   name="DBWriter")
    for i, sym in enumerate(SYMBOLS):
        _watch_thread(_ws_thread, args=(sym,),  name=f"WS-{sym}")
        # Stagger starts to avoid a startup burst of OTP REST requests on Render.
        if i < len(SYMBOLS) - 1:
            time.sleep(0.3)
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
            "Watch touch.log or Telegram for status.[/yellow]"
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

