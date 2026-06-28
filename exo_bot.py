#!/usr/bin/env python3
"""
Signal Bot v19 – Pro Hedge Signal System + Heartbeat
======================================================
- A+ straddle alerts every 15 min
- Heartbeat every 10 min (top near‑A+ coins)
- Fully stable for Termux & Render
"""
import os, time, asyncio, logging, threading, sys, json
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import requests
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from flask import Flask

# ══════════════════ CREDENTIALS ══════════════════
BOT_TOKEN = "8908331931:AAFHiBW7k_RSRENrhegpsqg8E-gl_nAaLx0"
CHAT_ID   = 6400145232

# ══════════════════ CONFIG ══════════════════
CANDIDATES = [
    "PEPEUSDT","BONKUSDT","WIFUSDT","FLOKIUSDT","BOMEUSDT",
    "MEMEUSDT","DOGEUSDT","SHIBUSDT","LUNCUSDT","JUPUSDT",
    "BANANAS31USDT","POPCATUSDT","MYROUSDT","WENUSDT","MOGUSDT",
    "TURBOUSDT","MEWUSDT","SLERFUSDT","SAMOUSDT","BRETTUSDT",
    "PONKEUSDT","BODENUSDT","TREMPUSDT","DEGENUSDT","ANDYUSDT",
    "WOLFUSDT","HARAMBEUSDT","NUBUSDT","PENGUSDT","MOODENGUSDT",
    "GOATSEUSDT","SIGMAUSDT","BITCOINUSDT","SPX6900USDT","GIGAUSDT",
    "MOTHERUSDT","BURGERUSDT","CATUSDT","DOGUSDT","AIDOGEUSDT",
    "BABYDOGEUSDT","CORGIAIUSDT","KISHUUSDT","SPXUSDT","HYPEUSDT",
    "APEUSDT","PEOPLEUSDT","LADYSUSDT","JENNERUSDT","MAGAUSDT",
    "TRUMPUSDT","BIDENUSDT","ELONUSDT","FOURUSDT","SATSUSDT",
    "RATSUSDT","ORDIUSDT","SILLYUSDT","DRAGONUSDT","KUJIUSDT",
    "MONUSDT","PORKUSDT","PUNDUUSDT","ONDOUSDT","STXUSDT",
    "INJUSDT","NEARUSDT","FETUSDT","GALAUSDT","SANDUSDT",
    "AVAXUSDT","SUIUSDT","ARBUSDT","OPUSDT","DOTUSDT",
    "LINKUSDT","ICPUSDT","TRXUSDT","SOLUSDT","TIAUSDT",
    "SEIUSDT","APTUSDT","RNDRUSDT","WOJAKUSDT","PEPE2USDT",
    "BENUSDT","PSYOPUSDT","LOLUSDT","BOBOMUSDT","DINGERUSDT",
    "MILADYUSDT","SAITAMAUSDT","VOLTUSDT",
]
BTC_PAIR = "BTCUSDT"

MEXC_URL = "https://api.mexc.com/api/v3"
SCAN_INTERVAL_MINUTES = 15
HEARTBEAT_INTERVAL_MINUTES = 10
HISTORY_DAYS = 2
ATR_PERIOD = 14
LOOKBACK_RANGE = 15
MIN_ADX = 10
DOLLAR_VOLUME_THRESHOLD = 50_000
FAIL_THRESHOLD = 3
RETRY_HOURS = 12
LOG_FILE = "bot.log"

# ── Pro filters ──
COOLDOWN_MINUTES = 30
MAX_ATR_PCT = 8.0
MAX_PRICE_CHANGE_15M = 5.0
BB_EXPANSION_MIN = 1.2
BB_EXPANSION_MAX = 2.5
MIN_CONSECUTIVE_VOL_SURGE = 2
BREAKOUT_MIN = 0.6
CHOPPY_ADX_THRESHOLD = 15
CHOPPY_ALERT_REDUCTION = 0.3
RELIABILITY_WINDOW = 20
RELIABILITY_WEIGHT = 0.05

# ══════════════════ LOGGING ══════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)]
)

# ══════════════════ HEALTH SERVER (optional) ══════════════════
health_app = Flask(__name__)
@health_app.route('/health')
def health(): return 'OK', 200
def run_health(): health_app.run(host='0.0.0.0', port=10000, debug=False, use_reloader=False)

# ══════════════════ SYMBOL MANAGEMENT ══════════════════
pairs_lock = threading.Lock()
inactive_symbols = {}

def get_active_pairs():
    with pairs_lock:
        return PAIRS.copy()

def mark_inactive(sym):
    with pairs_lock:
        if sym in PAIRS:
            PAIRS.remove(sym)
            inactive_symbols[sym] = datetime.now(timezone.utc)
            logging.warning(f"Marked {sym} inactive")

def reactivate_if_ready():
    now = datetime.now(timezone.utc)
    with pairs_lock:
        for sym, since in list(inactive_symbols.items()):
            if (now - since).total_seconds() > RETRY_HOURS * 3600:
                PAIRS.append(sym)
                del inactive_symbols[sym]
                logging.info(f"Reactivated {sym}")

def get_valid_symbols():
    try:
        resp = requests.get(f"{MEXC_URL}/defaultSymbols", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            all_syms = set(data.get("data", []))
            valid = [s for s in CANDIDATES if s in all_syms]
            logging.info(f"Valid symbols: {len(valid)}/{len(CANDIDATES)}")
            return valid
    except Exception as e:
        logging.error(f"Symbol validation failed: {e}")
    return CANDIDATES

PAIRS = []

# ══════════════════ DATA CACHE ══════════════════
data_cache = {}
ind_cache = {}
last_fetch = {}
fail_count = defaultdict(int)
data_lock = threading.Lock()

def fetch_candles(sym, start_time, end_time, retries=2):
    for attempt in range(retries+1):
        try:
            cursor = int(start_time.timestamp() * 1000)
            end_ms = int(end_time.timestamp() * 1000)
            rows = []
            sess = requests.Session()
            sess.headers.update({"User-Agent": "Mozilla/5.0"})
            while cursor < end_ms:
                r = sess.get(f"{MEXC_URL}/klines", params={
                    "symbol": sym, "interval": "1m",
                    "startTime": cursor, "endTime": end_ms, "limit": 500
                }, timeout=10)
                if r.status_code == 429:
                    logging.warning(f"Rate limit hit for {sym}, sleeping 30s")
                    time.sleep(30)
                    continue
                r.raise_for_status()
                data = r.json()
                if not isinstance(data, list) or not data: break
                rows.extend(data)
                cursor = data[-1][0] + 60_000
                if len(data) < 500: break
                time.sleep(0.02)
            if not rows:
                return pd.DataFrame()
            col_names = ["OpenTime","Open","High","Low","Close","Volume","CloseTime","QuoteVolume"]
            df = pd.DataFrame(rows, columns=col_names)
            df["OpenTime"] = pd.to_datetime(df["OpenTime"], unit="ms", utc=True)
            df.set_index("OpenTime", inplace=True)
            for c in ["Open","High","Low","Close","Volume"]:
                df[c] = pd.to_numeric(df[c])
            return df[["Open","High","Low","Close","Volume"]].sort_index()
        except Exception as e:
            if attempt < retries:
                time.sleep(2 ** attempt)
            else:
                logging.error(f"Failed to fetch {sym}: {e}")
                return pd.DataFrame()

def initial_data_download():
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=HISTORY_DAYS)
    active_syms = get_active_pairs() + [BTC_PAIR]
    logging.info(f"Downloading {HISTORY_DAYS}d history for {len(active_syms)-1} pairs + BTC...")
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {}
        for sym in active_syms:
            futures[pool.submit(fetch_candles, sym, start, end)] = sym
        downloaded = 0
        for fut in as_completed(futures):
            sym = futures[fut]
            df = fut.result()
            if not df.empty:
                with data_lock:
                    data_cache[sym] = df
                    ind_cache[sym] = add_indicators(df)
                    last_fetch[sym] = df.index.max()
                downloaded += 1
            else:
                logging.warning(f"No data for {sym}")
    logging.info(f"Initial download done. {downloaded}/{len(active_syms)} loaded.")

def incremental_update():
    now = datetime.now(timezone.utc)
    reactivate_if_ready()
    active_syms = get_active_pairs() + [BTC_PAIR]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {}
        for sym in active_syms:
            since = last_fetch.get(sym, now - timedelta(days=HISTORY_DAYS))
            since = since + timedelta(minutes=1) if since else since
            if since >= now: continue
            futures[pool.submit(fetch_candles, sym, since, now)] = sym
        for fut in as_completed(futures):
            sym = futures[fut]
            df_new = fut.result()
            if df_new.empty:
                fail_count[sym] += 1
                if fail_count[sym] >= FAIL_THRESHOLD:
                    mark_inactive(sym)
                continue
            else:
                fail_count[sym] = 0
            with data_lock:
                if sym in data_cache:
                    combined = pd.concat([data_cache[sym], df_new]).sort_index()
                    combined = combined[~combined.index.duplicated(keep='last')]
                    cutoff = now - timedelta(days=HISTORY_DAYS)
                    combined = combined[combined.index >= cutoff]
                    data_cache[sym] = combined
                else:
                    data_cache[sym] = df_new
                ind_cache[sym] = add_indicators(data_cache[sym])
                last_fetch[sym] = data_cache[sym].index.max()

# ══════════════════ INDICATORS ══════════════════
def add_indicators(df):
    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"]

    # True Range & ATR
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(ATR_PERIOD, min_periods=1).mean()
    df["ATR_pct"] = df["ATR"] / close * 100

    # ATR Expansion
    df["ATR_30min_ago"] = df["ATR"].shift(30)
    df["atr_expansion"] = df["ATR"] / df["ATR_30min_ago"].replace(0, np.nan)
    df["atr_expansion"] = df["atr_expansion"].clip(upper=5)

    # Relative Volume
    df["vol_avg"] = vol.rolling(20, min_periods=1).mean()
    df["vol_ratio"] = vol / df["vol_avg"]

    # Dollar volume surge
    df["dollar_vol"] = close * vol
    df["dollar_vol_avg20"] = df["dollar_vol"].rolling(20, min_periods=1).mean()
    df["vol_surge"] = df["dollar_vol"] / df["dollar_vol_avg20"]

    # Price velocity
    df["roc1"] = close.pct_change(1).abs() * 100
    df["roc3"] = close.pct_change(3).abs() * 100
    df["roc5"] = close.pct_change(5).abs() * 100
    df["price_velocity"] = (df["roc1"] + df["roc3"] + df["roc5"]) / 3

    # Breakout
    df["range_high_30"] = high.rolling(30, min_periods=1).max()
    df["range_low_30"] = low.rolling(30, min_periods=1).min()
    df["range_30"] = df["range_high_30"] - df["range_low_30"]
    df["breakout"] = (close - df["range_low_30"]) / df["range_30"].replace(0, np.nan)
    df["breakout"] = df["breakout"].clip(0, 1)

    # ADX
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0), index=df.index)
    atr_adx = tr.rolling(ATR_PERIOD, min_periods=1).mean()
    plus_di = 100 * plus_dm.rolling(ATR_PERIOD, min_periods=1).mean() / atr_adx
    minus_di = 100 * minus_dm.rolling(ATR_PERIOD, min_periods=1).mean() / atr_adx
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-6)
    df["ADX"] = dx.rolling(ATR_PERIOD, min_periods=1).mean()

    # Bollinger Bands
    df["sma20"] = close.rolling(20, min_periods=1).mean()
    df["std20"] = close.rolling(20, min_periods=1).std()
    df["upper_bb"] = df["sma20"] + 2 * df["std20"]
    df["lower_bb"] = df["sma20"] - 2 * df["std20"]
    df["bb_width"] = (df["upper_bb"] - df["lower_bb"]) / df["sma20"]
    df["bb_width_20min_ago"] = df["bb_width"].shift(20)
    df["bb_expansion"] = df["bb_width"] / df["bb_width_20min_ago"].replace(0, np.nan)
    df["bb_expansion"] = df["bb_expansion"].clip(upper=5)

    # 15‑min price change for no‑trade filter
    df["chg_15min"] = close.pct_change(periods=15) * 100

    return df

# ══════════════════ ORDER BOOK ══════════════════
def fetch_order_book(sym, retries=2):
    for attempt in range(retries+1):
        try:
            r = requests.get(f"{MEXC_URL}/depth", params={"symbol": sym, "limit": 20}, timeout=5)
            if r.status_code == 429:
                time.sleep(30)
                continue
            r.raise_for_status()
            data = r.json()
            bids = sum(float(b[1]) for b in data.get("bids", []))
            asks = sum(float(a[1]) for a in data.get("asks", []))
            total = bids + asks
            if total == 0:
                return 0, "Neutral"
            imbalance = (bids - asks) / total
            abs_imbalance = abs(imbalance)
            if imbalance > 0.15:
                side = "Bullish"
            elif imbalance < -0.15:
                side = "Bearish"
            else:
                side = "Neutral"
            return abs_imbalance, side
        except:
            if attempt < retries:
                time.sleep(1)
            else:
                return 0, "Neutral"

# ══════════════════ COIN MEMORY & RELIABILITY ══════════════════
reliability_store = defaultdict(lambda: deque(maxlen=RELIABILITY_WINDOW))
cooldown_tracker = {}

def update_reliability(sym, was_successful):
    reliability_store[sym].append(was_successful)

def get_reliability(sym):
    deq = reliability_store[sym]
    if not deq:
        return 0.5
    return sum(deq) / len(deq)

# ══════════════════ A+ SETUP CLASSIFIER ══════════════════
def check_ap_criteria(row, df):
    """Return a dict with pass/fail and missing items."""
    reasons = []
    passed = True

    bb_exp = row["bb_expansion"]
    if pd.isna(bb_exp) or bb_exp < BB_EXPANSION_MIN:
        reasons.append(f"BBexp({bb_exp:.1f}→{BB_EXPANSION_MIN})")
        passed = False
    elif bb_exp > BB_EXPANSION_MAX:
        reasons.append(f"BBexp overextended({bb_exp:.1f}>{BB_EXPANSION_MAX})")
        passed = False

    atr_exp = row["atr_expansion"]
    if pd.isna(atr_exp) or atr_exp < 1.1:
        reasons.append(f"ATRexp({atr_exp:.1f}→1.1)")
        passed = False

    recent_vol = df["vol_surge"].tail(10)
    consecutive = sum(1 for v in reversed(recent_vol.values) if v > 2.0)
    if consecutive < MIN_CONSECUTIVE_VOL_SURGE:
        reasons.append(f"VolSurge({consecutive}/{MIN_CONSECUTIVE_VOL_SURGE})")
        passed = False

    brk = row["breakout"]
    if pd.isna(brk) or brk < BREAKOUT_MIN:
        reasons.append(f"Break({brk:.2f}→{BREAKOUT_MIN})")
        passed = False

    adx_now = row["ADX"]
    adx_5min_ago = df["ADX"].iloc[-6] if len(df) > 5 else adx_now
    if adx_now <= adx_5min_ago:
        reasons.append("ADX flat")
        passed = False

    if row["ATR_pct"] > MAX_ATR_PCT:
        reasons.append(f"ATR% high({row['ATR_pct']:.1f}%>{MAX_ATR_PCT}%)")
        passed = False

    if abs(row["chg_15min"]) > MAX_PRICE_CHANGE_15M:
        reasons.append("Price overextended")
        passed = False

    return passed, reasons

def get_near_ap_coins(top_n=5):
    """Return coins sorted by composite score, with their missing criteria."""
    active_syms = get_active_pairs()
    results = []
    with data_lock:
        for sym in active_syms:
            if sym not in ind_cache: continue
            df = ind_cache[sym]
            if df.empty: continue
            row = df.iloc[-1]
            if pd.isna(row["ADX"]): continue
            if row["ADX"] < MIN_ADX: continue
            if row["dollar_vol_avg20"] < DOLLAR_VOLUME_THRESHOLD: continue

            # Compute composite score (same as A+ but without filtering)
            rvol_n = min(row["vol_ratio"] / 8, 1.0)
            atr_exp_n = min(row["atr_expansion"] / 3, 1.0)
            bb_exp_n = min(row["bb_expansion"] / 3, 1.0)
            vel_n = min(row["price_velocity"] / 5, 1.0)
            brk_n = row["breakout"]
            surge_n = min(row["vol_surge"] / 8, 1.0)
            adx_n = min(row["ADX"] / 50, 1.0)
            # order book not needed for near-A+; skip for speed
            rel_n = get_reliability(sym)
            score = (0.20 * rvol_n + 0.15 * atr_exp_n + 0.20 * bb_exp_n +
                     0.15 * vel_n + 0.10 * brk_n + 0.10 * surge_n +
                     0.05 * 0 + 0.03 * rel_n + 0.02 * adx_n) * 100

            passed, reasons = check_ap_criteria(row, df)
            results.append((sym, score, passed, reasons))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]

# ══════════════════ A+ STRADDLE ALERT (existing) ══════════════════
def is_ap_setup(row, df):
    passed, reasons = check_ap_criteria(row, df)
    if passed:
        bb_exp = row["bb_expansion"]
        setup_type = "SQUEEZE BREAKOUT" if bb_exp >= 1.5 else "VOLATILITY EXPANSION"
        return True, setup_type
    return False, reasons[0] if reasons else ""

def get_ap_signals(n=5):
    now = datetime.now(timezone.utc)
    active_syms = get_active_pairs()

    # Fetch order books only for A+ candidates (optimized)
    regime = get_market_regime()
    candidates = []
    with data_lock:
        for sym in active_syms:
            if sym not in ind_cache: continue
            df = ind_cache[sym]
            if df.empty: continue
            row = df.iloc[-1]
            if pd.isna(row["ADX"]): continue
            if row["ADX"] < MIN_ADX: continue
            if row["dollar_vol_avg20"] < DOLLAR_VOLUME_THRESHOLD: continue
            if sym in cooldown_tracker:
                if (now - cooldown_tracker[sym]).seconds < COOLDOWN_MINUTES * 60:
                    continue
            ok, reason = is_ap_setup(row, df)
            if not ok:
                continue

            # same scoring as before
            rvol_n = min(row["vol_ratio"] / 8, 1.0)
            atr_exp_n = min(row["atr_expansion"] / 3, 1.0)
            bb_exp_n = min(row["bb_expansion"] / 3, 1.0)
            vel_n = min(row["price_velocity"] / 5, 1.0)
            brk_n = row["breakout"]
            surge_n = min(row["vol_surge"] / 8, 1.0)
            adx_n = min(row["ADX"] / 50, 1.0)
            # fetch OB later
            candidates.append((sym, row, rvol_n, atr_exp_n, bb_exp_n, vel_n, brk_n, surge_n, adx_n))

    # get order books for passed candidates
    ob_data = {}
    syms_to_fetch = [c[0] for c in candidates]
    if syms_to_fetch:
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(fetch_order_book, sym): sym for sym in syms_to_fetch}
            for fut in as_completed(futures):
                sym = futures[fut]
                ob_data[sym] = fut.result()

    final_signals = []
    for sym, row, rvol_n, atr_exp_n, bb_exp_n, vel_n, brk_n, surge_n, adx_n in candidates:
        ob_abs, ob_side = ob_data.get(sym, (0, "Neutral"))
        ob_n = ob_abs
        rel = get_reliability(sym)
        rel_n = rel
        score = (0.20 * rvol_n + 0.15 * atr_exp_n + 0.20 * bb_exp_n +
                 0.15 * vel_n + 0.10 * brk_n + 0.10 * surge_n +
                 0.05 * ob_n + 0.03 * rel_n + 0.02 * adx_n) * 100
        exp_move = "HIGH" if score > 80 else "MED" if score > 60 else "LOW"
        setup_type = "SQUEEZE BREAKOUT" if row["bb_expansion"] >= 1.5 else "VOLATILITY EXPANSION"
        entry_end = now + timedelta(minutes=4)
        final_signals.append((
            sym, score, setup_type, exp_move, entry_end,
            row["bb_expansion"], atr_exp_n, rvol_n, vel_n,
            brk_n, ob_abs, ob_side, surge_n, row["ATR_pct"],
            row["chg_15min"], row["vol_ratio"]
        ))

    final_signals.sort(key=lambda x: x[1], reverse=True)
    if regime == "choppy":
        max_to_send = max(1, int(n * CHOPPY_ALERT_REDUCTION))
        final_signals = final_signals[:max_to_send]
    else:
        final_signals = final_signals[:n]

    for s in final_signals:
        cooldown_tracker[s[0]] = now

    return final_signals, regime

# ══════════════════ MARKET REGIME DETECTION ══════════════════
def get_market_regime():
    adxs = []
    with data_lock:
        for sym, df in ind_cache.items():
            if sym == BTC_PAIR or df.empty: continue
            row = df.iloc[-1]
            if not pd.isna(row["ADX"]):
                adxs.append(row["ADX"])
    if not adxs:
        return "trending"
    avg_adx = sum(adxs) / len(adxs)
    return "choppy" if avg_adx < CHOPPY_ADX_THRESHOLD else "trending"

# ══════════════════ TELEGRAM BOT ══════════════════
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("⚡ Top Setups", callback_data="top")],
        [InlineKeyboardButton("💓 Near A+", callback_data="near")],
        [InlineKeyboardButton("🔄 Refresh", callback_data="refresh")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Pro Hedge Straddle v19", reply_markup=reply_markup)

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    if data in ("top", "refresh"):
        setups, regime = get_ap_signals(5)
        msg = format_setups(setups, regime)
        await query.edit_message_text(text=msg, reply_markup=query.message.reply_markup, parse_mode="HTML")
    elif data == "near":
        near = get_near_ap_coins(5)
        msg = format_near_ap(near)
        await query.edit_message_text(text=msg, reply_markup=query.message.reply_markup, parse_mode="HTML")

def format_setups(setups, regime):
    now = datetime.now(timezone.utc).strftime("%H:%M UTC")
    if not setups:
        return f"⚡ <b>No A+ Setup</b> ({now})\nMarket: {regime.upper()}\nNo explosive coins right now."
    msg = f"⚡ <b>A+ STRADDLE SETUP</b> | {now}\n"
    if regime == "choppy":
        msg += "⚠️ <b>CHOPPY MARKET – SIZE DOWN</b>\n"
    msg += "\n"
    for s in setups:
        sym, score, setup_type, exp_move, entry_end, bb_exp, atr_exp, rvol, vel, brk, ob_abs, ob_side, surge, atr_pct, chg15, volr = s
        entry_str = entry_end.strftime("%H:%M UTC")
        msg += (
            f"🔥 <b>{sym}</b>\n"
            f"Score: {score:.0f} ({exp_move} CONFIDENCE)\n\n"
            f"Setup Type: {setup_type}\n"
            f"BB Expansion: YES ({bb_exp:.1f}x)\n"
            f"ATR Expansion: STARTING ({atr_exp:.1f}x)\n"
            f"Volume Surge: CONFIRMED ({volr:.1f}x)\n"
            f"Breakout Pressure: {brk:.2f}\n\n"
            f"Orderbook: {ob_side} (imbalance {ob_abs:.2f})\n\n"
            f"⏱ Valid Entry: until {entry_str}\n"
            f"❌ Ignore after: {entry_str}\n\n"
            f"💡 Expected Move: {exp_move}\n"
            f"📌 Open LONG + SHORT manually\n\n"
        )
    return msg

def format_near_ap(coins):
    now = datetime.now(timezone.utc).strftime("%H:%M UTC")
    if not coins:
        return f"💓 <b>Near A+</b> ({now})\nNo near‑A+ coins."
    msg = f"💓 <b>Near A+ Coins</b> ({now})\n\n"
    for sym, score, passed, reasons in coins:
        status = "✅ A+" if passed else f"❌ {', '.join(reasons)}"
        msg += f"<b>{sym}</b> Score:{score:.0f} {status}\n"
    return msg

async def periodic_scan(context: ContextTypes.DEFAULT_TYPE):
    bot = context.bot
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, incremental_update)
    except Exception as e:
        logging.error(f"Update failed: {e}")
    try:
        setups, regime = get_ap_signals(5)
        msg = format_setups(setups, regime)
        await bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="HTML")
    except Exception as e:
        logging.error(f"Send A+ failed: {e}")

async def heartbeat(context: ContextTypes.DEFAULT_TYPE):
    bot = context.bot
    try:
        near = get_near_ap_coins(5)
        msg = format_near_ap(near)
        await bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="HTML")
    except Exception as e:
        logging.error(f"Heartbeat failed: {e}")

async def health_watchdog(context: ContextTypes.DEFAULT_TYPE):
    bot = context.bot
    msg = f"✅ Pro Straddle Scanner – {len(get_active_pairs())} pairs"
    await bot.send_message(chat_id=CHAT_ID, text=msg)

async def send_startup(context: ContextTypes.DEFAULT_TYPE):
    msg = f"🚀 Pro Hedge Straddle v19 Online\nPairs: {len(get_active_pairs())}\nA+ Scan: {SCAN_INTERVAL_MINUTES} min\nHeartbeat: {HEARTBEAT_INTERVAL_MINUTES} min\nOnly A+ Setups"
    await context.bot.send_message(chat_id=CHAT_ID, text=msg)

# ══════════════════ MAIN (stable event loop) ══════════════════
if __name__ == "__main__":
    # Optional health server – comment out if port conflict on Termux
    threading.Thread(target=run_health, daemon=True).start()

    PAIRS = get_valid_symbols()
    if not PAIRS:
        logging.error("No valid symbols found, using all candidates.")
        PAIRS = CANDIDATES.copy()
    logging.info(f"Valid symbols: {len(PAIRS)}")

    initial_data_download()
    downloaded = sum(1 for s in get_active_pairs() + [BTC_PAIR] if s in data_cache)
    logging.info(f"Downloaded: {downloaded}")

    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CallbackQueryHandler(button_handler))

    job_queue = app.job_queue
    job_queue.run_once(send_startup, when=0)
    job_queue.run_repeating(periodic_scan, interval=SCAN_INTERVAL_MINUTES * 60, first=10)
    job_queue.run_repeating(heartbeat, interval=HEARTBEAT_INTERVAL_MINUTES * 60, first=5)
    job_queue.run_repeating(health_watchdog, interval=1800, first=60)

    app.run_polling(close_loop=False)
