#!/usr/bin/env python3
"""
Signal Bot v12 – All fixes applied (MEXC 8‑col, pandas freq, JobQueue, asyncio)
================================================================================
"""
import os, time, asyncio, logging, csv, threading, json, sys, traceback
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

SECTOR = defaultdict(list)
MEME = [
    "PEPEUSDT","BONKUSDT","WIFUSDT","FLOKIUSDT","BOMEUSDT",
    "MEMEUSDT","DOGEUSDT","SHIBUSDT","LUNCUSDT","BANANAS31USDT",
    "POPCATUSDT","MYROUSDT","WENUSDT","MOGUSDT","TURBOUSDT",
    "MEWUSDT","SLERFUSDT","SAMOUSDT","BRETTUSDT","PONKEUSDT",
    "BODENUSDT","TREMPUSDT","DEGENUSDT","ANDYUSDT","WOLFUSDT",
    "HARAMBEUSDT","NUBUSDT","PENGUSDT","MOODENGUSDT","GOATSEUSDT",
    "SIGMAUSDT","BITCOINUSDT","SPX6900USDT","GIGAUSDT","MOTHERUSDT",
    "BURGERUSDT","CATUSDT","DOGUSDT","AIDOGEUSDT","BABYDOGEUSDT",
    "CORGIAIUSDT","KISHUUSDT","WOJAKUSDT","PEPE2USDT","BENUSDT",
    "PSYOPUSDT","LOLUSDT","BOBOMUSDT","DINGERUSDT","MILADYUSDT",
    "SAITAMAUSDT","VOLTUSDT"
]
L1 = ["SOLUSDT","AVAXUSDT","SUIUSDT","NEARUSDT","INJUSDT","APTUSDT","SEIUSDT","TIAUSDT","ICPUSDT"]
AI = ["FETUSDT","RNDRUSDT"]
DEFI = ["LINKUSDT","UNIUSDT","AAVEUSDT","MKRUSDT"]
OTHER = [
    "ONDOUSDT","STXUSDT","GALAUSDT","SANDUSDT","ARBUSDT","OPUSDT","DOTUSDT",
    "TRXUSDT","JUPUSDT","SPXUSDT","HYPEUSDT","APEUSDT","PEOPLEUSDT","LADYSUSDT",
    "JENNERUSDT","MAGAUSDT","TRUMPUSDT","BIDENUSDT","ELONUSDT","FOURUSDT",
    "SATSUSDT","RATSUSDT","ORDIUSDT","SILLYUSDT","DRAGONUSDT","KUJIUSDT",
    "MONUSDT","PORKUSDT","PUNDUUSDT"
]
for s in MEME: SECTOR[s].append("meme")
for s in L1: SECTOR[s].append("L1")
for s in AI: SECTOR[s].append("AI")
for s in DEFI: SECTOR[s].append("DeFi")
for s in OTHER: SECTOR[s].append("other")

MEXC_URL = "https://api.mexc.com/api/v3"
CAPITAL = 5.0
LEVERAGE = 20
FEE = 0.0008
SLIPPAGE = 0.0002
STOP_PCT_MIN = 0.0188
TP_ATR_MULT = 2.0
SL_ATR_MULT = 1.2
TOP_N = 5
MIN_ADX = 15
MIN_EFFICIENCY = 0.3
ATR_PERIOD = 14
LOOKBACK_RANGE = 15
SCAN_INTERVAL_MINUTES = 15
MAX_MEME = 1
MAX_L1_AI_DEFI = 1
HISTORY_DAYS = 30
DOLLAR_VOLUME_THRESHOLD = 200_000
MARKET_BREADTH_LONG_MIN = 0.3
MARKET_BREADTH_SHORT_MAX = 0.7
OUTCOME_WINDOW_HOURS = 2
PERF_FILE = "signal_performance.csv"
ACTIVE_FILE = "active_signals.json"
STATS_FILE = "stats.json"
LOG_FILE = "bot.log"
FAIL_THRESHOLD = 3
RETRY_HOURS = 12
FORCE_BACKTEST = True

# ══════════════════ LOGGING ══════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)

# ══════════════════ HEALTH SERVER ══════════════════
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
            logging.warning(f"Marked {sym} as inactive (failed {FAIL_THRESHOLD} times)")

def reactivate_if_ready():
    now = datetime.now(timezone.utc)
    reactivated = []
    with pairs_lock:
        for sym, since in list(inactive_symbols.items()):
            if (now - since).total_seconds() > RETRY_HOURS * 3600:
                PAIRS.append(sym)
                del inactive_symbols[sym]
                reactivated.append(sym)
                logging.info(f"Reactivating {sym}")
    return reactivated

def remove_dead_coin(sym):
    with pairs_lock:
        if sym in PAIRS:
            PAIRS.remove(sym)
        if sym in inactive_symbols:
            del inactive_symbols[sym]
        if sym in data_cache: del data_cache[sym]
        if sym in ind_cache: del ind_cache[sym]
        if sym in last_fetch: del last_fetch[sym]
        logging.warning(f"Permanently removed {sym} after prolonged failure")

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
    """Fetch candles with rate‑limit handling. Works with MEXC's 8‑column response."""
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
                if not isinstance(data, list) or not data:
                    break
                rows.extend(data)
                cursor = data[-1][0] + 60_000
                if len(data) < 500:
                    break
                time.sleep(0.02)
            if not rows:
                return pd.DataFrame()

            # MEXC now returns 8 columns: [OpenTime, Open, High, Low, Close, Volume, CloseTime, QuoteVolume]
            col_names = ["OpenTime", "Open", "High", "Low", "Close", "Volume", "CloseTime", "QuoteVolume"]
            df = pd.DataFrame(rows, columns=col_names)

            df["OpenTime"] = pd.to_datetime(df["OpenTime"], unit="ms", utc=True)
            df.set_index("OpenTime", inplace=True)
            for c in ["Open", "High", "Low", "Close", "Volume"]:
                df[c] = pd.to_numeric(df[c])
            return df[["Open", "High", "Low", "Close", "Volume"]].sort_index()
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
    updated = False
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {}
        for sym in active_syms:
            since = last_fetch.get(sym, now - timedelta(days=HISTORY_DAYS))
            since = since + timedelta(minutes=1) if since else since
            if since >= now:
                continue
            futures[pool.submit(fetch_candles, sym, since, now)] = sym
        for fut in as_completed(futures):
            sym = futures[fut]
            df_new = fut.result()
            if df_new.empty:
                fail_count[sym] += 1
                if fail_count[sym] >= FAIL_THRESHOLD:
                    if sym in PAIRS:
                        mark_inactive(sym)
                    elif sym in inactive_symbols:
                        if (now - inactive_symbols[sym]).total_seconds() > 24 * 3600:
                            remove_dead_coin(sym)
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
            updated = True
    return updated

# ══════════════════ INDICATORS ══════════════════
def add_indicators(df):
    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"]

    tr = pd.concat([high-low,
                    (high-close.shift()).abs(),
                    (low-close.shift()).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(ATR_PERIOD, min_periods=1).mean()
    df["ATR_pct"] = df["ATR"] / close * 100
    df["range_15"] = (high.rolling(LOOKBACK_RANGE, min_periods=1).max() -
                      low.rolling(LOOKBACK_RANGE, min_periods=1).min()) / close * 100
    df["vol_avg"] = vol.rolling(20, min_periods=1).mean()
    df["vol_ratio"] = vol / df["vol_avg"]

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0), index=df.index)
    atr_adx = tr.rolling(ATR_PERIOD, min_periods=1).mean()
    plus_di = 100 * plus_dm.rolling(ATR_PERIOD, min_periods=1).mean() / atr_adx
    minus_di = 100 * minus_dm.rolling(ATR_PERIOD, min_periods=1).mean() / atr_adx
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-6)
    df["ADX"] = dx.rolling(ATR_PERIOD, min_periods=1).mean()

    net = close - close.shift(LOOKBACK_RANGE)
    total = close.diff().abs().rolling(LOOKBACK_RANGE, min_periods=1).sum()
    df["eff_ratio"] = np.abs(net) / total.replace(0, np.nan)

    atr_roll_mean = df["ATR"].rolling(50, min_periods=10).mean()
    atr_roll_std = df["ATR"].rolling(50, min_periods=10).std().replace(0, np.nan)
    df["atr_z"] = (df["ATR"] - atr_roll_mean) / atr_roll_std

    day = df.index.date
    cum_vp = (close * vol).groupby(day).cumsum()
    cum_vol = vol.groupby(day).cumsum()
    df["vwap_daily"] = cum_vp / cum_vol

    ohlc_15 = df.resample("15min").agg({"Close": "last", "Volume": "sum"}).dropna()
    ohlc_15["ema20"] = ohlc_15["Close"].ewm(span=20, adjust=False).mean()
    df["ema20_15m"] = ohlc_15["ema20"].reindex(df.index, method="ffill")

    df["dollar_vol"] = close * vol
    df["dollar_vol_avg20"] = df["dollar_vol"].rolling(20, min_periods=1).mean()
    return df

# ══════════════════ BACKTEST & PERFORMANCE ══════════════════
def load_rolling_performance(n=100):
    if not os.path.exists(PERF_FILE):
        return defaultdict(lambda: [0, 0])
    rows_by_sym = defaultdict(list)
    with open(PERF_FILE, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_by_sym[row["sym"]].append(row)
    perf = defaultdict(lambda: [0, 0])
    for sym, rows in rows_by_sym.items():
        recent = rows[-n:]
        wins = sum(1 for r in recent if r.get("outcome") == "win")
        perf[sym] = [wins, len(recent)]
    return perf

def backtest_pair(sym, df, btc_df):
    if df.empty or btc_df.empty:
        return []
    signals = []
    decision_times = df.resample("15min").last().dropna()
    perf = load_rolling_performance(100)
    for timestamp, row in decision_times.iterrows():
        if row["ADX"] < MIN_ADX or row["eff_ratio"] < MIN_EFFICIENCY:
            continue
        if pd.isna(row["vwap_daily"]) or pd.isna(row["ema20_15m"]):
            continue
        if row["Close"] > row["vwap_daily"] and row["Close"] > row["ema20_15m"]:
            direction = 1
        elif row["Close"] < row["vwap_daily"] and row["Close"] < row["ema20_15m"]:
            direction = -1
        else:
            continue

        avg_dollar_vol = row["dollar_vol_avg20"]
        if pd.isna(avg_dollar_vol) or avg_dollar_vol < DOLLAR_VOLUME_THRESHOLD:
            continue

        try:
            btc_slice = btc_df.loc[timestamp - timedelta(minutes=15):timestamp]
            btc_return = btc_slice["Close"].iloc[-1]/btc_slice["Close"].iloc[0]-1 if len(btc_slice)>=2 else 0
        except:
            btc_return = 0
        coin_slice = df.loc[timestamp - timedelta(minutes=15):timestamp]
        coin_ret = coin_slice["Close"].iloc[-1]/coin_slice["Close"].iloc[0]-1 if len(coin_slice)>=2 else 0
        rel_str = coin_ret - btc_return
        rs_score = 1/(1+np.exp(-rel_str*5)) if direction==1 else 1/(1+np.exp(rel_str*5))

        atr_z = row.get("atr_z", 0)
        atr_z = 0 if pd.isna(atr_z) else atr_z
        atr_z_score = (np.clip(atr_z,-2,2)+2)/4

        atr_val = row["ATR"]
        vwap_dist_score = 0.5
        if atr_val and atr_val > 0:
            vwap_dist = abs(row["Close"] - row["vwap_daily"])/atr_val
            vwap_dist_score = max(0, 1 - min(vwap_dist/2, 1))

        atr_n = min(row["ATR_pct"]/5,1)
        rng_n = min(row["range_15"]/3,1)
        vol_n = min(row["vol_ratio"]/5,1)
        adx_n = min(row["ADX"]/50,1)
        eff_n = min(row["eff_ratio"],1)

        wins, total = perf.get(sym, (0,0))
        hist_score = wins/total if total>=5 else 0.5

        score = (0.15*atr_n + 0.15*rng_n + 0.10*vol_n + 0.10*adx_n +
                 0.05*eff_n + 0.10*vwap_dist_score + 0.20*rs_score +
                 0.10*atr_z_score + 0.05*hist_score)*100
        if score < 10:
            continue

        atr_pct = row["ATR_pct"]
        sl_pct = max(SL_ATR_MULT*atr_pct/100, STOP_PCT_MIN)
        tp_pct = TP_ATR_MULT*atr_pct/100
        entry = row["Close"]
        sl = entry * (1 - direction*sl_pct)
        tp = entry * (1 + direction*tp_pct)

        if direction == 1:
            entry_real = entry*(1+SLIPPAGE)
            exit_tp = tp*(1-SLIPPAGE)
            exit_sl = sl*(1-SLIPPAGE)
        else:
            entry_real = entry*(1-SLIPPAGE)
            exit_tp = tp*(1+SLIPPAGE)
            exit_sl = sl*(1+SLIPPAGE)

        future = df[df.index > timestamp]
        outcome = "unresolved"
        tp_hit = sl_hit = False
        for _, f in future.iterrows():
            if direction == 1:
                if f["High"] >= tp: tp_hit = True
                if f["Low"] <= sl: sl_hit = True
            else:
                if f["Low"] <= tp: tp_hit = True
                if f["High"] >= sl: sl_hit = True
            if tp_hit and sl_hit:
                outcome = "ambiguous"; break
            if tp_hit and not sl_hit:
                outcome = "win"; break
            if sl_hit and not tp_hit:
                outcome = "loss"; break
            if (f.name - timestamp) > timedelta(hours=OUTCOME_WINDOW_HOURS):
                outcome = "timeout"; break

        if outcome == "win":
            net = abs(exit_tp/entry_real - 1) - 2*FEE
            if net <= 0:
                outcome = "loss"

        signals.append({
            "timestamp": timestamp.isoformat(),
            "sym": sym, "direction": direction,
            "price": entry, "sl": sl, "tp": tp,
            "score": score, "outcome": outcome
        })
    return signals

def run_backtest():
    logging.info("Running 30‑day backtest...")
    all_sigs = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = {}
        btc_df = ind_cache.get(BTC_PAIR, pd.DataFrame())
        active_syms = get_active_pairs()
        for sym, df in ind_cache.items():
            if sym == BTC_PAIR or sym not in active_syms: continue
            futs[pool.submit(backtest_pair, sym, df, btc_df)] = sym
        for fut in as_completed(futs):
            sym = futs[fut]
            try:
                sigs = fut.result()
                all_sigs.extend(sigs)
            except Exception as e:
                logging.error(f"Backtest error {sym}: {e}")

    if all_sigs:
        file_exists = os.path.isfile(PERF_FILE)
        with open(PERF_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp","sym","direction","entry","sl","tp","score","outcome"])
            for s in all_sigs:
                writer.writerow([s["timestamp"], s["sym"], s["direction"],
                                 s["price"], s["sl"], s["tp"], s["score"], s["outcome"]])
    return all_sigs

# ══════════════════ LIVE SIGNAL ENGINE ══════════════════
active_signals = {}
stats = {"daily_pnl": 0.0, "total_pnl": 0.0, "total_trades": 0, "wins": 0, "losses": 0, "last_reset_date": ""}
daily_closed = []
stats_lock = threading.Lock()

def load_stats():
    global stats
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, 'r') as f:
                stats = json.load(f)
            logging.info(f"Loaded stats: {stats}")
        except:
            pass

def save_stats():
    try:
        with open(STATS_FILE, 'w') as f:
            json.dump(stats, f)
    except Exception as e:
        logging.error(f"Failed to save stats: {e}")

def save_active_signals():
    try:
        with open(ACTIVE_FILE, 'w') as f:
            json.dump(active_signals, f, default=str)
    except Exception as e:
        logging.error(f"Failed to save active signals: {e}")

def load_active_signals():
    global active_signals
    if os.path.exists(ACTIVE_FILE):
        try:
            with open(ACTIVE_FILE, 'r') as f:
                active_signals = json.load(f)
            logging.info(f"Loaded {len(active_signals)} active signals from disk.")
        except Exception as e:
            logging.error(f"Failed to load active signals: {e}")

def check_active_signals(now):
    closed_notifications = []
    for sig_id, s in list(active_signals.items()):
        sym = s["sym"]
        with data_lock:
            if sym not in ind_cache: continue
            row = ind_cache[sym].iloc[-1]
        high = row["High"]
        low = row["Low"]
        direction = s["direction"]
        if direction == 1:
            if high >= s["tp"]:
                pnl = (s["tp"] / s["entry"] - 1) - 2*FEE
                outcome = "win"
            elif low <= s["sl"]:
                pnl = (s["sl"] / s["entry"] - 1) - 2*FEE
                outcome = "loss"
            else: continue
        else:
            if low <= s["tp"]:
                pnl = (s["entry"] / s["tp"] - 1) - 2*FEE
                outcome = "win"
            elif high >= s["sl"]:
                pnl = (s["entry"] / s["sl"] - 1) - 2*FEE
                outcome = "loss"
            else: continue

        pnl_amount = pnl * (CAPITAL * LEVERAGE)
        with stats_lock:
            stats["daily_pnl"] += pnl_amount
            stats["total_pnl"] += pnl_amount
            stats["total_trades"] += 1
            if outcome == "win":
                stats["wins"] += 1
            else:
                stats["losses"] += 1
            daily_closed.append((sym, direction, outcome, pnl_amount))

        closed_notifications.append((sym, direction, outcome, pnl_amount))
        del active_signals[sig_id]

        with open(PERF_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([now.isoformat(), sym, direction, s["entry"], s["sl"], s["tp"], s["score"], outcome])

    if closed_notifications:
        save_active_signals()
        save_stats()
    return closed_notifications

def live_score_and_signal(now=None):
    if now is None: now = datetime.now(timezone.utc)
    try:
        notifs = check_active_signals(now)
    except Exception as e:
        logging.error(f"Error checking active signals: {e}")
        notifs = []

    signals = []
    scored = []

    btc_df = ind_cache.get(BTC_PAIR, pd.DataFrame())
    btc_return = 0.0
    if not btc_df.empty:
        last_15 = btc_df.last("15min")
        if len(last_15) >= 2:
            btc_return = last_15["Close"].iloc[-1]/last_15["Close"].iloc[0] - 1

    breadth = 0.5
    total = 0; above = 0
    with data_lock:
        for sym, df in ind_cache.items():
            if sym == BTC_PAIR or df.empty: continue
            row = df.iloc[-1]
            if pd.isna(row["vwap_daily"]): continue
            total += 1
            if row["Close"] > row["vwap_daily"]: above += 1
    if total: breadth = above/total

    perf = load_rolling_performance(100)
    active_syms = get_active_pairs()

    with data_lock:
        for sym, df in ind_cache.items():
            if sym == BTC_PAIR or df.empty: continue
            if sym not in active_syms: continue
            row = df.iloc[-1]
            if row["ADX"] < MIN_ADX or row["eff_ratio"] < MIN_EFFICIENCY: continue
            if pd.isna(row["vwap_daily"]) or pd.isna(row["ema20_15m"]): continue
            if row["Close"] > row["vwap_daily"] and row["Close"] > row["ema20_15m"]:
                direction = 1
            elif row["Close"] < row["vwap_daily"] and row["Close"] < row["ema20_15m"]:
                direction = -1
            else: continue

            if any(sig["sym"] == sym for sig in active_signals.values()):
                continue

            avg_dollar_vol = row["dollar_vol_avg20"]
            if pd.isna(avg_dollar_vol) or avg_dollar_vol < DOLLAR_VOLUME_THRESHOLD: continue

            coin_15 = df.last("15min")
            coin_ret = 0.0
            if len(coin_15) >= 2:
                coin_ret = coin_15["Close"].iloc[-1]/coin_15["Close"].iloc[0] - 1
            rel_str = coin_ret - btc_return
            rs_score = 1/(1+np.exp(-rel_str*5)) if direction==1 else 1/(1+np.exp(rel_str*5))

            atr_z = row.get("atr_z",0)
            atr_z = 0 if pd.isna(atr_z) else atr_z
            atr_z_score = (np.clip(atr_z,-2,2)+2)/4

            atr_val = row["ATR"]
            vwap_dist_score = 0.5
            if atr_val and atr_val > 0:
                vwap_dist = abs(row["Close"] - row["vwap_daily"])/atr_val
                vwap_dist_score = max(0, 1 - min(vwap_dist/2,1))

            atr_n = min(row["ATR_pct"]/5,1)
            rng_n = min(row["range_15"]/3,1)
            vol_n = min(row["vol_ratio"]/5,1)
            adx_n = min(row["ADX"]/50,1)
            eff_n = min(row["eff_ratio"],1)

            wins, total = perf.get(sym, (0,0))
            hist_score = wins/total if total>=5 else 0.5

            score = (0.15*atr_n + 0.15*rng_n + 0.10*vol_n + 0.10*adx_n +
                     0.05*eff_n + 0.10*vwap_dist_score + 0.20*rs_score +
                     0.10*atr_z_score + 0.05*hist_score)*100
            if score < 10: continue

            atr_pct = row["ATR_pct"]
            sl_pct = max(SL_ATR_MULT*atr_pct/100, STOP_PCT_MIN)
            tp_pct = TP_ATR_MULT*atr_pct/100
            price = row["Close"]
            sl = price * (1 - direction*sl_pct)
            tp = price * (1 + direction*tp_pct)
            scored.append((sym, direction, price, score, atr_pct, sl, tp))

    scored.sort(key=lambda x: x[3], reverse=True)

    if breadth < MARKET_BREADTH_LONG_MIN:
        scored = [s for s in scored if s[1]==-1]
    if breadth > MARKET_BREADTH_SHORT_MAX:
        scored = [s for s in scored if s[1]==1]

    meme_count = 0; l1_ai_defi_count = 0; used = set()
    for sym, direction, price, score, atr_pct, sl, tp in scored:
        if sym in used: continue
        sectors = SECTOR.get(sym, ["other"])
        if "meme" in sectors:
            if meme_count >= MAX_MEME: continue
            meme_count += 1
        elif any(s in sectors for s in ["L1","AI","DeFi"]):
            if l1_ai_defi_count >= MAX_L1_AI_DEFI: continue
            l1_ai_defi_count += 1

        sig_id = f"{sym}_{now.timestamp()}"
        active_signals[sig_id] = {
            "sym": sym, "direction": direction, "entry": price,
            "sl": sl, "tp": tp, "score": score, "time": str(now)
        }
        used.add(sym)
        signals.append(active_signals[sig_id])
        save_active_signals()
        with open(PERF_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([now.isoformat(), sym, direction, price, sl, tp, score, ""])
        if len(signals) >= TOP_N: break

    return signals, notifs

# ══════════════════ TELEGRAM BOT ══════════════════
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("📈 Top Coins", callback_data="top")],
        [InlineKeyboardButton("🛑 Active Signals", callback_data="active")],
        [InlineKeyboardButton("💰 Daily PnL", callback_data="pnl")],
        [InlineKeyboardButton("📜 History", callback_data="history")],
        [InlineKeyboardButton("📊 Backtest", callback_data="backtest")],
        [InlineKeyboardButton("📊 Stats", callback_data="stats")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Signal Bot v12 ready. Choose an option:", reply_markup=reply_markup)

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    if data == "top":
        text = generate_top5()
    elif data == "active":
        text = generate_active()
    elif data == "pnl":
        with stats_lock:
            daily = stats["daily_pnl"]
            total = stats["total_pnl"]
        text = f"💰 Daily P&L: ${daily:.2f}\nTotal P&L: ${total:.2f}"
    elif data == "history":
        text = generate_history()
    elif data == "backtest":
        text = generate_backtest_summary()
    elif data == "stats":
        with stats_lock:
            s = stats
        text = (f"📊 Lifetime Stats\n"
                f"Trades: {s['total_trades']}\n"
                f"Wins: {s['wins']}\n"
                f"Losses: {s['losses']}\n"
                f"Win Rate: {s['wins']/s['total_trades']*100:.1f}%" if s['total_trades']>0 else "No trades yet")
    else:
        text = "Unknown."
    await query.edit_message_text(text=text, reply_markup=query.message.reply_markup, parse_mode="HTML")

def generate_top5():
    scored = []
    with data_lock:
        for sym, df in ind_cache.items():
            if sym == BTC_PAIR or df.empty: continue
            row = df.iloc[-1]
            atr_n = min(row["ATR_pct"]/5.0, 1.0)
            rng_n = min(row["range_15"]/3.0, 1.0)
            vol_n = min(row["vol_ratio"]/5.0, 1.0)
            adx_n = min(row["ADX"]/50.0, 1.0)
            eff_n = min(row["eff_ratio"], 1.0)
            score = (0.25*atr_n + 0.20*rng_n + 0.15*vol_n + 0.15*adx_n + 0.10*eff_n)*100
            scored.append((sym, score, row["Close"], row["vwap_daily"]))
    scored.sort(key=lambda x: x[1], reverse=True)
    top5 = scored[:5]
    now = datetime.now(timezone.utc).strftime("%H:%M UTC")
    msg = f"📡 <b>Top 5 Coins</b> ({now})\n"
    for sym, sc, price, vwap in top5:
        dir_str = "▲" if price > vwap else "▼"
        msg += f"{sym}: {dir_str} Score {sc:.0f}  Price {price:.6f}\n"
    return msg

def generate_active():
    if not active_signals:
        return "No active signals."
    msg = "🛑 <b>Active Signals:</b>\n"
    for sid, s in active_signals.items():
        dir_str = "LONG" if s["direction"]==1 else "SHORT"
        msg += f"{s['sym']} {dir_str} @ {s['entry']:.6f} TP:{s['tp']:.6f} SL:{s['sl']:.6f}\n"
    return msg

def generate_history():
    if not os.path.exists(PERF_FILE): return "No history file."
    with open(PERF_FILE, 'r') as f:
        lines = f.readlines()
    last = lines[-20:] if len(lines)>20 else lines
    return "📜 Last signals:\n<pre>" + "".join(last) + "</pre>"

def generate_backtest_summary():
    if not os.path.exists(PERF_FILE): return "No backtest data."
    wins = 0; total = 0
    with open(PERF_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("outcome") in ("win","loss"):
                total += 1
                if row["outcome"] == "win": wins += 1
    if total == 0: return "No closed trades yet."
    return f"📊 <b>Backtest/Live summary:</b>\n{wins}/{total} wins ({wins/total*100:.1f}%)"

async def send_trade_notification(bot, sym, direction, outcome, pnl_amount):
    dir_str = "LONG" if direction == 1 else "SHORT"
    if outcome == "win":
        icon = "✅ TP HIT"
    else:
        icon = "❌ SL HIT"
    with stats_lock:
        daily_pnl = stats["daily_pnl"]
    msg = (f"{icon}\n"
           f"{sym} {dir_str}\n"
           f"Profit: ${pnl_amount:+.2f}\n"
           f"Today's PnL: ${daily_pnl:.2f}")
    await bot.send_message(chat_id=CHAT_ID, text=msg)

async def periodic_scan(context: ContextTypes.DEFAULT_TYPE):
    bot = context.bot
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, incremental_update)
    except Exception as e:
        logging.error(f"Incremental update failed: {e}")

    try:
        signals, notifs = live_score_and_signal()
    except Exception as e:
        logging.error(f"Live scoring failed: {traceback.format_exc()}")
        signals, notifs = [], []

    for notif in notifs:
        try:
            await send_trade_notification(bot, *notif)
        except Exception as e:
            logging.error(f"Failed to send notification: {e}")

    for s in signals:
        try:
            dir_str = "🟢 LONG" if s["direction"]==1 else "🔴 SHORT"
            msg = (f"<b>📊 SIGNAL</b> {s['sym']} {dir_str}\n"
                   f"Entry: ${s['entry']:.6f}\n"
                   f"SL: ${s['sl']:.6f}\n"
                   f"TP: ${s['tp']:.6f}\n"
                   f"Score: {s['score']:.0f}/100")
            await bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="HTML")
        except Exception as e:
            logging.error(f"Failed to send signal alert: {e}")

async def daily_summary(context: ContextTypes.DEFAULT_TYPE):
    global daily_closed
    bot = context.bot
    now = datetime.now(timezone.utc)
    with stats_lock:
        closed_today = daily_closed.copy()
        daily_closed.clear()
        daily_pnl_snapshot = stats["daily_pnl"]
        stats["daily_pnl"] = 0.0
        stats["last_reset_date"] = now.strftime("%Y-%m-%d")
        save_stats()
    if not closed_today:
        msg = f"📅 Daily Summary ({now:%Y-%m-%d}): No closed trades today."
    else:
        wins = sum(1 for _,_,outcome,_ in closed_today if outcome == "win")
        losses = len(closed_today) - wins
        total_pnl = sum(p for _,_,_,p in closed_today)
        msg = (f"📅 <b>Daily Summary</b> ({now:%Y-%m-%d})\n"
               f"Signals closed: {len(closed_today)}\n"
               f"Wins: {wins} | Losses: {losses}\n"
               f"Win Rate: {wins/len(closed_today)*100:.1f}%\n"
               f"P&L: ${total_pnl:+.2f}\n"
               f"Total P&L: ${stats['total_pnl']:.2f}")
    await bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="HTML")

async def health_watchdog(context: ContextTypes.DEFAULT_TYPE):
    bot = context.bot
    with pairs_lock:
        active_count = len(PAIRS)
    msg = f"✅ Bot Alive\nActive signals: {len(active_signals)}\nPairs monitored: {active_count}"
    await bot.send_message(chat_id=CHAT_ID, text=msg)

# ══════════════════ MAIN ══════════════════

async def main():
    app = Application.builder().token(BOT_TOKEN).build()

    startup_msg = f"🚀 <b>Bot Online</b>
Pairs: {len(get_active_pairs())}
History: {HISTORY_DAYS} days
Active Signals: {len(active_signals)}"
    if bt_win_rate is not None:
        startup_msg += f"
Backtest Win Rate: {bt_win_rate:.1f}%"
    elif FORCE_BACKTEST or not os.path.exists(PERF_FILE):
        startup_msg += f"
Backtest: {bt_signal_count} signals found (0 wins)"
    with stats_lock:
        startup_msg += f"
Total P&L: ${stats["total_pnl"]:.2f}"

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CallbackQueryHandler(button_handler))

    job_queue = app.job_queue
    job_queue.run_repeating(periodic_scan, interval=SCAN_INTERVAL_MINUTES * 60, first=10)
    now = datetime.now(timezone.utc)
    next_midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    seconds_to_midnight = (next_midnight - now).total_seconds()
    job_queue.run_repeating(daily_summary, interval=86400, first=seconds_to_midnight)
    job_queue.run_repeating(health_watchdog, interval=1800, first=60)

    await app.bot.send_message(chat_id=CHAT_ID, text=startup_msg, parse_mode="HTML")
    await app.run_polling(close_loop=False)

if __name__ == "__main__":
    threading.Thread(target=run_health, daemon=True).start()

    # Symbol validation and setup
    global PAIRS
    PAIRS = get_valid_symbols()
    if not PAIRS:
        logging.error("No valid symbols found, using all candidates.")
        PAIRS = CANDIDATES.copy()
    logging.info(f"Valid symbols: {len(PAIRS)}")

    load_active_signals()
    load_stats()

    initial_data_download()
    downloaded = sum(1 for s in get_active_pairs() + [BTC_PAIR] if s in data_cache)
    logging.info(f"Downloaded: {downloaded}")

    bt_win_rate = None
    bt_signal_count = 0
    if FORCE_BACKTEST or not os.path.exists(PERF_FILE):
        bt_signals = run_backtest()
        bt_signal_count = len(bt_signals)
        wins = sum(1 for s in bt_signals if s["outcome"] == "win")
        if bt_signal_count:
            bt_win_rate = wins / bt_signal_count * 100
            logging.info(f"Backtest: {bt_win_rate:.1f}% ({wins}/{bt_signal_count})")
        else:
            logging.info("Backtest: 0 signals found.")
    else:
        logging.info("Backtest skipped (CSV exists).")

    # Run the bot
    asyncio.run(main())
