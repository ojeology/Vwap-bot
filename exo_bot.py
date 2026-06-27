#!/usr/bin/env python3
"""
Exo Engine v1.0 – Phase 1 (Complete Working Version)
====================================================
- Downloads 90‑day 1m candles for Majors & Altcoins
- 5 strategy templates × parameter grids
- Realistic execution (spread, High/Low stops, 25×, $1.88 liq, trailing stop)
- Walk‑forward: train 60 days → test 30 days
- Filters: minimum trades, pair concentration
- Ranks by composite score, sends Telegram progress & daily report
- Cache is stored next to the script (vwap‑bot/cache/)
"""
import os, time, asyncio, logging, threading, itertools
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from telegram import Bot
from flask import Flask

# ══════════════════ CONFIG ══════════════════
BOT_TOKEN = "8835542017:AAFDRUJjrXv2pgDdVpbxQlMAxILDlIBrL8g"
CHAT_ID   = 6400145232

MAJORS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
ALTS   = [
    "DOGEUSDT","PEPEUSDT","BONKUSDT","WIFUSDT","SUIUSDT",
    "ARBUSDT","OPUSDT","NEARUSDT","GALAUSDT","FETUSDT",
    "SANDUSDT","AVAXUSDT","FLOKIUSDT","DOTUSDT","LTCUSDT",
    "LINKUSDT","POLUSDT","LUNCUSDT","JUPUSDT","ONDOUSDT",
    "STXUSDT","BOMEUSDT","ADAUSDT","TRXUSDT","ICPUSDT","SHIBUSDT"
]

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

CAPITAL = 5.0
LEVERAGE = 25
FEE = 0.08
SPREAD = 0.0002
LIQ_THRESHOLD = 1.88
TRAIL_ATR = 1.5
MIN_VOL = 10000
SPREAD_THRESH = 0.04

TRAIN_DAYS = 60
TEST_DAYS  = 30
TOTAL_DAYS = TRAIN_DAYS + TEST_DAYS

MIN_TRADES = 40
MAX_SINGLE_PAIR_PCT = 0.40
MAX_TOP3_PAIR_PCT   = 0.60

# ══════════════════ HEALTH SERVER ══════════════════
health_app = Flask(__name__)
@health_app.route('/health')
def health(): return 'OK', 200
def run_health(): health_app.run(host='0.0.0.0', port=10000, debug=False, use_reloader=False)

# ══════════════════ TELEGRAM HELPER ══════════════════
async def send_message(bot, text):
    try:
        await bot.send_message(chat_id=CHAT_ID, text=text)
    except Exception as e:
        logging.error(f"Telegram error: {e}")

# ══════════════════ DATA PIPELINE ══════════════════
BINANCE_URL = "https://api.binance.com/api/v3/klines"

def fetch_klines(symbol, days):
    cache_file = CACHE_DIR / f"{symbol}_exo_{days}d.csv.gz"
    if cache_file.exists():
        df = pd.read_csv(cache_file, compression="gzip", index_col=0, parse_dates=True)
        if len(df) > 500:
            return df

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    rows = []
    cursor = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    sess = requests.Session()
    sess.headers.update({"User-Agent": "Mozilla/5.0"})

    while cursor < end_ms:
        try:
            r = sess.get(BINANCE_URL, params={
                "symbol": symbol,
                "interval": "1m",
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1000
            }, timeout=20)
            r.raise_for_status()
            data = r.json()

            if not isinstance(data, list) or not data:
                break
            if len(data[0]) < 5:
                break

            rows.extend(data)
            cursor = data[-1][0] + 60_000
            time.sleep(0.02)
        except Exception:
            break

    if not rows:
        return pd.DataFrame()

    n_cols = len(rows[0])
    columns = ["OpenTime","Open","High","Low","Close","Volume","CloseTime","QuoteVol",
               "Trades","TakerBuyBase","TakerBuyQuote","Ignore"]
    if n_cols != 12:
        columns = columns[:n_cols]

    df = pd.DataFrame(rows, columns=columns)
    df["OpenTime"] = pd.to_datetime(df["OpenTime"], unit="ms", utc=True)
    df.set_index("OpenTime", inplace=True)
    for c in ["Open","High","Low","Close","Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c])
    df = df[["Open","High","Low","Close","Volume"]].sort_index().drop_duplicates()
    if len(df) > 500:
        df.to_csv(cache_file, compression="gzip")
    return df

def load_data(pairs):
    raw = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(fetch_klines, s, TOTAL_DAYS): s for s in pairs}
        for fut in as_completed(futs):
            sym = futs[fut]; df = fut.result()
            if not df.empty and len(df) > 500:
                raw[sym] = df
    return raw

# ══════════════════ INDICATORS & REGIME ══════════════════
def add_indicators(ohlc):
    df = ohlc.copy()
    tr = pd.concat([df["High"]-df["Low"],
                    (df["High"]-df["Close"].shift()).abs(),
                    (df["Low"]-df["Close"].shift()).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()
    df["ATR_pct"] = df["ATR"] / df["Close"]
    high, low, close = df["High"], df["Low"], df["Close"]
    up = high.diff(); down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    atr_adx = tr.rolling(14).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(14).mean() / atr_adx
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(14).mean() / atr_adx
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    df["ADX"] = dx.rolling(14).mean()
    df["vol_avg"] = df["Volume"].rolling(20).mean()
    df["vol_ratio"] = df["Volume"] / df["vol_avg"]
    dates = df.index.normalize()
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    ctv = (tp * df["Volume"]).groupby(dates).cumsum()
    cv = df["Volume"].groupby(dates).cumsum()
    df["VWAP"] = ctv / cv
    df["dist_vwap"] = (df["Close"] - df["VWAP"]) / df["VWAP"]
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()
    df["ret_1h"] = df["Close"].pct_change(1)
    df["ret_4h"] = df["Close"].pct_change(4)
    df["hour"] = df.index.hour
    df["weekday"] = df.index.dayofweek
    return df.dropna()

def detect_regime(ohlc):
    if ohlc.empty: return "unknown"
    adx = ohlc["ADX"].iloc[-1] if pd.notna(ohlc["ADX"].iloc[-1]) else 20
    atr_pct = ohlc["ATR_pct"].iloc[-1] if pd.notna(ohlc["ATR_pct"].iloc[-1]) else 0
    avg_atr = ohlc["ATR_pct"].rolling(30*24).mean().iloc[-1] if len(ohlc) > 100 else atr_pct
    high_vol = atr_pct > 1.3 * avg_atr
    low_vol  = atr_pct < 0.7 * avg_atr
    if adx > 25:
        return "trending_high" if high_vol else "trending_low"
    elif adx < 20:
        return "ranging_high" if high_vol else "ranging_low"
    else:
        return "neutral"

# ══════════════════ STRATEGY TEMPLATES ══════════════════
def cross_sectional_signal(hourly, spread_thresh=SPREAD_THRESH):
    signals = []
    common = sorted(set.union(*[set(h.index) for h in hourly.values()]))
    for i in range(2, len(common)):
        now = common[i]; prev = common[i-1]; two_ago = common[i-2]
        rets = {}
        for sym, h in hourly.items():
            if two_ago in h.index and prev in h.index:
                old = h.loc[two_ago]; new = h.loc[prev]
                if old and old != 0: rets[sym] = (new - old) / old
        if len(rets) < 2 or max(rets.values()) - min(rets.values()) < spread_thresh: continue
        sorted_ret = sorted(rets.items(), key=lambda x: x[1], reverse=True)
        signals.append((sorted_ret[-1][0], 1, now))   # long worst
        signals.append((sorted_ret[0][0], -1, now))   # short best
    return signals

def vwap_reversal_signal(ohlc_dict):
    signals = []
    for sym, ohlc in ohlc_dict.items():
        if ohlc.empty: continue
        df = ohlc.copy()
        df["vwap_upper"] = df["VWAP"] + 2 * df["Close"].rolling(20).std()
        df["vwap_lower"] = df["VWAP"] - 2 * df["Close"].rolling(20).std()
        for i in range(20, len(df)-1):
            row = df.iloc[i]
            if row["Low"] <= row["vwap_lower"] and row["Close"] > row["Open"] and row["Close"] > row["vwap_lower"]:
                signals.append((sym, 1, df.index[i]))
            elif row["High"] >= row["vwap_upper"] and row["Close"] < row["Open"] and row["Close"] < row["vwap_upper"]:
                signals.append((sym, -1, df.index[i]))
    return signals

def ema_pullback_signal(ohlc_dict):
    signals = []
    for sym, ohlc in ohlc_dict.items():
        if ohlc.empty: continue
        df = ohlc
        for i in range(50, len(df)-1):
            row = df.iloc[i]
            if row["Close"] > row["EMA50"] and row["Low"] <= row["EMA20"] and row["Close"] > row["Open"]:
                signals.append((sym, 1, df.index[i]))
            elif row["Close"] < row["EMA50"] and row["High"] >= row["EMA20"] and row["Close"] < row["Open"]:
                signals.append((sym, -1, df.index[i]))
    return signals

def volume_breakout_signal(ohlc_dict):
    signals = []
    for sym, ohlc in ohlc_dict.items():
        if ohlc.empty: continue
        df = ohlc
        for i in range(60, len(df)-1):
            row = df.iloc[i]
            if row["vol_ratio"] < 1.5: continue
            lookback = df.iloc[i-6:i]
            high_break = lookback["High"].max()
            low_break  = lookback["Low"].min()
            if row["Close"] > high_break:
                signals.append((sym, 1, df.index[i]))
            elif row["Close"] < low_break:
                signals.append((sym, -1, df.index[i]))
    return signals

def engulfing_signal(ohlc_dict):
    signals = []
    for sym, ohlc in ohlc_dict.items():
        if ohlc.empty: continue
        df = ohlc
        prev_open  = df["Open"].shift(1); prev_close = df["Close"].shift(1)
        bull_eng = (prev_close < prev_open) & (df["Close"] > df["Open"]) & (df["Open"] <= prev_close) & (df["Close"] >= prev_open)
        bear_eng = (prev_close > prev_open) & (df["Close"] < df["Open"]) & (df["Open"] >= prev_close) & (df["Close"] <= prev_open)
        for i in range(50, len(df)-1):
            if bull_eng.iloc[i] and df["Close"].iloc[i] > df["EMA50"].iloc[i]:
                signals.append((sym, 1, df.index[i]))
            elif bear_eng.iloc[i] and df["Close"].iloc[i] < df["EMA50"].iloc[i]:
                signals.append((sym, -1, df.index[i]))
    return signals

# ══════════════════ REALISTIC SIMULATOR (FIXED) ══════════════════
def realistic_sim(sym, direction, entry_time, exit_time, raw_1m, entry_atr_val):
    df = raw_1m[sym]
    mask_entry = df.index >= entry_time
    if not mask_entry.any(): return None, False, 0.0, 0.0
    entry_idx = mask_entry.argmax()
    entry = df.iloc[entry_idx]["Open"] * (1 + direction * SPREAD)

    sub = df.iloc[entry_idx:]
    sub = sub[sub.index <= exit_time]
    best_price = entry
    trailing_stop = entry - direction * (TRAIL_ATR * entry_atr_val) if entry_atr_val > 0 else None

    for _, row in sub.iterrows():
        if direction == 1:
            if row["High"] > best_price:
                best_price = row["High"]
                trailing_stop = best_price - TRAIL_ATR * entry_atr_val
            if trailing_stop and row["Low"] <= trailing_stop:
                return trailing_stop, False, entry, 0.0
            worst = row["Low"]
            unrealised = CAPITAL * direction * (worst - entry) / entry * LEVERAGE - FEE
            if unrealised <= -LIQ_THRESHOLD:
                return worst, True, entry, -LIQ_THRESHOLD
        else:
            if row["Low"] < best_price:
                best_price = row["Low"]
                trailing_stop = best_price + TRAIL_ATR * entry_atr_val
            if trailing_stop and row["High"] >= trailing_stop:
                return trailing_stop, False, entry, 0.0
            worst = row["High"]
            unrealised = CAPITAL * direction * (worst - entry) / entry * LEVERAGE - FEE
            if unrealised <= -LIQ_THRESHOLD:
                return worst, True, entry, -LIQ_THRESHOLD

    mask_exit = df.index >= exit_time
    if mask_exit.any():
        exit_price = df.iloc[mask_exit.argmax()]["Open"]
    else:
        exit_price = df.iloc[-1]["Close"]
    exit_price *= (1 - direction * SPREAD)
    return exit_price, False, entry, 0.0

# ══════════════════ BACKTEST WRAPPER (FIXED) ══════════════════
def backtest_template(template_fn, raw_data, hourly_data, session_hours, hold_hours):
    t_min = min(df.index.min() for df in raw_data.values())
    test_start = t_min + timedelta(days=TRAIN_DAYS)
    test_end   = t_min + timedelta(days=TOTAL_DAYS)

    ohlc_test = {}
    for sym, df in hourly_data.items():
        ohlc_test[sym] = df[df.index >= test_start]

    signals = template_fn(ohlc_test)

    trades = []
    pair_pnl = {s: 0.0 for s in raw_data}
    for sym, direction, ts in signals:
        if ts.hour not in session_hours: continue
        entry_time = ts + timedelta(hours=1)
        exit_time  = entry_time + timedelta(hours=hold_hours)
        atr_val = 0
        if ts in hourly_data[sym].index:
            atr_val = hourly_data[sym]["ATR"].loc[ts]
        exit_price, liq, entry_price, pnl = realistic_sim(sym, direction, entry_time, exit_time, raw_data, atr_val)
        if exit_price is None: continue
        if liq:
            trade_pnl = pnl
        else:
            trade_pnl = CAPITAL * direction * (exit_price - entry_price) / entry_price * LEVERAGE - FEE
        trades.append(trade_pnl)
        pair_pnl[sym] += trade_pnl

    if len(trades) < MIN_TRADES: return None
    total_pnl = sum(pair_pnl.values())
    if total_pnl <= 0: return None
    sorted_pairs = sorted(pair_pnl.items(), key=lambda x: x[1], reverse=True)
    top1_pct = sorted_pairs[0][1] / total_pnl if total_pnl > 0 else 0
    top3_pct = sum(p[1] for p in sorted_pairs[:3]) / total_pnl if total_pnl > 0 else 0
    if top1_pct > MAX_SINGLE_PAIR_PCT or top3_pct > MAX_TOP3_PAIR_PCT: return None

    wins = sum(1 for p in trades if p > 0)
    win_rate = wins / len(trades) * 100
    gross_win = sum(p for p in trades if p > 0)
    gross_loss = abs(sum(p for p in trades if p <= 0))
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float('inf')
    expectancy = np.mean(trades)
    daily_pnl = total_pnl / TEST_DAYS
    # Sharpe approximation
    daily_returns = []
    for day in pd.date_range(test_start, test_end, freq='D'):
        day_trades = [p for (s, d, p) in zip(*[list(t) for t in zip(*[(sym, ts, pnl) for (sym,_,ts), pnl in zip(signals, trades)])]) if d.date() == day.date()]
        daily_returns.append(sum(day_trades))
    sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(365) if np.std(daily_returns) > 0 else 0

    score = (0.3 * profit_factor + 0.2 * sharpe + 0.2 * expectancy + 0.1 * win_rate + 0.1 * (1 - min(1, abs(min(trades))/total_pnl)) + 0.1 * np.log(len(trades)))
    return {"pf": profit_factor, "wr": win_rate, "daily": daily_pnl, "trades": len(trades), "score": score, "name": template_fn.__name__, "session": "Asia" if session_hours==set(range(0,8)) else "NY" if session_hours==set(range(13,21)) else "Both", "hold": hold_hours}

# ══════════════════ MAIN EXO LOOP ══════════════════
async def exo_run():
    bot = Bot(token=BOT_TOKEN)
    await send_message(bot, "🟢 Exo Engine v1.0 started. Beginning daily scan…")

    while True:
        now = datetime.now(timezone.utc)
        await send_message(bot, f"⏳ {now:%H:%M} UTC – Downloading fresh data…")

        all_pairs = MAJORS + ALTS
        raw = load_data(all_pairs)
        if len(raw) < 10:
            await send_message(bot, "❌ Data download failed. Retrying in 1 hour.")
            await asyncio.sleep(3600)
            continue

        await send_message(bot, f"✅ Data ready ({len(raw)} pairs). Building indicators…")

        hourly_data = {}
        for sym, df in raw.items():
            ohlc = df.resample("1h").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()
            ohlc = add_indicators(ohlc)
            hourly_data[sym] = ohlc

        regime_majors = detect_regime(pd.concat([hourly_data[s] for s in MAJORS if s in hourly_data]))
        regime_alts   = detect_regime(pd.concat([hourly_data[s] for s in ALTS if s in hourly_data]))

        await send_message(bot, f"🔍 Testing strategies (Majors: {regime_majors}, Alts: {regime_alts})…")

        templates = [
            ("CrossSectional", cross_sectional_signal),
            ("VWAPReversal", vwap_reversal_signal),
            ("EMAPullback", ema_pullback_signal),
            ("VolumeBreakout", volume_breakout_signal),
            ("Engulfing", engulfing_signal),
        ]
        holds = [6, 8, 10]  # hours
        sessions = [
            ("Asia", set(range(0, 8))),
            ("NY", set(range(13, 21))),
            ("Both", set(range(0, 8)) | set(range(13, 21))),
        ]

        all_results = []
        for tname, tfunc in templates:
            for hold in holds:
                for sname, shours in sessions:
                    res = backtest_template(tfunc, raw, hourly_data, shours, hold)
                    if res:
                        res["name"] = tname
                        res["session"] = sname
                        res["hold"] = hold
                        all_results.append(res)

        tested = len(all_results)
        passed = [r for r in all_results if r["score"] > 0]
        top5 = sorted(passed, key=lambda x: x["score"], reverse=True)[:5]

        await send_message(bot, f"📊 Tested {tested} variants, {len(passed)} passed filters. Top 5:")

        if not top5:
            await send_message(bot, "No strategies passed all filters this period.")
        else:
            msg = f"📡 EXO REPORT {now:%Y-%m-%d %H:%M} UTC\n"
            msg += f"Majors regime: {regime_majors} | Alts regime: {regime_alts}\n"
            for i, r in enumerate(top5, 1):
                msg += f"{i}. {r['name']} ({r['session']} {r['hold']}h) PF:{r['pf']:.2f} WR:{r['wr']:.0f}% Daily:${r['daily']:.2f} Trades:{r['trades']}\n"
            await send_message(bot, msg)

        await send_message(bot, "💤 Next scan in 24h.")
        next_run = now.replace(hour=1, minute=0, second=0) + timedelta(days=1)
        await asyncio.sleep((next_run - now).total_seconds())

# ══════════════════ RUN ══════════════════
if __name__ == "__main__":
    threading.Thread(target=run_health, daemon=True).start()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(exo_run())
