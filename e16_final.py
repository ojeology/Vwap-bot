#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E16 FINAL – Volume & Range Cap Tweaks on Pruned Pairs
Honest OOS: 3 windows × 15d train / 5d blind test.
"""
import time, itertools, json, math, warnings
import requests
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

# ═══════════ CONFIG ═══════════
PAIRS_PRUNED = [
    "PEPEUSDT","BONKUSDT","WIFUSDT","SUIUSDT","ARBUSDT","OPUSDT",
    "GALAUSDT","FETUSDT","SANDUSDT","MEMEUSDT","AVAXUSDT",
    "BNBUSDT","DOTUSDT","LTCUSDT","LINKUSDT","INJUSDT",
    "POLUSDT","STXUSDT","BOMEUSDT","ETHUSDT","ICPUSDT"
]

LOOKBACK_DAYS = 30
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
CACHE_TTL_H = 720   # 30 days – always use cache

MEXC_URL = "https://api.mexc.com/api/v3/klines"
MEXC_LIMIT = 500
API_TIMEOUT = 45
API_RETRIES = 3
API_DELAY = 3

CAPITAL = 5.0
LEVERAGE = 20
FEE = 0.08

BLOCKED_HOURS = {0,1,3,4,6,9,10,14,18,19,20,21}
BODY_PCT_MIN = 0.50
RSI_LONG_MAX = 40
RSI_SHORT_MIN = 60
SL_COOLDOWN = 5
DAILY_LOSS_LIMIT = 15.0

TP_GRID = [0.012,0.015,0.018,0.020,0.025,0.030,0.035,0.040]
SL_GRID = [0.007,0.010,0.012,0.015,0.018]

TRAIN_DAYS = 15
TEST_DAYS = 5
WF_WINDOWS = [
    {"train_offset":0, "test_offset":15},
    {"train_offset":5, "test_offset":20},
    {"train_offset":10,"test_offset":25},
]
W = 90

# ═══════════ DATA (CSV cache) ═══════════
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

def _load_any_cache(symbol, days):
    cp = _cache_path(symbol, days)
    if cp.exists():
        try:
            df = pd.read_csv(cp, compression="gzip", index_col=0, parse_dates=True)
            if len(df) > 500: return df, cp
        except: pass
    for f in sorted(CACHE_DIR.glob(f"{symbol}_*.csv.gz"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            df = pd.read_csv(f, compression="gzip", index_col=0, parse_dates=True)
            if len(df) > 500: return df, f
        except: pass
    return None, None

def fetch_pair(symbol, days):
    cp = _cache_path(symbol, days)
    age = time.time() - cp.stat().st_mtime if cp.exists() else 9e9
    if age < CACHE_TTL_H * 3600:
        df, _ = _load_any_cache(symbol, days)
        if df is not None: return df

    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = end_ms - days * 86_400_000
    rows, cursor = [], start_ms
    sess = requests.Session()
    sess.headers.update({"User-Agent": "Mozilla/5.0"})
    try:
        while cursor < end_ms:
            last_err = None; data = None
            for attempt in range(1, API_RETRIES+1):
                try:
                    r = sess.get(MEXC_URL, params={
                        "symbol": symbol, "interval": "1m",
                        "startTime": cursor, "endTime": end_ms,
                        "limit": MEXC_LIMIT}, timeout=API_TIMEOUT)
                    r.raise_for_status()
                    data = r.json(); break
                except Exception as e:
                    last_err = e
                    if attempt < API_RETRIES: time.sleep(API_DELAY * attempt)
            if data is None: raise RuntimeError(f"{symbol}: {last_err}")
            if not data: break
            rows.extend(data)
            last = data[-1][0]
            if last <= cursor: break
            cursor = last + 60_000
            if len(data) < MEXC_LIMIT: break
            time.sleep(0.08)
        df = _parse(rows)
        if len(df) > days * 3 * 60:
            df.to_csv(cp, compression="gzip")
        if len(df) > 500: return df
    except: pass
    df, src = _load_any_cache(symbol, days)
    return df if df is not None else pd.DataFrame()

def load_all(symbols, label=""):
    out = {}
    print(f"  Loading {len(symbols)} pairs {label}...")
    with ThreadPoolExecutor(max_workers=2) as pool:
        futs = {pool.submit(fetch_pair, s, LOOKBACK_DAYS): s for s in symbols}
        for fut in as_completed(futs):
            sym = futs[fut]
            try:
                df = fut.result()
                if df is not None and len(df) > 500:
                    out[sym] = df
            except Exception as e:
                print(f"  ✗ {sym} {str(e)[:50]}")
    return out

# ═══════════ INDICATORS ═══════════
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

# ═══════════ SIGNAL ═══════════
def get_signals(df, vol_mult=1.2, range_cap=False):
    c, o = df["Close"], df["Open"]
    t2lo = (df["Low"] <= df["vwap_2dn"]) & (c > df["vwap_2dn"])
    t2hi = (df["High"] >= df["vwap_2up"]) & (c < df["vwap_2up"])
    strong = df["body_pct"] > BODY_PCT_MIN
    rsi_ok_buy = df["rsi"] < RSI_LONG_MAX
    rsi_ok_sell = df["rsi"] > RSI_SHORT_MIN

    buy = t2lo & (c > o) & strong & rsi_ok_buy
    sell = t2hi & (c < o) & strong & rsi_ok_sell

    if vol_mult > 0:
        vol_spike = df["Volume"] > vol_mult * df["vol20"]
        buy = buy & vol_spike
        sell = sell & vol_spike

    if range_cap:
        range_ok = (df["High"] - df["Low"]) < 3 * df["atr"]
        buy = buy & range_ok
        sell = sell & range_ok

    s = pd.Series(0, index=df.index)
    s[buy] = 1
    s[sell] = -1
    s[df.index.hour.isin(BLOCKED_HOURS)] = 0
    return s

# ═══════════ SIMULATION ═══════════
def simulate(df, sigs, tp_pct, sl_pct):
    ca = df["Close"].values; ha = df["High"].values
    la = df["Low"].values;   oa = df["Open"].values
    sa = sigs.values;        n = len(df)
    idx = df.index
    trades = []; cooldown_until = 0; daily_pnl = {}
    i = 0
    while i < n - 1:
        if sa[i] == 0 or i < cooldown_until:
            i += 1; continue
        today = idx[i].date()
        if daily_pnl.get(today, 0.0) <= -DAILY_LOSS_LIMIT:
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
            j = ei+ti; ep = tp_p; reason = "TP"
        elif si < ti and si < n:
            j = ei+si; ep = sl_p; reason = "SL"
        else:
            j = n-1; ep = ca[j]; reason = "END"
        pnl = CAPITAL * d * (ep - entry) / entry * LEVERAGE - FEE
        daily_pnl[today] = daily_pnl.get(today, 0.0) + pnl
        trades.append({"pnl": pnl, "win": pnl>0, "reason": reason, "date": today})
        if reason == "SL": cooldown_until = j + SL_COOLDOWN
        i = j + 1
    return trades

def metrics(trades):
    if not trades:
        return {"trades":0,"win_rate":0.,"net_pnl":0.,"pf":0.,"max_dd":0.,"sharpe":0.}
    df = pd.DataFrame(trades)
    w = df[df["pnl"]>0]["pnl"]; l = df[df["pnl"]<=0]["pnl"].abs()
    gw, gl = w.sum(), l.sum()
    cum = df["pnl"].cumsum(); dd = (cum - cum.cummax()).min()
    dpnl = df.groupby("date")["pnl"].sum()
    sh = dpnl.mean()/dpnl.std()*math.sqrt(252) if len(dpnl)>1 and dpnl.std()>0 else 0
    return {"trades": len(df), "win_rate": 100*df["win"].mean(),
            "net_pnl": df["pnl"].sum(),
            "pf": gw/gl if gl>0 else (float("inf") if gw>0 else 0.),
            "max_dd": dd, "sharpe": round(sh,2)}

def run_wf(data_dict, pairs, vol_mult, range_cap):
    all_trades = []; win_sums = []; pair_oos = {p: [] for p in pairs}
    t_min = min(df.index.min() for df in data_dict.values())
    for wi, win in enumerate(WF_WINDOWS):
        train_start = t_min + pd.Timedelta(days=win["train_offset"])
        train_end   = train_start + pd.Timedelta(days=TRAIN_DAYS)
        oos_end     = train_end   + pd.Timedelta(days=TEST_DAYS)
        win_trades  = []
        for pair in pairs:
            df_full = data_dict[pair]
            df_train = df_full[(df_full.index >= train_start) & (df_full.index < train_end)]
            df_oos   = df_full[(df_full.index >= train_end) & (df_full.index < oos_end)]
            if len(df_train) < 200 or df_oos.empty: continue
            sigs_train = get_signals(df_train, vol_mult, range_cap)
            bests = []
            for tp, sl in itertools.product(TP_GRID, SL_GRID):
                if tp <= sl: continue
                t = simulate(df_train, sigs_train, tp, sl)
                bests.append({"pnl": sum(x["pnl"] for x in t), "tp": tp, "sl": sl})
            bests.sort(key=lambda x: x["pnl"], reverse=True)
            best_tp, best_sl = bests[0]["tp"], bests[0]["sl"]
            sigs_oos = get_signals(df_oos, vol_mult, range_cap)
            oos_t = simulate(df_oos, sigs_oos, best_tp, best_sl)
            for t in oos_t: t["pair"] = pair; t["window"] = wi+1
            win_trades.extend(oos_t); pair_oos[pair].extend(oos_t)
        wm = metrics(win_trades)
        win_sums.append(wm)
        all_trades.extend(win_trades)
    return all_trades, win_sums, pair_oos

# ═══════════ MAIN ═══════════
def main():
    t0 = time.time()
    all_syms = list(dict.fromkeys(PAIRS_PRUNED))
    raw = load_all(all_syms, "(pruned pairs)")
    data = {p: add_indicators(df) for p, df in raw.items()}
    avail = [p for p in PAIRS_PRUNED if p in data]

    combos = [
        {"label": "Vol1.2 Pruned",      "vol_mult": 1.2, "range_cap": False},
        {"label": "Vol1.5 Pruned",      "vol_mult": 1.5, "range_cap": False},
        {"label": "Vol1.2 Pruned +RangeCap", "vol_mult": 1.2, "range_cap": True},
    ]

    print(f"\n{'='*W}")
    print("  E16 FINAL — 3 Configurations (Pruned Pairs)")
    print(f"  Pairs loaded: {len(avail)}")
    print(f"  Data window: {data[avail[0]].index.min().date()} → {data[avail[0]].index.max().date()}")
    print(f"{'='*W}")

    results = []
    for idx, cfg in enumerate(combos):
        print(f"\n({idx+1}/3) {cfg['label']} ...", end=" ", flush=True)
        try:
            trades, win_sums, pair_oos = run_wf(
                data, avail, cfg["vol_mult"], cfg["range_cap"])
            m = metrics(trades)
            oos_days = len(WF_WINDOWS) * TEST_DAYS
            daily = m["net_pnl"] / oos_days if oos_days>0 else 0
            pos_wins = sum(1 for ws in win_sums if ws["net_pnl"]>0)
            results.append({
                **cfg,
                "trades": m["trades"], "wr": m["win_rate"], "net": m["net_pnl"],
                "pf": m["pf"], "daily": daily, "dd": m["max_dd"],
                "sharpe": m["sharpe"], "pos_wins": pos_wins,
                "win_pnls": [round(ws["net_pnl"],2) for ws in win_sums]
            })
            print(f"Done. PF:{pfs(m['pf'])} daily:${daily:.2f}")
        except Exception as e:
            print(f"FAILED: {e}")

    results.sort(key=lambda r: (r["pf"], r["daily"]), reverse=True)

    print(f"\n{'='*W}")
    print("  RESULTS — best first")
    print(f"{'='*W}")
    print(f"{'Rank':<5} {'Configuration':<30} {'PF':>7} {'Daily':>8} {'WR':>6} {'Trades':>6} {'MaxDD':>7} {'+Wins':>6}")
    print("-"*W)
    for i, r in enumerate(results, 1):
        print(f"{i:<5} {r['label']:<30} {pfs(r['pf'])} ${r['daily']:>7.2f} {r['wr']:>5.1f}% {r['trades']:>6} ${r['dd']:>6.2f} {r['pos_wins']:>2}/3")
    print("-"*W)
    best = results[0]
    print(f"\n  🏆 BEST: {best['label']} — PF {pfs(best['pf'])}  ${best['daily']:.2f}/day  ({best['pos_wins']}/3 windows)")

    with open("e16_final_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved → e16_final_results.json")
    print(f"  Run time: {time.time()-t0:.0f}s")

def sep(c="="): print(c*W)
def pfs(pf): return f"{pf:.3f}" if pf < 999 else "    ∞"

if __name__ == "__main__":
    main()
