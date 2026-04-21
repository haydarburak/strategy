"""
SAPAN STRATEJİSİ (SLINGSHOT STRATEGY) — v3  (COMPLETE)
Source: Kural Akademi - Sidar Demirgil (2021)

PDF 1 (DERS-10): Entry patterns — 2-candle reversal (original/inside/body-pierce) + 1-candle pinbar
PDF 2 (DERS-11): Trading rules  — indicator filters (EMA stack, Stoch, MACD, ATR), entry/exit mechanics
PDF 3 (DERS-12): Quality filters — higher lows/lower highs, EMA100 exception,
                                   counter-trend break, resistance zone avoidance

══════════════════════════════════════════════════════════════
COMPLETE 7-POINT CHECKLIST (LONG)
──────────────────────────────────────────────────────────────
1. Trend    : EMA20 > EMA50 > EMA100 > EMA200  (full stack)
2. Pattern  : 2-candle OR 1-candle pinbar reversal at EMA20
3. Confirm  : Bullish confirmation candle closes above reversal high
4. MACD     : MACD(50,100,9) bullish  OR  turned bearish ≤5 bars ago
5. Stoch    : Stochastic(5,3,3) K < 30  (oversold)
6. Higher   : Reversal low > previous reversal low  (yükselen dip)
              Exception: if reversal low touches EMA100 → still valid
7. Counter  : Confirmation bar closes above max high of last 5 pullback bars
              (ters trend kırılımı)
  + Bonus   : No major resistance between entry and TP (direnç bölgesi)

SHORT is the exact mirror image (lower highs, stoch >70, MACD bearish).

Entry  : Stop order at confirmation HIGH (long) / LOW (short)
Stop   : Reversal candle LOW (long) / HIGH (short)
Target : Entry ± 2R
ATR    : 1R must be between 0.5×ATR and 2.0×ATR
══════════════════════════════════════════════════════════════
"""

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────
CONFIG = {
    "symbol":            "SPY",
    "start":             "2018-01-01",
    "end":               "2024-12-31",
    "interval":          "1d",
    "initial_capital":   10_000,
    "risk_per_trade":    0.01,          # 1% of capital per trade
    # Indicators
    "stoch_k":           5,
    "stoch_ks":          3,
    "stoch_d":           3,
    "stoch_ob":          70,
    "stoch_os":          30,
    "macd_fast":         50,
    "macd_slow":         100,
    "macd_sig":          9,
    "macd_max_bars":     5,
    "atr_period":        14,
    "atr_min_mult":      0.5,
    "atr_max_mult":      2.0,
    "pinbar_wick_ratio": 2.0,
    # PDF-3 filters
    "higher_low_lookback":   40,        # bars to look back for previous reversal low
    "counter_trend_bars":    6,         # pullback bars checked for counter-trend break
    "resistance_lookback":   30,        # bars to scan for resistance zones
    "resistance_clearance":  0.003,     # TP must clear resistance by this fraction
}


# ─────────────────────────────────────────────────────────────
#  INDICATORS
# ─────────────────────────────────────────────────────────────
def add_indicators(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    # EMAs
    for p in [20, 50, 100, 200]:
        df[f"ema{p}"] = df["Close"].ewm(span=p, adjust=False).mean()

    # Trend: all 4 EMAs stacked
    df["uptrend"]   = ((df["ema20"] > df["ema50"]) &
                       (df["ema50"] > df["ema100"]) &
                       (df["ema100"] > df["ema200"]))
    df["downtrend"] = ((df["ema20"] < df["ema50"]) &
                       (df["ema50"] < df["ema100"]) &
                       (df["ema100"] < df["ema200"]))

    # Stochastic (5, 3, 3)
    lo = df["Low"].rolling(cfg["stoch_k"]).min()
    hi = df["High"].rolling(cfg["stoch_k"]).max()
    raw = 100 * (df["Close"] - lo) / (hi - lo + 1e-12)
    df["stoch_k_val"] = raw.rolling(cfg["stoch_ks"]).mean()
    df["stoch_d_val"] = df["stoch_k_val"].rolling(cfg["stoch_d"]).mean()

    # MACD (50, 100, 9) + consecutive bars counter
    ema_f           = df["Close"].ewm(span=cfg["macd_fast"], adjust=False).mean()
    ema_s           = df["Close"].ewm(span=cfg["macd_slow"], adjust=False).mean()
    df["macd_line"] = ema_f - ema_s
    df["macd_sig"]  = df["macd_line"].ewm(span=cfg["macd_sig"], adjust=False).mean()
    df["macd_hist"] = df["macd_line"] - df["macd_sig"]
    signs   = np.sign(df["macd_hist"].fillna(0).values)
    consec  = np.ones(len(signs), dtype=int)
    for i in range(1, len(signs)):
        consec[i] = consec[i - 1] + 1 if signs[i] == signs[i - 1] else 1
    df["macd_consec"] = consec

    # ATR (14)
    hl  = df["High"] - df["Low"]
    hpc = (df["High"] - df["Close"].shift(1)).abs()
    lpc = (df["Low"]  - df["Close"].shift(1)).abs()
    df["atr"] = pd.concat([hl, hpc, lpc], axis=1).max(axis=1).ewm(
        span=cfg["atr_period"], adjust=False).mean()

    return df


# ─────────────────────────────────────────────────────────────
#  PATTERN HELPERS
# ─────────────────────────────────────────────────────────────
def _bt(c):   return max(c["Open"], c["Close"])
def _bb(c):   return min(c["Open"], c["Close"])
def _lw(c):   return _bb(c) - c["Low"]
def _uw(c):   return c["High"] - _bt(c)
def _bd(c):   return abs(c["Close"] - c["Open"])
def _rg(c):   return c["High"] - c["Low"]
def _bull(c): return c["Close"] > c["Open"]
def _bear(c): return c["Close"] < c["Open"]


def _counter_trend_level(df, c2_iloc, c3_iloc, ct_bars, side="long"):
    """
    Pullback içindeki 2 en son pivot high (long) veya pivot low (short)
    noktasından doğrusal trendline çizer ve c3 konumundaki değeri döner.
    """
    start = max(0, c2_iloc - ct_bars)
    end   = c2_iloc + 1
    col   = "High" if side == "long" else "Low"
    vals  = df[col].iloc[start:end].values
    n     = len(vals)

    pivots = []
    for j in range(1, n - 1):
        if side == "long":
            if vals[j] >= vals[j - 1] and vals[j] >= vals[j + 1]:
                pivots.append((start + j, float(vals[j])))
        else:
            if vals[j] <= vals[j - 1] and vals[j] <= vals[j + 1]:
                pivots.append((start + j, float(vals[j])))

    if len(pivots) >= 2:
        p1, p2 = pivots[-2], pivots[-1]
    else:
        order = sorted(range(n), key=lambda j: -vals[j] if side == "long" else vals[j])
        if len(order) >= 2:
            a, b = sorted([start + order[0], start + order[1]])
            p1 = (a, float(df[col].iloc[a]))
            p2 = (b, float(df[col].iloc[b]))
        else:
            return float(df[col].iloc[c2_iloc])

    if p1[0] == p2[0]:
        return p2[1]

    slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
    return p2[1] + slope * (c3_iloc - p2[0])


def _is_pinbar_long(c, ema, ratio):
    if _rg(c) == 0:
        return False
    lw = _lw(c); b = _bd(c)
    return (c["Close"] > ema and c["Low"] < ema and
            ((lw >= ratio * b if b > 0 else lw > 0) or lw / _rg(c) >= 0.60))


def _is_pinbar_short(c, ema, ratio):
    if _rg(c) == 0:
        return False
    uw = _uw(c); b = _bd(c)
    return (c["Close"] < ema and c["High"] > ema and
            ((uw >= ratio * b if b > 0 else uw > 0) or uw / _rg(c) >= 0.60))


# ─────────────────────────────────────────────────────────────
#  SIGNAL DETECTION  (all 7 filters)
# ─────────────────────────────────────────────────────────────
def detect_signals(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Bar layout at detection point i:
      c2 = df[i-2]  reversal candle
      c3 = df[i-1]  confirmation candle  ← signal stamped here
      entry triggered next bar (i) if stop order is hit
    """
    ratio        = cfg["pinbar_wick_ratio"]
    stoch_os     = cfg["stoch_os"]
    stoch_ob     = cfg["stoch_ob"]
    macd_max     = cfg["macd_max_bars"]
    atr_min      = cfg["atr_min_mult"]
    atr_max      = cfg["atr_max_mult"]
    hl_lookback  = cfg["higher_low_lookback"]
    ct_bars      = cfg["counter_trend_bars"]
    res_lb       = cfg["resistance_lookback"]
    res_clear    = cfg["resistance_clearance"]
    use_idx      = "idx_uptrend" in df.columns   # BIST 100 index filter active

    # Stateful: track last confirmed reversal extremes for higher-low/lower-high rule
    last_rev_low_long   = -np.inf
    last_rev_high_short =  np.inf

    signals, entries, stops, targets, stypes = [], [], [], [], []
    filter_stats = {"stoch": 0, "macd": 0, "atr": 0,
                    "higher_low": 0, "counter_trend": 0, "resistance": 0,
                    "index_trend": 0, "passed": 0}

    min_idx = 205
    pad     = min_idx

    for i in range(min_idx, len(df)):
        c3 = df.iloc[i - 1]   # confirmation
        c2 = df.iloc[i - 2]   # reversal
        c1 = df.iloc[i - 3]   # prior bar

        atr    = c3["atr"]

        sig   = 0
        entry = stop = tp = stype = None

        # ═══════════════════  LONG  ═══════════════════════
        if c2["uptrend"]:

            # ── Index trend filter: BIST 100 must be in uptrend ─
            if use_idx and not c2.get("idx_uptrend", True):
                filter_stats["index_trend"] += 1
                signals.append(0); entries.append(None); stops.append(None)
                targets.append(None); stypes.append(None)
                continue

            # ── Filter 4: Stochastic oversold ──────────────
            if not (c2["stoch_k_val"] < stoch_os):
                filter_stats["stoch"] += 1
                signals.append(0); entries.append(None); stops.append(None)
                targets.append(None); stypes.append(None)
                continue

            # ── Filter 5: MACD direction ────────────────────
            if not ((c2["macd_hist"] > 0) or (c2["macd_consec"] <= macd_max)):
                filter_stats["macd"] += 1
                signals.append(0); entries.append(None); stops.append(None)
                targets.append(None); stypes.append(None)
                continue

            # ── Filters 2 & 3: Pattern + Confirmation ───────
            # EMA20 → EMA50 → EMA100 → EMA200 sırasıyla denenir,
            # ilk eşleşen EMA ve formasyon kabul edilir.
            conf_bull      = _bull(c3)
            conf_above_rev = c3["Close"] > c2["High"]
            rev_below_c1   = c2["Low"] < c1["Low"]

            for ema_p in (20, 50, 100, 200):
                ema_val = c2[f"ema{ema_p}"]
                lbl     = f"ema{ema_p}"

                rev_body_above = _bb(c2) > ema_val
                rev_wick_below = c2["Low"] < ema_val

                if (rev_body_above and rev_wick_below and rev_below_c1
                        and conf_bull and conf_above_rev):
                    sig, stype = 1, f"2c-original-long-{lbl}"; break

                if (_bb(c2) < ema_val < _bt(c2)
                        and rev_below_c1 and conf_bull and conf_above_rev):
                    sig, stype = 1, f"2c-body-pierce-long-{lbl}"; break

                if (c2["High"] < c1["High"] and rev_below_c1
                        and rev_body_above and rev_wick_below
                        and conf_bull and conf_above_rev):
                    sig, stype = 1, f"2c-inside-long-{lbl}"; break

                if (_is_pinbar_long(c2, ema_val, ratio)
                        and conf_bull and c3["Low"] > c2["Low"]
                        and c3["Close"] > c2["High"]):
                    sig, stype = 1, f"1c-pinbar-long-{lbl}"; break

            if sig == 1:
                entry = c3["High"]
                stop  = c2["Low"]
                risk  = entry - stop
                if risk <= 0:
                    sig = 0
                else:
                    tp = entry + 2 * risk

                    # ── Filter: ATR risk size ───────────────
                    if not (atr_min * atr <= risk <= atr_max * atr):
                        filter_stats["atr"] += 1; sig = 0

                    # ── Filter 6: Higher lows ───────────────
                    # Reversal pattern tespit edildi → referansı hemen kaydet.
                    # Böylece MACD/Stoch/ATR nedeniyle reddedilen reversallar da
                    # bir sonraki higher-low karşılaştırmasına dahil edilir.
                    if sig == 1:
                        prev_rev_low  = last_rev_low_long
                        last_rev_low_long = c2["Low"]   # her pattern tespitinde güncelle
                        any_ema_touch = any(
                            c2["Low"] <= c2[f"ema{p}"] * 1.01
                            for p in (20, 50, 100, 200)
                        )
                        higher_low_ok = (c2["Low"] > prev_rev_low) or any_ema_touch
                        if not higher_low_ok:
                            filter_stats["higher_low"] += 1; sig = 0

                    # ── Filter 7: Counter-trend break ───────
                    # Pullback'teki 2 pivot high'tan çizilen trendline hesaplanır;
                    # teyit mumu bu çizginin üzerinde kapanmalı.
                    if sig == 1:
                        tl_val = _counter_trend_level(df, i - 2, i - 1, ct_bars, "long")
                        if not (c3["Close"] > tl_val):
                            filter_stats["counter_trend"] += 1; sig = 0

                    # ── Filter: Resistance zone ─────────────
                    # No significant swing high between entry and TP
                    if sig == 1:
                        res_start = max(0, i - res_lb - 5)
                        res_end   = max(0, i - 5)
                        if res_start < res_end:
                            zone_highs = df["High"].iloc[res_start:res_end]
                            obstacle   = zone_highs.max()
                            # Resistance inside path = high is between entry and TP*(1-clearance)
                            if entry * (1 + res_clear) < obstacle < tp * (1 - res_clear):
                                filter_stats["resistance"] += 1; sig = 0

                    if sig == 1:
                        filter_stats["passed"] += 1

        # ═══════════════════  SHORT  ══════════════════════
        elif c2["downtrend"]:

            # ── Index trend filter: BIST 100 must be in downtrend ─
            if use_idx and not c2.get("idx_downtrend", True):
                filter_stats["index_trend"] += 1
                signals.append(0); entries.append(None); stops.append(None)
                targets.append(None); stypes.append(None)
                continue

            if not (c2["stoch_k_val"] > stoch_ob):
                filter_stats["stoch"] += 1
                signals.append(0); entries.append(None); stops.append(None)
                targets.append(None); stypes.append(None)
                continue

            if not ((c2["macd_hist"] < 0) or (c2["macd_consec"] <= macd_max)):
                filter_stats["macd"] += 1
                signals.append(0); entries.append(None); stops.append(None)
                targets.append(None); stypes.append(None)
                continue

            conf_bear      = _bear(c3)
            conf_below_rev = c3["Close"] < c2["Low"]
            rev_above_c1   = c2["High"] > c1["High"]

            for ema_p in (20, 50, 100, 200):
                ema_val = c2[f"ema{ema_p}"]
                lbl     = f"ema{ema_p}"

                rev_body_below = _bt(c2) < ema_val
                rev_wick_above = c2["High"] > ema_val

                if (rev_body_below and rev_wick_above and rev_above_c1
                        and conf_bear and conf_below_rev):
                    sig, stype = -1, f"2c-original-short-{lbl}"; break

                if (_bb(c2) < ema_val < _bt(c2)
                        and rev_above_c1 and conf_bear and conf_below_rev):
                    sig, stype = -1, f"2c-body-pierce-short-{lbl}"; break

                if (c2["Low"] > c1["Low"] and rev_above_c1
                        and rev_body_below and rev_wick_above
                        and conf_bear and conf_below_rev):
                    sig, stype = -1, f"2c-inside-short-{lbl}"; break

                if (_is_pinbar_short(c2, ema_val, ratio)
                        and conf_bear and c3["High"] < c2["High"]
                        and c3["Close"] < c2["Low"]):
                    sig, stype = -1, f"1c-pinbar-short-{lbl}"; break

            if sig == -1:
                entry = c3["Low"]
                stop  = c2["High"]
                risk  = stop - entry
                if risk <= 0:
                    sig = 0
                else:
                    tp = entry - 2 * risk

                    if not (atr_min * atr <= risk <= atr_max * atr):
                        filter_stats["atr"] += 1; sig = 0

                    if sig == -1:
                        prev_rev_high      = last_rev_high_short
                        last_rev_high_short = c2["High"]   # her pattern tespitinde güncelle
                        any_ema_touch = any(
                            c2["High"] >= c2[f"ema{p}"] * 0.99
                            for p in (20, 50, 100, 200)
                        )
                        lower_high_ok = (c2["High"] < prev_rev_high) or any_ema_touch
                        if not lower_high_ok:
                            filter_stats["higher_low"] += 1; sig = 0

                    if sig == -1:
                        # Pullback'teki 2 pivot low'dan çizilen trendline hesaplanır;
                        # teyit mumu bu çizginin altında kapanmalı.
                        tl_val = _counter_trend_level(df, i - 2, i - 1, ct_bars, "short")
                        if not (c3["Close"] < tl_val):
                            filter_stats["counter_trend"] += 1; sig = 0

                    if sig == -1:
                        res_start = max(0, i - res_lb - 5)
                        res_end   = max(0, i - 5)
                        if res_start < res_end:
                            zone_lows = df["Low"].iloc[res_start:res_end]
                            support   = zone_lows.min()
                            if tp * (1 + res_clear) < support < entry * (1 - res_clear):
                                filter_stats["resistance"] += 1; sig = 0

                    if sig == -1:
                        filter_stats["passed"] += 1

        signals.append(sig)
        entries.append(entry if sig != 0 else None)
        stops.append(stop   if sig != 0 else None)
        targets.append(tp   if sig != 0 else None)
        stypes.append(stype if sig != 0 else None)

    df["signal"]      = [0]    * pad + signals
    df["entry_price"] = [None] * pad + entries
    df["stop_loss"]   = [None] * pad + stops
    df["take_profit"] = [None] * pad + targets
    df["signal_type"] = [None] * pad + stypes

    print(f"\n  Filtre istatistikleri:")
    if filter_stats["index_trend"] > 0:
        print(f"    Endeks trend filtresi  : {filter_stats['index_trend']:>5} bar atlandı")
    print(f"    Stochastic filtresi    : {filter_stats['stoch']:>5} bar atlandı")
    print(f"    MACD filtresi          : {filter_stats['macd']:>5} bar atlandı")
    print(f"    ATR risk filtresi      : {filter_stats['atr']:>5} sinyal atlandı")
    print(f"    Yükselen dip filtresi  : {filter_stats['higher_low']:>5} sinyal atlandı")
    print(f"    Ters trend filtresi    : {filter_stats['counter_trend']:>5} sinyal atlandı")
    print(f"    Direnç bölgesi filtresi: {filter_stats['resistance']:>5} sinyal atlandı")
    print(f"    Geçen sinyal           : {filter_stats['passed']:>5}")

    return df


# ─────────────────────────────────────────────────────────────
#  BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────
def run_backtest(df: pd.DataFrame, cfg: dict) -> tuple:
    """
    Stop-order entry: long triggers when next bar's High >= entry_price.
    SL checked before TP on same bar. One position at a time.
    """
    capital   = cfg["initial_capital"]
    risk_frac = cfg["risk_per_trade"]
    trades, equity, in_trade, trade = [], [], False, {}

    for idx, row in df.iterrows():
        if in_trade:
            d   = trade["direction"]
            rsk = trade["risk_amount"]

            if d == "long":
                if trade.get("pending"):
                    trade["pending_bars"] = trade.get("pending_bars", 0) + 1
                    if row["High"] >= trade["entry_price"]:
                        trade["pending"] = False
                    elif trade["pending_bars"] >= 3:
                        trade.update(exit_price=trade["entry_price"], exit_date=idx,
                                     pnl_dollar=0, result="EXPIRED", exit_capital=capital)
                        trades.append(trade); in_trade = False
                else:
                    if row["Low"] <= trade["sl"]:
                        capital += -rsk
                        trade.update(exit_price=trade["sl"], exit_date=idx,
                                     pnl_dollar=-rsk, result="SL", exit_capital=capital)
                        trades.append(trade); in_trade = False
                    elif row["High"] >= trade["tp"]:
                        capital += 2 * rsk
                        trade.update(exit_price=trade["tp"], exit_date=idx,
                                     pnl_dollar=2 * rsk, result="TP", exit_capital=capital)
                        trades.append(trade); in_trade = False
            else:
                if trade.get("pending"):
                    trade["pending_bars"] = trade.get("pending_bars", 0) + 1
                    if row["Low"] <= trade["entry_price"]:
                        trade["pending"] = False
                    elif trade["pending_bars"] >= 3:
                        trade.update(exit_price=trade["entry_price"], exit_date=idx,
                                     pnl_dollar=0, result="EXPIRED", exit_capital=capital)
                        trades.append(trade); in_trade = False
                else:
                    if row["High"] >= trade["sl"]:
                        capital += -rsk
                        trade.update(exit_price=trade["sl"], exit_date=idx,
                                     pnl_dollar=-rsk, result="SL", exit_capital=capital)
                        trades.append(trade); in_trade = False
                    elif row["Low"] <= trade["tp"]:
                        capital += 2 * rsk
                        trade.update(exit_price=trade["tp"], exit_date=idx,
                                     pnl_dollar=2 * rsk, result="TP", exit_capital=capital)
                        trades.append(trade); in_trade = False

        if (not in_trade
                and row["signal"] != 0
                and row["entry_price"] is not None):
            rsk   = capital * risk_frac
            d     = "long" if row["signal"] == 1 else "short"
            trade = {
                "entry_date":    idx,
                "entry_price":   row["entry_price"],
                "sl":            row["stop_loss"],
                "tp":            row["take_profit"],
                "direction":     d,
                "signal_type":   row["signal_type"],
                "risk_amount":   rsk,
                "entry_capital": capital,
                "pending":       True,
            }
            in_trade = True

        equity.append(capital)

    if in_trade:
        last = df.iloc[-1]; lp = last["Close"]
        d = trade["direction"]; rm = abs(trade["entry_price"] - trade["sl"])
        pm = (lp - trade["entry_price"]) if d == "long" else (trade["entry_price"] - lp)
        pnl = (pm / rm * trade["risk_amount"]) if rm > 0 else 0
        capital += pnl
        trade.update(exit_price=lp, exit_date=last.name,
                     pnl_dollar=pnl, result="OPEN", exit_capital=capital)
        trades.append(trade)

    return trades, equity


# ─────────────────────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────────────────────
def compute_metrics(trades, initial_capital, equity):
    closed = [t for t in trades if t["result"] in ("TP", "SL")]
    if not closed:
        return {}
    wins   = [t for t in closed if t["result"] == "TP"]
    losses = [t for t in closed if t["result"] == "SL"]
    gp     = sum(t["pnl_dollar"] for t in wins)
    gl     = abs(sum(t["pnl_dollar"] for t in losses))
    longs  = [t for t in closed if t["direction"] == "long"]
    shorts = [t for t in closed if t["direction"] == "short"]
    eq     = np.array(equity)
    roll   = np.maximum.accumulate(eq)
    mdd    = ((roll - eq) / (roll + 1e-12)).max() * 100
    final  = equity[-1]
    def _wr(lst): return sum(1 for t in lst if t["result"] == "TP") / len(lst) * 100 if lst else 0.0
    by_type: dict = {}
    for t in closed:
        st = t.get("signal_type") or "unknown"
        bt = by_type.setdefault(st, {"wins": 0, "losses": 0, "pnl": 0.0})
        bt["pnl"] += t["pnl_dollar"]
        bt["wins" if t["result"] == "TP" else "losses"] += 1
    return {
        "total_trades":   len(closed),   "long_trades":    len(longs),
        "short_trades":   len(shorts),   "wins":           len(wins),
        "losses":         len(losses),   "win_rate":       _wr(closed),
        "long_win_rate":  _wr(longs),    "short_win_rate": _wr(shorts),
        "gross_profit":   gp,            "gross_loss":     gl,
        "profit_factor":  gp / gl if gl else float("inf"),
        "avg_win":        gp / len(wins) if wins else 0.0,
        "avg_loss":       gl / len(losses) if losses else 0.0,
        "max_drawdown":   mdd,
        "total_return":   (final - initial_capital) / initial_capital * 100,
        "initial_capital": initial_capital, "final_capital": final,
        "by_signal_type": by_type,
    }


# ─────────────────────────────────────────────────────────────
#  REPORTING
# ─────────────────────────────────────────────────────────────
def print_report(m, trades, cfg):
    s = "=" * 66
    print(f"\n{s}")
    print(f"   SAPAN STRATEJİSİ v3 (TAM) — BACKTEST RAPORU")
    print(f"   {cfg['symbol']} | {cfg['start']} → {cfg['end']} | {cfg['interval']}")
    print(s)
    print(f"\n  {'Başlangıç Sermayesi':<32} ${m['initial_capital']:>10,.2f}")
    print(f"  {'Bitiş Sermayesi':<32} ${m['final_capital']:>10,.2f}")
    print(f"  {'Toplam Getiri':<32} {m['total_return']:>10.2f}%")
    print(f"  {'Maksimum Drawdown':<32} {m['max_drawdown']:>10.2f}%")
    print(f"  {'Kâr Faktörü':<32} {m['profit_factor']:>10.2f}")
    print(f"\n  {'Toplam İşlem':<32} {m['total_trades']:>10}")
    print(f"  {'  Long':<32} {m['long_trades']:>10}")
    print(f"  {'  Short':<32} {m['short_trades']:>10}")
    print(f"  {'Kârlı / Zararlı':<32} {m['wins']:>5} / {m['losses']}")
    print(f"  {'Kazanma Oranı':<32} {m['win_rate']:>10.1f}%")
    print(f"  {'  Long Kazanma Oranı':<32} {m['long_win_rate']:>10.1f}%")
    print(f"  {'  Short Kazanma Oranı':<32} {m['short_win_rate']:>10.1f}%")
    print(f"  {'Ortalama Kâr':<32} ${m['avg_win']:>10,.2f}")
    print(f"  {'Ortalama Zarar':<32} ${m['avg_loss']:>10,.2f}")
    print(f"\n  {'7 NOKTA FİLTRE ÖZETI':^62}")
    print(f"  {'-'*62}")
    print(f"  {'1. EMA Stack (20>50>100>200)':<40} ✓ uygulandı")
    print(f"  {'2. Price Action Paterni':<40} ✓ uygulandı")
    print(f"  {'3. Teyit Mum Çubuğu':<40} ✓ uygulandı")
    print(f"  {'4. MACD(50,100,9) yön uyumu':<40} ✓ uygulandı")
    print(f"  {'5. Stochastic(5,3,3) aşırı al/sat':<40} ✓ uygulandı")
    print(f"  {'6. Yükselen dip / Düşen tepe':<40} ✓ uygulandı (EMA100 istisnası)")
    print(f"  {'7. Ters trend kırılımı':<40} ✓ uygulandı")
    print(f"  {'+ Direnç bölgesi':<40} ✓ uygulandı")
    print(f"\n  {'SİNYAL TİPİ BAZINDA PERFORMANS':^62}")
    print(f"  {'-'*62}")
    print(f"  {'Sinyal Tipi':<28} {'İşlem':>6}  {'Kazanma':>8}  {'PnL':>12}")
    print(f"  {'-'*62}")
    for st, d in m["by_signal_type"].items():
        tot = d["wins"] + d["losses"]
        wr  = d["wins"] / tot * 100 if tot else 0
        print(f"  {st:<28} {tot:>6}  {wr:>7.1f}%  ${d['pnl']:>10,.2f}")
    print(f"\n  SON 10 İŞLEM:")
    print(f"  {'-'*84}")
    print(f"  {'#':<4} {'Tarih':<12} {'Yön':<6} {'Tip':<26} {'Giriş':>8} {'SL':>8} {'Sonuç':<5} {'PnL':>9}")
    print(f"  {'-'*84}")
    for k, t in enumerate(trades[-10:], 1):
        dt = str(t["entry_date"])[:10]
        print(f"  {k:<4} {dt:<12} {t['direction']:<6} {str(t.get('signal_type','')):<26} "
              f"{t['entry_price']:>8.2f} {t['sl']:>8.2f} {t['result']:<5} ${t.get('pnl_dollar',0):>8.2f}")
    print(s)


# ─────────────────────────────────────────────────────────────
#  VISUALISATION
# ─────────────────────────────────────────────────────────────
def plot_results(df, trades, equity, m, cfg):
    sym = cfg["symbol"]
    fig = plt.figure(figsize=(24, 22))
    fig.suptitle(
        f"Sapan Stratejisi v3 (Tam 7-Nokta) — {sym}  ({cfg['start']} → {cfg['end']})\n"
        f"EMA Stack | Stoch(5,3,3) | MACD(50,100,9) | ATR | Yükselen Dip | Ters Trend | Direnç",
        fontsize=13, fontweight="bold", y=0.99)

    gs = gridspec.GridSpec(5, 2, hspace=0.48, wspace=0.28,
                           top=0.94, bottom=0.05, left=0.06, right=0.97)
    ax_p    = fig.add_subplot(gs[0, :])
    ax_macd = fig.add_subplot(gs[1, :])
    ax_sto  = fig.add_subplot(gs[2, :])
    ax_eq   = fig.add_subplot(gs[3, :])
    ax_wr   = fig.add_subplot(gs[4, 0])
    ax_pl   = fig.add_subplot(gs[4, 1])

    # ── Price + EMAs ─────────────────────────────────────────
    ax_p.plot(df.index, df["Close"],   color="black",   lw=0.9, label="Close")
    ax_p.plot(df.index, df["ema20"],   color="#27ae60", lw=1.3, label="EMA20",  alpha=0.9)
    ax_p.plot(df.index, df["ema50"],   color="#2980b9", lw=1.3, label="EMA50",  alpha=0.9)
    ax_p.plot(df.index, df["ema100"],  color="#e67e22", lw=1.0, label="EMA100", alpha=0.8)
    ax_p.plot(df.index, df["ema200"],  color="#7f8c8d", lw=1.0, label="EMA200", alpha=0.7)
    for t in trades:
        mk  = "^" if t["direction"] == "long" else "v"
        clr = "#27ae60" if t["result"] == "TP" else ("#c0392b" if t["result"] == "SL" else "#95a5a6")
        ax_p.scatter(t["entry_date"], t["entry_price"], marker=mk,
                     color=clr, s=80, zorder=5, edgecolors="white", linewidths=0.5)
    ax_p.set_title("Price + EMAs + Entries  (▲=Long | ▼=Short | ■green=TP | ■red=SL)", fontweight="bold")
    ax_p.legend(loc="upper left", fontsize=8, ncol=5)
    ax_p.grid(alpha=0.2)

    # ── MACD ─────────────────────────────────────────────────
    ax_macd.plot(df.index, df["macd_line"], color="#2980b9", lw=1.0, label="MACD(50,100)")
    ax_macd.plot(df.index, df["macd_sig"],  color="#e74c3c", lw=1.0, label="Signal(9)")
    ax_macd.bar(df.index, df["macd_hist"],
                color=["#27ae60" if v >= 0 else "#c0392b" for v in df["macd_hist"]],
                alpha=0.4, label="Hist")
    ax_macd.axhline(0, color="black", lw=0.7)
    ax_macd.set_title("MACD (50, 100, 9)", fontweight="bold")
    ax_macd.legend(loc="upper left", fontsize=8)
    ax_macd.grid(alpha=0.2)

    # ── Stochastic ────────────────────────────────────────────
    ax_sto.plot(df.index, df["stoch_k_val"], color="#8e44ad", lw=1.0, label="K(5,3)")
    ax_sto.plot(df.index, df["stoch_d_val"], color="#f39c12", lw=1.0, label="D(3)", alpha=0.8)
    ax_sto.axhline(cfg["stoch_ob"], color="#c0392b", ls="--", lw=0.8, alpha=0.7)
    ax_sto.axhline(cfg["stoch_os"], color="#27ae60", ls="--", lw=0.8, alpha=0.7)
    ax_sto.fill_between(df.index, df["stoch_k_val"], cfg["stoch_ob"],
                        where=df["stoch_k_val"] > cfg["stoch_ob"], color="#c0392b", alpha=0.1)
    ax_sto.fill_between(df.index, df["stoch_k_val"], cfg["stoch_os"],
                        where=df["stoch_k_val"] < cfg["stoch_os"], color="#27ae60", alpha=0.1)
    ax_sto.set_ylim(0, 100)
    ax_sto.set_title("Stochastic (5, 3, 3)  — OS<30 (long) | OB>70 (short)", fontweight="bold")
    ax_sto.legend(loc="upper left", fontsize=8)
    ax_sto.grid(alpha=0.2)

    # ── Equity ───────────────────────────────────────────────
    eq_idx = df.index[:len(equity)]
    init   = cfg["initial_capital"]
    ax_eq.plot(eq_idx, equity, color="#2980b9", lw=1.5)
    ax_eq.axhline(init, color="gray", ls="--", alpha=0.5)
    ax_eq.fill_between(eq_idx, equity, init,
                       where=[e >= init for e in equity], color="#27ae60", alpha=0.15)
    ax_eq.fill_between(eq_idx, equity, init,
                       where=[e <  init for e in equity], color="#c0392b", alpha=0.15)
    box = (f"Getiri  : {m['total_return']:.1f}%\n"
           f"Max DD  : {m['max_drawdown']:.1f}%\n"
           f"PF      : {m['profit_factor']:.2f}\n"
           f"Kazanma : {m['win_rate']:.1f}%\n"
           f"İşlem   : {m['total_trades']}")
    ax_eq.text(0.01, 0.97, box, transform=ax_eq.transAxes, fontsize=8.5,
               va="top", fontfamily="monospace",
               bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.85))
    ax_eq.set_title("Equity Curve", fontweight="bold")
    ax_eq.set_ylabel("Sermaye ($)")
    ax_eq.grid(alpha=0.2)

    # ── Win Rate by Type ─────────────────────────────────────
    bt = m.get("by_signal_type", {})
    if bt:
        labs = list(bt.keys())
        wrs  = [bt[l]["wins"] / (bt[l]["wins"] + bt[l]["losses"]) * 100
                if (bt[l]["wins"] + bt[l]["losses"]) > 0 else 0 for l in labs]
        bars = ax_wr.bar(range(len(labs)), wrs,
                         color=["#27ae60" if w >= 50 else "#c0392b" for w in wrs],
                         alpha=0.8, edgecolor="white")
        ax_wr.set_xticks(range(len(labs)))
        ax_wr.set_xticklabels(labs, rotation=28, ha="right", fontsize=7.5)
        ax_wr.axhline(50, color="gray", ls="--", alpha=0.6)
        ax_wr.set_ylim(0, 100)
        ax_wr.set_title("Sinyal Tipi Kazanma Oranı (%)", fontweight="bold")
        for bar, val in zip(bars, wrs):
            ax_wr.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                       f"{val:.0f}%", ha="center", va="bottom", fontsize=8)
        ax_wr.grid(alpha=0.2, axis="y")

    # ── P&L per Trade ─────────────────────────────────────────
    closed = [t for t in trades if t["result"] in ("TP", "SL")]
    if closed:
        pnls = [t["pnl_dollar"] for t in closed]
        ax_pl.bar(range(len(pnls)), pnls,
                  color=["#27ae60" if p > 0 else "#c0392b" for p in pnls],
                  alpha=0.75, edgecolor="white", lw=0.3)
        ax_pl.axhline(0, color="black", lw=0.8)
        ax_pl.set_title("İşlem Başına P&L ($)", fontweight="bold")
        ax_pl.set_xlabel("İşlem #")
        ax_pl.grid(alpha=0.2, axis="y")
        ax2 = ax_pl.twinx()
        ax2.plot(range(len(pnls)), np.cumsum(pnls), color="#8e44ad", lw=1.3)
        ax2.set_ylabel("Kümülatif P&L ($)", color="#8e44ad")
        ax2.tick_params(axis="y", labelcolor="#8e44ad")

    fname = f"sapan_backtest_{sym}_v3.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nGrafik kaydedildi: {fname}")


# ─────────────────────────────────────────────────────────────
#  DETAILED TRADE CHART
# ─────────────────────────────────────────────────────────────
def plot_trade_details(symbol, index_symbol, cfg, context_bars=80, output_file=None):
    """
    Detailed single-stock chart showing every trade on the price series.
    Layout (top→bottom):
      1. Candlestick + EMAs + trade levels (entry / SL / TP)
      2. Stochastic (5,3,3)
      3. MACD (50,100,9)
      4. Index (index_symbol) + EMAs + uptrend shading
    """
    import matplotlib.patches as mpatches
    import matplotlib.dates  as mdates

    print(f"\nDetaylı trade grafiği oluşturuluyor: {symbol} | endeks: {index_symbol}")

    # ── Download & process stock ──────────────────────────────
    df = yf.download(symbol, start=cfg["start"], end=cfg["end"],
                     interval=cfg["interval"], auto_adjust=True, progress=False)
    if df.empty:
        print(f"  [HATA] {symbol} verisi bulunamadı."); return
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = add_indicators(df, cfg)

    # ── Download & process index ──────────────────────────────
    idx = yf.download(index_symbol, start=cfg["start"], end=cfg["end"],
                      interval=cfg["interval"], auto_adjust=True, progress=False)
    if isinstance(idx.columns, pd.MultiIndex):
        idx.columns = idx.columns.get_level_values(0)
    for p in [20, 50, 100, 200]:
        idx[f"ema{p}"] = idx["Close"].ewm(span=p, adjust=False).mean()
    idx["uptrend"] = ((idx["ema20"] > idx["ema50"]) &
                      (idx["ema50"] > idx["ema100"]) &
                      (idx["ema100"] > idx["ema200"]))

    # Merge index trend into stock df
    df = df.join(idx[["idx_uptrend","idx_downtrend"]] if "idx_uptrend" in idx.columns
                 else idx[["uptrend"]].rename(columns={"uptrend":"idx_uptrend"}),
                 how="left")
    df["idx_uptrend"]   = df.get("idx_uptrend", pd.Series(False, index=df.index)).ffill().fillna(False)
    df["idx_downtrend"] = df.get("idx_downtrend", pd.Series(False, index=df.index)).ffill().fillna(False)

    df = detect_signals(df, cfg)
    trades, equity = run_backtest(df, cfg)
    if not trades:
        print("  Bu hisse için hiç işlem bulunamadı."); return

    closed = [t for t in trades if t["result"] in ("TP", "SL", "OPEN")]
    print(f"  {len(closed)} işlem bulundu.")

    # ── Zoom window: context_bars before first entry, after last exit ─
    entry_dates = [t["entry_date"] for t in closed]
    exit_dates  = [t["exit_date"]  for t in closed]
    all_dates   = list(df.index)

    first_idx = max(0, all_dates.index(min(entry_dates)) - context_bars)
    last_idx  = min(len(all_dates) - 1, all_dates.index(max(exit_dates)) + context_bars)
    dfz = df.iloc[first_idx : last_idx + 1]        # zoomed stock df
    idxz = idx.loc[dfz.index[0] : dfz.index[-1]]   # zoomed index df

    # ── Figure layout ─────────────────────────────────────────
    fig = plt.figure(figsize=(26, 20))
    fig.suptitle(
        f"Sapan Stratejisi — {symbol}  |  Endeks: {index_symbol}\n"
        f"{dfz.index[0].date()} → {dfz.index[-1].date()}  |  "
        f"{len(closed)} işlem",
        fontsize=14, fontweight="bold", y=0.99)

    gs = gridspec.GridSpec(
        4, 1, height_ratios=[5, 1.5, 1.5, 2],
        hspace=0.08, top=0.94, bottom=0.05, left=0.07, right=0.97)

    ax_price = fig.add_subplot(gs[0])
    ax_sto   = fig.add_subplot(gs[1], sharex=ax_price)
    ax_macd  = fig.add_subplot(gs[2], sharex=ax_price)
    ax_idx   = fig.add_subplot(gs[3])   # NOT shared — different dates possible

    # ─── Panel 1: Candlestick + EMAs ─────────────────────────
    def _draw_candles(ax, dfc):
        for dt, row in dfc.iterrows():
            bull = row["Close"] >= row["Open"]
            col  = "#26a69a" if bull else "#ef5350"
            # Body
            body_bot = min(row["Open"], row["Close"])
            body_h   = abs(row["Close"] - row["Open"])
            ax.add_patch(mpatches.Rectangle(
                (mdates.date2num(dt.to_pydatetime()) - 0.3, body_bot),
                0.6, max(body_h, row["Close"] * 0.0005),
                color=col, zorder=3))
            # Wick
            ax.vlines(mdates.date2num(dt.to_pydatetime()),
                      row["Low"], row["High"], color=col, lw=0.7, zorder=2)

    _draw_candles(ax_price, dfz)

    # EMAs
    ax_price.plot(dfz.index, dfz["ema20"],  color="#27ae60", lw=1.5, label="EMA20",  zorder=4)
    ax_price.plot(dfz.index, dfz["ema50"],  color="#2980b9", lw=1.5, label="EMA50",  zorder=4)
    ax_price.plot(dfz.index, dfz["ema100"], color="#e67e22", lw=1.2, label="EMA100", zorder=4)
    ax_price.plot(dfz.index, dfz["ema200"], color="#95a5a6", lw=1.0, label="EMA200", zorder=4)

    # ─── Trades overlay ──────────────────────────────────────
    TP_CLR  = "#27ae60"
    SL_CLR  = "#c0392b"
    OP_CLR  = "#7f8c8d"

    for k, t in enumerate(closed, 1):
        ed   = t["entry_date"]
        xd   = t["exit_date"]
        ep   = t["entry_price"]
        sl   = t["sl"]
        tp   = t["tp"]
        res  = t["result"]
        d    = t["direction"]
        clr  = TP_CLR if res == "TP" else (SL_CLR if res == "SL" else OP_CLR)

        # Skip if outside zoom window
        if ed < dfz.index[0] or xd > dfz.index[-1]:
            continue

        # Shaded trade zone
        ax_price.axvspan(ed, xd, alpha=0.08, color=clr, zorder=1)

        # Entry / SL / TP horizontal lines (only within trade span)
        for lvl, lclr, ls, lbl in [
            (ep, "#2980b9", "-",  f"Giriş #{k}"),
            (sl, SL_CLR,   "--", f"SL #{k}"),
            (tp, TP_CLR,   "--", f"TP #{k}"),
        ]:
            ax_price.hlines(lvl, ed, xd, colors=lclr, linestyles=ls, lw=1.4,
                            alpha=0.9, zorder=5)

        # Entry marker
        mk = "^" if d == "long" else "v"
        ax_price.scatter(ed, ep, marker=mk, s=150, color="#2980b9",
                         edgecolors="white", lw=0.8, zorder=7)

        # Exit marker
        exit_price = sl if res == "SL" else (tp if res == "TP" else t.get("exit_price", ep))
        ax_price.scatter(xd, exit_price, marker="o", s=100, color=clr,
                         edgecolors="white", lw=0.8, zorder=7)

        # Annotation box
        txt = (f"#{k} {d.upper()}  {t.get('signal_type','')}\n"
               f"Giriş: {ep:.2f}  SL: {sl:.2f}  TP: {tp:.2f}\n"
               f"Sonuç: {res}  PnL: ${t.get('pnl_dollar',0):.0f}")
        y_ann = ep * (1.01 if d == "long" else 0.99)
        ax_price.annotate(txt, xy=(ed, ep), xytext=(ed, y_ann),
                          fontsize=7.5, color="white",
                          bbox=dict(boxstyle="round,pad=0.3", fc=clr, alpha=0.82),
                          zorder=8)

    ax_price.set_xlim(dfz.index[0], dfz.index[-1])
    ax_price.xaxis_date()
    ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_price.set_ylabel("Fiyat", fontsize=10)
    ax_price.set_title(f"{symbol}  —  Fiyat + EMA Stack + İşlemler", fontweight="bold")
    ax_price.legend(loc="upper left", fontsize=8, ncol=5)
    ax_price.grid(alpha=0.15)
    plt.setp(ax_price.get_xticklabels(), visible=False)

    # ─── Panel 2: Stochastic ─────────────────────────────────
    ax_sto.plot(dfz.index, dfz["stoch_k_val"], color="#8e44ad", lw=1.0, label="K(5,3)")
    ax_sto.plot(dfz.index, dfz["stoch_d_val"], color="#f39c12", lw=0.8, label="D(3)")
    ax_sto.axhline(cfg["stoch_ob"], color=SL_CLR, ls="--", lw=0.8, alpha=0.7)
    ax_sto.axhline(cfg["stoch_os"], color=TP_CLR, ls="--", lw=0.8, alpha=0.7)
    ax_sto.fill_between(dfz.index, dfz["stoch_k_val"], cfg["stoch_os"],
                        where=dfz["stoch_k_val"] < cfg["stoch_os"], color=TP_CLR, alpha=0.12)
    ax_sto.fill_between(dfz.index, dfz["stoch_k_val"], cfg["stoch_ob"],
                        where=dfz["stoch_k_val"] > cfg["stoch_ob"], color=SL_CLR, alpha=0.12)
    ax_sto.set_ylim(0, 100)
    ax_sto.set_ylabel("Stoch", fontsize=8)
    ax_sto.legend(loc="upper left", fontsize=7)
    ax_sto.grid(alpha=0.15)
    plt.setp(ax_sto.get_xticklabels(), visible=False)

    # ─── Panel 3: MACD ───────────────────────────────────────
    ax_macd.plot(dfz.index, dfz["macd_line"], color="#2980b9", lw=1.0, label="MACD(50,100)")
    ax_macd.plot(dfz.index, dfz["macd_sig"],  color="#e74c3c", lw=0.8, label="Sinyal(9)")
    ax_macd.bar(dfz.index, dfz["macd_hist"],
                color=["#27ae60" if v >= 0 else "#c0392b" for v in dfz["macd_hist"]],
                alpha=0.45, label="Hist")
    ax_macd.axhline(0, color="black", lw=0.7)
    ax_macd.set_ylabel("MACD", fontsize=8)
    ax_macd.legend(loc="upper left", fontsize=7)
    ax_macd.grid(alpha=0.15)
    ax_macd.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax_macd.get_xticklabels(), visible=False)

    # ─── Panel 4: Index chart ────────────────────────────────
    ax_idx.plot(idxz.index, idxz["Close"],  color="black", lw=1.0, label="Close", zorder=3)
    ax_idx.plot(idxz.index, idxz["ema20"],  color="#27ae60", lw=1.2, label="EMA20",  zorder=4)
    ax_idx.plot(idxz.index, idxz["ema50"],  color="#2980b9", lw=1.2, label="EMA50",  zorder=4)
    ax_idx.plot(idxz.index, idxz["ema100"], color="#e67e22", lw=1.0, label="EMA100", zorder=4)
    ax_idx.plot(idxz.index, idxz["ema200"], color="#95a5a6", lw=0.9, label="EMA200", zorder=4)

    # Shade uptrend / downtrend background
    for i in range(len(idxz) - 1):
        dt0, dt1 = idxz.index[i], idxz.index[i + 1]
        if idxz["uptrend"].iloc[i]:
            ax_idx.axvspan(dt0, dt1, alpha=0.07, color=TP_CLR, zorder=1)
        elif not idxz["uptrend"].iloc[i]:
            ax_idx.axvspan(dt0, dt1, alpha=0.05, color=SL_CLR, zorder=1)

    # Mark trade entry/exit dates on index too
    for t in closed:
        if dfz.index[0] <= t["entry_date"] <= dfz.index[-1]:
            ax_idx.axvline(t["entry_date"], color="#2980b9", ls=":", lw=1.0, alpha=0.7)
        if dfz.index[0] <= t["exit_date"] <= dfz.index[-1]:
            clr = TP_CLR if t["result"] == "TP" else SL_CLR
            ax_idx.axvline(t["exit_date"], color=clr, ls=":", lw=1.0, alpha=0.7)

    ax_idx.set_xlim(idxz.index[0], idxz.index[-1])
    ax_idx.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_idx.set_ylabel("Fiyat", fontsize=10)
    ax_idx.set_title(f"{index_symbol}  —  Endeks Trendi  (yeşil=uptrend / kırmızı=downtrend)",
                     fontweight="bold")
    ax_idx.legend(loc="upper left", fontsize=8, ncol=5)
    ax_idx.grid(alpha=0.15)
    plt.setp(ax_idx.get_xticklabels(), rotation=30, ha="right")

    fname = output_file or f"sapan_trade_detail_{symbol.replace('^','').replace('.','_')}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Grafik kaydedildi: {fname}")
    return fname


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
def main(cfg=CONFIG):
    print(f"\nSapan Stratejisi v3 (Tam 7-Nokta Filtre)")
    print(f"Sembol : {cfg['symbol']} | {cfg['start']} → {cfg['end']} | {cfg['interval']}")
    print(f"Risk   : {cfg['risk_per_trade']*100:.1f}%/işlem | Sermaye: ${cfg['initial_capital']:,}")
    print("Veri indiriliyor...\n")

    df = yf.download(cfg["symbol"], start=cfg["start"], end=cfg["end"],
                     interval=cfg["interval"], auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"Veri bulunamadı: {cfg['symbol']}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    print(f"{len(df)} mum: {df.index[0].date()} → {df.index[-1].date()}")

    df = add_indicators(df, cfg)
    df = detect_signals(df, cfg)

    n_long  = (df["signal"] ==  1).sum()
    n_short = (df["signal"] == -1).sum()
    print(f"\nGeçen sinyaller → Long: {n_long} | Short: {n_short}")

    trades, equity = run_backtest(df, cfg)
    if not trades:
        print("Hiç işlem gerçekleşmedi.")
        return df, [], {}

    m = compute_metrics(trades, cfg["initial_capital"], equity)
    print_report(m, trades, cfg)
    plot_results(df, trades, equity, m, cfg)
    return df, trades, m


# ─────────────────────────────────────────────────────────────
#  OHLCV RESAMPLER
# ─────────────────────────────────────────────────────────────
def resample_ohlcv(df: pd.DataFrame, rule: str = "4h") -> pd.DataFrame:
    """Resample minute/hour OHLCV data to a larger timeframe (e.g. '4h')."""
    resampled = df.resample(rule, label="left", closed="left").agg({
        "Open":   "first",
        "High":   "max",
        "Low":    "min",
        "Close":  "last",
        "Volume": "sum",
    }).dropna(subset=["Close"])
    return resampled


# ─────────────────────────────────────────────────────────────
#  BIST MULTI-STOCK SCANNER
# ─────────────────────────────────────────────────────────────
BIST_TICKERS = [
    # Bankacılık & Finans
    "AKBNK.IS", "GARAN.IS", "ISCTR.IS", "YKBNK.IS", "HALKB.IS",
    "VAKBN.IS", "SKBNK.IS", "ALBRK.IS", "KLNMA.IS", "ISFIN.IS",
    # Holding
    "KCHOL.IS", "SAHOL.IS", "DOHOL.IS", "AGHOL.IS", "GLYHO.IS",
    "ALARK.IS", "NTHOL.IS", "POLHO.IS",
    # Sanayi / Çelik / Metal
    "EREGL.IS", "KRDMD.IS", "SARKY.IS", "TMSN.IS", "ALKIM.IS",
    # Enerji & Petrokimya
    "TUPRS.IS", "PETKM.IS", "AYGAZ.IS", "AKSEN.IS", "ENJSA.IS",
    "ODAS.IS", "CWENE.IS", "KONTR.IS", "SASA.IS",
    # Otomotiv
    "FROTO.IS", "TOASO.IS", "TTRAK.IS", "OTKAR.IS", "DOAS.IS",
    "BRISA.IS", "TOGG.IS",
    # Havacılık & Lojistik
    "THYAO.IS", "PGSUS.IS", "TAVHL.IS",
    # Teknoloji & Telecom & Savunma
    "ASELS.IS", "TCELL.IS", "TTKOM.IS", "NETAS.IS", "LOGO.IS",
    "INDES.IS", "SELEC.IS", "SMART.IS",
    # Perakende & Gıda & Tüketim
    "BIMAS.IS", "MGROS.IS", "SOKM.IS", "ULKER.IS", "AEFES.IS",
    "MAVI.IS", "TKFEN.IS", "YATAS.IS",
    # GYO (Gayrimenkul)
    "EKGYO.IS", "TRGYO.IS", "YKGYO.IS", "ZRGYO.IS", "MPARK.IS",
    # Cam & Çimento & İnşaat
    "SISE.IS", "TRKCM.IS", "CIMSA.IS", "CEMTS.IS", "SODA.IS",
    "ENKAI.IS",
    # Diğer Sanayi
    "ARCLK.IS", "VESTL.IS", "KORDS.IS", "HEKTS.IS", "GUBRF.IS",
    "GESAN.IS", "PRKAB.IS", "BFREN.IS", "IPEKE.IS", "BERA.IS",
    "OYAKC.IS", "KOZAL.IS", "KOZAA.IS", "TURSG.IS",
    "SNPAM.IS", "QUAGR.IS", "PRKME.IS", "FORTS.IS", "DNISI.IS",
]

BIST_CONFIG_BASE = {
    "start":             "2018-01-01",
    "end":               "2024-12-31",
    "interval":          "1d",
    "initial_capital":   10_000,
    "risk_per_trade":    0.01,
    "stoch_k": 5, "stoch_ks": 3, "stoch_d": 3,
    "stoch_ob": 70, "stoch_os": 30,
    "macd_fast": 50, "macd_slow": 100, "macd_sig": 9,
    "macd_max_bars": 5,
    "atr_period": 14, "atr_min_mult": 0.5, "atr_max_mult": 2.0,
    "pinbar_wick_ratio": 2.0,
    "higher_low_lookback":   40,
    "counter_trend_bars":    6,
    "resistance_lookback":   30,
    "resistance_clearance":  0.003,
}

# 4-saatlik tarama konfigürasyonu
# yfinance 60m verisi son 730 günü destekler → 4H'e resample
# NOT: BIST (.IS) hisseleri için Yahoo Finance intraday desteklemiyor.
#      Bu config yalnızca US/EU hisseleri için geçerlidir.
SCAN_4H_CONFIG = {
    **BIST_CONFIG_BASE,
    "start":    "2024-04-25",   # yfinance 730-gün 60m sınırı içinde
    "end":      "2026-04-18",
    "interval": "60m",          # indir
    "resample": "4h",           # 4 saatliğe çevir
}
BIST_4H_CONFIG = SCAN_4H_CONFIG   # backward compat alias


def _load_index_trend(index_symbol, base_cfg):
    """Download index and return DataFrame with idx_uptrend / idx_downtrend columns."""
    print(f"  Endeks verisi ({index_symbol}) indiriliyor...")
    idx = yf.download(index_symbol, start=base_cfg["start"], end=base_cfg["end"],
                      interval=base_cfg["interval"], auto_adjust=True, progress=False)
    if idx.empty:
        print(f"  [UYARI] {index_symbol} verisi alınamadı — endeks filtresi devre dışı.")
        return None
    if isinstance(idx.columns, pd.MultiIndex):
        idx.columns = idx.columns.get_level_values(0)
    if base_cfg.get("resample"):
        idx = resample_ohlcv(idx, base_cfg["resample"])
    for p in [20, 50, 100, 200]:
        idx[f"ema{p}"] = idx["Close"].ewm(span=p, adjust=False).mean()
    idx["idx_uptrend"]   = ((idx["ema20"] > idx["ema50"]) &
                            (idx["ema50"] > idx["ema100"]) &
                            (idx["ema100"] > idx["ema200"]))
    idx["idx_downtrend"] = ((idx["ema20"] < idx["ema50"]) &
                            (idx["ema50"] < idx["ema100"]) &
                            (idx["ema100"] < idx["ema200"]))
    tf = base_cfg.get("resample", base_cfg["interval"])
    print(f"  {index_symbol}: {len(idx)} mum ({idx.index[0].date()} → {idx.index[-1].date()}) [{tf}]")
    return idx[["idx_uptrend", "idx_downtrend"]]


# ─────────────────────────────────────────────────────────────
#  TICKER FETCHERS
# ─────────────────────────────────────────────────────────────
def _wiki_tables(url):
    """Fetch Wikipedia page and parse HTML tables (bypasses 403 with User-Agent)."""
    import io, urllib.request
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (compatible; SapanStrategy/1.0)"
    })
    with urllib.request.urlopen(req, timeout=20) as resp:
        html = resp.read()
    return pd.read_html(io.BytesIO(html))


def fetch_sp500_tickers():
    """Fetch current S&P 500 component tickers from Wikipedia."""
    try:
        tables = _wiki_tables("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        table  = tables[0]
        # Yahoo Finance uses '-' instead of '.' (e.g. BRK.B → BRK-B)
        tickers = table["Symbol"].str.replace(".", "-", regex=False).tolist()
        print(f"  S&P 500: {len(tickers)} hisse Wikipedia'dan alındı.")
        return tickers
    except Exception as e:
        print(f"  [UYARI] S&P 500 listesi alınamadı: {e}")
        return []


def fetch_nasdaq100_tickers():
    """Fetch current NASDAQ 100 component tickers from Wikipedia."""
    try:
        tables = _wiki_tables("https://en.wikipedia.org/wiki/Nasdaq-100")
        for t in tables:
            for col in ("Ticker", "Symbol"):
                if col in t.columns:
                    tickers = [str(x) for x in t[col].dropna().tolist()
                               if str(x) not in ("nan", "")]
                    if len(tickers) > 50:
                        print(f"  NASDAQ 100: {len(tickers)} hisse Wikipedia'dan alındı.")
                        return tickers
        raise ValueError("Uygun sütun bulunamadı")
    except Exception as e:
        print(f"  [UYARI] NASDAQ 100 listesi alınamadı: {e}")
        return []


def fetch_dax_tickers():
    """Fetch current DAX 40 component tickers from Wikipedia."""
    try:
        tables = _wiki_tables("https://en.wikipedia.org/wiki/DAX")
        for t in tables:
            if "Ticker" in t.columns and len(t) >= 30:
                tickers = [str(x) for x in t["Ticker"].dropna().tolist()
                           if str(x) not in ("nan", "")]
                print(f"  DAX 40: {len(tickers)} hisse Wikipedia'dan alındı.")
                return tickers
        raise ValueError("Ticker sütunu bulunamadı")
    except Exception as e:
        print(f"  [UYARI] DAX listesi alınamadı: {e}")
        return []


# ─────────────────────────────────────────────────────────────
#  GENERIC MARKET SCANNER
# ─────────────────────────────────────────────────────────────
def run_market_scan(tickers, index_symbol, base_cfg, market_label="Piyasa",
                    save_charts=False):
    """
    Universal Sapan strategy scanner.
      tickers       : list of Yahoo Finance ticker strings
      index_symbol  : benchmark index ticker (for trend filter)
      base_cfg      : strategy config dict
      market_label  : display name used in reports and filenames
      save_charts   : save per-stock PNG charts (slow for large universes)
    """
    summary_rows = []
    sep = "=" * 90
    tf = base_cfg.get("resample", base_cfg["interval"])

    print(f"\n{sep}")
    print(f"   SAPAN STRATEJİSİ — {market_label} TARAMASI  ({len(tickers)} hisse)")
    print(f"   {base_cfg['start']} → {base_cfg['end']} | {tf}")
    print(f"   Endeks filtresi: {index_symbol}  (EMA20 > EMA50 > EMA100 > EMA200)")
    print(sep)

    idx_trend = _load_index_trend(index_symbol, base_cfg)

    for i, ticker in enumerate(tickers, 1):
        cfg = {**base_cfg, "symbol": ticker}
        print(f"\n[{i:>3}/{len(tickers)}] {ticker}")

        try:
            df = yf.download(ticker, start=cfg["start"], end=cfg["end"],
                             interval=cfg["interval"], auto_adjust=True, progress=False)
            if df.empty:
                summary_rows.append({"Hisse": ticker, "İşlem": 0, "Kazanma%": 0,
                                     "PF": 0, "Getiri%": 0, "MaxDD%": 0, "Durum": "VERİ YOK"})
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if cfg.get("resample"):
                df = resample_ohlcv(df, cfg["resample"])
            if len(df) < 220:
                summary_rows.append({"Hisse": ticker, "İşlem": 0, "Kazanma%": 0,
                                     "PF": 0, "Getiri%": 0, "MaxDD%": 0, "Durum": "AZ VERİ"})
                continue

            df = add_indicators(df, cfg)

            if idx_trend is not None:
                df = df.join(idx_trend, how="left")
                df["idx_uptrend"]   = df["idx_uptrend"].ffill().fillna(False)
                df["idx_downtrend"] = df["idx_downtrend"].ffill().fillna(False)

            df = detect_signals(df, cfg)
            trades, equity = run_backtest(df, cfg)

            if not trades:
                summary_rows.append({"Hisse": ticker, "İşlem": 0, "Kazanma%": 0,
                                     "PF": 0, "Getiri%": 0, "MaxDD%": 0, "Durum": "SİNYAL YOK"})
                continue

            m = compute_metrics(trades, cfg["initial_capital"], equity)
            if not m:
                summary_rows.append({"Hisse": ticker, "İşlem": 0, "Kazanma%": 0,
                                     "PF": 0, "Getiri%": 0, "MaxDD%": 0, "Durum": "KAPALI YOK"})
                continue

            print_report(m, trades, cfg)
            if save_charts:
                plot_results(df, trades, equity, m, cfg)

            summary_rows.append({
                "Hisse":    ticker,
                "İşlem":    m["total_trades"],
                "Long":     m["long_trades"],
                "Short":    m["short_trades"],
                "Kazanma%": round(m["win_rate"], 1),
                "PF":       round(m["profit_factor"], 2) if m["profit_factor"] != float("inf") else 999,
                "Getiri%":  round(m["total_return"], 2),
                "MaxDD%":   round(m["max_drawdown"], 2),
                "Durum":    "OK",
            })

        except Exception as e:
            summary_rows.append({"Hisse": ticker, "İşlem": 0, "Kazanma%": 0,
                                 "PF": 0, "Getiri%": 0, "MaxDD%": 0, "Durum": f"HATA"})

    # ── Summary table ───────────────────────────────────────────
    ok_rows = [r for r in summary_rows if r["Durum"] == "OK"]
    skip_rows = [r for r in summary_rows if r["Durum"] != "OK"]

    print(f"\n\n{sep}")
    print(f"   {market_label} ÖZET RAPORU")
    print(f"   Toplam: {len(tickers)} | İşlem yapan: {len(ok_rows)} | Atlanan: {len(skip_rows)}")
    print(sep)
    print(f"  {'Hisse':<12} {'İşlem':>6} {'Long':>5} {'Short':>5} "
          f"{'Kazanma%':>9} {'PF':>7} {'Getiri%':>9} {'MaxDD%':>7}")
    print(f"  {'-'*72}")
    for r in sorted(ok_rows, key=lambda x: -x["Getiri%"]):
        pf_str = f"{r['PF']:.2f}" if r["PF"] < 999 else "  ∞  "
        print(f"  {r['Hisse']:<12} {r['İşlem']:>6} {r.get('Long',0):>5} {r.get('Short',0):>5}"
              f"  {r['Kazanma%']:>8.1f}% {pf_str:>7} {r['Getiri%']:>8.2f}% {r['MaxDD%']:>6.2f}%")
    if not ok_rows:
        print("  Geçerli sonuç bulunan hisse yok.")

    # Aggregate stats
    if ok_rows:
        total_trades = sum(r["İşlem"] for r in ok_rows)
        total_wins   = sum(round(r["İşlem"] * r["Kazanma%"] / 100) for r in ok_rows)
        total_losses = total_trades - total_wins
        agg_wr       = total_wins / total_trades * 100 if total_trades else 0
        net_r        = total_wins * 2 - total_losses
        print(f"\n  ─── KÜMÜLATİF İSTATİSTİKLER ───────────────────────────────")
        print(f"  Toplam işlem      : {total_trades}")
        print(f"  TP / SL           : {total_wins} / {total_losses}")
        print(f"  Kazanma oranı     : {agg_wr:.1f}%")
        print(f"  Net R kazancı     : +{net_r}R  ({total_wins}×2R − {total_losses}×1R)")

    # Skipped summary (brief)
    skip_counts: dict = {}
    for r in skip_rows:
        skip_counts[r["Durum"]] = skip_counts.get(r["Durum"], 0) + 1
    if skip_counts:
        print(f"\n  Atlananlar: " + " | ".join(f"{k}: {v}" for k, v in skip_counts.items()))

    # ── Summary chart ────────────────────────────────────────────
    if ok_rows:
        _plot_market_summary(ok_rows, market_label, base_cfg)

    print(f"{sep}\n")
    return summary_rows


def _plot_market_summary(ok_rows, market_label, base_cfg):
    """Save a 3-panel summary chart for a market scan."""
    fig, axes = plt.subplots(1, 3, figsize=(22, max(6, len(ok_rows) * 0.25 + 2)))
    fig.suptitle(
        f"Sapan Stratejisi — {market_label} Özeti  "
        f"({base_cfg['start']} → {base_cfg['end']})",
        fontsize=13, fontweight="bold")

    order   = sorted(range(len(ok_rows)), key=lambda i: -ok_rows[i]["Getiri%"])
    syms    = [ok_rows[i]["Hisse"] for i in order]
    returns = [ok_rows[i]["Getiri%"] for i in order]
    pfs     = [min(ok_rows[i]["PF"], 10) for i in order]

    # Return bars
    ax = axes[0]
    colors = ["#27ae60" if r >= 0 else "#c0392b" for r in returns]
    ax.barh(syms, returns, color=colors, alpha=0.8, edgecolor="white")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("Toplam Getiri (%)", fontweight="bold")
    ax.set_xlabel("%")
    ax.tick_params(axis="y", labelsize=max(4, min(8, 200 // len(ok_rows))))
    ax.grid(alpha=0.2, axis="x")

    # Profit Factor bars
    ax = axes[1]
    ax.barh(syms, pfs, color="#2980b9", alpha=0.8, edgecolor="white")
    ax.axvline(1, color="gray", ls="--", lw=0.8)
    ax.set_title("Kâr Faktörü (maks 10)", fontweight="bold")
    ax.set_xlabel("PF")
    ax.tick_params(axis="y", labelsize=max(4, min(8, 200 // len(ok_rows))))
    ax.grid(alpha=0.2, axis="x")

    # Win rate vs trades scatter
    ax = axes[2]
    trades_list = [ok_rows[i]["İşlem"] for i in range(len(ok_rows))]
    wr_list     = [ok_rows[i]["Kazanma%"] for i in range(len(ok_rows))]
    ret_list    = [ok_rows[i]["Getiri%"] for i in range(len(ok_rows))]
    scatter = ax.scatter(trades_list, wr_list, c=ret_list,
                         cmap="RdYlGn", s=80, edgecolors="black", linewidths=0.4,
                         vmin=-20, vmax=20, zorder=3)
    for r in ok_rows:
        if r["İşlem"] >= 3:
            ax.annotate(r["Hisse"], (r["İşlem"], r["Kazanma%"]),
                        textcoords="offset points", xytext=(4, 2), fontsize=6)
    plt.colorbar(scatter, ax=ax, label="Getiri%")
    ax.axhline(50, color="gray", ls="--", lw=0.8)
    ax.set_title("İşlem Sayısı vs Kazanma Oranı", fontweight="bold")
    ax.set_xlabel("İşlem Sayısı")
    ax.set_ylabel("Kazanma Oranı (%)")
    ax.grid(alpha=0.2)

    plt.tight_layout()
    fname = f"sapan_{market_label.lower().replace(' ', '_')}_ozet.png"
    plt.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n  Özet grafik kaydedildi: {fname}")


if __name__ == "__main__":
    import sys

    if "--plot" in sys.argv:
        # Usage: --plot TICKER --index INDEX_SYMBOL
        pidx = sys.argv.index("--plot")
        sym  = sys.argv[pidx + 1] if pidx + 1 < len(sys.argv) else "GE"
        # Determine index automatically from suffix, else use --index arg
        if "--index" in sys.argv:
            iidx  = sys.argv.index("--index")
            isym  = sys.argv[iidx + 1]
        elif sym.endswith(".IS"):
            isym = "XU100.IS"
        elif sym.endswith(".DE") or sym.endswith(".PA"):
            isym = "^GDAXI"
        else:
            isym = "SPY"   # default for US stocks
        plot_trade_details(sym, isym, CONFIG)

    elif "--all-4h" in sys.argv:
        # BIST: Yahoo Finance BIST için intraday desteklemiyor → atlanıyor
        print("\n[BİLGİ] BIST hisseleri için Yahoo Finance 60m veri sağlamıyor.")
        print("        4H taraması yalnızca S&P500 / NASDAQ100 / DAX40 için çalışır.\n")
        # ── S&P 500 ───────────────────────────────────────────
        sp500 = fetch_sp500_tickers()
        if sp500:
            run_market_scan(sp500, "SPY", SCAN_4H_CONFIG, "SP500-4H")
        # ── NASDAQ 100 ────────────────────────────────────────
        ndx100 = fetch_nasdaq100_tickers()
        if ndx100:
            run_market_scan(ndx100, "QQQ", SCAN_4H_CONFIG, "NASDAQ100-4H")
        # ── DAX 40 ────────────────────────────────────────────
        dax = fetch_dax_tickers()
        if dax:
            run_market_scan(dax, "^GDAXI", SCAN_4H_CONFIG, "DAX40-4H")

    elif "--all" in sys.argv:
        # ── BIST 100 ──────────────────────────────────────────
        run_market_scan(BIST_TICKERS, "XU100.IS", BIST_CONFIG_BASE, "BIST100")
        # ── S&P 500 (NYSE) ────────────────────────────────────
        sp500 = fetch_sp500_tickers()
        if sp500:
            run_market_scan(sp500, "SPY", BIST_CONFIG_BASE, "SP500")
        # ── NASDAQ 100 ────────────────────────────────────────
        ndx100 = fetch_nasdaq100_tickers()
        if ndx100:
            run_market_scan(ndx100, "QQQ", BIST_CONFIG_BASE, "NASDAQ100")
        # ── DAX 40 ────────────────────────────────────────────
        dax = fetch_dax_tickers()
        if dax:
            run_market_scan(dax, "^GDAXI", BIST_CONFIG_BASE, "DAX40")

    elif "--bist" in sys.argv:
        run_market_scan(BIST_TICKERS, "XU100.IS", BIST_CONFIG_BASE, "BIST100")

    elif "--sp500" in sys.argv:
        sp500 = fetch_sp500_tickers()
        run_market_scan(sp500, "SPY", BIST_CONFIG_BASE, "SP500")

    elif "--nasdaq100" in sys.argv:
        ndx100 = fetch_nasdaq100_tickers()
        run_market_scan(ndx100, "QQQ", BIST_CONFIG_BASE, "NASDAQ100")

    elif "--dax" in sys.argv:
        dax = fetch_dax_tickers()
        run_market_scan(dax, "^GDAXI", BIST_CONFIG_BASE, "DAX40")

    elif "--bist-4h" in sys.argv:
        run_market_scan(BIST_TICKERS, "XU100.IS", BIST_4H_CONFIG, "BIST100-4H")

    else:
        df, trades, metrics = main()
