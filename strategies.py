"""
TRITON Strategy Family — Poseidon Research Engine

TRITON = Trend-Regime-Integrated Time-Optimised Nexus

Core thesis: Most strategies here fail because they ignore regime and time.
Ross wins because it implicitly captures high-impulse breakouts in sideways/up BTC markets.
TRITON extends this with:
1. Multi-timeframe structural bias (1H + 15m alignment)
2. Volume surge confirmation before entry
3. Time-of-day gate (only trade proven UTC windows)
4. Adaptive stops based on volatility regime
5. Clean R-multiple exits, not trailing

Strategy families tested:
A. TRITON-VSMO: Volume Surge + Momentum Oscillator (RSI divergence + vol)
B. TRITON-STAB: Squeeze-and-Break (BB/KC compression → expansion)
C. TRITON-MTALIGN: Multi-Timeframe Alignment (1H + 15m + 5m trend stack)
D. TRITON-RVOL: Relative Volume Leader (buy the coin with highest vol surge vs BTC)
E. TRITON-HVOL: High-Volatility Regime breakouts (ADX trend filter)
F. TRITON-EMASTACK: EMA stack alignment with volume confirmation
"""
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from indicators import (
    ema, sma, atr, rsi, macd, volume_ratio, volume_surge,
    rate_of_change, bb_width, keltner_width, stochastic,
    donchian, adx, cci, vwma, squeeze_score, obv
)
from backtester import Backtester, BacktestResult


# ─── Shared Utilities ────────────────────────────────────────────────────────

def _prep(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    """Add all indicators to a dataframe given params dict."""
    d = df.copy()
    # EMAs
    for n in [9, 21, 50, 100, 200]:
        d[f"ema{n}"] = ema(d["close"], n)
    # ATR
    for per in [7, 14, 20]:
        d[f"atr{per}"] = atr(d["high"], d["low"], d["close"], per)
    # RSI
    d["rsi14"] = rsi(d["close"], 14)
    d["rsi7"] = rsi(d["close"], 7)
    # MACD
    d["macd_line"], d["macd_sig"], d["macd_hist"] = macd(d["close"], 12, 26, 9)
    # Volume
    d["vr20"] = volume_ratio(d["vol_usdt"], 20)
    d["vr10"] = volume_ratio(d["vol_usdt"], 10)
    # ROC
    d["roc5"] = rate_of_change(d["close"], 5)
    d["roc10"] = rate_of_change(d["close"], 10)
    # BB width (compression)
    d["bb_width20"] = bb_width(d["close"], 20, 2.0)
    d["kc_width20"] = keltner_width(d["high"], d["low"], d["close"], 20, 1.5)
    d["squeeze"] = (d["kc_width20"] > d["bb_width20"]).astype(int)
    # ADX
    d["adx14"], d["di_plus"], d["di_minus"] = adx(d["high"], d["low"], d["close"], 14)
    # Donchian
    d["don20_hi"], d["don20_lo"], d["don20_mid"] = donchian(d["high"], d["low"], 20)
    d["don10_hi"], d["don10_lo"], d["don10_mid"] = donchian(d["high"], d["low"], 10)
    # Stochastic
    d["stoch_k"], d["stoch_d"] = stochastic(d["high"], d["low"], d["close"], 14, 3)
    # OBV momentum
    d["obv"] = obv(d["close"], d["vol_usdt"])
    d["obv_ema10"] = ema(d["obv"], 10)
    # VWMA
    d["vwma20"] = vwma(d["close"], d["vol_usdt"], 20)
    # Time features
    d["hour"] = d.index.hour
    d["dayofweek"] = d.index.dayofweek  # 0=Mon
    return d


def _time_ok(hour: int, allowed_hours: set) -> bool:
    """Check if trading hour is in the allowed set."""
    if not allowed_hours:
        return True
    return hour in allowed_hours


def _btc_regime_scalar(btc_row, trend_thresh=0.0) -> int:
    """Returns +1 (up/sideways) or -1 (down) based on BTC EMA relationship."""
    if btc_row is None:
        return 1
    # Positive regime: ema21 > ema50 (even slightly), or close > ema50
    return 1 if btc_row.get("ema21", 1) >= btc_row.get("ema50", 1) * (1 - 0.003) else -1


# ─── Strategy A: TRITON-VSMO ────────────────────────────────────────────────

def signal_vsmo(row, prev_row, params: dict) -> tuple:
    """
    Volume Surge + Momentum Oscillator.
    Entry: RSI crosses up from oversold while volume surges. EMA trend up.
    Exit: RSI overbought OR MACD cross down.
    Returns (signal, stop_price)
    """
    p = params
    entry_sig = 0
    stop = None

    if prev_row is None:
        return 0, None

    # Trend filter
    trend_ok = row["ema21"] > row["ema50"] * (1 - p.get("trend_slack", 0.005))
    # Volume surge
    vol_ok = row["vr20"] >= p.get("vol_thresh", 2.0)
    # RSI crossing up from oversold
    rsi_entry = (prev_row["rsi14"] < p.get("rsi_lo", 40)) and (row["rsi14"] >= p.get("rsi_lo", 40))
    # Price above short EMA
    price_ok = row["close"] > row["ema9"] * (1 - p.get("ema_slack", 0.002))
    # Time gate
    time_ok = _time_ok(row["hour"], set(p.get("allowed_hours", [])))
    # BTC regime
    regime_ok = row.get("btc_regime", 1) >= 0

    if trend_ok and vol_ok and rsi_entry and price_ok and time_ok and regime_ok:
        entry_sig = 1
        stop = row["close"] * (1 - p.get("stop_atr_mult", 1.5) * row["atr14"] / row["close"])

    # Exit: RSI overbought or MACD histogram turns negative
    if row["rsi14"] >= p.get("rsi_hi", 70):
        entry_sig = -1
    elif row["macd_hist"] < 0 and prev_row["macd_hist"] >= 0:
        entry_sig = -1

    return entry_sig, stop


# ─── Strategy B: TRITON-STAB ────────────────────────────────────────────────

def signal_stab(row, prev_row, state: dict, params: dict) -> tuple:
    """
    Squeeze-and-Break.
    Entry: 3+ bars of squeeze (BB inside KC) → squeeze fires (BB breaks out of KC)
           + price breaks above 20-bar high + volume surge.
    Exit: 2× ATR target or trail.
    Returns (signal, stop_price, new_state)
    """
    p = params
    entry_sig = 0
    stop = None

    if prev_row is None:
        return 0, None, state

    # Track squeeze bars
    if prev_row.get("squeeze", 0) == 1:
        state["squeeze_bars"] = state.get("squeeze_bars", 0) + 1
    else:
        state["squeeze_bars"] = 0

    # Entry: squeeze just fired + break above donchian high
    squeeze_fired = (prev_row.get("squeeze", 0) == 1) and (row.get("squeeze", 0) == 0)
    enough_bars = state.get("squeeze_bars", 0) >= p.get("min_squeeze_bars", 3)
    price_breakout = row["close"] > row["don20_hi"].shift(1) if hasattr(row["don20_hi"], "shift") else row["close"] > prev_row["don20_hi"]
    vol_ok = row["vr20"] >= p.get("vol_thresh", 1.5)
    time_ok = _time_ok(row["hour"], set(p.get("allowed_hours", [])))

    if squeeze_fired and enough_bars and vol_ok and time_ok:
        entry_sig = 1
        stop = row["close"] - p.get("stop_atr_mult", 2.0) * row["atr14"]

    # Exit: ADX weakens or price below EMA
    if row["adx14"] < p.get("adx_exit", 15) and row["close"] < row["ema21"]:
        entry_sig = -1

    return entry_sig, stop, state


# ─── Strategy C: TRITON-MTALIGN ─────────────────────────────────────────────

def signal_mtalign(row, prev_row, htf_row, params: dict) -> tuple:
    """
    Multi-Timeframe Alignment.
    Requires 1H EMA stack bullish AND 5m momentum entry.
    HTF row is the corresponding 1H candle.
    """
    p = params
    entry_sig = 0
    stop = None

    if prev_row is None or htf_row is None:
        return 0, None

    # 1H structure bullish
    htf_bull = (htf_row.get("ema9", 0) > htf_row.get("ema21", 0) > htf_row.get("ema50", 0))
    # 5m momentum entry: pullback to EMA21 + volume + RSI not overbought
    pb_to_ema = abs(row["close"] - row["ema21"]) / row["ema21"] < p.get("pb_thresh", 0.005)
    bounce = row["close"] > prev_row["close"] and row["low"] >= prev_row["low"]
    vol_ok = row["vr10"] >= p.get("vol_thresh", 1.2)
    rsi_ok = 35 <= row["rsi14"] <= 65
    time_ok = _time_ok(row["hour"], set(p.get("allowed_hours", [])))

    if htf_bull and pb_to_ema and bounce and vol_ok and rsi_ok and time_ok:
        entry_sig = 1
        stop = min(row["low"], prev_row["low"]) * (1 - p.get("stop_extra", 0.002))

    # Exit: close below EMA50
    if row["close"] < row["ema50"]:
        entry_sig = -1

    return entry_sig, stop


# ─── Strategy D: TRITON-RVOL ────────────────────────────────────────────────

def signal_rvol(row, prev_row, params: dict) -> tuple:
    """
    Relative Volume Leader.
    This runs per-pair. Entry: coin's vol surge is top decile AND price momentum
    is positive relative to BTC momentum.
    """
    p = params
    entry_sig = 0
    stop = None

    if prev_row is None:
        return 0, None

    # High relative volume
    vol_ok = row["vr20"] >= p.get("vol_thresh", 3.0)
    # Strong short-term momentum
    mom_ok = row["roc5"] >= p.get("min_roc", 1.5)
    # Not extended — RSI not overbought
    rsi_ok = row["rsi14"] <= p.get("max_rsi", 72)
    # Trend aligned
    trend_ok = row["ema9"] > row["ema21"] > row["ema50"]
    # ADX showing trend strength
    adx_ok = row["adx14"] >= p.get("min_adx", 18)
    # Time
    time_ok = _time_ok(row["hour"], set(p.get("allowed_hours", [])))
    # BTC regime
    regime_ok = row.get("btc_regime", 1) >= 0

    if vol_ok and mom_ok and rsi_ok and trend_ok and adx_ok and time_ok and regime_ok:
        entry_sig = 1
        stop = row["close"] - p.get("stop_atr_mult", 2.0) * row["atr14"]

    # Exit: RSI hits overbought or price below EMA9
    if row["rsi14"] >= p.get("rsi_exit", 75) or row["close"] < row["ema9"] * (1 - 0.003):
        entry_sig = -1

    return entry_sig, stop


# ─── Strategy E: TRITON-HVOL (High-Volatility Regime) ──────────────────────

def signal_hvol(row, prev_row, params: dict) -> tuple:
    """
    ADX-filtered high volatility breakout.
    Entry: Strong ADX trend + Donchian 10-bar break + vol surge.
    Exit: Price breaks below Donchian 5-bar low or EMA cross.
    """
    p = params
    entry_sig = 0
    stop = None

    if prev_row is None:
        return 0, None

    # ADX trend strength
    adx_ok = row["adx14"] >= p.get("min_adx", 25)
    di_ok = row["di_plus"] > row["di_minus"]
    # Donchian break
    don_break = row["close"] > prev_row["don10_hi"]
    # Not already extended
    ext = (row["close"] - prev_row["don10_hi"]) / prev_row["don10_hi"]
    not_extended = ext <= p.get("max_ext", 0.03)
    # Volume
    vol_ok = row["vr20"] >= p.get("vol_thresh", 1.5)
    # Time
    time_ok = _time_ok(row["hour"], set(p.get("allowed_hours", [])))

    if adx_ok and di_ok and don_break and not_extended and vol_ok and time_ok:
        entry_sig = 1
        stop = row["don10_lo"]
        if stop <= 0 or stop >= row["close"]:
            stop = row["close"] - 2.0 * row["atr14"]

    # Exit: Donchian 5-bar low or ADX weakens
    if row["close"] < row["don10_lo"] * (1 + 0.001):
        entry_sig = -1
    elif row["adx14"] < p.get("adx_exit", 20) and row["close"] < row["ema21"]:
        entry_sig = -1

    return entry_sig, stop


# ─── Strategy F: TRITON-EMASTACK ────────────────────────────────────────────

def signal_emastack(row, prev_row, params: dict) -> tuple:
    """
    Perfect EMA stack (9 > 21 > 50) + pullback + vol confirm.
    The simplest possible trend-following entry, optimised for crypto.
    """
    p = params
    entry_sig = 0
    stop = None

    if prev_row is None:
        return 0, None

    # Perfect stack
    stack_ok = row["ema9"] > row["ema21"] > row["ema50"]
    # Pullback: price dipped to within X% of EMA21 recently
    pb_depth = (row["ema21"] - row["low"]) / row["ema21"]
    pb_ok = 0 <= pb_depth <= p.get("max_pb", 0.015)
    # Recovery: close above EMA9
    recov_ok = row["close"] > row["ema9"]
    # Volume confirm
    vol_ok = row["vr20"] >= p.get("vol_thresh", 1.3)
    # RSI in range
    rsi_ok = p.get("rsi_lo", 40) <= row["rsi14"] <= p.get("rsi_hi", 68)
    # ADX showing some trend
    adx_ok = row["adx14"] >= p.get("min_adx", 18)
    # Time
    time_ok = _time_ok(row["hour"], set(p.get("allowed_hours", [])))
    # BTC regime
    regime_ok = row.get("btc_regime", 1) >= 0

    if stack_ok and pb_ok and recov_ok and vol_ok and rsi_ok and adx_ok and time_ok and regime_ok:
        entry_sig = 1
        stop = row["ema50"] * (1 - p.get("stop_extra", 0.002))

    # Exit: EMA9 crosses below EMA21 or RSI hits top
    if row["ema9"] < row["ema21"] * (1 - 0.001):
        entry_sig = -1
    elif row["rsi14"] >= p.get("rsi_exit", 72):
        entry_sig = -1

    return entry_sig, stop


# ─── Master Runner ──────────────────────────────────────────────────────────

def run_strategy(df: pd.DataFrame, strategy: str, params: dict,
                 pair: str = "PAIR", capital: float = 3000.0,
                 htf_df: pd.DataFrame = None) -> BacktestResult:
    """
    Run a single strategy on a prepared DataFrame.
    df must already have indicators added via _prep().
    """
    bt = Backtester(
        capital=capital,
        risk_per_trade=params.get("risk_pct", 0.01),
        max_position_frac=params.get("max_pos_frac", 0.25),
        max_hold_bars=params.get("max_hold_bars", 200),
    )
    result = BacktestResult(pair=pair, strategy=strategy, params=params,
                            start_capital=capital)
    state = {}
    bars_open = 0

    rows = list(df.itertuples())
    n = len(rows)

    # Pre-compute take profit levels
    tp_mult = params.get("tp_atr_mult", None)
    use_trail = params.get("use_trail", False)
    trail_ema = params.get("trail_ema", 9)
    trail_pct = params.get("trail_pct", None)

    current_stop = None
    current_tp = None
    trail_high = None

    for i in range(1, n):
        row = rows[i]
        prev_row = rows[i-1]
        row_d = row._asdict()
        prev_d = prev_row._asdict()

        # Get HTF row if available
        htf_row = None
        if htf_df is not None:
            htf_idx = htf_df.index.searchsorted(pd.Timestamp(row_d.get("Index", None))) - 1
            if 0 <= htf_idx < len(htf_df):
                htf_row = htf_df.iloc[htf_idx].to_dict()

        # Generate signal
        if strategy == "VSMO":
            sig, stop = signal_vsmo(row_d, prev_d, params)
        elif strategy == "STAB":
            sig, stop, state = signal_stab(row_d, prev_d, state, params)
        elif strategy == "MTALIGN":
            sig, stop = signal_mtalign(row_d, prev_d, htf_row, params)
        elif strategy == "RVOL":
            sig, stop = signal_rvol(row_d, prev_d, params)
        elif strategy == "HVOL":
            sig, stop = signal_hvol(row_d, prev_d, params)
        elif strategy == "EMASTACK":
            sig, stop = signal_emastack(row_d, prev_d, params)
        else:
            sig, stop = 0, None

        # Update trailing stop
        effective_stop = current_stop
        if bt.open_trade and use_trail:
            if trail_high is None:
                trail_high = row_d["close"]
            trail_high = max(trail_high, row_d["high"])
            if trail_pct:
                effective_stop = trail_high * (1 - trail_pct)
            else:
                ema_key = f"ema{trail_ema}"
                effective_stop = row_d.get(ema_key, current_stop or 0)
        elif bt.open_trade:
            effective_stop = current_stop

        # Track bars open
        if bt.open_trade:
            bars_open += 1
        else:
            bars_open = 0

        # Run bar
        bt.run_bar(
            ts=int(row_d.get("Index", 0).timestamp() * 1000) if hasattr(row_d.get("Index", 0), "timestamp") else i,
            o=row_d["open"], h=row_d["high"], l=row_d["low"], c=row_d["close"],
            signal=sig,
            stop_price=effective_stop or (row_d["close"] * 0.99),
            pair=pair,
            take_profit=current_tp,
            trail_stop=effective_stop if use_trail else None,
            bars_open=bars_open,
        )

        # Update stop/tp for new entries
        if sig == 1 and not bt.open_trade is None and bt.open_trade.open_:
            current_stop = stop
            trail_high = row_d["close"]
            if tp_mult and stop:
                risk_dist = bt.open_trade.entry_price - stop
                current_tp = bt.open_trade.entry_price + tp_mult * risk_dist
            else:
                current_tp = None

        if not bt.open_trade:
            current_stop = None
            current_tp = None
            trail_high = None

    # Close any open trade at last bar
    if bt.open_trade:
        last = rows[-1]._asdict()
        bt.exit_long(i, last["close"], "end_of_data")

    result.trades = bt.closed_trades
    result.equity_curve = bt.equity_curve
    return result
