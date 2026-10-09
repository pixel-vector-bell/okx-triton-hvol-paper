"""
Indicator Library — Poseidon Research Engine
Vectorised indicators on pandas DataFrames.
All indicators operate on OHLCV columns: open, high, low, close, vol, vol_usdt
"""
import numpy as np
import pandas as pd


# ─── Basic Utilities ─────────────────────────────────────────────────────────

def as_df(candles: list) -> pd.DataFrame:
    """Convert list of dicts → DataFrame with datetime index."""
    df = pd.DataFrame(candles)
    df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("datetime").sort_index()
    df = df.astype({"open": float, "high": float, "low": float, "close": float,
                    "vol": float, "vol_usdt": float})
    return df


def returns(close: pd.Series) -> pd.Series:
    return close.pct_change()


# ─── Moving Averages ─────────────────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def vwma(close: pd.Series, vol: pd.Series, period: int) -> pd.Series:
    return (close * vol).rolling(period).sum() / vol.rolling(period).sum()


def hull_ma(close: pd.Series, period: int) -> pd.Series:
    half = max(1, period // 2)
    sqrt_p = max(1, int(np.sqrt(period)))
    wma_half = close.rolling(half).apply(lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.arange(1, len(x)+1).sum(), raw=True)
    wma_full = close.rolling(period).apply(lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.arange(1, len(x)+1).sum(), raw=True)
    raw = 2 * wma_half - wma_full
    return raw.rolling(sqrt_p).apply(lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.arange(1, len(x)+1).sum(), raw=True)


# ─── Volatility ──────────────────────────────────────────────────────────────

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def bb_width(close: pd.Series, period: int = 20, std: float = 2.0) -> pd.Series:
    mid = sma(close, period)
    s = close.rolling(period).std()
    upper = mid + std * s
    lower = mid - std * s
    return (upper - lower) / mid


def historical_vol(close: pd.Series, period: int = 20) -> pd.Series:
    """Annualised log-return volatility."""
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(period).std() * np.sqrt(365 * 24)  # hourly candles → annual


def keltner_width(high: pd.Series, low: pd.Series, close: pd.Series,
                  period: int = 20, mult: float = 2.0) -> pd.Series:
    mid = ema(close, period)
    _atr = atr(high, low, close, period)
    return (2 * mult * _atr) / mid


def squeeze_score(close: pd.Series, high: pd.Series, low: pd.Series,
                  bb_period: int = 20, kc_period: int = 20,
                  bb_std: float = 2.0, kc_mult: float = 1.5) -> pd.Series:
    """Compression score 0–4: higher = more compressed (BB inside KC)."""
    bb_w = bb_width(close, bb_period, bb_std)
    kc_w = keltner_width(high, low, close, kc_period, kc_mult)
    return (kc_w > bb_w).astype(int) * 4  # simplified binary squeeze


# ─── Momentum ────────────────────────────────────────────────────────────────

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def rate_of_change(close: pd.Series, period: int = 10) -> pd.Series:
    return (close - close.shift(period)) / close.shift(period) * 100


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
               k_period: int = 14, d_period: int = 3) -> tuple:
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return k, d


def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    typical = (high + low + close) / 3
    mean_dev = typical.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (typical - typical.rolling(period).mean()) / (0.015 * mean_dev.replace(0, np.nan))


# ─── Volume Analysis ─────────────────────────────────────────────────────────

def volume_ratio(vol: pd.Series, period: int = 20) -> pd.Series:
    """Current volume / rolling average volume."""
    return vol / vol.rolling(period).mean()


def obv(close: pd.Series, vol: pd.Series) -> pd.Series:
    direction = np.sign(close.diff())
    return (direction * vol).fillna(0).cumsum()


def volume_surge(vol: pd.Series, period: int = 20, threshold: float = 2.0) -> pd.Series:
    """Boolean: current vol > threshold × average."""
    return vol > (vol.rolling(period).mean() * threshold)


def money_flow_index(high: pd.Series, low: pd.Series, close: pd.Series,
                     vol: pd.Series, period: int = 14) -> pd.Series:
    typical = (high + low + close) / 3
    raw_mf = typical * vol
    pos_mf = raw_mf.where(typical > typical.shift(1), 0)
    neg_mf = raw_mf.where(typical < typical.shift(1), 0)
    pos_sum = pos_mf.rolling(period).sum()
    neg_sum = neg_mf.rolling(period).sum()
    mfr = pos_sum / neg_sum.replace(0, np.nan)
    return 100 - 100 / (1 + mfr)


def vwap_daily(high: pd.Series, low: pd.Series, close: pd.Series,
               vol: pd.Series) -> pd.Series:
    """Rolling VWAP reset daily (UTC)."""
    typical = (high + low + close) / 3
    cum_typical_vol = (typical * vol).groupby(typical.index.date).cumsum()
    cum_vol = vol.groupby(vol.index.date).cumsum()
    return cum_typical_vol / cum_vol.replace(0, np.nan)


# ─── Trend / Structure ───────────────────────────────────────────────────────

def donchian(high: pd.Series, low: pd.Series, period: int = 20) -> tuple:
    upper = high.rolling(period).max()
    lower = low.rolling(period).min()
    mid = (upper + lower) / 2
    return upper, lower, mid


def supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
               period: int = 10, multiplier: float = 3.0) -> tuple:
    _atr = atr(high, low, close, period)
    src = (high + low) / 2
    upper_band = src + multiplier * _atr
    lower_band = src - multiplier * _atr

    st = pd.Series(np.nan, index=close.index)
    trend = pd.Series(1, index=close.index)

    for i in range(1, len(close)):
        prev_upper = upper_band.iloc[i-1] if not np.isnan(upper_band.iloc[i-1]) else upper_band.iloc[i]
        prev_lower = lower_band.iloc[i-1] if not np.isnan(lower_band.iloc[i-1]) else lower_band.iloc[i]

        upper_band.iloc[i] = upper_band.iloc[i] if upper_band.iloc[i] < prev_upper or close.iloc[i-1] > prev_upper else prev_upper
        lower_band.iloc[i] = lower_band.iloc[i] if lower_band.iloc[i] > prev_lower or close.iloc[i-1] < prev_lower else prev_lower

        if trend.iloc[i-1] == 1:
            trend.iloc[i] = -1 if close.iloc[i] < lower_band.iloc[i] else 1
        else:
            trend.iloc[i] = 1 if close.iloc[i] > upper_band.iloc[i] else -1

        st.iloc[i] = lower_band.iloc[i] if trend.iloc[i] == 1 else upper_band.iloc[i]

    return st, trend


def pivot_highs(high: pd.Series, window: int = 5) -> pd.Series:
    """Returns True at local pivot high (high > surrounding window bars)."""
    left = high.rolling(window, center=True).max()
    return (high == left) & (high == high.rolling(window, center=False).max())


def pivot_lows(low: pd.Series, window: int = 5) -> pd.Series:
    """Returns True at local pivot low."""
    right = low.rolling(window, center=True).min()
    return (low == right) & (low == low.rolling(window, center=False).min())


def higher_highs_higher_lows(high: pd.Series, low: pd.Series, period: int = 3) -> pd.Series:
    """Simple HH/HL trend detection over last N pivot points. Returns +1/-1/0."""
    hh = (high.rolling(period).max() == high)
    ll = (low.rolling(period).min() == low)
    return pd.Series(
        np.where(hh, 1, np.where(ll, -1, 0)),
        index=high.index
    )


# ─── Market Regime ───────────────────────────────────────────────────────────

def btc_regime(btc_df: pd.DataFrame, fast: int = 8, slow: int = 21,
               vol_period: int = 20, vol_high_thresh: float = 1.5) -> pd.DataFrame:
    """
    Returns DataFrame with columns: trend (+1/0/-1), vol_regime ('high'/'normal').
    Uses BTC as macro filter.
    """
    close = btc_df["close"]
    fast_e = ema(close, fast)
    slow_e = ema(close, slow)
    trend = pd.Series(
        np.where(fast_e > slow_e * 1.002, 1,
                 np.where(fast_e < slow_e * 0.998, -1, 0)),
        index=close.index, name="trend"
    )
    vol_r = volume_ratio(btc_df["vol_usdt"], vol_period)
    vol_regime = pd.Series(
        np.where(vol_r >= vol_high_thresh, "high", "normal"),
        index=close.index, name="vol_regime"
    )
    return pd.DataFrame({"btc_trend": trend, "btc_vol": vol_regime})


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> tuple:
    """Returns (adx, +DI, -DI)."""
    _atr = atr(high, low, close, period)
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    pos_di = 100 * pd.Series(pos_dm, index=close.index).ewm(span=period, adjust=False).mean() / _atr.replace(0, np.nan)
    neg_di = 100 * pd.Series(neg_dm, index=close.index).ewm(span=period, adjust=False).mean() / _atr.replace(0, np.nan)
    dx = 100 * (pos_di - neg_di).abs() / (pos_di + neg_di).replace(0, np.nan)
    adx_val = dx.ewm(span=period, adjust=False).mean()
    return adx_val, pos_di, neg_di


# ─── Multi-Timeframe Alignment ───────────────────────────────────────────────

def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample a DataFrame to a higher timeframe."""
    return df.resample(rule).agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
        "vol": "sum", "vol_usdt": "sum"
    }).dropna()
