"""
Backtesting Engine — Poseidon Research Engine
Realistic event-driven backtester with:
- Maker/taker fee model (0.08% + 0.10% = 0.18% round-trip, matching OKX)
- Slippage simulation (0.05% per side)
- Position sizing (fixed-fraction or fixed-dollar)
- MFE/MAE tracking
- Walk-forward split support
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

# ─── Cost Model ──────────────────────────────────────────────────────────────
TAKER_FEE = 0.0008     # 0.08% (matching Ross: 0.08%)
SLIPPAGE   = 0.0005    # 0.05% per side (matching Ross)
ROUND_TRIP_COST = (TAKER_FEE + SLIPPAGE) * 2  # both sides

# ─── Data Classes ────────────────────────────────────────────────────────────

@dataclass
class Trade:
    pair: str
    entry_ts: int
    exit_ts: Optional[int]
    entry_price: float
    exit_price: Optional[float]
    qty: float
    side: str  # 'long'
    exit_reason: str = ""
    mfe: float = 0.0   # max favourable excursion (fraction)
    mae: float = 0.0   # max adverse excursion (fraction, positive = bad)
    pnl_raw: float = 0.0   # before fees
    pnl_net: float = 0.0   # after fees + slip
    fee_paid: float = 0.0
    open_: bool = True


@dataclass
class BacktestResult:
    pair: str
    strategy: str
    params: dict
    trades: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)  # [(ts, equity)]
    start_capital: float = 3000.0

    @property
    def closed_trades(self):
        return [t for t in self.trades if not t.open_]

    @property
    def n_trades(self):
        return len(self.closed_trades)

    @property
    def wins(self):
        return [t for t in self.closed_trades if t.pnl_net > 0]

    @property
    def win_rate(self):
        if self.n_trades == 0:
            return 0.0
        return len(self.wins) / self.n_trades

    @property
    def total_pnl(self):
        return sum(t.pnl_net for t in self.closed_trades)

    @property
    def profit_factor(self):
        gross_profit = sum(t.pnl_net for t in self.closed_trades if t.pnl_net > 0)
        gross_loss = abs(sum(t.pnl_net for t in self.closed_trades if t.pnl_net < 0))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    @property
    def expectancy(self):
        if self.n_trades == 0:
            return 0.0
        return self.total_pnl / self.n_trades

    @property
    def max_drawdown(self):
        if not self.equity_curve:
            return 0.0
        equity = [e for _, e in self.equity_curve]
        peak = equity[0]
        max_dd = 0.0
        for e in equity:
            if e > peak:
                peak = e
            dd = (peak - e) / peak
            max_dd = max(max_dd, dd)
        return max_dd

    @property
    def sharpe(self):
        """Simplified Sharpe using trade returns."""
        if self.n_trades < 5:
            return 0.0
        rets = [t.pnl_net / self.start_capital for t in self.closed_trades]
        if np.std(rets) == 0:
            return 0.0
        return np.mean(rets) / np.std(rets) * np.sqrt(252)  # annualised approx

    @property
    def avg_mfe(self):
        if self.n_trades == 0:
            return 0.0
        return np.mean([t.mfe for t in self.closed_trades])

    @property
    def avg_mae(self):
        if self.n_trades == 0:
            return 0.0
        return np.mean([t.mae for t in self.closed_trades])

    def summary(self) -> dict:
        return {
            "pair": self.pair,
            "strategy": self.strategy,
            "params": self.params,
            "n_trades": self.n_trades,
            "win_rate": round(self.win_rate, 4),
            "total_pnl": round(self.total_pnl, 4),
            "profit_factor": round(self.profit_factor, 4),
            "expectancy": round(self.expectancy, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "sharpe": round(self.sharpe, 4),
            "avg_mfe": round(self.avg_mfe, 4),
            "avg_mae": round(self.avg_mae, 4),
            "final_equity": round(self.start_capital + self.total_pnl, 2),
        }


# ─── Backtester ──────────────────────────────────────────────────────────────

class Backtester:
    """
    Event-driven backtester operating on 1m/5m candles.
    Strategies emit signals; backtester handles execution, stops, and exits.
    """

    def __init__(self, capital: float = 3000.0, risk_per_trade: float = 0.01,
                 max_position_frac: float = 0.25, fee_rate: float = TAKER_FEE,
                 slippage: float = SLIPPAGE, max_hold_bars: int = 200):
        self.capital = capital
        self.start_capital = capital
        self.risk_per_trade = risk_per_trade          # fraction of equity to risk
        self.max_position_frac = max_position_frac    # max fraction of equity in one trade
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.max_hold_bars = max_hold_bars
        self.equity = capital
        self.peak_equity = capital
        self.open_trade: Optional[Trade] = None
        self.closed_trades: list = []
        self.equity_curve: list = []

    def reset(self):
        self.equity = self.start_capital
        self.peak_equity = self.start_capital
        self.open_trade = None
        self.closed_trades = []
        self.equity_curve = []

    def _fill_price_buy(self, price: float) -> float:
        return price * (1 + self.slippage)

    def _fill_price_sell(self, price: float) -> float:
        return price * (1 - self.slippage)

    def _position_size(self, entry_price: float, stop_price: float) -> float:
        """Risk-based sizing: risk X% of equity per trade."""
        risk_amount = self.equity * self.risk_per_trade
        stop_dist = abs(entry_price - stop_price) / entry_price
        if stop_dist <= 0:
            stop_dist = 0.01
        max_size_by_risk = risk_amount / (stop_dist * entry_price)
        max_size_by_capital = (self.equity * self.max_position_frac) / entry_price
        return min(max_size_by_risk, max_size_by_capital)

    def enter_long(self, ts: int, price: float, stop: float, pair: str) -> Optional[Trade]:
        if self.open_trade is not None:
            return None
        fill = self._fill_price_buy(price)
        qty = self._position_size(fill, stop)
        fee = fill * qty * self.fee_rate
        cost = fill * qty + fee
        if cost > self.equity:
            qty = (self.equity * (1 - self.fee_rate)) / fill
            fee = fill * qty * self.fee_rate
        self.equity -= (fill * qty + fee)
        trade = Trade(
            pair=pair, entry_ts=ts, exit_ts=None,
            entry_price=fill, exit_price=None,
            qty=qty, side="long",
            mfe=0.0, mae=0.0,
            fee_paid=fee, open_=True
        )
        self.open_trade = trade
        return trade

    def exit_long(self, ts: int, price: float, reason: str) -> Optional[Trade]:
        if self.open_trade is None:
            return None
        t = self.open_trade
        fill = self._fill_price_sell(price)
        fee = fill * t.qty * self.fee_rate
        proceeds = fill * t.qty - fee
        pnl_raw = (fill - t.entry_price) * t.qty
        pnl_net = pnl_raw - (t.fee_paid + fee)
        t.exit_ts = ts
        t.exit_price = fill
        t.exit_reason = reason
        t.pnl_raw = pnl_raw
        t.pnl_net = pnl_net
        t.fee_paid += fee
        t.open_ = False
        self.equity += proceeds
        self.peak_equity = max(self.peak_equity, self.equity)
        self.closed_trades.append(t)
        self.open_trade = None
        return t

    def update_mfe_mae(self, high: float, low: float):
        if self.open_trade is None:
            return
        t = self.open_trade
        mfe = (high - t.entry_price) / t.entry_price
        mae = (t.entry_price - low) / t.entry_price
        t.mfe = max(t.mfe, mfe)
        t.mae = max(t.mae, mae)

    def run_bar(self, ts: int, o: float, h: float, l: float, c: float,
                signal: int, stop_price: float, pair: str,
                take_profit: Optional[float] = None,
                trail_stop: Optional[float] = None,
                bars_open: int = 0) -> list:
        """
        Process one bar:
        signal: +1=enter, 0=hold, -1=exit
        Returns list of events ('enter', 'exit_stop', 'exit_tp', 'exit_signal', 'exit_timeout')
        """
        events = []
        # Update MFE/MAE
        self.update_mfe_mae(h, l)
        # Update equity curve
        current_equity = self.equity
        if self.open_trade:
            unrealised = (c - self.open_trade.entry_price) * self.open_trade.qty
            current_equity += unrealised
        self.equity_curve.append((ts, current_equity))

        # Handle open trade stops/exits
        if self.open_trade:
            effective_stop = trail_stop if trail_stop is not None else stop_price
            # Stop hit
            if effective_stop and l <= effective_stop:
                self.exit_long(ts, effective_stop, "stop")
                events.append("exit_stop")
                return events
            # Take profit hit
            if take_profit and h >= take_profit:
                self.exit_long(ts, take_profit, "tp")
                events.append("exit_tp")
                return events
            # Strategy exit signal
            if signal == -1:
                self.exit_long(ts, c, "signal")
                events.append("exit_signal")
                return events
            # Timeout
            if bars_open >= self.max_hold_bars:
                self.exit_long(ts, c, "timeout")
                events.append("exit_timeout")
                return events

        # Entry
        if signal == 1 and self.open_trade is None:
            self.enter_long(ts, c, stop_price, pair)
            events.append("enter")

        return events


# ─── Walk-Forward Splitter ───────────────────────────────────────────────────

def walk_forward_splits(df: pd.DataFrame, n_splits: int = 5,
                        train_frac: float = 0.7) -> list:
    """
    Returns list of (train_df, test_df) tuples for walk-forward validation.
    """
    n = len(df)
    fold_size = n // n_splits
    splits = []
    for i in range(n_splits):
        start = i * fold_size
        end = start + fold_size
        train_end = start + int(fold_size * train_frac)
        train = df.iloc[start:train_end]
        test = df.iloc[train_end:end]
        if len(train) > 50 and len(test) > 20:
            splits.append((train, test))
    return splits
