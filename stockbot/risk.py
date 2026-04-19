from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RiskConfig:
    """Configuration for risk management rules."""
    stop_loss_pct: float = 0.05          # exit if position drops 5% from entry
    take_profit_pct: float = 0.15        # exit if position rises 15% from entry
    max_drawdown_pct: float = 0.10       # halt trading if portfolio drops 10% from peak
    max_position_pct: float = 0.25        # never let one position exceed 25% of equity
    max_daily_trades: int = 10            # max trades per symbol per day
    min_trade_interval_minutes: int = 5   # minimum time between trades for same symbol
    volatility_position_scale: bool = True # reduce position size in high-vol regimes
    volatility_lookback: int = 20         # days for vol-based position scaling
    target_volatility: float = 0.15        # annualized vol target for position sizing


@dataclass
class PortfolioState:
    """Track portfolio-level state for risk checks."""
    peak_equity: float = 0.0
    daily_trade_count: dict[str, int] = field(default_factory=dict)
    last_trade_time: dict[str, float] = field(default_factory=dict)
    entry_prices: dict[str, float] = field(default_factory=dict)
    halted: bool = False
    halt_reason: str = ""


class RiskManager:
    """Evaluates trade requests against risk rules before execution."""

    def __init__(self, config: Optional[RiskConfig] = None) -> None:
        self.config = config or RiskConfig()
        self.state = PortfolioState()

    def check_stop_loss(self, symbol: str, entry_price: float, current_price: float) -> bool:
        if entry_price <= 0:
            return False
        decline = (entry_price - current_price) / entry_price
        return decline >= self.config.stop_loss_pct

    def check_take_profit(self, symbol: str, entry_price: float, current_price: float) -> bool:
        if entry_price <= 0:
            return False
        gain = (current_price - entry_price) / entry_price
        return gain >= self.config.take_profit_pct

    def check_drawdown(self, current_equity: float) -> bool:
        """Return True if max drawdown exceeded."""
        if current_equity > self.state.peak_equity:
            self.state.peak_equity = current_equity
        if self.state.peak_equity <= 0:
            return False
        drawdown = (self.state.peak_equity - current_equity) / self.state.peak_equity
        return drawdown >= self.config.max_drawdown_pct

    def check_position_limit(self, symbol: str, qty: float, price: float, total_equity: float) -> bool:
        """Return True if position would exceed max_position_pct."""
        position_value = qty * price
        if total_equity <= 0:
            return False
        return (position_value / total_equity) > self.config.max_position_pct

    def scale_position_by_volatility(
        self, base_qty: float, recent_volatility: float
    ) -> float:
        """Scale position size inversely with volatility."""
        if not self.config.volatility_position_scale or recent_volatility <= 0:
            return base_qty
        # Annualize daily vol: vol_annual = vol_daily * sqrt(252)
        vol_annual = recent_volatility * (252 ** 0.5)
        if vol_annual < 0.01:
            return base_qty
        scale = min(self.config.target_volatility / vol_annual, 2.0)
        scale = max(scale, 0.2)  # floor at 20%
        return base_qty * scale

    def record_trade(self, symbol: str, timestamp: float) -> None:
        self.state.daily_trade_count[symbol] = self.state.daily_trade_count.get(symbol, 0) + 1
        self.state.last_trade_time[symbol] = timestamp

    def reset_daily_counters(self) -> None:
        self.state.daily_trade_count = {}

    def evaluate(
        self,
        symbol: str,
        action: str,
        qty: float,
        price: float,
        equity: float,
        entry_price: float,
        current_time: float,
        recent_volatility: float = 0.0,
    ) -> tuple[bool, str]:
        """Full risk evaluation. Returns (allowed, reason)."""
        if self.state.halted:
            return False, f"Trading halted: {self.state.halt_reason}"

        if self.check_drawdown(equity):
            self.state.halted = True
            self.state.halt_reason = f"Max drawdown exceeded ({self.config.max_drawdown_pct:.0%})"
            return False, self.state.halt_reason

        if action == "buy":
            if self.check_position_limit(symbol, qty, price, equity):
                return False, f"Position would exceed {self.config.max_position_pct:.0%} of equity"

        if action == "sell" and entry_price > 0:
            if self.check_stop_loss(symbol, entry_price, price):
                return True, "Stop-loss triggered"
            if self.check_take_profit(symbol, entry_price, price):
                return True, "Take-profit triggered"

        daily_count = self.state.daily_trade_count.get(symbol, 0)
        if daily_count >= self.config.max_daily_trades:
            return False, f"Daily trade limit reached for {symbol} ({daily_count})"

        if symbol in self.state.last_trade_time:
            elapsed = current_time - self.state.last_trade_time[symbol]
            if elapsed < self.config.min_trade_interval_minutes * 60:
                return False, f"Minimum trade interval not met for {symbol}"

        if action == "buy" and recent_volatility > 0:
            scaled = self.scale_position_by_volatility(qty, recent_volatility)
            if scaled < qty * 0.2:
                return False, "Volatility too high, position scaled below minimum"

        return True, "OK"
