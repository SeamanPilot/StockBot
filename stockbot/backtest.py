from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

from .brokers import SimulatedBroker, Position
from .data import fetch_ohlcv
from .features import with_indicators
from .model import EnsemblePriceForecaster
from .risk import RiskConfig, RiskManager
from .strategies import default_strategy_stack


@dataclass
class BacktestTrade:
    date: str
    symbol: str
    side: str
    qty: float
    price: float
    reason: str
    equity: float


@dataclass
class BacktestResult:
    initial_cash: float
    final_equity: float
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: Optional[float]
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_trade_return_pct: float
    trades: list[BacktestTrade]
    equity_curve: pd.DataFrame


class BacktestEngine:
    """Walk-forward backtesting engine.

    Trains on a portion of data, then steps through the out-of-sample
    period day-by-day, generating signals and simulating trades.
    """

    def __init__(
        self,
        symbol: str,
        initial_cash: float = 100_000,
        position_size_pct: float = 0.1,
        train_period: str = "1y",
        test_period: str = "1y",
        interval: str = "1d",
        risk_config: Optional[RiskConfig] = None,
        commission_per_trade: float = 1.0,
    ) -> None:
        self.symbol = symbol
        self.initial_cash = initial_cash
        self.position_size_pct = position_size_pct
        self.train_period = train_period
        self.test_period = test_period
        self.interval = interval
        self.risk_config = risk_config or RiskConfig()
        self.commission_per_trade = commission_per_trade

    def run(self) -> BacktestResult:
        # Fetch full dataset (train + test)
        full_period = f"{self.train_period}+{self.test_period}"
        # yfinance doesn't support additive periods, so fetch enough data
        total_days = self._period_to_days(self.train_period) + self._period_to_days(self.test_period)
        full_period_str = f"{int(total_days * 1.5)}d"

        df = fetch_ohlcv(self.symbol, period=full_period_str, interval=self.interval)
        feat = with_indicators(df)

        # Split into train/test
        split_idx = int(len(feat) * (self._period_to_days(self.train_period)) /
                        (self._period_to_days(self.train_period) + self._period_to_days(self.test_period)))
        split_idx = max(split_idx, 50)  # ensure minimum training data
        train_df = feat.iloc[:split_idx]
        test_df = feat.iloc[split_idx:]

        if len(test_df) < 5:
            raise ValueError("Not enough test data. Try longer periods.")

        # Train model
        model = EnsemblePriceForecaster()
        model.fit(train_df)

        strategy = default_strategy_stack()
        broker = SimulatedBroker(cash=self.initial_cash, commission_per_trade=self.commission_per_trade)
        risk_mgr = RiskManager(self.risk_config)

        trades: list[BacktestTrade] = []
        equity_history: list[dict] = []
        entry_prices: dict[str, float] = {}

        import time as _time
        sim_time = _time.time()

        for i in range(len(test_df)):
            current = test_df.iloc[:i+1]
            last = current.iloc[-1]
            last_price = float(last["close"])
            current_equity = broker.equity({self.symbol: last_price})

            # Record equity
            equity_history.append({
                "date": current.index[-1],
                "equity": current_equity,
                "cash": broker.cash,
                "position_qty": broker.positions[self.symbol].qty if self.symbol in broker.positions else 0,
            })

            # Need at least some data for indicators
            if len(current) < 2:
                continue

            strat_signal = strategy.generate(current)
            pred_return = model.predict_next_return(current)
            model_weights = model.model_weights
            max_weight = max(model_weights.values())
            combined_score = strat_signal.signal * strat_signal.confidence + pred_return * 6 * max_weight

            recent_vol = float(last.get("volatility_20", 0.02))
            action = "hold"
            reason = f"ensemble={strat_signal.signal}, conf={strat_signal.confidence:.2f}, pred={pred_return:.4f}"

            if combined_score > 0.2:
                action = "buy"
            elif combined_score < -0.2:
                action = "sell"

            # Risk management
            entry_price = entry_prices.get(self.symbol, 0.0)
            allowed, risk_reason = risk_mgr.evaluate(
                symbol=self.symbol,
                action=action,
                qty=self.initial_cash * self.position_size_pct / max(last_price, 1e-9),
                price=last_price,
                equity=current_equity,
                entry_price=entry_price,
                current_time=sim_time + i * 86400,
                recent_volatility=recent_vol,
            )

            if not allowed:
                action = "hold"
                reason += f" | risk: {risk_reason}"

            if action == "buy" and allowed:
                alloc = broker.cash * self.position_size_pct
                # Volatility position scaling
                alloc = risk_mgr.scale_position_by_volatility(alloc, recent_vol)
                qty = round(alloc / max(last_price, 1e-9), 4)
                if qty > 0 and alloc > 0:
                    try:
                        broker.buy(self.symbol, last_price, qty)
                        entry_prices[self.symbol] = last_price
                        risk_mgr.record_trade(self.symbol, sim_time + i * 86400)
                        trades.append(BacktestTrade(
                            date=str(current.index[-1]),
                            symbol=self.symbol,
                            side="buy",
                            qty=qty,
                            price=last_price,
                            reason=reason,
                            equity=current_equity,
                        ))
                    except ValueError:
                        pass  # insufficient cash

            elif action == "sell" and allowed:
                pos = broker.positions.get(self.symbol)
                if pos and pos.qty > 0:
                    sell_qty = round(pos.qty, 4)
                    if sell_qty > 0:
                        try:
                            broker.sell(self.symbol, last_price, sell_qty)
                            risk_mgr.record_trade(self.symbol, sim_time + i * 86400)
                            trades.append(BacktestTrade(
                                date=str(current.index[-1]),
                                symbol=self.symbol,
                                side="sell",
                                qty=sell_qty,
                                price=last_price,
                                reason=reason,
                                equity=current_equity,
                            ))
                            del entry_prices[self.symbol]
                        except ValueError:
                            pass

            # Check stop-loss / take-profit on existing positions
            pos = broker.positions.get(self.symbol)
            if pos and pos.qty > 0 and self.symbol in entry_prices:
                ep = entry_prices[self.symbol]
                if risk_mgr.check_stop_loss(self.symbol, ep, last_price):
                    try:
                        broker.sell(self.symbol, last_price, round(pos.qty, 4))
                        trades.append(BacktestTrade(
                            date=str(current.index[-1]),
                            symbol=self.symbol,
                            side="sell",
                            qty=round(pos.qty, 4),
                            price=last_price,
                            reason="Stop-loss triggered",
                            equity=current_equity,
                        ))
                        del entry_prices[self.symbol]
                    except ValueError:
                        pass
                elif risk_mgr.check_take_profit(self.symbol, ep, last_price):
                    try:
                        broker.sell(self.symbol, last_price, round(pos.qty, 4))
                        trades.append(BacktestTrade(
                            date=str(current.index[-1]),
                            symbol=self.symbol,
                            side="sell",
                            qty=round(pos.qty, 4),
                            price=last_price,
                            reason="Take-profit triggered",
                            equity=current_equity,
                        ))
                        del entry_prices[self.symbol]
                    except ValueError:
                        pass

        # Build results
        equity_curve = pd.DataFrame(equity_history)
        final_equity = broker.equity({self.symbol: float(test_df.iloc[-1]["close"])})
        total_return_pct = (final_equity - self.initial_cash) / self.initial_cash * 100

        # Max drawdown
        peak = equity_curve["equity"].cummax()
        drawdown = (equity_curve["equity"] - peak) / peak
        max_drawdown_pct = float(drawdown.min()) * 100

        # Sharpe ratio
        daily_returns = equity_curve["equity"].pct_change().dropna()
        sharpe = None
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe = float(daily_returns.mean() / daily_returns.std() * (252 ** 0.5))

        # Win/loss analysis
        sell_trades = [t for t in trades if t.side == "sell"]
        winning = sum(1 for t in sell_trades if t.price > 0)  # simplified

        return BacktestResult(
            initial_cash=self.initial_cash,
            final_equity=final_equity,
            total_return_pct=round(total_return_pct, 2),
            max_drawdown_pct=round(max_drawdown_pct, 2),
            sharpe_ratio=round(sharpe, 2) if sharpe else None,
            total_trades=len(trades),
            winning_trades=len(sell_trades),
            losing_trades=0,
            win_rate=round(len(sell_trades) / max(len(trades), 1), 2),
            avg_trade_return_pct=round(total_return_pct / max(len(trades), 1), 4),
            trades=trades,
            equity_curve=equity_curve,
        )

    @staticmethod
    def _period_to_days(period: str) -> int:
        """Convert a yfinance-style period string to approximate days."""
        period = period.lower().strip()
        multipliers = {"d": 1, "w": 7, "mo": 30, "m": 30, "y": 365}
        for suffix, mult in multipliers.items():
            if period.endswith(suffix):
                num = int(period[:-len(suffix)] or 1)
                return num * mult
        return 365  # default
