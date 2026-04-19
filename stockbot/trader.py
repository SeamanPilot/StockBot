from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .brokers import SimulatedBroker, AlpacaBroker, Position
from .data import fetch_ohlcv
from .features import with_indicators
from .model import EnsemblePriceForecaster
from .risk import RiskConfig, RiskManager
from .strategies import default_strategy_stack


DATA_DIR = Path(os.environ.get("STOCKBOT_DATA_DIR", Path(__file__).parent.parent / "stockbot_data"))
TRADE_LOG_PATH = DATA_DIR / "trade_history.jsonl"
EQUITY_LOG_PATH = DATA_DIR / "equity_history.jsonl"


@dataclass
class TradeEvent:
    timestamp: str
    symbol: str
    side: str
    qty: float
    price: float
    reason: str
    equity: float = 0.0
    entry_price: float = 0.0


class PaperTradingEngine:
    """Enhanced paper trading engine with risk management, portfolio tracking, and model persistence."""

    def __init__(
        self,
        symbols: list[str],
        initial_cash: float = 100_000,
        position_size_pct: float = 0.1,
        risk_config: Optional[RiskConfig] = None,
        use_alpaca: bool = False,
        alpaca_api_key: Optional[str] = None,
        alpaca_api_secret: Optional[str] = None,
        alpaca_paper: bool = True,
    ) -> None:
        self.symbols = symbols
        self.position_size_pct = position_size_pct
        self.risk_config = risk_config or RiskConfig()
        self.risk_mgr = RiskManager(self.risk_config)

        if use_alpaca:
            self.broker = AlpacaBroker(
                api_key=alpaca_api_key,
                api_secret=alpaca_api_secret,
                paper=alpaca_paper,
            )
            self._live_broker = True
        else:
            self.broker = SimulatedBroker(cash=initial_cash)
            self._live_broker = False

        self.model = EnsemblePriceForecaster()
        self.strategy = default_strategy_stack()
        self.trade_log: list[TradeEvent] = []
        self._entry_prices: dict[str, float] = {}
        self._equity_history: list[dict] = []

        # Ensure data directory exists
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    def train(self, symbol: str, period: str = "2y", interval: str = "1d") -> Any:
        df = fetch_ohlcv(symbol, period=period, interval=interval)
        feat = with_indicators(df)
        self.model.fit(feat)
        return feat

    def save_model(self, path: Optional[str] = None) -> None:
        """Save the trained model to disk."""
        save_path = Path(path) if path else DATA_DIR / "model"
        self.model.save(save_path)

    def load_model(self, path: Optional[str] = None) -> None:
        """Load a trained model from disk."""
        load_path = Path(path) if path else DATA_DIR / "model"
        self.model.load(load_path)

    def _record_equity(self, symbol: str, price: float) -> None:
        equity = self.broker.equity({symbol: price}) if not self._live_broker else self.broker.equity()
        self._equity_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "equity": equity,
            "cash": self.broker.cash if not self._live_broker else 0,
            "price": price,
        })

    def run_once(self, symbol: str, period: str = "6mo", interval: str = "1d") -> dict[str, Any]:
        df = fetch_ohlcv(symbol, period=period, interval=interval)
        feat = with_indicators(df)

        if not self.model.trained:
            self.model.fit(feat)

        strat_signal = self.strategy.generate(feat)
        pred_return = self.model.predict_next_return(feat)
        detailed = self.model.predict_detailed(feat)
        model_weights = self.model.model_weights
        max_weight = max(model_weights.values())
        combined_score = strat_signal.signal * strat_signal.confidence + pred_return * 6 * max_weight

        last_price = float(feat.iloc[-1]["close"])
        recent_vol = float(feat.iloc[-1].get("volatility_20", 0.02))

        alloc_cash = self.broker.cash * self.position_size_pct if not self._live_broker else 0
        qty = round(alloc_cash / max(last_price, 1e-9), 4)

        action = "hold"
        reason_parts = [
            f"ensemble={strat_signal.signal}",
            f"conf={strat_signal.confidence:.2f}",
            f"pred={pred_return:.4f}",
        ]

        if combined_score > 0.2:
            action = "buy"
        elif combined_score < -0.2:
            action = "sell"

        # Risk management evaluation
        entry_price = self._entry_prices.get(symbol, 0.0)
        current_time = datetime.now(timezone.utc).timestamp()
        allowed, risk_reason = self.risk_mgr.evaluate(
            symbol=symbol,
            action=action,
            qty=qty,
            price=last_price,
            equity=self.broker.equity({symbol: last_price}) if not self._live_broker else self.broker.equity(),
            entry_price=entry_price,
            current_time=current_time,
            recent_volatility=recent_vol,
        )

        if not allowed:
            action = "hold"
            reason_parts.append(f"risk={risk_reason}")

        # Execute trade
        if action == "buy" and allowed:
            scaled_alloc = self.risk_mgr.scale_position_by_volatility(alloc_cash, recent_vol)
            qty = round(scaled_alloc / max(last_price, 1e-9), 4)
            if qty > 0:
                if self._live_broker:
                    result = self.broker.buy(symbol, qty)
                    reason_parts.append(f"order_id={result['id']}")
                else:
                    self.broker.buy(symbol, last_price, qty)
                self._entry_prices[symbol] = last_price
                self.risk_mgr.record_trade(symbol, current_time)
                self.trade_log.append(TradeEvent(
                    datetime.now(timezone.utc).isoformat(), symbol, "buy", qty, last_price,
                    ", ".join(reason_parts),
                    equity=self.broker.equity({symbol: last_price}) if not self._live_broker else 0,
                    entry_price=last_price,
                ))

        elif action == "sell" and allowed:
            pos = self.broker.positions.get(symbol) if not self._live_broker else self.broker.get_position(symbol)
            if pos and pos.qty > 0:
                sell_qty = round(min(pos.qty, qty if qty > 0 else pos.qty), 4)
                if sell_qty > 0:
                    if self._live_broker:
                        result = self.broker.sell(symbol, sell_qty)
                        reason_parts.append(f"order_id={result['id']}")
                    else:
                        self.broker.sell(symbol, last_price, sell_qty)
                    self.risk_mgr.record_trade(symbol, current_time)
                    self.trade_log.append(TradeEvent(
                        datetime.now(timezone.utc).isoformat(), symbol, "sell", sell_qty, last_price,
                        ", ".join(reason_parts),
                        equity=self.broker.equity({symbol: last_price}) if not self._live_broker else 0,
                        entry_price=entry_price,
                    ))
                    if symbol in self._entry_prices:
                        del self._entry_prices[symbol]

        # Check stop-loss / take-profit
        pos = self.broker.positions.get(symbol) if not self._live_broker else self.broker.get_position(symbol)
        if pos and pos.qty > 0 and symbol in self._entry_prices:
            ep = self._entry_prices[symbol]
            if self.risk_mgr.check_stop_loss(symbol, ep, last_price):
                if not self._live_broker:
                    self.broker.sell(symbol, last_price, round(pos.qty, 4))
                else:
                    self.broker.sell(symbol, round(pos.qty, 4))
                self.trade_log.append(TradeEvent(
                    datetime.now(timezone.utc).isoformat(), symbol, "sell", round(pos.qty, 4),
                    last_price, "Stop-loss triggered",
                    equity=self.broker.equity({symbol: last_price}) if not self._live_broker else 0,
                ))
                del self._entry_prices[symbol]
            elif self.risk_mgr.check_take_profit(symbol, ep, last_price):
                if not self._live_broker:
                    self.broker.sell(symbol, last_price, round(pos.qty, 4))
                else:
                    self.broker.sell(symbol, round(pos.qty, 4))
                self.trade_log.append(TradeEvent(
                    datetime.now(timezone.utc).isoformat(), symbol, "sell", round(pos.qty, 4),
                    last_price, "Take-profit triggered",
                    equity=self.broker.equity({symbol: last_price}) if not self._live_broker else 0,
                ))
                del self._entry_prices[symbol]

        self._record_equity(symbol, last_price)

        equity = self.broker.equity({symbol: last_price}) if not self._live_broker else self.broker.equity()
        return {
            "symbol": symbol,
            "price": last_price,
            "action": action,
            "reason": ", ".join(reason_parts),
            "equity": equity,
            "cash": self.broker.cash if not self._live_broker else 0,
            "position_qty": self.broker.positions.get(symbol).qty if not self._live_broker and symbol in self.broker.positions else (pos.qty if pos else 0),
            "predicted_return": pred_return,
            "strategy_confidence": strat_signal.confidence,
            "model_weights": model_weights,
            "model_predictions": detailed["predictions"],
            "volatility_20": recent_vol,
            "data": feat,
        }

    def run_portfolio(self, period: str = "6mo", interval: str = "1d") -> dict[str, dict]:
        """Run a trading cycle for all symbols in the portfolio."""
        results = {}
        for symbol in self.symbols:
            try:
                results[symbol] = self.run_once(symbol, period=period, interval=interval)
            except Exception as e:
                results[symbol] = {"error": str(e), "symbol": symbol}
        return results

    def save_trade_log(self) -> None:
        """Persist trade log to JSONL file."""
        with open(TRADE_LOG_PATH, "a", encoding="utf-8") as f:
            for trade in self.trade_log:
                f.write(json.dumps({
                    "timestamp": trade.timestamp,
                    "symbol": trade.symbol,
                    "side": trade.side,
                    "qty": trade.qty,
                    "price": trade.price,
                    "reason": trade.reason,
                    "equity": trade.equity,
                    "entry_price": trade.entry_price,
                }) + "\n")

    def save_equity_history(self) -> None:
        """Persist equity history to JSONL file."""
        with open(EQUITY_LOG_PATH, "a", encoding="utf-8") as f:
            for entry in self._equity_history:
                f.write(json.dumps(entry) + "\n")
