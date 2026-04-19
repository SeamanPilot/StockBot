from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional


SUPPORTED_BROKERS = {
    "paper": "Local simulation broker",
    "alpaca": "Alpaca Trading API",
    "interactive_brokers": "Interactive Brokers (gateway/TWS integration placeholder)",
    "tradier": "Tradier REST API placeholder",
}


@dataclass
class Position:
    qty: float = 0.0
    avg_price: float = 0.0

    @property
    def cost_basis(self) -> float:
        return self.qty * self.avg_price


@dataclass
class SimulatedBroker:
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    commission_per_trade: float = 0.0

    def buy(self, symbol: str, price: float, qty: float) -> None:
        cost = price * qty + self.commission_per_trade
        if cost > self.cash:
            raise ValueError(f"Insufficient cash: have {self.cash:.2f}, need {cost:.2f}")
        pos = self.positions.get(symbol, Position())
        total_qty = pos.qty + qty
        pos.avg_price = ((pos.avg_price * pos.qty) + price * qty) / max(total_qty, 1e-9)
        pos.qty = total_qty
        self.positions[symbol] = pos
        self.cash -= cost

    def sell(self, symbol: str, price: float, qty: float) -> None:
        pos = self.positions.get(symbol)
        if pos is None or pos.qty < qty:
            raise ValueError(f"Insufficient position: have {pos.qty if pos else 0}, need {qty}")
        pos.qty -= qty
        self.cash += price * qty - self.commission_per_trade

    def equity(self, market_prices: dict[str, float]) -> float:
        value = self.cash
        for sym, pos in self.positions.items():
            value += pos.qty * market_prices.get(sym, pos.avg_price)
        return value


class AlpacaBroker:
    """Live/paper trading via Alpaca API with market, limit, and stop orders."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        paper: bool = True,
        base_url: Optional[str] = None,
    ) -> None:
        try:
            import alpaca_trade_api as alpaca
        except ImportError:
            raise ImportError(
                "alpaca-trade-api is required for live trading. "
                "Install with: pip install alpaca-trade-api"
            )

        self.api_key = api_key or os.environ.get("ALPACA_API_KEY", "")
        self.api_secret = api_secret or os.environ.get("ALPACA_API_SECRET", "")
        if not self.api_key or not self.api_secret:
            raise ValueError("Alpaca API key and secret are required.")

        if base_url:
            url = base_url
        elif paper:
            url = "https://paper-api.alpaca.markets"
        else:
            url = "https://api.alpaca.markets"

        self.api = alpaca.REST(
            key_id=self.api_key,
            secret_key=self.api_secret,
            base_url=url,
        )
        self.paper = paper

    def buy(self, symbol: str, qty: float, order_type: str = "market", limit_price: Optional[float] = None) -> dict:
        kwargs = {
            "symbol": symbol,
            "qty": qty,
            "side": "buy",
            "type": order_type,
            "time_in_force": "day",
        }
        if order_type == "limit" and limit_price:
            kwargs["limit_price"] = str(limit_price)
        order = self.api.submit_order(**kwargs)
        return {
            "id": order.id,
            "symbol": order.symbol,
            "qty": float(order.qty),
            "side": order.side,
            "type": order.type,
            "status": order.status,
        }

    def sell(self, symbol: str, qty: float, order_type: str = "market", limit_price: Optional[float] = None, stop_price: Optional[float] = None) -> dict:
        kwargs = {
            "symbol": symbol,
            "qty": qty,
            "side": "sell",
            "type": order_type,
            "time_in_force": "day",
        }
        if order_type == "limit" and limit_price:
            kwargs["limit_price"] = str(limit_price)
        if order_type == "stop" and stop_price:
            kwargs["stop_price"] = str(stop_price)
            kwargs["type"] = "stop"
        if order_type == "stop_limit" and limit_price and stop_price:
            kwargs["limit_price"] = str(limit_price)
            kwargs["stop_price"] = str(stop_price)
            kwargs["type"] = "stop_limit"
        order = self.api.submit_order(**kwargs)
        return {
            "id": order.id,
            "symbol": order.symbol,
            "qty": float(order.qty),
            "side": order.side,
            "type": order.type,
            "status": order.status,
        }

    def get_position(self, symbol: str) -> Optional[Position]:
        try:
            pos = self.api.get_position(symbol)
            return Position(qty=float(pos.qty), avg_price=float(pos.avg_entry_price))
        except Exception:
            return None

    def get_cash(self) -> float:
        account = self.api.get_account()
        return float(account.cash)

    def equity(self, market_prices: Optional[dict] = None) -> float:
        account = self.api.get_account()
        return float(account.equity)

    def cancel_all_orders(self, symbol: Optional[str] = None) -> None:
        if symbol:
            orders = self.api.list_orders(status="open", symbols=[symbol])
            for order in orders:
                self.api.cancel_order(order.id)
        else:
            self.api.cancel_all_orders()
