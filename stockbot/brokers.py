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
    commission_per_trade: float = 0.0  # flat fee per trade

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
    """Live/paper trading via Alpaca API.

    Requires ALPACA_API_KEY and ALPACA_API_SECRET env vars,
    or pass them to the constructor.
    """

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

    def buy(self, symbol: str, qty: float, side: str = "buy") -> dict:
        order = self.api.submit_order(
            symbol=symbol,
            qty=qty,
            side="buy",
            type="market",
            time_in_force="day",
        )
        return {
            "id": order.id,
            "symbol": order.symbol,
            "qty": float(order.qty),
            "side": order.side,
            "status": order.status,
        }

    def sell(self, symbol: str, qty: float) -> dict:
        order = self.api.submit_order(
            symbol=symbol,
            qty=qty,
            side="sell",
            type="market",
            time_in_force="day",
        )
        return {
            "id": order.id,
            "symbol": order.symbol,
            "qty": float(order.qty),
            "side": order.side,
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
