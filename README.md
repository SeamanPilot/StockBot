# StockBot

Paper-trading stock bot with deep-learning forecasts, multi-strategy ensemble signals, and a Streamlit dashboard.

## Quick Start

```bash
git clone https://github.com/SeamanPilot/StockBot.git
cd StockBot
python -m pip install -r requirements.txt
python -m stockbot.wizard   # configure broker + symbols
streamlit run run_stockbot.py
```

## Architecture

- **stockbot/config.py** — Broker credentials and bot settings (dataclasses → JSON)
- **stockbot/data.py** — OHLCV data via yfinance
- **stockbot/features.py** — Technical indicators (SMA, RSI, MACD, Bollinger Bands)
- **stockbot/model.py** — MLPRegressor deep-learning price forecaster
- **stockbot/strategies.py** — SMA crossover, RSI mean-reversion, breakout → ensemble
- **stockbot/trader.py** — PaperTradingEngine combining strategy + model signals
- **stockbot/brokers.py** — SimulatedBroker (Alpaca/IB/Tradier stubs)
- **stockbot/dashboard.py** — Streamlit UI with candlestick charts
- **stockbot/wizard.py** — CLI setup wizard
- **run_stockbot.py** — Entry point

## License

Apache 2.0
