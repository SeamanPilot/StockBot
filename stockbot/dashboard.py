from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from .backtest import BacktestEngine, BacktestResult
from .config import CONFIG_PATH, load_config, BotSettings, BrokerCredentials
from .risk import RiskConfig
from .trader import PaperTradingEngine, DATA_DIR, TRADE_LOG_PATH, EQUITY_LOG_PATH


@st.cache_resource
def get_engine(
    symbols: list[str],
    initial_cash: float,
    position_size_pct: float,
    risk_config: Optional[RiskConfig] = None,
) -> PaperTradingEngine:
    return PaperTradingEngine(
        symbols=symbols,
        initial_cash=initial_cash,
        position_size_pct=position_size_pct,
        risk_config=risk_config,
    )


def _price_figure(df: pd.DataFrame, symbol: str) -> go.Figure:
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=[f"{symbol} Price", "Volume", "RSI"],
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="Price",
    ), row=1, col=1)

    # Moving averages
    if "sma_fast" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["sma_fast"], mode="lines", name="SMA 10", line=dict(width=1)), row=1, col=1)
    if "sma_slow" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["sma_slow"], mode="lines", name="SMA 30", line=dict(width=1)), row=1, col=1)

    # Bollinger Bands
    if "bb_upper" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["bb_upper"], mode="lines", name="BB Upper", line=dict(dash="dot", width=1)), row=1, col=1)
    if "bb_lower" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["bb_lower"], mode="lines", name="BB Lower", line=dict(dash="dot", width=1)), row=1, col=1)

    # Volume
    colors = ["#26a69a" if c >= o else "#ef5350" for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["volume"], name="Volume", marker_color=colors, showlegend=False), row=2, col=1)

    # RSI
    if "rsi" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["rsi"], mode="lines", name="RSI", line=dict(color="#9C27B0", width=1)), row=3, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)

    fig.update_layout(height=700, xaxis_rangeslider_visible=False, showlegend=True)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    return fig


def _equity_figure(equity_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity_df["date"], y=equity_df["equity"],
        mode="lines", name="Equity", line=dict(color="#00E676", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=equity_df["date"], y=equity_df["cash"],
        mode="lines", name="Cash", line=dict(color="#FFC107", width=1, dash="dot"),
    ))
    fig.update_layout(title="Equity Curve", xaxis_title="Date", yaxis_title="USD", height=400)
    return fig


def _pnl_figure(trades_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=trades_df["timestamp"], y=trades_df["pnl"],
        name="P&L per trade", marker_color=trades_df["pnl"].apply(lambda x: "#26a69a" if x >= 0 else "#ef5350"),
    ))
    fig.update_layout(title="Trade P&L", xaxis_title="Date", yaxis_title="P&L USD", height=300)
    return fig


def _load_trade_log() -> pd.DataFrame:
    if TRADE_LOG_PATH.exists():
        lines = TRADE_LOG_PATH.read_text(encoding="utf-8").strip().split("\n")
        records = [json.loads(line) for line in lines if line.strip()]
        return pd.DataFrame(records)
    return pd.DataFrame()


def _load_equity_history() -> pd.DataFrame:
    if EQUITY_LOG_PATH.exists():
        lines = EQUITY_LOG_PATH.read_text(encoding="utf-8").strip().split("\n")
        records = [json.loads(line) for line in lines if line.strip()]
        return pd.DataFrame(records)
    return pd.DataFrame()


def run_dashboard() -> None:
    st.set_page_config(layout="wide", page_title="StockBot")
    st.title("StockBot — Ensemble ML Paper Trading")

    # Sidebar - Configuration
    st.sidebar.header("Configuration")

    if CONFIG_PATH.exists():
        creds, settings = load_config()
    else:
        st.sidebar.warning("Run setup wizard first: `python -m stockbot.wizard`")
        creds = BrokerCredentials(broker="paper")
        settings = BotSettings(symbols=["AAPL"])

    symbol = st.sidebar.selectbox("Symbol", settings.symbols)
    st.sidebar.caption(f"Broker: {creds.broker}")

    # Risk config sidebar
    st.sidebar.subheader("Risk Management")
    stop_loss = st.sidebar.slider("Stop Loss %", 1.0, 20.0, 5.0, 0.5) / 100
    take_profit = st.sidebar.slider("Take Profit %", 5.0, 50.0, 15.0, 1.0) / 100
    max_drawdown = st.sidebar.slider("Max Drawdown %", 5.0, 30.0, 10.0, 1.0) / 100
    max_position = st.sidebar.slider("Max Position %", 10.0, 50.0, 25.0, 5.0) / 100

    risk_config = RiskConfig(
        stop_loss_pct=stop_loss,
        take_profit_pct=take_profit,
        max_drawdown_pct=max_drawdown,
        max_position_pct=max_position,
    )

    engine = get_engine(settings.symbols, settings.initial_cash, settings.position_size_pct, risk_config)

    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Trade", "Portfolio", "Backtest", "Model Info", "History"
    ])

    # ─── Tab 1: Trade ───
    with tab1:
        if st.button("Run Trading Cycle", type="primary"):
            with st.spinner("Fetching data and generating signals..."):
                result = engine.run_once(symbol=symbol, period=settings.lookback, interval=settings.timeframe)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Action", result["action"])
            c2.metric("Equity", f"${result['equity']:,.2f}")
            c3.metric("Cash", f"${result['cash']:,.2f}")
            c4.metric("Pred Return", f"{result['predicted_return'] * 100:.2f}%")

            st.info(result["reason"])

            # Model weights
            weights = result.get("model_weights", {})
            if weights:
                st.subheader("Model Weights")
                w_cols = st.columns(len(weights))
                for col, (name, w) in zip(w_cols, weights.items()):
                    col.metric(name.upper(), f"{w:.1%}")

            # Model predictions
            preds = result.get("model_predictions", {})
            if preds:
                st.subheader("Individual Predictions")
                p_cols = st.columns(len(preds))
                for col, (name, val) in zip(p_cols, preds.items()):
                    col.metric(name.upper(), f"{val*100:.3f}%")

            # Chart
            chart_data = result["data"].tail(160)
            fig = _price_figure(chart_data, symbol)
            st.plotly_chart(fig, use_container_width=True)

        if engine.trade_log:
            st.subheader("Trade Log (Session)")
            st.dataframe(pd.DataFrame([t.__dict__ for t in engine.trade_log[-20:]]), use_container_width=True)

            # Save button
            if st.button("Save Trade Log"):
                engine.save_trade_log()
                st.success(f"Trade log saved to {TRADE_LOG_PATH}")

    # ─── Tab 2: Portfolio ───
    with tab2:
        if st.button("Run All Symbols"):
            with st.spinner("Running portfolio cycle..."):
                results = engine.run_portfolio(period=settings.lookback, interval=settings.timeframe)

            for sym, res in results.items():
                if "error" in res:
                    st.error(f"{sym}: {res['error']}")
                    continue
                with st.expander(f"{sym} — {res['action'].upper()} @ ${res['price']:.2f}"):
                    st.write(f"**Reason:** {res['reason']}")
                    st.write(f"**Predicted Return:** {res['predicted_return']*100:.3f}%")
                    st.write(f"**Equity:** ${res['equity']:,.2f}")

    # ─── Tab 3: Backtest ───
    with tab3:
        st.subheader("Walk-Forward Backtest")
        bt_cols = st.columns(3)
        bt_symbol = bt_cols[0].text_input("Symbol", value=symbol)
        bt_train = bt_cols[1].selectbox("Train Period", ["6mo", "1y", "2y"], index=1)
        bt_test = bt_cols[2].selectbox("Test Period", ["3mo", "6mo", "1y"], index=1)
        bt_commission = st.slider("Commission per trade ($)", 0.0, 10.0, 1.0, 0.5)

        if st.button("Run Backtest"):
            with st.spinner("Running backtest..."):
                bt = BacktestEngine(
                    symbol=bt_symbol,
                    train_period=bt_train,
                    test_period=bt_test,
                    risk_config=risk_config,
                    commission_per_trade=bt_commission,
                )
                result = bt.run()

            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Total Return", f"{result.total_return_pct:.2f}%")
            r2.metric("Max Drawdown", f"{result.max_drawdown_pct:.2f}%")
            r3.metric("Sharpe Ratio", f"{result.sharpe_ratio:.2f}" if result.sharpe_ratio else "N/A")
            r4.metric("Total Trades", result.total_trades)

            r5, r6, r7 = st.columns(3)
            r5.metric("Win Rate", f"{result.win_rate:.0%}")
            r6.metric("Final Equity", f"${result.final_equity:,.2f}")
            r7.metric("Avg Trade", f"{result.avg_trade_return_pct:.4f}%")

            if not result.equity_curve.empty:
                eq_fig = _equity_figure(result.equity_curve)
                st.plotly_chart(eq_fig, use_container_width=True)

            if result.trades:
                st.subheader("Backtest Trades")
                trades_df = pd.DataFrame([{
                    "date": t.date, "symbol": t.symbol, "side": t.side,
                    "qty": t.qty, "price": t.price, "reason": t.reason, "equity": t.equity,
                } for t in result.trades])
                st.dataframe(trades_df, use_container_width=True)

    # ─── Tab 4: Model Info ───
    with tab4:
        if st.button("Train Model"):
            with st.spinner("Training ensemble model..."):
                engine.train(symbol, period=settings.lookback, interval=settings.timeframe)
                st.success("Model trained!")

        if st.button("Save Model"):
            engine.save_model()
            st.success("Model saved!")

        if st.button("Load Model"):
            engine.load_model()
            st.success("Model loaded!")

        if engine.model.trained:
            st.subheader("Model Weights")
            weights = engine.model.model_weights
            for name, w in weights.items():
                st.progress(min(w, 1.0), text=f"{name.upper()}: {w:.1%}")

            scores = engine.model._validation_scores
            if scores:
                st.subheader("Walk-Forward Validation Scores")
                for name, score in scores.items():
                    st.write(f"**{name.upper()}**: R² = {score:.4f}")

    # ─── Tab 5: History ───
    with tab5:
        st.subheader("Equity History")
        equity_df = _load_equity_history()
        if not equity_df.empty:
            st.dataframe(equity_df.tail(50), use_container_width=True)
            eq_fig = _price_figure(chart_data, symbol) if engine.model.trained else None
        else:
            st.info("No equity history yet. Run trading cycles to build history.")

        st.subheader("Trade History")
        trade_df = _load_trade_log()
        if not trade_df.empty:
            st.dataframe(trade_df.tail(50), use_container_width=True)
        else:
            st.info("No trade history yet. Run and save trade logs.")


if __name__ == "__main__":
    run_dashboard()
