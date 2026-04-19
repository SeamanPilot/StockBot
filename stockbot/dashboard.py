from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from .backtest import BacktestEngine
from .config import CONFIG_PATH, load_config, BotSettings, BrokerCredentials
from .feature_importance import gbr_feature_importance, rf_feature_importance, permutation_importance
from .multi_tf import analyze_multi_timeframe, TIMEFRAMES
from .risk import RiskConfig
from .trader import PaperTradingEngine, DATA_DIR, TRADE_LOG_PATH, EQUITY_LOG_PATH


# Dark theme config
DARK_THEME = {
    "bgcolor": "#0e1117",
    "plot_bgcolor": "#0e1117",
    "font_color": "#e0e0e0",
    "grid_color": "#262730",
    "green": "#00e676",
    "red": "#ff5252",
    "blue": "#448aff",
    "yellow": "#ffd740",
}


@st.cache_resource
def get_engine(
    symbols: list[str],
    initial_cash: float,
    position_size_pct: float,
    risk_config: Optional[RiskConfig] = None,
) -> PaperTradingEngine:
    return PaperTradingEngine(
        symbols=symbols, initial_cash=initial_cash,
        position_size_pct=position_size_pct, risk_config=risk_config,
    )


def _dark_fig(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        plot_bgcolor=DARK_THEME["plot_bgcolor"],
        paper_bgcolor=DARK_THEME["bgcolor"],
        font_color=DARK_THEME["font_color"],
    )
    fig.update_xaxes(gridcolor=DARK_THEME["grid_color"])
    fig.update_yaxes(gridcolor=DARK_THEME["grid_color"])
    return fig


def _price_figure(df: pd.DataFrame, symbol: str) -> go.Figure:
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=[f"{symbol} Price", "Volume", "RSI"],
    )
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="Price",
        increasing_line_color=DARK_THEME["green"], decreasing_line_color=DARK_THEME["red"],
    ), row=1, col=1)
    if "sma_fast" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["sma_fast"], mode="lines", name="SMA 10",
                                 line=dict(width=1, color=DARK_THEME["blue"])), row=1, col=1)
    if "sma_slow" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["sma_slow"], mode="lines", name="SMA 30",
                                 line=dict(width=1, color=DARK_THEME["yellow"])), row=1, col=1)
    if "bb_upper" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["bb_upper"], mode="lines", name="BB Upper",
                                 line=dict(dash="dot", width=1, color="#888")), row=1, col=1)
    if "bb_lower" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["bb_lower"], mode="lines", name="BB Lower",
                                 line=dict(dash="dot", width=1, color="#888")), row=1, col=1)
    colors = [DARK_THEME["green"] if c >= o else DARK_THEME["red"] for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["volume"], name="Volume", marker_color=colors, showlegend=False), row=2, col=1)
    if "rsi" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["rsi"], mode="lines", name="RSI",
                                 line=dict(color="#ce93d8", width=1)), row=3, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color=DARK_THEME["red"], row=3, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color=DARK_THEME["green"], row=3, col=1)
    fig.update_layout(height=700, xaxis_rangeslider_visible=False, showlegend=True)
    return _dark_fig(fig)


def _equity_figure(equity_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity_df["date"], y=equity_df["equity"],
                             mode="lines", name="Equity", line=dict(color=DARK_THEME["green"], width=2)))
    fig.add_trace(go.Scatter(x=equity_df["date"], y=equity_df["cash"],
                             mode="lines", name="Cash", line=dict(color=DARK_THEME["yellow"], width=1, dash="dot")))
    fig.update_layout(title="Equity Curve", xaxis_title="Date", yaxis_title="USD", height=400)
    return _dark_fig(fig)


def _importance_figure(imp_df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=imp_df["importance"],
        y=imp_df.index if "feature" not in imp_df.columns else imp_df["feature"],
        orientation="h",
        marker_color=DARK_THEME["blue"],
    ))
    fig.update_layout(title=title, height=500, yaxis_title="Feature", xaxis_title="Importance")
    return _dark_fig(fig)


def _load_trade_log() -> pd.DataFrame:
    if TRADE_LOG_PATH.exists():
        lines = TRADE_LOG_PATH.read_text(encoding="utf-8").strip().split("\n")
        return pd.DataFrame([json.loads(l) for l in lines if l.strip()])
    return pd.DataFrame()


def _load_equity_history() -> pd.DataFrame:
    if EQUITY_LOG_PATH.exists():
        lines = EQUITY_LOG_PATH.read_text(encoding="utf-8").strip().split("\n")
        return pd.DataFrame([json.loads(l) for l in lines if l.strip()])
    return pd.DataFrame()


def run_dashboard() -> None:
    st.set_page_config(layout="wide", page_title="StockBot", page_icon="📊")
    st.markdown("""
    <style>
    .stApp {background-color: #0e1117; color: #e0e0e0;}
    .stSidebar {background-color: #1a1a2e;}
    </style>
    """, unsafe_allow_html=True)
    st.title("📊 StockBot — Ensemble ML Trading")

    # Config
    if CONFIG_PATH.exists():
        creds, settings = load_config()
    else:
        st.sidebar.warning("Run setup wizard: `python -m stockbot.wizard`")
        creds = BrokerCredentials(broker="paper")
        settings = BotSettings(symbols=["AAPL"])

    symbol = st.sidebar.selectbox("Symbol", settings.symbols)
    st.sidebar.caption(f"Broker: {creds.broker}")

    # Risk config
    st.sidebar.subheader("Risk Management")
    stop_loss = st.sidebar.slider("Stop Loss %", 1.0, 20.0, 5.0, 0.5) / 100
    take_profit = st.sidebar.slider("Take Profit %", 5.0, 50.0, 15.0, 1.0) / 100
    max_drawdown = st.sidebar.slider("Max Drawdown %", 5.0, 30.0, 10.0, 1.0) / 100
    max_position = st.sidebar.slider("Max Position %", 10.0, 50.0, 25.0, 5.0) / 100
    risk_config = RiskConfig(stop_loss_pct=stop_loss, take_profit_pct=take_profit,
                             max_drawdown_pct=max_drawdown, max_position_pct=max_position)

    engine = get_engine(settings.symbols, settings.initial_cash, settings.position_size_pct, risk_config)

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📈 Trade", "💼 Portfolio", "🔄 Backtest", "🧠 Model", "📊 Features", "🕐 Multi-TF", "📜 History"
    ])

    # ── Trade ──
    with tab1:
        if st.button("Run Trading Cycle", type="primary"):
            with st.spinner("Generating signals..."):
                result = engine.run_once(symbol=symbol, period=settings.lookback, interval=settings.timeframe)
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Action", result["action"])
            c2.metric("Equity", f"${result['equity']:,.2f}")
            c3.metric("Cash", f"${result['cash']:,.2f}")
            c4.metric("Pred Return", f"{result['predicted_return'] * 100:.2f}%")
            c5.metric("Vol 20d", f"{result.get('volatility_20', 0):.4f}")
            st.info(result["reason"])
            weights = result.get("model_weights", {})
            if weights:
                st.subheader("Model Weights")
                w_cols = st.columns(len(weights))
                for col, (name, w) in zip(w_cols, weights.items()):
                    col.metric(name.upper(), f"{w:.1%}")
            preds = result.get("model_predictions", {})
            if preds:
                st.subheader("Individual Predictions")
                p_cols = st.columns(len(preds))
                for col, (name, val) in zip(p_cols, preds.items()):
                    col.metric(name.upper(), f"{val*100:.3f}%")
            chart_data = result["data"].tail(160)
            st.plotly_chart(_price_figure(chart_data, symbol), use_container_width=True)
        if engine.trade_log:
            st.subheader("Trade Log (Session)")
            st.dataframe(pd.DataFrame([t.__dict__ for t in engine.trade_log[-20:]]), use_container_width=True)
            if st.button("Save Trade Log"):
                engine.save_trade_log()
                st.success(f"Saved to {TRADE_LOG_PATH}")

    # ── Portfolio ──
    with tab2:
        if st.button("Run All Symbols"):
            with st.spinner("Running portfolio..."):
                results = engine.run_portfolio(period=settings.lookback, interval=settings.timeframe)
            for sym, res in results.items():
                if "error" in res:
                    st.error(f"{sym}: {res['error']}")
                    continue
                with st.expander(f"{sym} — {res['action'].upper()} @ ${res['price']:.2f}"):
                    st.write(f"**Reason:** {res['reason']}")
                    st.write(f"**Pred:** {res['predicted_return']*100:.3f}%")
                    st.write(f"**Equity:** ${res['equity']:,.2f}")

    # ── Backtest ──
    with tab3:
        st.subheader("Walk-Forward Backtest")
        bt_cols = st.columns(3)
        bt_symbol = bt_cols[0].text_input("Symbol", value=symbol)
        bt_train = bt_cols[1].selectbox("Train", ["6mo", "1y", "2y"], index=1)
        bt_test = bt_cols[2].selectbox("Test", ["3mo", "6mo", "1y"], index=1)
        bt_commission = st.slider("Commission ($)", 0.0, 10.0, 1.0, 0.5)
        if st.button("Run Backtest"):
            with st.spinner("Running backtest..."):
                bt = BacktestEngine(symbol=bt_symbol, train_period=bt_train, test_period=bt_test,
                                    risk_config=risk_config, commission_per_trade=bt_commission)
                result = bt.run()
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Return", f"{result.total_return_pct:.2f}%")
            r2.metric("Max DD", f"{result.max_drawdown_pct:.2f}%")
            r3.metric("Sharpe", f"{result.sharpe_ratio:.2f}" if result.sharpe_ratio else "N/A")
            r4.metric("Trades", result.total_trades)
            r5, r6, r7 = st.columns(3)
            r5.metric("Win Rate", f"{result.win_rate:.0%}")
            r6.metric("Final Equity", f"${result.final_equity:,.2f}")
            r7.metric("Avg Trade", f"{result.avg_trade_return_pct:.4f}%")
            if not result.equity_curve.empty:
                st.plotly_chart(_equity_figure(result.equity_curve), use_container_width=True)
            if result.trades:
                st.dataframe(pd.DataFrame([{
                    "date": t.date, "symbol": t.symbol, "side": t.side,
                    "qty": t.qty, "price": t.price, "reason": t.reason, "equity": t.equity,
                } for t in result.trades]), use_container_width=True)

    # ── Model ──
    with tab4:
        c1, c2, c3 = st.columns(3)
        if c1.button("Train Model"):
            with st.spinner("Training..."):
                engine.train(symbol, period=settings.lookback, interval=settings.timeframe)
            c1.success("Trained!")
        if c2.button("Save Model"):
            engine.save_model()
            c2.success("Saved!")
        if c3.button("Load Model"):
            engine.load_model()
            c3.success("Loaded!")
        if engine.model.trained:
            st.subheader("Ensemble Weights")
            weights = engine.model.model_weights
            for name, w in weights.items():
                st.progress(min(w, 1.0), text=f"{name.upper()}: {w:.1%}")
            scores = engine.model._validation_scores
            if scores:
                st.subheader("Walk-Forward R²")
                for name, score in scores.items():
                    st.write(f"**{name.upper()}**: {score:.4f}")

    # ── Feature Importance ──
    with tab5:
        st.subheader("Feature Importance")
        if not engine.model.trained:
            st.warning("Train model first.")
        else:
            imp_type = st.radio("Method", ["GBR (fast)", "RF (fast)", "Permutation (slow)"], horizontal=True)
            if st.button("Compute Importance"):
                with st.spinner("Computing..."):
                    df_feat = engine.train(symbol, period=settings.lookback, interval=settings.timeframe) if not engine.model.trained else None
                    if imp_type == "GBR (fast)":
                        imp = gbr_feature_importance(engine.model)
                    elif imp_type == "RF (fast)":
                        imp = rf_feature_importance(engine.model)
                    else:
                        imp = permutation_importance(engine.model, df_feat)
                st.plotly_chart(_importance_figure(imp, f"Feature Importance ({imp_type.split(' ')[0]})"), use_container_width=True)
                st.dataframe(imp, use_container_width=True)

    # ── Multi-Timeframe ──
    with tab6:
        st.subheader("Multi-Timeframe Analysis")
        tf_symbol = st.text_input("MTF Symbol", value=symbol)
        if st.button("Analyze Timeframes"):
            with st.spinner("Analyzing..."):
                if not engine.model.trained:
                    engine.train(tf_symbol, period=settings.lookback, interval=settings.timeframe)
                mtf_result = analyze_multi_timeframe(tf_symbol, engine.model)
            st.write(f"**Aggregate Action:** {mtf_result.aggregate_action.upper()}")
            st.write(f"**Aggregate Score:** {mtf_result.aggregate_score:.4f}")
            st.write(f"**Dominant TF:** {mtf_result.dominant_timeframe}")
            for tf, sig in mtf_result.signals.items():
                with st.expander(f"{tf} — Score: {sig.combined_score:.4f}"):
                    st.write(f"Predicted Return: {sig.predicted_return*100:.3f}%")
                    st.write(f"Strategy Signal: {sig.strategy_signal} (conf: {sig.strategy_confidence:.2f})")
                    st.write(f"Volatility: {sig.volatility:.6f}")
                    if sig.model_predictions:
                        for mn, mv in sig.model_predictions.items():
                            st.write(f"  {mn.upper()}: {mv*100:.4f}%")

    # ── History ──
    with tab7:
        st.subheader("Equity History")
        eq_df = _load_equity_history()
        if not eq_df.empty:
            st.dataframe(eq_df.tail(50), use_container_width=True)
        else:
            st.info("No equity history yet.")
        st.subheader("Trade History")
        tr_df = _load_trade_log()
        if not tr_df.empty:
            st.dataframe(tr_df.tail(50), use_container_width=True)
        else:
            st.info("No trade history yet.")


if __name__ == "__main__":
    run_dashboard()
