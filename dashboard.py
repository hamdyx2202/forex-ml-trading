import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import sqlite3
import numpy as np
from src.trader import Trader
from src.predictor import Predictor
from src.risk_manager import RiskManager
import os
from dotenv import load_dotenv

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Forex ML Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .success {
        color: #00cc44;
    }
    .danger {
        color: #ff4444;
    }
</style>
""", unsafe_allow_html=True)


class TradingDashboard:
    def __init__(self):
        self.config = self._load_config()
        self.db_path = self.config["database"]["path"]
        
    def _load_config(self):
        with open("config/config.json", 'r') as f:
            return json.load(f)
    
    def get_account_summary(self):
        """Get account summary data"""
        # This would connect to MT5 in real implementation
        # For demo, return mock data
        return {
            'balance': 10000,
            'equity': 10500,
            'margin': 1000,
            'free_margin': 9500,
            'profit': 500,
            'margin_level': 1050
        }
    
    def get_open_positions(self):
        """Get open positions"""
        # Mock data for demonstration
        positions = pd.DataFrame({
            'Symbol': ['EURUSD', 'GBPUSD', 'XAUUSD'],
            'Type': ['BUY', 'SELL', 'BUY'],
            'Volume': [0.1, 0.05, 0.02],
            'Open Price': [1.0950, 1.2650, 1850.50],
            'Current Price': [1.0965, 1.2635, 1852.30],
            'P&L': [15.00, 7.50, 3.60],
            'Open Time': [datetime.now() - timedelta(hours=i) for i in range(3)]
        })
        return positions
    
    def get_trade_history(self, days=30):
        """Get trade history from database"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT * FROM trade_history 
            WHERE close_time >= datetime('now', '-{} days')
            ORDER BY close_time DESC
        """.format(days)
        
        try:
            df = pd.read_sql_query(query, conn)
            if not df.empty:
                df['close_time'] = pd.to_datetime(df['close_time'])
                df['open_time'] = pd.to_datetime(df['open_time'])
        except:
            df = pd.DataFrame()
        
        conn.close()
        return df
    
    def get_predictions(self):
        """Get latest predictions"""
        # Mock data for demonstration
        predictions = pd.DataFrame({
            'Symbol': ['EURUSD', 'GBPUSD', 'XAUUSD'],
            'Timeframe': ['H1', 'H1', 'H4'],
            'Prediction': ['BUY', 'SELL', 'BUY'],
            'Confidence': [0.78, 0.82, 0.71],
            'Signal Strength': [75, 82, 68],
            'Time': [datetime.now() for _ in range(3)]
        })
        return predictions
    
    def plot_equity_curve(self, trades_df):
        """Plot equity curve"""
        if trades_df.empty:
            return None
        
        trades_df = trades_df.sort_values('close_time')
        trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trades_df['close_time'],
            y=trades_df['cumulative_pnl'],
            mode='lines',
            name='Equity',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title='Equity Curve',
            xaxis_title='Date',
            yaxis_title='Cumulative P&L ($)',
            height=400
        )
        
        return fig
    
    def plot_win_rate_chart(self, trades_df):
        """Plot win rate chart"""
        if trades_df.empty:
            return None
        
        wins = len(trades_df[trades_df['pnl'] > 0])
        losses = len(trades_df[trades_df['pnl'] < 0])
        
        fig = go.Figure(data=[
            go.Bar(name='Wins', x=['Trades'], y=[wins], marker_color='green'),
            go.Bar(name='Losses', x=['Trades'], y=[losses], marker_color='red')
        ])
        
        fig.update_layout(
            title='Win/Loss Distribution',
            barmode='stack',
            height=300
        )
        
        return fig
    
    def plot_profit_by_symbol(self, trades_df):
        """Plot profit by symbol"""
        if trades_df.empty:
            return None
        
        profit_by_symbol = trades_df.groupby('symbol')['pnl'].sum().reset_index()
        
        fig = px.bar(
            profit_by_symbol,
            x='symbol',
            y='pnl',
            color='pnl',
            color_continuous_scale=['red', 'yellow', 'green'],
            title='Profit by Symbol'
        )
        
        fig.update_layout(height=350)
        
        return fig
    
    def calculate_performance_metrics(self, trades_df):
        """Calculate performance metrics"""
        if trades_df.empty:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'average_win': 0,
                'average_loss': 0,
                'total_pnl': 0
            }
        
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]
        
        metrics = {
            'total_trades': len(trades_df),
            'win_rate': len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0,
            'profit_factor': abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0 else 0,
            'sharpe_ratio': self._calculate_sharpe_ratio(trades_df['pnl'].values),
            'max_drawdown': self._calculate_max_drawdown(trades_df),
            'average_win': winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0,
            'average_loss': losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0,
            'total_pnl': trades_df['pnl'].sum()
        }
        
        return metrics
    
    def _calculate_sharpe_ratio(self, returns):
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0
        
        sharpe_ratio = (mean_return / std_return) * np.sqrt(252)
        return sharpe_ratio
    
    def _calculate_max_drawdown(self, trades_df):
        """Calculate maximum drawdown"""
        if trades_df.empty:
            return 0
        
        trades_df = trades_df.sort_values('close_time')
        cumulative_pnl = trades_df['pnl'].cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = (cumulative_pnl - running_max) / running_max.abs()
        
        return drawdown.min()


def main():
    st.title("🤖 Forex ML Trading Dashboard")
    st.markdown("---")
    
    # Initialize dashboard
    dashboard = TradingDashboard()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Time period selector
        period = st.selectbox(
            "Time Period",
            ["Today", "Last 7 Days", "Last 30 Days", "All Time"]
        )
        
        # Auto refresh
        auto_refresh = st.checkbox("Auto Refresh (60s)", value=False)
        if auto_refresh:
            st.empty()
        
        st.markdown("---")
        
        # Trading controls
        st.header("🎮 Trading Controls")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Start Bot", type="primary", use_container_width=True):
                st.success("Bot started!")
        
        with col2:
            if st.button("⏹️ Stop Bot", type="secondary", use_container_width=True):
                st.warning("Bot stopped!")
        
        st.markdown("---")
        
        # System status
        st.header("🔧 System Status")
        st.success("✅ MT5 Connected")
        st.success("✅ Database Active")
        st.success("✅ Models Loaded")
    
    # Main content
    # Row 1: Account Summary
    st.header("💰 Account Summary")
    account = dashboard.get_account_summary()
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("Balance", f"${account['balance']:,.2f}")
    
    with col2:
        st.metric("Equity", f"${account['equity']:,.2f}")
    
    with col3:
        delta = account['profit']
        st.metric("Profit", f"${account['profit']:,.2f}", 
                 delta=f"{delta/account['balance']*100:.2f}%")
    
    with col4:
        st.metric("Margin", f"${account['margin']:,.2f}")
    
    with col5:
        st.metric("Free Margin", f"${account['free_margin']:,.2f}")
    
    with col6:
        st.metric("Margin Level", f"{account['margin_level']:.0f}%")
    
    st.markdown("---")
    
    # Row 2: Trading Performance
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 Trading Performance")
        
        # Get trade history
        days_map = {"Today": 1, "Last 7 Days": 7, "Last 30 Days": 30, "All Time": 365}
        days = days_map.get(period, 30)
        trades_df = dashboard.get_trade_history(days)
        
        # Equity curve
        equity_fig = dashboard.plot_equity_curve(trades_df)
        if equity_fig:
            st.plotly_chart(equity_fig, use_container_width=True)
        else:
            st.info("No trading data available")
    
    with col2:
        st.header("📈 Performance Metrics")
        
        metrics = dashboard.calculate_performance_metrics(trades_df)
        
        st.metric("Total Trades", metrics['total_trades'])
        st.metric("Win Rate", f"{metrics['win_rate']:.1%}")
        st.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
        st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
        st.metric("Max Drawdown", f"{metrics['max_drawdown']:.1%}")
        st.metric("Total P&L", f"${metrics['total_pnl']:,.2f}")
    
    st.markdown("---")
    
    # Row 3: Analysis Charts
    col1, col2 = st.columns(2)
    
    with col1:
        win_rate_fig = dashboard.plot_win_rate_chart(trades_df)
        if win_rate_fig:
            st.plotly_chart(win_rate_fig, use_container_width=True)
    
    with col2:
        profit_by_symbol_fig = dashboard.plot_profit_by_symbol(trades_df)
        if profit_by_symbol_fig:
            st.plotly_chart(profit_by_symbol_fig, use_container_width=True)
    
    st.markdown("---")
    
    # Row 4: Open Positions and Predictions
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📋 Open Positions")
        positions = dashboard.get_open_positions()
        
        if not positions.empty:
            # Style the dataframe
            def color_pnl(val):
                color = 'green' if val > 0 else 'red'
                return f'color: {color}'
            
            styled_positions = positions.style.applymap(
                color_pnl, subset=['P&L']
            ).format({
                'Volume': '{:.2f}',
                'Open Price': '{:.5f}',
                'Current Price': '{:.5f}',
                'P&L': '${:.2f}'
            })
            
            st.dataframe(styled_positions, use_container_width=True)
        else:
            st.info("No open positions")
    
    with col2:
        st.header("🔮 Latest Predictions")
        predictions = dashboard.get_predictions()
        
        if not predictions.empty:
            # Style predictions
            def color_prediction(val):
                if val == 'BUY':
                    return 'background-color: #90EE90'
                elif val == 'SELL':
                    return 'background-color: #FFB6C1'
                return ''
            
            styled_predictions = predictions.style.applymap(
                color_prediction, subset=['Prediction']
            ).format({
                'Confidence': '{:.1%}',
                'Signal Strength': '{:.0f}'
            })
            
            st.dataframe(styled_predictions, use_container_width=True)
        else:
            st.info("No predictions available")
    
    st.markdown("---")
    
    # Row 5: Recent Trades
    st.header("📜 Recent Trades")
    
    if not trades_df.empty:
        recent_trades = trades_df.head(10)[['symbol', 'direction', 'volume', 
                                           'open_price', 'close_price', 'pnl', 
                                           'close_time']]
        
        # Style the dataframe
        def color_pnl(val):
            color = 'green' if val > 0 else 'red'
            return f'color: {color}'
        
        styled_trades = recent_trades.style.applymap(
            color_pnl, subset=['pnl']
        ).format({
            'volume': '{:.2f}',
            'open_price': '{:.5f}',
            'close_price': '{:.5f}',
            'pnl': '${:.2f}',
            'close_time': lambda x: x.strftime('%Y-%m-%d %H:%M')
        })
        
        st.dataframe(styled_trades, use_container_width=True)
    else:
        st.info("No recent trades")
    
    # Auto refresh
    if auto_refresh:
        st.rerun()


if __name__ == "__main__":
    main()