#!/usr/bin/env python3
"""
Performance Tracker for ML Trading System
تتبع أداء نظام التداول
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import sqlite3
from loguru import logger

class PerformanceTracker:
    """تتبع وتحليل أداء النماذج والصفقات"""
    
    def __init__(self, db_path="trading_performance.db", lookback_days=30):
        self.db_path = db_path
        self.lookback_days = lookback_days
        self.init_database()
        
    def init_database(self):
        """إنشاء جداول قاعدة البيانات"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # جدول أداء النماذج
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    win_rate REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    avg_profit REAL,
                    avg_loss REAL,
                    total_trades INTEGER,
                    avg_sl_pips REAL,
                    avg_tp_pips REAL,
                    avg_risk_reward REAL
                )
            """)
            
            # جدول الصفقات
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    entry_time DATETIME,
                    exit_time DATETIME,
                    entry_price REAL,
                    exit_price REAL,
                    sl_price REAL,
                    tp_price REAL,
                    result TEXT,
                    pnl_pips REAL,
                    pnl_amount REAL,
                    confidence REAL
                )
            """)
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            
    def record_trade(self, trade_data):
        """تسجيل صفقة جديدة"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO trades (
                    symbol, timeframe, signal, entry_time, exit_time,
                    entry_price, exit_price, sl_price, tp_price,
                    result, pnl_pips, pnl_amount, confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_data['symbol'],
                trade_data['timeframe'],
                trade_data['signal'],
                trade_data.get('entry_time'),
                trade_data.get('exit_time'),
                trade_data.get('entry_price'),
                trade_data.get('exit_price'),
                trade_data.get('sl_price'),
                trade_data.get('tp_price'),
                trade_data.get('result'),
                trade_data.get('pnl_pips'),
                trade_data.get('pnl_amount'),
                trade_data.get('confidence')
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error recording trade: {str(e)}")
            
    def update_model_performance(self, symbol, timeframe, metrics):
        """تحديث أداء النموذج"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO model_performance (
                    symbol, timeframe, win_rate, sharpe_ratio, max_drawdown,
                    avg_profit, avg_loss, total_trades, avg_sl_pips, 
                    avg_tp_pips, avg_risk_reward
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol,
                timeframe,
                metrics.get('win_rate', 0),
                metrics.get('sharpe_ratio', 0),
                metrics.get('max_drawdown', 0),
                metrics.get('avg_profit', 0),
                metrics.get('avg_loss', 0),
                metrics.get('total_trades', 0),
                metrics.get('avg_sl_pips', 0),
                metrics.get('avg_tp_pips', 0),
                metrics.get('avg_risk_reward', 0)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error updating model performance: {str(e)}")
            
    def get_pair_performance(self, symbol, timeframe, days=None):
        """الحصول على أداء زوج معين"""
        if days is None:
            days = self.lookback_days
            
        try:
            conn = sqlite3.connect(self.db_path)
            
            # آخر أداء مسجل
            query = """
                SELECT * FROM model_performance 
                WHERE symbol = ? AND timeframe = ?
                AND timestamp > datetime('now', '-{} days')
                ORDER BY timestamp DESC
                LIMIT 1
            """.format(days)
            
            df = pd.read_sql_query(query, conn, params=(symbol, timeframe))
            conn.close()
            
            if len(df) > 0:
                return df.iloc[0].to_dict()
            
            # إذا لم يوجد، احسب من الصفقات
            return self.calculate_performance_from_trades(symbol, timeframe, days)
            
        except Exception as e:
            logger.error(f"Error getting pair performance: {str(e)}")
            return None
            
    def calculate_performance_from_trades(self, symbol, timeframe, days):
        """حساب الأداء من سجل الصفقات"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = """
                SELECT * FROM trades 
                WHERE symbol = ? AND timeframe = ?
                AND entry_time > datetime('now', '-{} days')
            """.format(days)
            
            df = pd.read_sql_query(query, conn, params=(symbol, timeframe))
            conn.close()
            
            if len(df) == 0:
                return None
                
            # حساب المقاييس
            wins = df[df['result'] == 'win']
            losses = df[df['result'] == 'loss']
            
            win_rate = len(wins) / len(df) if len(df) > 0 else 0
            
            # Sharpe Ratio
            returns = df['pnl_pips'].values
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
            
            # Max Drawdown
            cumulative_pnl = df['pnl_pips'].cumsum()
            running_max = cumulative_pnl.cummax()
            drawdown = running_max - cumulative_pnl
            max_drawdown = drawdown.max()
            
            # متوسطات
            avg_profit = wins['pnl_pips'].mean() if len(wins) > 0 else 0
            avg_loss = abs(losses['pnl_pips'].mean()) if len(losses) > 0 else 0
            
            # SL/TP
            avg_sl = df['sl_price'].mean() if 'sl_price' in df else 0
            avg_tp = df['tp_price'].mean() if 'tp_price' in df else 0
            
            return {
                'win_rate': win_rate,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'avg_profit': avg_profit,
                'avg_loss': avg_loss,
                'total_trades': len(df),
                'avg_sl_pips': avg_sl,
                'avg_tp_pips': avg_tp,
                'avg_risk_reward': avg_tp / avg_sl if avg_sl > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance: {str(e)}")
            return None
            
    def get_overall_performance(self, days=None):
        """الحصول على الأداء الإجمالي"""
        if days is None:
            days = self.lookback_days
            
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = """
                SELECT 
                    COUNT(DISTINCT symbol || '_' || timeframe) as total_pairs,
                    AVG(win_rate) as avg_win_rate,
                    AVG(sharpe_ratio) as avg_sharpe,
                    MAX(max_drawdown) as worst_drawdown,
                    SUM(total_trades) as total_trades
                FROM model_performance
                WHERE timestamp > datetime('now', '-{} days')
            """.format(days)
            
            result = conn.execute(query).fetchone()
            conn.close()
            
            return {
                'total_pairs': result[0] or 0,
                'avg_win_rate': result[1] or 0,
                'avg_sharpe': result[2] or 0,
                'worst_drawdown': result[3] or 0,
                'total_trades': result[4] or 0
            }
            
        except Exception as e:
            logger.error(f"Error getting overall performance: {str(e)}")
            return None
            
    def get_performance_report(self, days=None):
        """تقرير شامل للأداء"""
        if days is None:
            days = self.lookback_days
            
        try:
            conn = sqlite3.connect(self.db_path)
            
            # أفضل الأزواج
            best_query = """
                SELECT symbol, timeframe, win_rate, sharpe_ratio
                FROM model_performance
                WHERE timestamp > datetime('now', '-{} days')
                ORDER BY win_rate DESC
                LIMIT 10
            """.format(days)
            
            best_pairs = pd.read_sql_query(best_query, conn)
            
            # أسوأ الأزواج
            worst_query = """
                SELECT symbol, timeframe, win_rate, max_drawdown
                FROM model_performance
                WHERE timestamp > datetime('now', '-{} days')
                ORDER BY win_rate ASC
                LIMIT 10
            """.format(days)
            
            worst_pairs = pd.read_sql_query(worst_query, conn)
            
            conn.close()
            
            return {
                'best_performers': best_pairs.to_dict('records'),
                'worst_performers': worst_pairs.to_dict('records'),
                'overall': self.get_overall_performance(days)
            }
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            return None