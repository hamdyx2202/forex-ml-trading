#!/usr/bin/env python3
"""
Linux-Compatible Data Collector
يعمل مع Yahoo Finance على Linux ومع MT5 على Windows
"""

import sys
import platform

# Import Linux compatibility first if on Linux
if platform.system() == 'Linux':
    try:
        import src.linux_compatibility
        from src.linux_compatibility import mt5
    except:
        import linux_compatibility
        from linux_compatibility import mt5
else:
    import MetaTrader5 as mt5

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import sqlite3
from loguru import logger
from typing import List, Dict, Optional, Tuple
import time
from pathlib import Path
import os
from dotenv import load_dotenv

# For Linux - use yfinance for real data
if platform.system() == 'Linux':
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not installed - using mock data only")
        yf = None

load_dotenv()

class DataCollector:
    """جمع البيانات - متوافق مع Linux و Windows"""
    
    def __init__(self, config_path: str = "config/config.json"):
        self.config = self._load_config(config_path)
        self.db_path = self.config["database"]["path"]
        self.is_linux = platform.system() == 'Linux'
        self._ensure_database()
        logger.add("logs/data_collector.log", rotation="1 day", retention="30 days")
        
    def _load_config(self, config_path: str) -> dict:
        """تحميل ملف الإعدادات"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # تحديث بيانات MT5 من متغيرات البيئة (Windows فقط)
        if not self.is_linux:
            config["mt5"]["login"] = int(os.getenv("MT5_LOGIN", config["mt5"]["login"]))
            config["mt5"]["password"] = os.getenv("MT5_PASSWORD", config["mt5"]["password"])
            config["mt5"]["server"] = os.getenv("MT5_SERVER", config["mt5"]["server"])
            config["mt5"]["path"] = os.getenv("MT5_PATH", config["mt5"]["path"])
        
        return config
    
    def _ensure_database(self):
        """إنشاء قاعدة البيانات والجداول إذا لم تكن موجودة"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # جدول البيانات التاريخية
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS price_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                time INTEGER NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                spread INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timeframe, time)
            )
        """)
        
        # جدول حالة التحديث
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS update_status (
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                last_update INTEGER NOT NULL,
                status TEXT,
                PRIMARY KEY (symbol, timeframe)
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    
    def connect_mt5(self) -> bool:
        """الاتصال بـ MetaTrader 5 أو المحاكي"""
        if self.is_linux:
            logger.info("Running on Linux - using data simulator")
            return True
            
        try:
            # Initialize MT5 (Windows only)
            if not mt5.initialize(
                path=self.config["mt5"]["path"] if self.config["mt5"]["path"] else None,
                login=self.config["mt5"]["login"],
                password=self.config["mt5"]["password"],
                server=self.config["mt5"]["server"],
                timeout=self.config["mt5"]["timeout"]
            ):
                logger.error(f"Failed to initialize MT5: {mt5.last_error()}")
                return False
                
            logger.info(f"Successfully connected to MT5 - Account: {self.config['mt5']['login']}")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to MT5: {str(e)}")
            return False
    
    def disconnect_mt5(self):
        """قطع الاتصال بـ MetaTrader 5"""
        if not self.is_linux:
            mt5.shutdown()
            logger.info("Disconnected from MT5")
    
    def _convert_timeframe(self, mt5_timeframe: str) -> str:
        """تحويل MT5 timeframe إلى Yahoo Finance interval"""
        mapping = {
            'M1': '1m',
            'M5': '5m', 
            'M15': '15m',
            'M30': '30m',
            'H1': '1h',
            'H4': '4h',
            'D1': '1d',
            'W1': '1wk',
            'MN1': '1mo'
        }
        return mapping.get(mt5_timeframe, '1h')
    
    def _get_yfinance_symbol(self, symbol: str) -> str:
        """تحويل رمز Forex إلى Yahoo Finance"""
        # Remove 'm' suffix if exists
        symbol = symbol.rstrip('m')
        
        forex_mapping = {
            'EURUSD': 'EURUSD=X',
            'GBPUSD': 'GBPUSD=X',
            'USDJPY': 'USDJPY=X',
            'AUDUSD': 'AUDUSD=X',
            'USDCAD': 'USDCAD=X',
            'USDCHF': 'USDCHF=X',
            'NZDUSD': 'NZDUSD=X',
            'XAUUSD': 'GC=F',  # Gold futures
            'XAGUSD': 'SI=F',  # Silver futures
        }
        return forex_mapping.get(symbol, symbol + '=X')
    
    def get_historical_data_linux(self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """جلب البيانات من Yahoo Finance للـ Linux"""
        if not yf:
            logger.warning("yfinance not available - returning empty dataframe")
            return pd.DataFrame()
            
        try:
            yf_symbol = self._get_yfinance_symbol(symbol)
            interval = self._convert_timeframe(timeframe)
            
            logger.info(f"Fetching {yf_symbol} data from Yahoo Finance...")
            ticker = yf.Ticker(yf_symbol)
            
            # جلب البيانات
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval
            )
            
            if df.empty:
                logger.warning(f"No data received for {symbol}")
                return pd.DataFrame()
            
            # تحويل إلى تنسيق MT5
            df['time'] = df.index.astype('int64') // 10**9  # Convert to Unix timestamp
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            df['spread'] = 2  # Default spread
            df['symbol'] = symbol
            df['timeframe'] = timeframe
            
            return df[['time', 'open', 'high', 'low', 'close', 'volume', 'spread', 'symbol', 'timeframe']]
            
        except Exception as e:
            logger.error(f"Error fetching data from Yahoo Finance: {e}")
            return pd.DataFrame()
    
    def get_historical_data(self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """جلب البيانات التاريخية"""
        if self.is_linux:
            return self.get_historical_data_linux(symbol, timeframe, start_date, end_date)
            
        # Windows - use MT5
        try:
            timeframe_map = {
                'M1': mt5.TIMEFRAME_M1,
                'M5': mt5.TIMEFRAME_M5,
                'M15': mt5.TIMEFRAME_M15,
                'M30': mt5.TIMEFRAME_M30,
                'H1': mt5.TIMEFRAME_H1,
                'H4': mt5.TIMEFRAME_H4,
                'D1': mt5.TIMEFRAME_D1,
                'W1': mt5.TIMEFRAME_W1,
                'MN1': mt5.TIMEFRAME_MN1
            }
            
            tf = timeframe_map.get(timeframe, mt5.TIMEFRAME_H1)
            
            # Select symbol
            if not mt5.symbol_select(symbol, True):
                logger.error(f"Failed to select symbol {symbol}")
                return pd.DataFrame()
            
            # Get rates
            rates = mt5.copy_rates_range(symbol, tf, start_date, end_date)
            
            if rates is None or len(rates) == 0:
                logger.warning(f"No data received for {symbol} {timeframe}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['symbol'] = symbol
            df['timeframe'] = timeframe
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting historical data: {str(e)}")
            return pd.DataFrame()
    
    def save_to_database(self, df: pd.DataFrame) -> int:
        """حفظ البيانات في قاعدة البيانات"""
        if df.empty:
            return 0
            
        conn = sqlite3.connect(self.db_path)
        
        try:
            # حفظ البيانات
            saved_count = 0
            for _, row in df.iterrows():
                try:
                    conn.execute("""
                        INSERT OR REPLACE INTO price_data 
                        (symbol, timeframe, time, open, high, low, close, volume, spread)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        row['symbol'], row['timeframe'], int(row['time']),
                        row['open'], row['high'], row['low'], row['close'],
                        int(row['volume']), int(row.get('spread', 0))
                    ))
                    saved_count += 1
                except Exception as e:
                    logger.debug(f"Error saving row: {e}")
                    
            conn.commit()
            logger.info(f"Saved {saved_count} records to database")
            return saved_count
            
        except Exception as e:
            logger.error(f"Database error: {str(e)}")
            return 0
        finally:
            conn.close()
    
    def update_symbol_data(self, symbol: str, timeframe: str, days: int = 30) -> bool:
        """تحديث بيانات رمز معين"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            logger.info(f"Updating {symbol} {timeframe} for last {days} days")
            
            # جلب البيانات
            df = self.get_historical_data(symbol, timeframe, start_date, end_date)
            
            if df.empty:
                logger.warning(f"No data received for {symbol} {timeframe}")
                return False
            
            # حفظ في قاعدة البيانات
            saved = self.save_to_database(df)
            
            # تحديث حالة التحديث
            self._update_status(symbol, timeframe, end_date)
            
            return saved > 0
            
        except Exception as e:
            logger.error(f"Error updating {symbol} {timeframe}: {str(e)}")
            return False
    
    def _update_status(self, symbol: str, timeframe: str, last_update: datetime):
        """تحديث حالة التحديث"""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                INSERT OR REPLACE INTO update_status (symbol, timeframe, last_update, status)
                VALUES (?, ?, ?, ?)
            """, (symbol, timeframe, int(last_update.timestamp()), 'success'))
            conn.commit()
        finally:
            conn.close()
    
    def update_all_pairs(self) -> Dict[str, bool]:
        """تحديث جميع الأزواج"""
        results = {}
        
        if not self.connect_mt5():
            logger.error("Failed to connect to data source")
            return results
        
        try:
            for symbol in self.config["trading"]["pairs"]:
                for timeframe in self.config["trading"]["timeframes"]:
                    key = f"{symbol}_{timeframe}"
                    results[key] = self.update_symbol_data(
                        symbol, 
                        timeframe, 
                        days=self.config["data"]["history_days"]
                    )
                    time.sleep(1)  # Rate limiting
                    
        finally:
            self.disconnect_mt5()
            
        # Summary
        success = sum(1 for v in results.values() if v)
        logger.info(f"Update complete: {success}/{len(results)} successful")
        
        return results
    
    def get_latest_data(self, symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
        """جلب أحدث البيانات من قاعدة البيانات"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            query = """
                SELECT * FROM price_data 
                WHERE symbol = ? AND timeframe = ?
                ORDER BY time DESC
                LIMIT ?
            """
            
            df = pd.read_sql_query(query, conn, params=(symbol, timeframe, limit))
            
            if not df.empty:
                df = df.sort_values('time')
                df['datetime'] = pd.to_datetime(df['time'], unit='s')
                
            return df
            
        except Exception as e:
            logger.error(f"Error reading from database: {str(e)}")
            return pd.DataFrame()
        finally:
            conn.close()

# للتوافق مع الكود القديم
MT5DataCollector = DataCollector