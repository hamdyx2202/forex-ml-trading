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

load_dotenv()

class MT5DataCollector:
    """جمع البيانات من MetaTrader 5"""
    
    def __init__(self, config_path: str = "config/config.json"):
        self.config = self._load_config(config_path)
        self.db_path = self.config["database"]["path"]
        self._ensure_database()
        logger.add("logs/data_collector.log", rotation="1 day", retention="30 days")
        
    def _load_config(self, config_path: str) -> dict:
        """تحميل ملف الإعدادات"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # تحديث بيانات MT5 من متغيرات البيئة
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
                spread INTEGER,
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
        """الاتصال بـ MetaTrader 5"""
        try:
            # Initialize MT5
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
        mt5.shutdown()
        logger.info("Disconnected from MT5")
    
    def get_timeframe_value(self, timeframe: str) -> int:
        """تحويل الإطار الزمني إلى قيمة MT5"""
        timeframe_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
            "W1": mt5.TIMEFRAME_W1,
            "MN1": mt5.TIMEFRAME_MN1
        }
        return timeframe_map.get(timeframe, mt5.TIMEFRAME_H1)
    
    def fetch_historical_data(self, symbol: str, timeframe: str, days: int = None) -> pd.DataFrame:
        """جلب البيانات التاريخية من MT5"""
        if days is None:
            days = self.config["data"]["history_days"]
            
        timeframe_val = self.get_timeframe_value(timeframe)
        
        # Calculate date range
        utc_to = datetime.now()
        utc_from = utc_to - timedelta(days=days)
        
        try:
            # Get rates
            rates = mt5.copy_rates_range(symbol, timeframe_val, utc_from, utc_to)
            
            if rates is None:
                logger.error(f"Failed to get rates for {symbol} {timeframe}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df['symbol'] = symbol
            df['timeframe'] = timeframe
            
            logger.info(f"Fetched {len(df)} bars for {symbol} {timeframe}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol} {timeframe}: {str(e)}")
            return pd.DataFrame()
    
    def save_to_database(self, df: pd.DataFrame):
        """حفظ البيانات في قاعدة البيانات"""
        if df.empty:
            return
            
        conn = sqlite3.connect(self.db_path)
        
        # Prepare data for insertion
        df_to_save = df[['symbol', 'timeframe', 'time', 'open', 'high', 'low', 'close', 'volume', 'spread']].copy()
        df_to_save['time'] = df_to_save['time'].astype('int64') // 10**9  # Convert to Unix timestamp
        
        # Save to database with replace to handle duplicates
        df_to_save.to_sql('price_data', conn, if_exists='append', index=False, 
                          method='multi', chunksize=1000)
        
        # Update status
        last_time = df_to_save['time'].max()
        symbol = df_to_save['symbol'].iloc[0]
        timeframe = df_to_save['timeframe'].iloc[0]
        
        conn.execute("""
            INSERT OR REPLACE INTO update_status (symbol, timeframe, last_update, status)
            VALUES (?, ?, ?, ?)
        """, (symbol, timeframe, last_time, 'success'))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Saved {len(df_to_save)} records for {symbol} {timeframe}")
    
    def update_all_pairs(self):
        """تحديث جميع الأزواج والإطارات الزمنية"""
        if not self.connect_mt5():
            logger.error("Failed to connect to MT5")
            return
        
        pairs = self.config["trading"]["pairs"]
        timeframes = self.config["trading"]["timeframes"]
        
        for symbol in pairs:
            for timeframe in timeframes:
                logger.info(f"Updating {symbol} {timeframe}")
                
                # Get last update time
                last_update = self._get_last_update(symbol, timeframe)
                
                if last_update:
                    # Calculate days since last update
                    days_since = (datetime.now().timestamp() - last_update) / 86400
                    days_to_fetch = int(days_since) + 1
                else:
                    # First time, fetch all history
                    days_to_fetch = self.config["data"]["history_days"]
                
                # Fetch and save data
                df = self.fetch_historical_data(symbol, timeframe, days_to_fetch)
                if not df.empty:
                    self.save_to_database(df)
                
                # Small delay to avoid overwhelming the server
                time.sleep(0.5)
        
        self.disconnect_mt5()
        logger.info("Update completed for all pairs")
    
    def _get_last_update(self, symbol: str, timeframe: str) -> Optional[int]:
        """الحصول على آخر وقت تحديث من قاعدة البيانات"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        result = cursor.execute("""
            SELECT last_update FROM update_status 
            WHERE symbol = ? AND timeframe = ?
        """, (symbol, timeframe)).fetchone()
        
        conn.close()
        
        return result[0] if result else None
    
    def get_latest_data(self, symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
        """الحصول على أحدث البيانات من قاعدة البيانات"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT * FROM price_data 
            WHERE symbol = ? AND timeframe = ?
            ORDER BY time DESC
            LIMIT ?
        """
        
        df = pd.read_sql_query(query, conn, params=(symbol, timeframe, limit))
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df.sort_values('time')
        
        conn.close()
        
        return df
    
    def run_continuous_update(self, interval_seconds: int = None):
        """تشغيل التحديث المستمر"""
        if interval_seconds is None:
            interval_seconds = self.config["data"]["update_interval"]
            
        logger.info(f"Starting continuous update with interval: {interval_seconds} seconds")
        
        while True:
            try:
                self.update_all_pairs()
                logger.info(f"Next update in {interval_seconds} seconds...")
                time.sleep(interval_seconds)
                
            except KeyboardInterrupt:
                logger.info("Stopping continuous update...")
                break
            except Exception as e:
                logger.error(f"Error in continuous update: {str(e)}")
                time.sleep(60)  # Wait 1 minute on error


if __name__ == "__main__":
    # مثال على الاستخدام
    collector = MT5DataCollector()
    
    # تحديث جميع البيانات مرة واحدة
    collector.update_all_pairs()
    
    # أو تشغيل التحديث المستمر
    # collector.run_continuous_update()