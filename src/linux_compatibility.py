#!/usr/bin/env python3
"""
Linux Compatibility Module
يوفر بدائل وهمية للمكتبات التي لا تعمل على Linux
"""

import sys
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

class MT5Mock:
    """Mock لـ MetaTrader5 للعمل على Linux"""
    
    # Constants
    TIMEFRAME_M1 = 1
    TIMEFRAME_M5 = 5
    TIMEFRAME_M15 = 15
    TIMEFRAME_M30 = 30
    TIMEFRAME_H1 = 60
    TIMEFRAME_H4 = 240
    TIMEFRAME_D1 = 1440
    TIMEFRAME_W1 = 10080
    TIMEFRAME_MN1 = 43200
    
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    TRADE_ACTION_DEAL = 1
    ORDER_TIME_GTC = 0
    ORDER_FILLING_IOC = 1
    POSITION_TYPE_BUY = 0
    POSITION_TYPE_SELL = 1
    TRADE_RETCODE_DONE = 10009
    
    def __init__(self):
        self.connected = False
    
    def initialize(self, *args, **kwargs):
        self.connected = True
        return True
    
    def shutdown(self):
        self.connected = False
        return True
    
    def symbol_info_tick(self, symbol):
        """Mock tick data"""
        class Tick:
            def __init__(self):
                self.bid = 1.1000 + np.random.rand() * 0.01
                self.ask = self.bid + 0.0002
                self.time = int(datetime.now().timestamp())
        return Tick()
    
    def account_info(self):
        """Mock account info"""
        class AccountInfo:
            login = 12345
            server = "MockServer"
            balance = 10000.0
            equity = 10000.0
            margin = 0.0
            margin_free = 10000.0
            leverage = 100
            profit = 0.0
            currency = "USD"
        return AccountInfo()
    
    def positions_get(self, **kwargs):
        """Mock positions"""
        return []
    
    def copy_rates_range(self, symbol, timeframe, date_from, date_to):
        """Mock historical data"""
        # إنشاء بيانات وهمية للاختبار
        periods = int((date_to - date_from).total_seconds() / 3600)  # عدد الساعات
        times = pd.date_range(start=date_from, end=date_to, freq='H')[:periods]
        
        data = []
        base_price = 1.1000
        for i, t in enumerate(times):
            open_price = base_price + np.sin(i/10) * 0.01 + np.random.rand() * 0.001
            high = open_price + np.random.rand() * 0.0005
            low = open_price - np.random.rand() * 0.0005
            close = np.random.uniform(low, high)
            
            data.append({
                'time': int(t.timestamp()),
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'tick_volume': int(1000 + np.random.rand() * 500),
                'spread': 2,
                'real_volume': 0
            })
        
        return np.array(data, dtype=[
            ('time', 'i8'), ('open', 'f8'), ('high', 'f8'), 
            ('low', 'f8'), ('close', 'f8'), ('tick_volume', 'i8'),
            ('spread', 'i4'), ('real_volume', 'i8')
        ])
    
    def copy_rates_from_pos(self, *args):
        return None
    
    def last_error(self):
        return (0, "No error")
    
    def symbol_select(self, symbol, enable):
        return True
    
    def order_send(self, request):
        """Mock order send"""
        class Result:
            retcode = 10009  # TRADE_RETCODE_DONE
            order = 12345
            comment = "Mock order executed"
        return Result()


class TalibMock:
    """Mock لـ talib - نستخدم pandas-ta بدلاً منه"""
    
    @staticmethod
    def RSI(close, timeperiod=14):
        """حساب RSI باستخدام pandas"""
        delta = pd.Series(close).diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=timeperiod).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=timeperiod).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.values
    
    @staticmethod
    def MACD(close, fastperiod=12, slowperiod=26, signalperiod=9):
        """حساب MACD باستخدام pandas"""
        close_series = pd.Series(close)
        ema_fast = close_series.ewm(span=fastperiod).mean()
        ema_slow = close_series.ewm(span=slowperiod).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signalperiod).mean()
        hist = macd - signal
        return macd.values, signal.values, hist.values
    
    @staticmethod
    def BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2):
        """حساب Bollinger Bands باستخدام pandas"""
        close_series = pd.Series(close)
        sma = close_series.rolling(window=timeperiod).mean()
        std = close_series.rolling(window=timeperiod).std()
        upper = sma + (std * nbdevup)
        lower = sma - (std * nbdevdn)
        return upper.values, sma.values, lower.values
    
    @staticmethod
    def ATR(high, low, close, timeperiod=14):
        """حساب ATR باستخدام pandas"""
        high_s = pd.Series(high)
        low_s = pd.Series(low)
        close_s = pd.Series(close)
        
        tr1 = high_s - low_s
        tr2 = abs(high_s - close_s.shift())
        tr3 = abs(low_s - close_s.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=timeperiod).mean()
        return atr.values
    
    @staticmethod
    def SMA(close, timeperiod=20):
        """حساب SMA باستخدام pandas"""
        return pd.Series(close).rolling(window=timeperiod).mean().values
    
    @staticmethod
    def EMA(close, timeperiod=20):
        """حساب EMA باستخدام pandas"""
        return pd.Series(close).ewm(span=timeperiod).mean().values
    
    @staticmethod
    def STOCH(high, low, close, fastk_period=5, slowk_period=3, slowd_period=3):
        """حساب Stochastic باستخدام pandas"""
        high_s = pd.Series(high)
        low_s = pd.Series(low)
        close_s = pd.Series(close)
        
        lowest_low = low_s.rolling(window=fastk_period).min()
        highest_high = high_s.rolling(window=fastk_period).max()
        
        fastk = 100 * ((close_s - lowest_low) / (highest_high - lowest_low))
        slowk = fastk.rolling(window=slowk_period).mean()
        slowd = slowk.rolling(window=slowd_period).mean()
        
        return slowk.values, slowd.values


# تثبيت المحاكيات
if 'MetaTrader5' not in sys.modules:
    sys.modules['MetaTrader5'] = MT5Mock()
    
if 'talib' not in sys.modules:
    sys.modules['talib'] = TalibMock()

# للاستيراد المباشر
mt5 = sys.modules['MetaTrader5']
talib = sys.modules['talib']