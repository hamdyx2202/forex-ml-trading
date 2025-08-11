import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import json
import sqlite3
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
import talib
from src.data_collector import MT5DataCollector
from src.feature_engineer import FeatureEngineer


class AdvancedLearner:
    """نظام التعلم المتقدم من التجارب التاريخية"""
    
    def __init__(self, config_path: str = "config/config.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.db_path = self.config["database"]["path"]
        logger.add("logs/advanced_learner.log", rotation="1 day", retention="30 days")
        
        # قاعدة بيانات التجارب الافتراضية
        self._init_virtual_trades_db()
        
        # معايير التعلم
        self.learning_patterns = {
            'candlestick_patterns': {},
            'volume_patterns': {},
            'indicator_combinations': {},
            'time_patterns': {},
            'market_context': {}
        }
        
    def _init_virtual_trades_db(self):
        """إنشاء قاعدة بيانات التجارب الافتراضية"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS virtual_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                entry_time INTEGER NOT NULL,
                entry_price REAL NOT NULL,
                direction TEXT NOT NULL,
                exit_price REAL NOT NULL,
                exit_time INTEGER NOT NULL,
                pnl_pips REAL NOT NULL,
                success INTEGER NOT NULL,
                
                -- سياق السوق
                trend_strength REAL,
                volatility REAL,
                volume_ratio REAL,
                buyer_seller_ratio REAL,
                
                -- الشموع
                candle_pattern TEXT,
                candle_size_ratio REAL,
                wick_body_ratio REAL,
                
                -- المؤشرات
                rsi_value REAL,
                macd_signal TEXT,
                bb_position REAL,
                ma_alignment TEXT,
                stochastic_value REAL,
                adx_value REAL,
                
                -- السياق الزمني
                hour INTEGER,
                day_of_week INTEGER,
                trading_session TEXT,
                
                -- الأسباب
                entry_reasons TEXT,
                exit_reasons TEXT,
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def analyze_historical_opportunities(self, symbol: str, timeframe: str, 
                                       lookback_days: int = 365):
        """تحليل الفرص التاريخية وإنشاء تجارب افتراضية"""
        logger.info(f"Analyzing historical opportunities for {symbol} {timeframe}")
        
        # جمع البيانات التاريخية
        collector = MT5DataCollector()
        df = collector.get_latest_data(symbol, timeframe, limit=lookback_days * 24)
        
        if df.empty:
            logger.error("No historical data available")
            return
        
        # إضافة جميع المؤشرات والسياق
        df = self._add_advanced_features(df)
        
        # البحث عن نقاط دخول محتملة
        entry_points = self._find_entry_points(df)
        
        # محاكاة التجارب
        virtual_trades = []
        for entry in entry_points:
            trade = self._simulate_trade(df, entry)
            if trade:
                virtual_trades.append(trade)
        
        # حفظ التجارب الافتراضية
        self._save_virtual_trades(virtual_trades, symbol, timeframe)
        
        # التعلم من النتائج
        self._learn_from_trades(virtual_trades)
        
        logger.info(f"Analyzed {len(virtual_trades)} virtual trades")
    
    def _add_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """إضافة مؤشرات وميزات متقدمة"""
        
        # حجم الشموع ونسبها
        df['candle_size'] = df['high'] - df['low']
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['wick_body_ratio'] = (df['upper_wick'] + df['lower_wick']) / (df['body_size'] + 0.0001)
        
        # نسبة البائع والمشتري (تقديرية من الحجم والحركة)
        df['price_change'] = df['close'] - df['open']
        df['buyer_pressure'] = df['volume'] * (df['price_change'] > 0).astype(int)
        df['seller_pressure'] = df['volume'] * (df['price_change'] < 0).astype(int)
        df['buyer_seller_ratio'] = df['buyer_pressure'] / (df['seller_pressure'] + 1)
        
        # أنماط الشموع المتقدمة
        df['doji'] = (df['body_size'] < df['candle_size'] * 0.1).astype(int)
        df['hammer'] = ((df['lower_wick'] > df['body_size'] * 2) & 
                       (df['upper_wick'] < df['body_size'] * 0.3)).astype(int)
        df['shooting_star'] = ((df['upper_wick'] > df['body_size'] * 2) & 
                              (df['lower_wick'] < df['body_size'] * 0.3)).astype(int)
        df['engulfing'] = ((df['body_size'] > df['body_size'].shift(1) * 1.5) & 
                          (df['price_change'] * df['price_change'].shift(1) < 0)).astype(int)
        
        # مؤشرات متقدمة
        # RSI
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        
        # MACD
        macd, signal, hist = talib.MACD(df['close'])
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = hist
        df['macd_bullish'] = (df['macd'] > df['macd_signal']).astype(int)
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(df['close'])
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Moving Averages
        df['ma_20'] = talib.SMA(df['close'], 20)
        df['ma_50'] = talib.SMA(df['close'], 50)
        df['ma_200'] = talib.SMA(df['close'], 200)
        df['ma_alignment_bullish'] = ((df['ma_20'] > df['ma_50']) & 
                                      (df['ma_50'] > df['ma_200'])).astype(int)
        
        # ADX للترند
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'])
        df['strong_trend'] = (df['adx'] > 25).astype(int)
        
        # Stochastic
        slowk, slowd = talib.STOCH(df['high'], df['low'], df['close'])
        df['stoch_k'] = slowk
        df['stoch_d'] = slowd
        df['stoch_oversold'] = (df['stoch_k'] < 20).astype(int)
        df['stoch_overbought'] = (df['stoch_k'] > 80).astype(int)
        
        # Volume analysis
        df['volume_sma'] = talib.SMA(df['volume'], 20)
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['high_volume'] = (df['volume_ratio'] > 1.5).astype(int)
        
        # ATR للتذبذب
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'])
        df['volatility'] = df['atr'] / df['close']
        
        # Support/Resistance
        df['resistance'] = df['high'].rolling(20).max()
        df['support'] = df['low'].rolling(20).min()
        df['near_resistance'] = (abs(df['close'] - df['resistance']) < df['atr'] * 0.5).astype(int)
        df['near_support'] = (abs(df['close'] - df['support']) < df['atr'] * 0.5).astype(int)
        
        return df
    
    def _find_entry_points(self, df: pd.DataFrame) -> List[Dict]:
        """البحث عن نقاط دخول محتملة بناءً على معايير متعددة"""
        entry_points = []
        
        # تخطي الصفوف الأولى للمؤشرات
        df = df.iloc[200:].copy()
        
        for idx, row in df.iterrows():
            # معايير الشراء
            buy_score = 0
            buy_reasons = []
            
            # 1. RSI oversold + momentum
            if row['rsi_oversold'] and row['rsi'] > df.loc[idx-1, 'rsi']:
                buy_score += 2
                buy_reasons.append("RSI oversold reversal")
            
            # 2. MACD bullish crossover
            if row['macd_bullish'] and not df.loc[idx-1, 'macd_bullish']:
                buy_score += 2
                buy_reasons.append("MACD bullish crossover")
            
            # 3. Bounce from support
            if row['near_support'] and row['close'] > row['open']:
                buy_score += 2
                buy_reasons.append("Bounce from support")
            
            # 4. Bullish candle patterns
            if row['hammer'] and row['near_support']:
                buy_score += 3
                buy_reasons.append("Hammer at support")
            
            if row['engulfing'] and row['close'] > row['open']:
                buy_score += 2
                buy_reasons.append("Bullish engulfing")
            
            # 5. MA alignment
            if row['ma_alignment_bullish']:
                buy_score += 1
                buy_reasons.append("Bullish MA alignment")
            
            # 6. Volume confirmation
            if row['high_volume'] and row['close'] > row['open']:
                buy_score += 1
                buy_reasons.append("High volume bullish")
            
            # 7. Strong trend
            if row['strong_trend'] and row['close'] > row['ma_20']:
                buy_score += 1
                buy_reasons.append("Strong uptrend")
            
            # معايير البيع
            sell_score = 0
            sell_reasons = []
            
            # 1. RSI overbought + momentum
            if row['rsi_overbought'] and row['rsi'] < df.loc[idx-1, 'rsi']:
                sell_score += 2
                sell_reasons.append("RSI overbought reversal")
            
            # 2. MACD bearish crossover
            if not row['macd_bullish'] and df.loc[idx-1, 'macd_bullish']:
                sell_score += 2
                sell_reasons.append("MACD bearish crossover")
            
            # 3. Rejection from resistance
            if row['near_resistance'] and row['close'] < row['open']:
                sell_score += 2
                sell_reasons.append("Rejection from resistance")
            
            # 4. Bearish candle patterns
            if row['shooting_star'] and row['near_resistance']:
                sell_score += 3
                sell_reasons.append("Shooting star at resistance")
            
            if row['engulfing'] and row['close'] < row['open']:
                sell_score += 2
                sell_reasons.append("Bearish engulfing")
            
            # قرار الدخول
            if buy_score >= 5:
                entry_points.append({
                    'index': idx,
                    'time': row['time'],
                    'price': row['close'],
                    'direction': 'BUY',
                    'score': buy_score,
                    'reasons': buy_reasons,
                    'context': self._get_market_context(df, idx)
                })
            elif sell_score >= 5:
                entry_points.append({
                    'index': idx,
                    'time': row['time'],
                    'price': row['close'],
                    'direction': 'SELL',
                    'score': sell_score,
                    'reasons': sell_reasons,
                    'context': self._get_market_context(df, idx)
                })
        
        return entry_points
    
    def _get_market_context(self, df: pd.DataFrame, idx: int) -> Dict:
        """الحصول على سياق السوق الكامل"""
        row = df.loc[idx]
        
        return {
            'trend_strength': float(row['adx']) if not pd.isna(row['adx']) else 0,
            'volatility': float(row['volatility']),
            'volume_ratio': float(row['volume_ratio']),
            'buyer_seller_ratio': float(row['buyer_seller_ratio']),
            'candle_pattern': self._identify_candle_pattern(row),
            'candle_size_ratio': float(row['candle_size'] / row['atr']),
            'wick_body_ratio': float(row['wick_body_ratio']),
            'rsi_value': float(row['rsi']),
            'macd_signal': 'bullish' if row['macd_bullish'] else 'bearish',
            'bb_position': float(row['bb_position']),
            'ma_alignment': 'bullish' if row['ma_alignment_bullish'] else 'bearish',
            'stochastic_value': float(row['stoch_k']),
            'hour': row['time'].hour if hasattr(row['time'], 'hour') else 0,
            'day_of_week': row['time'].weekday() if hasattr(row['time'], 'weekday') else 0,
            'trading_session': self._get_trading_session(row['time'])
        }
    
    def _identify_candle_pattern(self, row) -> str:
        """تحديد نمط الشمعة"""
        if row['doji']:
            return 'doji'
        elif row['hammer']:
            return 'hammer'
        elif row['shooting_star']:
            return 'shooting_star'
        elif row['engulfing']:
            return 'engulfing'
        elif row['body_size'] > row['candle_size'] * 0.7:
            return 'marubozu'
        else:
            return 'normal'
    
    def _get_trading_session(self, time) -> str:
        """تحديد جلسة التداول"""
        if hasattr(time, 'hour'):
            hour = time.hour
            if 0 <= hour < 8:
                return 'asian'
            elif 8 <= hour < 16:
                return 'london'
            elif 16 <= hour < 22:
                return 'newyork'
            else:
                return 'aftermarket'
        return 'unknown'
    
    def _simulate_trade(self, df: pd.DataFrame, entry: Dict) -> Optional[Dict]:
        """محاكاة صفقة من نقطة الدخول"""
        entry_idx = entry['index']
        entry_price = entry['price']
        direction = entry['direction']
        
        # حساب SL و TP
        atr = df.loc[entry_idx, 'atr']
        if direction == 'BUY':
            stop_loss = entry_price - (atr * 2)
            take_profit = entry_price + (atr * 3)
        else:
            stop_loss = entry_price + (atr * 2)
            take_profit = entry_price - (atr * 3)
        
        # البحث عن نقطة الخروج
        exit_idx = None
        exit_price = None
        exit_reason = None
        
        for i in range(entry_idx + 1, min(entry_idx + 100, len(df))):
            high = df.loc[i, 'high']
            low = df.loc[i, 'low']
            close = df.loc[i, 'close']
            
            if direction == 'BUY':
                # Check SL
                if low <= stop_loss:
                    exit_idx = i
                    exit_price = stop_loss
                    exit_reason = 'stop_loss'
                    break
                # Check TP
                elif high >= take_profit:
                    exit_idx = i
                    exit_price = take_profit
                    exit_reason = 'take_profit'
                    break
                # Check exit signals
                elif self._check_exit_signal(df, i, direction):
                    exit_idx = i
                    exit_price = close
                    exit_reason = 'signal_exit'
                    break
            else:  # SELL
                # Check SL
                if high >= stop_loss:
                    exit_idx = i
                    exit_price = stop_loss
                    exit_reason = 'stop_loss'
                    break
                # Check TP
                elif low <= take_profit:
                    exit_idx = i
                    exit_price = take_profit
                    exit_reason = 'take_profit'
                    break
                # Check exit signals
                elif self._check_exit_signal(df, i, direction):
                    exit_idx = i
                    exit_price = close
                    exit_reason = 'signal_exit'
                    break
        
        if exit_idx is None:
            return None
        
        # حساب النتيجة
        if direction == 'BUY':
            pnl_pips = (exit_price - entry_price) / 0.0001
        else:
            pnl_pips = (entry_price - exit_price) / 0.0001
        
        success = 1 if pnl_pips > 0 else 0
        
        # إنشاء سجل التجارة
        trade = {
            'entry_time': int(entry['time'].timestamp()) if hasattr(entry['time'], 'timestamp') else 0,
            'entry_price': entry_price,
            'direction': direction,
            'exit_time': int(df.loc[exit_idx, 'time'].timestamp()) if hasattr(df.loc[exit_idx, 'time'], 'timestamp') else 0,
            'exit_price': exit_price,
            'pnl_pips': pnl_pips,
            'success': success,
            'entry_reasons': json.dumps(entry['reasons']),
            'exit_reasons': exit_reason,
            **entry['context']
        }
        
        return trade
    
    def _check_exit_signal(self, df: pd.DataFrame, idx: int, direction: str) -> bool:
        """فحص إشارات الخروج"""
        row = df.loc[idx]
        
        if direction == 'BUY':
            # Exit conditions for long positions
            if row['rsi_overbought']:
                return True
            if row['shooting_star'] and row['near_resistance']:
                return True
            if not row['macd_bullish'] and df.loc[idx-1, 'macd_bullish']:
                return True
        else:  # SELL
            # Exit conditions for short positions
            if row['rsi_oversold']:
                return True
            if row['hammer'] and row['near_support']:
                return True
            if row['macd_bullish'] and not df.loc[idx-1, 'macd_bullish']:
                return True
        
        return False
    
    def _save_virtual_trades(self, trades: List[Dict], symbol: str, timeframe: str):
        """حفظ التجارب الافتراضية في قاعدة البيانات"""
        if not trades:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for trade in trades:
            trade['symbol'] = symbol
            trade['timeframe'] = timeframe
            
            columns = ', '.join(trade.keys())
            placeholders = ', '.join(['?' for _ in trade])
            
            cursor.execute(f"""
                INSERT INTO virtual_trades ({columns})
                VALUES ({placeholders})
            """, list(trade.values()))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Saved {len(trades)} virtual trades to database")
    
    def _learn_from_trades(self, trades: List[Dict]):
        """التعلم من نتائج التجارب"""
        if not trades:
            return
        
        successful_trades = [t for t in trades if t['success'] == 1]
        failed_trades = [t for t in trades if t['success'] == 0]
        
        logger.info(f"Learning from {len(successful_trades)} successful and {len(failed_trades)} failed trades")
        
        # تحليل الأنماط الناجحة
        for trade in successful_trades:
            # تحديث أنماط الشموع الناجحة
            candle_pattern = trade.get('candle_pattern', 'unknown')
            if candle_pattern not in self.learning_patterns['candlestick_patterns']:
                self.learning_patterns['candlestick_patterns'][candle_pattern] = {'success': 0, 'total': 0}
            self.learning_patterns['candlestick_patterns'][candle_pattern]['success'] += 1
            self.learning_patterns['candlestick_patterns'][candle_pattern]['total'] += 1
            
            # تحديث أنماط الحجم
            if trade.get('volume_ratio', 0) > 1.5:
                if 'high_volume' not in self.learning_patterns['volume_patterns']:
                    self.learning_patterns['volume_patterns']['high_volume'] = {'success': 0, 'total': 0}
                self.learning_patterns['volume_patterns']['high_volume']['success'] += 1
                self.learning_patterns['volume_patterns']['high_volume']['total'] += 1
        
        # حفظ التعلم
        self._save_learning_patterns()
    
    def _save_learning_patterns(self):
        """حفظ أنماط التعلم"""
        with open('data/learning_patterns.json', 'w') as f:
            json.dump(self.learning_patterns, f, indent=2)
        
        logger.info("Saved learning patterns")
    
    def find_high_quality_opportunities(self, symbol: str, timeframe: str) -> List[Dict]:
        """البحث عن فرص عالية الجودة بناءً على التعلم"""
        logger.info(f"Searching for high quality opportunities in {symbol} {timeframe}")
        
        # جمع البيانات الحالية
        collector = MT5DataCollector()
        df = collector.get_latest_data(symbol, timeframe, limit=500)
        
        if df.empty:
            return []
        
        # إضافة المؤشرات
        df = self._add_advanced_features(df)
        
        # الحصول على أفضل الأنماط من التعلم
        best_patterns = self._get_best_patterns()
        
        # البحث عن الفرص
        opportunities = []
        
        # تحليل آخر 50 شمعة فقط
        for idx in range(len(df) - 50, len(df)):
            row = df.loc[idx]
            opportunity_score = 0
            reasons = []
            
            # فحص الأنماط الناجحة
            current_pattern = self._identify_candle_pattern(row)
            if current_pattern in best_patterns['candlestick']:
                opportunity_score += 3
                reasons.append(f"Successful pattern: {current_pattern}")
            
            # فحص الحجم
            if row['volume_ratio'] > 1.5 and 'high_volume' in best_patterns.get('volume', {}):
                opportunity_score += 2
                reasons.append("High volume pattern")
            
            # فحص المؤشرات المتعددة
            indicators_aligned = 0
            if row['rsi_oversold']:
                indicators_aligned += 1
            if row['macd_bullish']:
                indicators_aligned += 1
            if row['near_support']:
                indicators_aligned += 1
            if row['stoch_oversold']:
                indicators_aligned += 1
            
            if indicators_aligned >= 3:
                opportunity_score += 4
                reasons.append(f"{indicators_aligned} indicators aligned")
            
            # فحص السياق الزمني
            if self._is_optimal_time(row):
                opportunity_score += 1
                reasons.append("Optimal trading time")
            
            # إضافة الفرصة إذا كانت جيدة
            if opportunity_score >= 6:
                opportunities.append({
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'time': row['time'],
                    'price': row['close'],
                    'score': opportunity_score,
                    'reasons': reasons,
                    'direction': self._determine_direction(df, idx),
                    'confidence': min(opportunity_score / 10, 0.9),
                    'suggested_sl': self._calculate_sl(df, idx),
                    'suggested_tp': self._calculate_tp(df, idx)
                })
        
        # ترتيب حسب الجودة
        opportunities.sort(key=lambda x: x['score'], reverse=True)
        
        return opportunities[:5]  # أفضل 5 فرص فقط
    
    def _get_best_patterns(self) -> Dict:
        """الحصول على أفضل الأنماط من التعلم"""
        try:
            with open('data/learning_patterns.json', 'r') as f:
                patterns = json.load(f)
        except:
            return {}
        
        best = {'candlestick': {}, 'volume': {}}
        
        # أفضل أنماط الشموع
        for pattern, stats in patterns.get('candlestick_patterns', {}).items():
            if stats['total'] > 10:  # على الأقل 10 تجارب
                success_rate = stats['success'] / stats['total']
                if success_rate > 0.6:  # نسبة نجاح > 60%
                    best['candlestick'][pattern] = success_rate
        
        # أفضل أنماط الحجم
        for pattern, stats in patterns.get('volume_patterns', {}).items():
            if stats['total'] > 10:
                success_rate = stats['success'] / stats['total']
                if success_rate > 0.6:
                    best['volume'][pattern] = success_rate
        
        return best
    
    def _is_optimal_time(self, row) -> bool:
        """فحص إذا كان الوقت مناسب للتداول"""
        if hasattr(row['time'], 'hour'):
            hour = row['time'].hour
            # أفضل أوقات التداول: جلسات لندن ونيويورك
            return 8 <= hour <= 20
        return True
    
    def _determine_direction(self, df: pd.DataFrame, idx: int) -> str:
        """تحديد اتجاه الصفقة"""
        row = df.loc[idx]
        
        buy_signals = 0
        sell_signals = 0
        
        if row['rsi_oversold']:
            buy_signals += 1
        if row['rsi_overbought']:
            sell_signals += 1
        
        if row['macd_bullish']:
            buy_signals += 1
        else:
            sell_signals += 1
        
        if row['close'] > row['ma_20']:
            buy_signals += 1
        else:
            sell_signals += 1
        
        return 'BUY' if buy_signals > sell_signals else 'SELL'
    
    def _calculate_sl(self, df: pd.DataFrame, idx: int) -> float:
        """حساب Stop Loss المقترح"""
        row = df.loc[idx]
        atr = row['atr']
        
        if self._determine_direction(df, idx) == 'BUY':
            return row['close'] - (atr * 2)
        else:
            return row['close'] + (atr * 2)
    
    def _calculate_tp(self, df: pd.DataFrame, idx: int) -> float:
        """حساب Take Profit المقترح"""
        row = df.loc[idx]
        atr = row['atr']
        
        if self._determine_direction(df, idx) == 'BUY':
            return row['close'] + (atr * 3)
        else:
            return row['close'] - (atr * 3)
    
    def get_learning_report(self) -> Dict:
        """تقرير عن التعلم والأداء"""
        conn = sqlite3.connect(self.db_path)
        
        # إحصائيات عامة
        query = """
            SELECT 
                COUNT(*) as total_trades,
                SUM(success) as successful_trades,
                AVG(pnl_pips) as avg_pips,
                MAX(pnl_pips) as best_trade,
                MIN(pnl_pips) as worst_trade
            FROM virtual_trades
        """
        
        stats = pd.read_sql_query(query, conn).iloc[0].to_dict()
        
        # أفضل الأنماط
        query = """
            SELECT 
                candle_pattern,
                COUNT(*) as count,
                SUM(success) as wins,
                AVG(pnl_pips) as avg_pips
            FROM virtual_trades
            GROUP BY candle_pattern
            HAVING count > 5
            ORDER BY (CAST(wins AS FLOAT) / count) DESC
            LIMIT 5
        """
        
        best_patterns = pd.read_sql_query(query, conn).to_dict('records')
        
        # أفضل الأوقات
        query = """
            SELECT 
                hour,
                COUNT(*) as count,
                SUM(success) as wins,
                AVG(pnl_pips) as avg_pips
            FROM virtual_trades
            GROUP BY hour
            HAVING count > 10
            ORDER BY (CAST(wins AS FLOAT) / count) DESC
            LIMIT 5
        """
        
        best_hours = pd.read_sql_query(query, conn).to_dict('records')
        
        conn.close()
        
        report = {
            'general_stats': stats,
            'best_patterns': best_patterns,
            'best_hours': best_hours,
            'success_rate': stats['successful_trades'] / stats['total_trades'] if stats['total_trades'] > 0 else 0
        }
        
        return report