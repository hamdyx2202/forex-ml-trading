#!/usr/bin/env python3
"""
🎯 Market Analysis Engine - محرك تحليل السوق المتقدم
📊 يحلل السياق الكامل للسوق قبل اتخاذ القرار
"""

import numpy as np
import pandas as pd
import talib
from datetime import datetime, timedelta
import sqlite3
import logging

logger = logging.getLogger(__name__)

class MarketAnalysisEngine:
    """محرك تحليل السوق الشامل"""
    
    def __init__(self, db_path='./data/forex_ml.db'):
        self.db_path = db_path
        self.major_levels = {}  # مستويات الدعم والمقاومة الرئيسية
        self.market_sessions = {
            'Sydney': (22, 6),
            'Tokyo': (0, 8),
            'London': (8, 16),
            'NewYork': (13, 21)
        }
        
    def analyze_complete_market_context(self, symbol, current_candles, timeframe='M15'):
        """تحليل شامل للسوق"""
        try:
            context = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'timeframe': timeframe,
                'score': 0,  # -100 to +100
                'signals': [],
                'warnings': [],
                'strength': 'NEUTRAL'
            }
            
            # 1. تحليل الترند على فريمات متعددة
            trend_analysis = self.analyze_multi_timeframe_trend(symbol, current_candles)
            context['trend'] = trend_analysis
            
            # 2. تحديد مستويات الدعم والمقاومة الحقيقية
            sr_levels = self.identify_support_resistance(symbol, current_candles)
            context['support_resistance'] = sr_levels
            
            # 3. تحليل حجم التداول والسيولة
            volume_analysis = self.analyze_volume_profile(current_candles)
            context['volume'] = volume_analysis
            
            # 4. تحليل جلسات التداول
            session_analysis = self.analyze_trading_sessions()
            context['session'] = session_analysis
            
            # 5. تحليل الزخم والقوة
            momentum_analysis = self.analyze_momentum(current_candles)
            context['momentum'] = momentum_analysis
            
            # 6. تحليل الأنماط السعرية
            pattern_analysis = self.detect_price_patterns(current_candles)
            context['patterns'] = pattern_analysis
            
            # 7. تحليل التقلبات
            volatility_analysis = self.analyze_volatility(current_candles)
            context['volatility'] = volatility_analysis
            
            # 8. حساب النقاط النهائية
            context['score'] = self.calculate_market_score(context)
            context['strength'] = self.determine_signal_strength(context['score'])
            
            return context
            
        except Exception as e:
            logger.error(f"Error in market analysis: {e}")
            return None
    
    def analyze_multi_timeframe_trend(self, symbol, current_candles):
        """تحليل الترند على فريمات متعددة"""
        trends = {}
        
        # تحليل M15 (الحالي)
        df_m15 = pd.DataFrame(current_candles)
        trends['M15'] = self._analyze_single_trend(df_m15)
        
        # جلب وتحليل H1
        df_h1 = self._get_higher_timeframe_data(symbol, 'H1')
        if df_h1 is not None:
            trends['H1'] = self._analyze_single_trend(df_h1)
        
        # جلب وتحليل H4
        df_h4 = self._get_higher_timeframe_data(symbol, 'H4')
        if df_h4 is not None:
            trends['H4'] = self._analyze_single_trend(df_h4)
        
        # جلب وتحليل D1
        df_d1 = self._get_higher_timeframe_data(symbol, 'D1')
        if df_d1 is not None:
            trends['D1'] = self._analyze_single_trend(df_d1)
        
        # تحديد الاتجاه العام
        trend_scores = {
            'bullish': 0,
            'bearish': 0,
            'neutral': 0
        }
        
        weights = {'M15': 1, 'H1': 2, 'H4': 3, 'D1': 4}
        
        for tf, trend in trends.items():
            if trend['direction'] == 'BULLISH':
                trend_scores['bullish'] += weights.get(tf, 1)
            elif trend['direction'] == 'BEARISH':
                trend_scores['bearish'] += weights.get(tf, 1)
            else:
                trend_scores['neutral'] += weights.get(tf, 1)
        
        # الاتجاه النهائي
        if trend_scores['bullish'] > trend_scores['bearish'] * 1.5:
            overall_trend = 'STRONG_BULLISH'
        elif trend_scores['bullish'] > trend_scores['bearish']:
            overall_trend = 'BULLISH'
        elif trend_scores['bearish'] > trend_scores['bullish'] * 1.5:
            overall_trend = 'STRONG_BEARISH'
        elif trend_scores['bearish'] > trend_scores['bullish']:
            overall_trend = 'BEARISH'
        else:
            overall_trend = 'NEUTRAL'
        
        return {
            'timeframes': trends,
            'overall': overall_trend,
            'scores': trend_scores,
            'alignment': self._check_trend_alignment(trends)
        }
    
    def _analyze_single_trend(self, df):
        """تحليل الترند لفريم واحد"""
        if len(df) < 50:
            return {'direction': 'NEUTRAL', 'strength': 0}
        
        # حساب المتوسطات
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['sma_200'] = df['close'].rolling(200).mean() if len(df) > 200 else df['sma_50']
        
        # حساب EMA
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['signal'] = df['macd'].ewm(span=9).mean()
        
        # ADX لقوة الترند
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        if len(close) > 14:
            try:
                adx = talib.ADX(high, low, close, timeperiod=14)
                current_adx = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0
            except:
                current_adx = 0
        else:
            current_adx = 0
        
        # تحديد الاتجاه
        latest = df.iloc[-1]
        
        trend_signals = 0
        
        # إشارات الترند
        if latest['close'] > latest['sma_20']:
            trend_signals += 1
        else:
            trend_signals -= 1
            
        if latest['sma_20'] > latest['sma_50']:
            trend_signals += 2
        else:
            trend_signals -= 2
            
        if latest['sma_50'] > latest['sma_200']:
            trend_signals += 3
        else:
            trend_signals -= 3
            
        if latest['macd'] > latest['signal']:
            trend_signals += 1
        else:
            trend_signals -= 1
        
        # تحديد الاتجاه والقوة
        if trend_signals >= 4:
            direction = 'BULLISH'
            strength = min(100, trend_signals * 10 + current_adx)
        elif trend_signals <= -4:
            direction = 'BEARISH'
            strength = min(100, abs(trend_signals) * 10 + current_adx)
        else:
            direction = 'NEUTRAL'
            strength = current_adx
        
        return {
            'direction': direction,
            'strength': strength,
            'adx': current_adx,
            'signals': trend_signals
        }
    
    def identify_support_resistance(self, symbol, candles):
        """تحديد مستويات الدعم والمقاومة الحقيقية"""
        df = pd.DataFrame(candles)
        levels = {
            'support': [],
            'resistance': [],
            'pivots': {}
        }
        
        # 1. حساب Pivot Points
        if len(df) > 0:
            last_high = df['high'].iloc[-1]
            last_low = df['low'].iloc[-1]
            last_close = df['close'].iloc[-1]
            
            pivot = (last_high + last_low + last_close) / 3
            
            levels['pivots'] = {
                'PP': pivot,
                'R1': 2 * pivot - last_low,
                'R2': pivot + (last_high - last_low),
                'R3': last_high + 2 * (pivot - last_low),
                'S1': 2 * pivot - last_high,
                'S2': pivot - (last_high - last_low),
                'S3': last_low - 2 * (last_high - pivot)
            }
        
        # 2. تحديد القمم والقيعان المحلية
        if len(df) > 20:
            # القمم
            highs = df['high'].rolling(window=10, center=True).max()
            peaks = df[df['high'] == highs]['high'].unique()
            
            # القيعان
            lows = df['low'].rolling(window=10, center=True).min()
            troughs = df[df['low'] == lows]['low'].unique()
            
            # فلترة المستويات القريبة
            current_price = df['close'].iloc[-1]
            
            # المقاومات (فوق السعر الحالي)
            for peak in peaks:
                if peak > current_price:
                    levels['resistance'].append({
                        'price': float(peak),
                        'strength': self._calculate_level_strength(df, peak, 'resistance'),
                        'touches': self._count_level_touches(df, peak)
                    })
            
            # الدعوم (تحت السعر الحالي)
            for trough in troughs:
                if trough < current_price:
                    levels['support'].append({
                        'price': float(trough),
                        'strength': self._calculate_level_strength(df, trough, 'support'),
                        'touches': self._count_level_touches(df, trough)
                    })
        
        # 3. ترتيب المستويات حسب القوة
        levels['resistance'] = sorted(levels['resistance'], 
                                    key=lambda x: x['strength'], 
                                    reverse=True)[:5]
        levels['support'] = sorted(levels['support'], 
                                 key=lambda x: x['strength'], 
                                 reverse=True)[:5]
        
        # 4. تحديد أقرب مستويات
        if levels['resistance']:
            levels['nearest_resistance'] = min(levels['resistance'], 
                                             key=lambda x: x['price'])
        if levels['support']:
            levels['nearest_support'] = max(levels['support'], 
                                          key=lambda x: x['price'])
        
        return levels
    
    def _calculate_level_strength(self, df, level, level_type):
        """حساب قوة المستوى"""
        touches = self._count_level_touches(df, level)
        recency = self._calculate_level_recency(df, level)
        volume_at_level = self._calculate_volume_at_level(df, level)
        
        strength = (touches * 30) + (recency * 40) + (volume_at_level * 30)
        return min(100, strength)
    
    def _count_level_touches(self, df, level, tolerance=0.0002):
        """عد عدد مرات ملامسة المستوى"""
        touches = 0
        
        for i in range(len(df)):
            high = df['high'].iloc[i]
            low = df['low'].iloc[i]
            
            # التحقق من الملامسة
            if abs(high - level) <= level * tolerance:
                touches += 1
            elif abs(low - level) <= level * tolerance:
                touches += 1
            elif low <= level <= high:
                touches += 1
        
        return touches
    
    def _calculate_level_recency(self, df, level, tolerance=0.0002):
        """حساب حداثة المستوى"""
        last_touch = 0
        
        for i in range(len(df)-1, -1, -1):
            high = df['high'].iloc[i]
            low = df['low'].iloc[i]
            
            if (abs(high - level) <= level * tolerance or 
                abs(low - level) <= level * tolerance or
                low <= level <= high):
                last_touch = len(df) - i
                break
        
        if last_touch == 0:
            return 0
        elif last_touch < 10:
            return 100
        elif last_touch < 50:
            return 70
        elif last_touch < 100:
            return 40
        else:
            return 20
    
    def _calculate_volume_at_level(self, df, level, tolerance=0.0002):
        """حساب الحجم عند المستوى"""
        if 'volume' not in df.columns:
            return 50  # قيمة افتراضية
        
        total_volume = 0
        level_volume = 0
        
        for i in range(len(df)):
            high = df['high'].iloc[i]
            low = df['low'].iloc[i]
            volume = df['volume'].iloc[i]
            
            total_volume += volume
            
            if (abs(high - level) <= level * tolerance or 
                abs(low - level) <= level * tolerance or
                low <= level <= high):
                level_volume += volume
        
        if total_volume > 0:
            return min(100, (level_volume / total_volume) * 1000)
        return 0
    
    def analyze_volume_profile(self, candles):
        """تحليل ملف الحجم"""
        df = pd.DataFrame(candles)
        
        if 'volume' not in df.columns:
            return {
                'average_volume': 0,
                'current_volume': 0,
                'volume_trend': 'NORMAL',
                'volume_signal': 'NEUTRAL'
            }
        
        # حساب متوسط الحجم
        avg_volume = df['volume'].rolling(20).mean().iloc[-1]
        current_volume = df['volume'].iloc[-1]
        
        # تحليل اتجاه الحجم
        volume_sma = df['volume'].rolling(10).mean()
        
        if current_volume > avg_volume * 2:
            volume_trend = 'VERY_HIGH'
        elif current_volume > avg_volume * 1.5:
            volume_trend = 'HIGH'
        elif current_volume < avg_volume * 0.5:
            volume_trend = 'VERY_LOW'
        elif current_volume < avg_volume * 0.7:
            volume_trend = 'LOW'
        else:
            volume_trend = 'NORMAL'
        
        # إشارة الحجم مع السعر
        price_change = df['close'].pct_change().iloc[-1]
        volume_change = df['volume'].pct_change().iloc[-1]
        
        if price_change > 0 and volume_change > 0:
            volume_signal = 'BULLISH_CONFIRMATION'
        elif price_change < 0 and volume_change > 0:
            volume_signal = 'BEARISH_CONFIRMATION'
        elif price_change > 0 and volume_change < 0:
            volume_signal = 'BULLISH_DIVERGENCE'
        elif price_change < 0 and volume_change < 0:
            volume_signal = 'BEARISH_DIVERGENCE'
        else:
            volume_signal = 'NEUTRAL'
        
        return {
            'average_volume': float(avg_volume),
            'current_volume': float(current_volume),
            'volume_ratio': float(current_volume / avg_volume) if avg_volume > 0 else 1,
            'volume_trend': volume_trend,
            'volume_signal': volume_signal
        }
    
    def analyze_trading_sessions(self):
        """تحليل جلسات التداول"""
        current_hour = datetime.now().hour
        active_sessions = []
        session_quality = 'LOW'
        
        for session, (start, end) in self.market_sessions.items():
            if start <= end:
                if start <= current_hour < end:
                    active_sessions.append(session)
            else:  # للجلسات التي تعبر منتصف الليل
                if current_hour >= start or current_hour < end:
                    active_sessions.append(session)
        
        # تحديد جودة الوقت للتداول
        if 'London' in active_sessions and 'NewYork' in active_sessions:
            session_quality = 'EXCELLENT'
        elif 'London' in active_sessions or 'NewYork' in active_sessions:
            session_quality = 'GOOD'
        elif 'Tokyo' in active_sessions:
            session_quality = 'MODERATE'
        else:
            session_quality = 'LOW'
        
        return {
            'current_hour': current_hour,
            'active_sessions': active_sessions,
            'session_quality': session_quality,
            'is_news_time': self._check_news_time(),
            'is_weekend': datetime.now().weekday() >= 5
        }
    
    def _check_news_time(self):
        """التحقق من أوقات الأخبار المهمة"""
        current_time = datetime.now()
        
        # أوقات الأخبار الرئيسية (UTC)
        news_times = [
            (8, 30),   # EU news
            (13, 30),  # US news
            (15, 0),   # US news
            (19, 0)    # FOMC
        ]
        
        for hour, minute in news_times:
            news_time = current_time.replace(hour=hour, minute=minute)
            time_diff = abs((current_time - news_time).total_seconds() / 60)
            
            if time_diff <= 30:  # 30 دقيقة قبل أو بعد الخبر
                return True
        
        return False
    
    def analyze_momentum(self, candles):
        """تحليل الزخم"""
        df = pd.DataFrame(candles)
        
        if len(df) < 50:
            return {'momentum': 'NEUTRAL', 'strength': 0}
        
        # حساب مؤشرات الزخم
        # RSI
        rsi = talib.RSI(df['close'].values, timeperiod=14)
        current_rsi = rsi[-1] if len(rsi) > 0 and not pd.isna(rsi[-1]) else 50
        
        # Stochastic
        slowk, slowd = talib.STOCH(df['high'].values, 
                                   df['low'].values, 
                                   df['close'].values,
                                   fastk_period=14,
                                   slowk_period=3,
                                   slowd_period=3)
        current_stoch = slowk[-1] if len(slowk) > 0 and not pd.isna(slowk[-1]) else 50
        
        # CCI
        cci = talib.CCI(df['high'].values,
                       df['low'].values,
                       df['close'].values,
                       timeperiod=20)
        current_cci = cci[-1] if len(cci) > 0 and not pd.isna(cci[-1]) else 0
        
        # تحليل الزخم
        momentum_score = 0
        
        # RSI
        if current_rsi > 70:
            momentum_score -= 2
        elif current_rsi > 60:
            momentum_score += 1
        elif current_rsi < 30:
            momentum_score += 2
        elif current_rsi < 40:
            momentum_score -= 1
        
        # Stochastic
        if current_stoch > 80:
            momentum_score -= 2
        elif current_stoch > 60:
            momentum_score += 1
        elif current_stoch < 20:
            momentum_score += 2
        elif current_stoch < 40:
            momentum_score -= 1
        
        # CCI
        if current_cci > 100:
            momentum_score += 2
        elif current_cci > 0:
            momentum_score += 1
        elif current_cci < -100:
            momentum_score -= 2
        elif current_cci < 0:
            momentum_score -= 1
        
        # تحديد حالة الزخم
        if momentum_score >= 3:
            momentum = 'STRONG_BULLISH'
        elif momentum_score >= 1:
            momentum = 'BULLISH'
        elif momentum_score <= -3:
            momentum = 'STRONG_BEARISH'
        elif momentum_score <= -1:
            momentum = 'BEARISH'
        else:
            momentum = 'NEUTRAL'
        
        return {
            'momentum': momentum,
            'strength': abs(momentum_score) * 20,
            'rsi': current_rsi,
            'stochastic': current_stoch,
            'cci': current_cci,
            'score': momentum_score
        }
    
    def detect_price_patterns(self, candles):
        """اكتشاف الأنماط السعرية"""
        df = pd.DataFrame(candles)
        patterns = []
        
        if len(df) < 50:
            return patterns
        
        # تحويل البيانات لـ numpy arrays
        open_prices = df['open'].values
        high_prices = df['high'].values
        low_prices = df['low'].values
        close_prices = df['close'].values
        
        # أنماط الشموع اليابانية
        # Hammer
        hammer = talib.CDLHAMMER(open_prices, high_prices, low_prices, close_prices)
        if hammer[-1] != 0:
            patterns.append({
                'name': 'Hammer',
                'type': 'reversal',
                'direction': 'bullish',
                'strength': abs(hammer[-1])
            })
        
        # Doji
        doji = talib.CDLDOJI(open_prices, high_prices, low_prices, close_prices)
        if doji[-1] != 0:
            patterns.append({
                'name': 'Doji',
                'type': 'indecision',
                'direction': 'neutral',
                'strength': abs(doji[-1])
            })
        
        # Engulfing
        engulfing = talib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices)
        if engulfing[-1] > 0:
            patterns.append({
                'name': 'Bullish Engulfing',
                'type': 'reversal',
                'direction': 'bullish',
                'strength': engulfing[-1]
            })
        elif engulfing[-1] < 0:
            patterns.append({
                'name': 'Bearish Engulfing',
                'type': 'reversal',
                'direction': 'bearish',
                'strength': abs(engulfing[-1])
            })
        
        # Morning/Evening Star
        morning_star = talib.CDLMORNINGSTAR(open_prices, high_prices, low_prices, close_prices)
        if morning_star[-1] != 0:
            patterns.append({
                'name': 'Morning Star',
                'type': 'reversal',
                'direction': 'bullish',
                'strength': abs(morning_star[-1])
            })
        
        evening_star = talib.CDLEVENINGSTAR(open_prices, high_prices, low_prices, close_prices)
        if evening_star[-1] != 0:
            patterns.append({
                'name': 'Evening Star',
                'type': 'reversal',
                'direction': 'bearish',
                'strength': abs(evening_star[-1])
            })
        
        # أنماط الأسعار
        # Double Top/Bottom
        double_pattern = self._detect_double_patterns(df)
        if double_pattern:
            patterns.append(double_pattern)
        
        # Head and Shoulders
        hs_pattern = self._detect_head_shoulders(df)
        if hs_pattern:
            patterns.append(hs_pattern)
        
        return patterns
    
    def _detect_double_patterns(self, df):
        """اكتشاف نماذج القمة/القاع المزدوج"""
        if len(df) < 40:
            return None
        
        # البحث عن القمم والقيعان
        highs = []
        lows = []
        
        for i in range(5, len(df) - 5):
            # قمة محلية
            if (df['high'].iloc[i] > df['high'].iloc[i-1] and 
                df['high'].iloc[i] > df['high'].iloc[i+1] and
                df['high'].iloc[i] > df['high'].iloc[i-2] and
                df['high'].iloc[i] > df['high'].iloc[i+2]):
                highs.append((i, df['high'].iloc[i]))
            
            # قاع محلي
            if (df['low'].iloc[i] < df['low'].iloc[i-1] and 
                df['low'].iloc[i] < df['low'].iloc[i+1] and
                df['low'].iloc[i] < df['low'].iloc[i-2] and
                df['low'].iloc[i] < df['low'].iloc[i+2]):
                lows.append((i, df['low'].iloc[i]))
        
        # التحقق من القمة المزدوجة
        if len(highs) >= 2:
            last_two_highs = highs[-2:]
            price_diff = abs(last_two_highs[0][1] - last_two_highs[1][1]) / last_two_highs[0][1]
            
            if price_diff < 0.02:  # فرق أقل من 2%
                return {
                    'name': 'Double Top',
                    'type': 'reversal',
                    'direction': 'bearish',
                    'strength': 80,
                    'level': max(last_two_highs[0][1], last_two_highs[1][1])
                }
        
        # التحقق من القاع المزدوج
        if len(lows) >= 2:
            last_two_lows = lows[-2:]
            price_diff = abs(last_two_lows[0][1] - last_two_lows[1][1]) / last_two_lows[0][1]
            
            if price_diff < 0.02:  # فرق أقل من 2%
                return {
                    'name': 'Double Bottom',
                    'type': 'reversal',
                    'direction': 'bullish',
                    'strength': 80,
                    'level': min(last_two_lows[0][1], last_two_lows[1][1])
                }
        
        return None
    
    def _detect_head_shoulders(self, df):
        """اكتشاف نموذج الرأس والكتفين"""
        # تطبيق مبسط - يحتاج لتحسين
        return None
    
    def analyze_volatility(self, candles):
        """تحليل التقلبات"""
        df = pd.DataFrame(candles)
        
        if len(df) < 20:
            return {
                'atr': 0,
                'volatility_level': 'NORMAL',
                'volatility_trend': 'STABLE'
            }
        
        # حساب ATR
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        atr = talib.ATR(high, low, close, timeperiod=14)
        current_atr = atr[-1] if len(atr) > 0 and not pd.isna(atr[-1]) else 0
        avg_atr = np.mean(atr[-20:]) if len(atr) >= 20 else current_atr
        
        # حساب Bollinger Bands width
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20)
        bb_width = bb_upper - bb_lower
        current_bb_width = bb_width[-1] if len(bb_width) > 0 else 0
        
        # تحديد مستوى التقلب
        atr_ratio = current_atr / close[-1] if close[-1] > 0 else 0
        
        if atr_ratio > 0.03:
            volatility_level = 'VERY_HIGH'
        elif atr_ratio > 0.02:
            volatility_level = 'HIGH'
        elif atr_ratio < 0.005:
            volatility_level = 'VERY_LOW'
        elif atr_ratio < 0.01:
            volatility_level = 'LOW'
        else:
            volatility_level = 'NORMAL'
        
        # اتجاه التقلب
        if current_atr > avg_atr * 1.2:
            volatility_trend = 'INCREASING'
        elif current_atr < avg_atr * 0.8:
            volatility_trend = 'DECREASING'
        else:
            volatility_trend = 'STABLE'
        
        return {
            'atr': float(current_atr),
            'atr_percentage': float(atr_ratio * 100),
            'bb_width': float(current_bb_width),
            'volatility_level': volatility_level,
            'volatility_trend': volatility_trend
        }
    
    def calculate_market_score(self, context):
        """حساب النقاط الإجمالية للسوق"""
        score = 0
        
        # 1. نقاط الترند (30%)
        trend = context.get('trend', {})
        if trend.get('overall') == 'STRONG_BULLISH':
            score += 30
        elif trend.get('overall') == 'BULLISH':
            score += 15
        elif trend.get('overall') == 'STRONG_BEARISH':
            score -= 30
        elif trend.get('overall') == 'BEARISH':
            score -= 15
        
        # محاذاة الترند
        if trend.get('alignment', False):
            score += 10
        
        # 2. نقاط الزخم (20%)
        momentum = context.get('momentum', {})
        if momentum.get('momentum') == 'STRONG_BULLISH':
            score += 20
        elif momentum.get('momentum') == 'BULLISH':
            score += 10
        elif momentum.get('momentum') == 'STRONG_BEARISH':
            score -= 20
        elif momentum.get('momentum') == 'BEARISH':
            score -= 10
        
        # 3. نقاط الحجم (15%)
        volume = context.get('volume', {})
        if volume.get('volume_signal') == 'BULLISH_CONFIRMATION':
            score += 15
        elif volume.get('volume_signal') == 'BEARISH_CONFIRMATION':
            score -= 15
        elif volume.get('volume_signal') == 'BULLISH_DIVERGENCE':
            score -= 5
        elif volume.get('volume_signal') == 'BEARISH_DIVERGENCE':
            score += 5
        
        # 4. نقاط الجلسة (10%)
        session = context.get('session', {})
        if session.get('session_quality') == 'EXCELLENT':
            score += 10
        elif session.get('session_quality') == 'GOOD':
            score += 5
        elif session.get('session_quality') == 'LOW':
            score -= 5
        
        # خصم للأخبار
        if session.get('is_news_time', False):
            score -= 10
        
        # 5. نقاط الأنماط (15%)
        patterns = context.get('patterns', [])
        for pattern in patterns:
            if pattern['direction'] == 'bullish':
                score += pattern['strength'] / 100 * 15
            elif pattern['direction'] == 'bearish':
                score -= pattern['strength'] / 100 * 15
        
        # 6. نقاط التقلب (10%)
        volatility = context.get('volatility', {})
        if volatility.get('volatility_level') == 'VERY_HIGH':
            score -= 10
        elif volatility.get('volatility_level') == 'HIGH':
            score -= 5
        elif volatility.get('volatility_level') == 'VERY_LOW':
            score -= 10
        
        return max(-100, min(100, score))
    
    def determine_signal_strength(self, score):
        """تحديد قوة الإشارة"""
        if score >= 70:
            return 'VERY_STRONG_BUY'
        elif score >= 40:
            return 'STRONG_BUY'
        elif score >= 20:
            return 'BUY'
        elif score <= -70:
            return 'VERY_STRONG_SELL'
        elif score <= -40:
            return 'STRONG_SELL'
        elif score <= -20:
            return 'SELL'
        else:
            return 'NEUTRAL'
    
    def _get_higher_timeframe_data(self, symbol, timeframe):
        """جلب بيانات فريم أعلى من قاعدة البيانات"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # تحديد عدد الشموع حسب الفريم
            candle_limits = {
                'H1': 200,
                'H4': 200,
                'D1': 100
            }
            
            limit = candle_limits.get(timeframe, 100)
            
            # محاولة جلب البيانات
            query = f"""
            SELECT * FROM price_data 
            WHERE symbol LIKE '%{symbol}%'
            AND timeframe = '{timeframe}'
            ORDER BY time DESC
            LIMIT {limit}
            """
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if df.empty:
                return None
            
            # تحويل الأعمدة
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            df = df.sort_index()
            
            for col in ['open', 'high', 'low', 'close']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting {timeframe} data: {e}")
            return None
    
    def _check_trend_alignment(self, trends):
        """التحقق من محاذاة الترند عبر الفريمات"""
        if not trends:
            return False
        
        directions = []
        for tf, trend in trends.items():
            if trend and 'direction' in trend:
                directions.append(trend['direction'])
        
        # التحقق من التوافق
        bullish_count = sum(1 for d in directions if 'BULLISH' in d)
        bearish_count = sum(1 for d in directions if 'BEARISH' in d)
        
        # محاذاة صاعدة
        if bullish_count >= len(directions) * 0.7:
            return True
        # محاذاة هابطة
        elif bearish_count >= len(directions) * 0.7:
            return True
        
        return False