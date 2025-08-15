#!/usr/bin/env python3
"""
نظام حساب مستويات الدعم والمقاومة المتقدم
Advanced Support & Resistance Calculation System

يحسب مستويات الدعم والمقاومة باستخدام 5 طرق مختلفة:
1. القمم والقيعان (Peaks & Troughs)
2. نقاط البيفوت اليومية (Daily Pivot Points)
3. المتوسطات المتحركة الرئيسية (Major Moving Averages)
4. مستويات فيبوناتشي (Fibonacci Levels)
5. المستويات النفسية (Psychological Levels)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from loguru import logger
import json
from scipy.signal import argrelextrema
import warnings
warnings.filterwarnings('ignore')

class SupportResistanceCalculator:
    """حاسب مستويات الدعم والمقاومة المتقدم"""
    
    def __init__(self):
        """تهيئة الحاسب"""
        self.lookback_periods = {
            'peaks_troughs': 100,      # آخر 100 شمعة
            'pivot': 1,                 # بيانات يوم واحد
            'fibonacci': 50,            # آخر 50 شمعة للترند
            'ma': 200                   # للمتوسط المتحرك 200
        }
        
        # إعدادات حساسية القمم والقيعان
        self.peak_order = 5  # عدد الشموع على كل جانب للتأكيد
        
        # المتوسطات المتحركة المستخدمة
        self.ma_periods = {
            'ema_50': 50,
            'sma_100': 100,
            'sma_200': 200
        }
        
        # نسب فيبوناتشي
        self.fib_levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        
    def calculate_all_levels(self, df: pd.DataFrame, symbol: str) -> Dict:
        """
        حساب جميع مستويات الدعم والمقاومة
        
        Args:
            df: DataFrame يحتوي على OHLCV
            symbol: رمز الزوج
            
        Returns:
            dict: مستويات الدعم والمقاومة مع القوة والنوع
        """
        try:
            # التأكد من وجود البيانات المطلوبة
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                logger.error(f"Missing required columns in DataFrame")
                return self._empty_levels()
            
            # التأكد من وجود بيانات كافية
            if len(df) < self.lookback_periods['ma']:
                logger.warning(f"Not enough data for {symbol}. Need at least {self.lookback_periods['ma']} candles")
                return self._empty_levels()
            
            # حساب المستويات بكل طريقة
            levels = {
                'support': [],
                'resistance': [],
                'current_price': float(df['close'].iloc[-1]),
                'timestamp': datetime.now().isoformat()
            }
            
            # 1. القمم والقيعان
            peaks_troughs = self._calculate_peaks_troughs(df)
            levels['support'].extend(peaks_troughs['support'])
            levels['resistance'].extend(peaks_troughs['resistance'])
            
            # 2. نقاط البيفوت
            pivot_levels = self._calculate_pivot_points(df)
            levels['support'].extend(pivot_levels['support'])
            levels['resistance'].extend(pivot_levels['resistance'])
            
            # 3. المتوسطات المتحركة
            ma_levels = self._calculate_ma_levels(df)
            levels['support'].extend(ma_levels['support'])
            levels['resistance'].extend(ma_levels['resistance'])
            
            # 4. مستويات فيبوناتشي
            fib_levels = self._calculate_fibonacci_levels(df)
            levels['support'].extend(fib_levels['support'])
            levels['resistance'].extend(fib_levels['resistance'])
            
            # 5. المستويات النفسية
            psych_levels = self._calculate_psychological_levels(df, symbol)
            levels['support'].extend(psych_levels['support'])
            levels['resistance'].extend(psych_levels['resistance'])
            
            # ترتيب وتنظيف المستويات
            levels = self._organize_levels(levels)
            
            # حساب أقرب المستويات
            levels['nearest'] = self._find_nearest_levels(levels)
            
            # حساب الميزات للنموذج (5 ميزات جديدة)
            levels['features'] = self._calculate_features(levels, df)
            
            logger.info(f"✅ Calculated S/R levels for {symbol}: "
                       f"{len(levels['support'])} support, {len(levels['resistance'])} resistance")
            
            return levels
            
        except Exception as e:
            logger.error(f"Error calculating S/R levels for {symbol}: {str(e)}")
            return self._empty_levels()
    
    def _calculate_peaks_troughs(self, df: pd.DataFrame) -> Dict:
        """حساب القمم والقيعان من آخر 100 شمعة"""
        try:
            # استخدام آخر 100 شمعة
            recent_df = df.tail(self.lookback_periods['peaks_troughs']).copy()
            
            # إيجاد القمم (المقاومات)
            peaks = argrelextrema(recent_df['high'].values, np.greater, order=self.peak_order)[0]
            
            # إيجاد القيعان (الدعوم)
            troughs = argrelextrema(recent_df['low'].values, np.less, order=self.peak_order)[0]
            
            # تحويل للمستويات
            resistance_levels = []
            support_levels = []
            
            for peak_idx in peaks:
                level = float(recent_df['high'].iloc[peak_idx])
                strength = self._calculate_touch_strength(df, level, is_resistance=True)
                resistance_levels.append({
                    'level': level,
                    'type': 'peak',
                    'strength': strength,
                    'touches': self._count_touches(df, level)
                })
            
            for trough_idx in troughs:
                level = float(recent_df['low'].iloc[trough_idx])
                strength = self._calculate_touch_strength(df, level, is_resistance=False)
                support_levels.append({
                    'level': level,
                    'type': 'trough',
                    'strength': strength,
                    'touches': self._count_touches(df, level)
                })
            
            return {
                'support': support_levels,
                'resistance': resistance_levels
            }
            
        except Exception as e:
            logger.error(f"Error in peaks/troughs calculation: {str(e)}")
            return {'support': [], 'resistance': []}
    
    def _calculate_pivot_points(self, df: pd.DataFrame) -> Dict:
        """حساب نقاط البيفوت اليومية"""
        try:
            # استخدام بيانات اليوم السابق
            last_high = float(df['high'].iloc[-1])
            last_low = float(df['low'].iloc[-1])
            last_close = float(df['close'].iloc[-1])
            
            # حساب البيفوت الرئيسي
            pivot = (last_high + last_low + last_close) / 3
            
            # حساب مستويات الدعم والمقاومة
            r1 = 2 * pivot - last_low
            r2 = pivot + (last_high - last_low)
            r3 = r1 + (last_high - last_low)
            
            s1 = 2 * pivot - last_high
            s2 = pivot - (last_high - last_low)
            s3 = s1 - (last_high - last_low)
            
            resistance_levels = [
                {'level': r1, 'type': 'pivot_r1', 'strength': 0.7, 'touches': 0},
                {'level': r2, 'type': 'pivot_r2', 'strength': 0.6, 'touches': 0},
                {'level': r3, 'type': 'pivot_r3', 'strength': 0.5, 'touches': 0}
            ]
            
            support_levels = [
                {'level': s1, 'type': 'pivot_s1', 'strength': 0.7, 'touches': 0},
                {'level': s2, 'type': 'pivot_s2', 'strength': 0.6, 'touches': 0},
                {'level': s3, 'type': 'pivot_s3', 'strength': 0.5, 'touches': 0}
            ]
            
            # إضافة البيفوت نفسه
            current_price = float(df['close'].iloc[-1])
            if current_price > pivot:
                support_levels.append({'level': pivot, 'type': 'pivot', 'strength': 0.8, 'touches': 0})
            else:
                resistance_levels.append({'level': pivot, 'type': 'pivot', 'strength': 0.8, 'touches': 0})
            
            return {
                'support': support_levels,
                'resistance': resistance_levels
            }
            
        except Exception as e:
            logger.error(f"Error in pivot calculation: {str(e)}")
            return {'support': [], 'resistance': []}
    
    def _calculate_ma_levels(self, df: pd.DataFrame) -> Dict:
        """حساب مستويات المتوسطات المتحركة"""
        try:
            current_price = float(df['close'].iloc[-1])
            resistance_levels = []
            support_levels = []
            
            # حساب EMA 50
            if len(df) >= self.ma_periods['ema_50']:
                ema_50 = df['close'].ewm(span=self.ma_periods['ema_50'], adjust=False).mean().iloc[-1]
                level_data = {
                    'level': float(ema_50),
                    'type': 'ema_50',
                    'strength': 0.7,
                    'touches': self._count_ma_touches(df, ema_50)
                }
                
                if current_price > ema_50:
                    support_levels.append(level_data)
                else:
                    resistance_levels.append(level_data)
            
            # حساب SMA 100
            if len(df) >= self.ma_periods['sma_100']:
                sma_100 = df['close'].rolling(window=self.ma_periods['sma_100']).mean().iloc[-1]
                level_data = {
                    'level': float(sma_100),
                    'type': 'sma_100',
                    'strength': 0.8,
                    'touches': self._count_ma_touches(df, sma_100)
                }
                
                if current_price > sma_100:
                    support_levels.append(level_data)
                else:
                    resistance_levels.append(level_data)
            
            # حساب SMA 200
            if len(df) >= self.ma_periods['sma_200']:
                sma_200 = df['close'].rolling(window=self.ma_periods['sma_200']).mean().iloc[-1]
                level_data = {
                    'level': float(sma_200),
                    'type': 'sma_200',
                    'strength': 0.9,
                    'touches': self._count_ma_touches(df, sma_200)
                }
                
                if current_price > sma_200:
                    support_levels.append(level_data)
                else:
                    resistance_levels.append(level_data)
            
            return {
                'support': support_levels,
                'resistance': resistance_levels
            }
            
        except Exception as e:
            logger.error(f"Error in MA calculation: {str(e)}")
            return {'support': [], 'resistance': []}
    
    def _calculate_fibonacci_levels(self, df: pd.DataFrame) -> Dict:
        """حساب مستويات فيبوناتشي"""
        try:
            # استخدام آخر 50 شمعة لتحديد الترند
            recent_df = df.tail(self.lookback_periods['fibonacci'])
            
            # إيجاد أعلى وأدنى نقطة
            high_point = float(recent_df['high'].max())
            low_point = float(recent_df['low'].min())
            
            # حساب المدى
            price_range = high_point - low_point
            
            # تحديد اتجاه الترند
            current_price = float(df['close'].iloc[-1])
            is_uptrend = current_price > (high_point + low_point) / 2
            
            resistance_levels = []
            support_levels = []
            
            for fib_ratio in self.fib_levels:
                if is_uptrend:
                    # في الترند الصاعد، مستويات الفيبو تكون دعوم
                    level = high_point - (price_range * fib_ratio)
                    if level < current_price and fib_ratio not in [0.0, 1.0]:
                        support_levels.append({
                            'level': level,
                            'type': f'fib_{int(fib_ratio*100)}',
                            'strength': 0.6 + (0.2 if fib_ratio in [0.382, 0.618] else 0),
                            'touches': 0
                        })
                else:
                    # في الترند الهابط، مستويات الفيبو تكون مقاومات
                    level = low_point + (price_range * fib_ratio)
                    if level > current_price and fib_ratio not in [0.0, 1.0]:
                        resistance_levels.append({
                            'level': level,
                            'type': f'fib_{int(fib_ratio*100)}',
                            'strength': 0.6 + (0.2 if fib_ratio in [0.382, 0.618] else 0),
                            'touches': 0
                        })
            
            return {
                'support': support_levels,
                'resistance': resistance_levels
            }
            
        except Exception as e:
            logger.error(f"Error in Fibonacci calculation: {str(e)}")
            return {'support': [], 'resistance': []}
    
    def _calculate_psychological_levels(self, df: pd.DataFrame, symbol: str) -> Dict:
        """حساب المستويات النفسية (الأرقام المدورة)"""
        try:
            current_price = float(df['close'].iloc[-1])
            
            # تحديد دقة التقريب بناءً على السعر
            if current_price < 1:
                # للأزواج مثل EURUSD
                round_to = 0.0050  # كل 50 نقطة
            elif current_price < 10:
                # للأزواج مثل USDCAD
                round_to = 0.0100  # كل 100 نقطة
            elif current_price < 100:
                # للأزواج مثل USDJPY
                round_to = 0.5000  # كل 50 سنت
            elif current_price < 1000:
                # للذهب
                round_to = 5.0000  # كل 5 دولار
            else:
                # للمؤشرات
                round_to = 50.0000  # كل 50 نقطة
            
            # حساب المستويات القريبة
            resistance_levels = []
            support_levels = []
            
            # حساب 5 مستويات فوق وتحت السعر الحالي
            for i in range(1, 6):
                # مستويات المقاومة
                resistance = current_price + (i * round_to)
                resistance = round(resistance / round_to) * round_to
                resistance_levels.append({
                    'level': resistance,
                    'type': 'psychological',
                    'strength': 0.5 + (0.1 if resistance % (round_to * 10) == 0 else 0),
                    'touches': 0
                })
                
                # مستويات الدعم
                support = current_price - (i * round_to)
                support = round(support / round_to) * round_to
                support_levels.append({
                    'level': support,
                    'type': 'psychological',
                    'strength': 0.5 + (0.1 if support % (round_to * 10) == 0 else 0),
                    'touches': 0
                })
            
            return {
                'support': support_levels,
                'resistance': resistance_levels
            }
            
        except Exception as e:
            logger.error(f"Error in psychological levels calculation: {str(e)}")
            return {'support': [], 'resistance': []}
    
    def _calculate_touch_strength(self, df: pd.DataFrame, level: float, is_resistance: bool) -> float:
        """حساب قوة المستوى بناءً على عدد اللمسات"""
        touches = self._count_touches(df, level)
        
        # قوة أساسية
        base_strength = 0.5
        
        # إضافة قوة لكل لمسة (حتى 0.4 إضافية)
        touch_bonus = min(touches * 0.1, 0.4)
        
        # قوة نهائية
        return min(base_strength + touch_bonus, 1.0)
    
    def _count_touches(self, df: pd.DataFrame, level: float, tolerance_pct: float = 0.0005) -> int:
        """عد عدد مرات لمس السعر للمستوى"""
        tolerance = level * tolerance_pct
        touches = 0
        
        for _, row in df.iterrows():
            # فحص إذا كان السعر لمس المستوى
            if (abs(row['high'] - level) <= tolerance or 
                abs(row['low'] - level) <= tolerance or
                abs(row['close'] - level) <= tolerance):
                touches += 1
        
        return touches
    
    def _count_ma_touches(self, df: pd.DataFrame, ma_value: float, lookback: int = 20) -> int:
        """عد عدد مرات لمس السعر للمتوسط المتحرك"""
        recent_df = df.tail(lookback)
        tolerance = ma_value * 0.001  # 0.1%
        touches = 0
        
        for _, row in recent_df.iterrows():
            if (abs(row['high'] - ma_value) <= tolerance or 
                abs(row['low'] - ma_value) <= tolerance):
                touches += 1
        
        return touches
    
    def _organize_levels(self, levels: Dict) -> Dict:
        """تنظيم وترتيب المستويات"""
        try:
            current_price = levels['current_price']
            
            # دمج المستويات القريبة من بعضها
            levels['support'] = self._merge_nearby_levels(levels['support'])
            levels['resistance'] = self._merge_nearby_levels(levels['resistance'])
            
            # ترتيب حسب القرب من السعر الحالي
            levels['support'].sort(key=lambda x: current_price - x['level'])
            levels['resistance'].sort(key=lambda x: x['level'] - current_price)
            
            # الاحتفاظ بأقوى 10 مستويات فقط لكل جانب
            levels['support'] = levels['support'][:10]
            levels['resistance'] = levels['resistance'][:10]
            
            return levels
            
        except Exception as e:
            logger.error(f"Error organizing levels: {str(e)}")
            return levels
    
    def _merge_nearby_levels(self, levels: List[Dict], merge_threshold: float = 0.001) -> List[Dict]:
        """دمج المستويات القريبة من بعضها"""
        if not levels:
            return []
        
        # ترتيب حسب المستوى
        sorted_levels = sorted(levels, key=lambda x: x['level'])
        merged = []
        
        current_group = [sorted_levels[0]]
        
        for level in sorted_levels[1:]:
            # حساب المسافة النسبية
            distance_pct = abs(level['level'] - current_group[-1]['level']) / current_group[-1]['level']
            
            if distance_pct <= merge_threshold:
                # إضافة للمجموعة الحالية
                current_group.append(level)
            else:
                # دمج المجموعة الحالية وبدء مجموعة جديدة
                merged.append(self._merge_group(current_group))
                current_group = [level]
        
        # دمج آخر مجموعة
        if current_group:
            merged.append(self._merge_group(current_group))
        
        return merged
    
    def _merge_group(self, group: List[Dict]) -> Dict:
        """دمج مجموعة من المستويات في مستوى واحد"""
        # حساب المتوسط المرجح بالقوة
        total_strength = sum(level['strength'] for level in group)
        weighted_level = sum(level['level'] * level['strength'] for level in group) / total_strength
        
        # أقوى قوة في المجموعة
        max_strength = max(level['strength'] for level in group)
        
        # مجموع اللمسات
        total_touches = sum(level['touches'] for level in group)
        
        # النوع الأكثر أهمية
        type_priority = ['sma_200', 'sma_100', 'ema_50', 'peak', 'trough', 'pivot', 'fib_618', 'fib_382']
        best_type = 'merged'
        for t in type_priority:
            if any(level['type'] == t for level in group):
                best_type = t
                break
        
        return {
            'level': weighted_level,
            'type': best_type,
            'strength': min(max_strength * 1.2, 1.0),  # زيادة القوة للمستويات المدمجة
            'touches': total_touches,
            'merged_count': len(group)
        }
    
    def _find_nearest_levels(self, levels: Dict) -> Dict:
        """إيجاد أقرب مستويات الدعم والمقاومة"""
        current_price = levels['current_price']
        
        nearest = {
            'support': None,
            'resistance': None,
            'support_distance': None,
            'resistance_distance': None,
            'support_distance_pct': None,
            'resistance_distance_pct': None
        }
        
        # أقرب دعم
        if levels['support']:
            nearest['support'] = levels['support'][0]['level']
            nearest['support_distance'] = current_price - nearest['support']
            nearest['support_distance_pct'] = (nearest['support_distance'] / current_price) * 100
        
        # أقرب مقاومة
        if levels['resistance']:
            nearest['resistance'] = levels['resistance'][0]['level']
            nearest['resistance_distance'] = nearest['resistance'] - current_price
            nearest['resistance_distance_pct'] = (nearest['resistance_distance'] / current_price) * 100
        
        return nearest
    
    def _calculate_features(self, levels: Dict, df: pd.DataFrame) -> Dict:
        """
        حساب 5 ميزات جديدة للنموذج
        
        Returns:
            dict: الميزات الجديدة
        """
        features = {}
        
        try:
            current_price = levels['current_price']
            nearest = levels['nearest']
            
            # 1. المسافة النسبية لأقرب دعم (%)
            if nearest['support_distance_pct'] is not None:
                features['distance_to_support_pct'] = round(nearest['support_distance_pct'], 4)
            else:
                features['distance_to_support_pct'] = 10.0  # قيمة افتراضية
            
            # 2. المسافة النسبية لأقرب مقاومة (%)
            if nearest['resistance_distance_pct'] is not None:
                features['distance_to_resistance_pct'] = round(nearest['resistance_distance_pct'], 4)
            else:
                features['distance_to_resistance_pct'] = 10.0  # قيمة افتراضية
            
            # 3. قوة أقرب دعم (0-1)
            if levels['support']:
                features['nearest_support_strength'] = round(levels['support'][0]['strength'], 4)
            else:
                features['nearest_support_strength'] = 0.0
            
            # 4. قوة أقرب مقاومة (0-1)
            if levels['resistance']:
                features['nearest_resistance_strength'] = round(levels['resistance'][0]['strength'], 4)
            else:
                features['nearest_resistance_strength'] = 0.0
            
            # 5. نسبة الموقع الحالي في النطاق (0-1)
            # 0 = عند أدنى دعم، 1 = عند أعلى مقاومة
            if nearest['support'] is not None and nearest['resistance'] is not None:
                range_size = nearest['resistance'] - nearest['support']
                if range_size > 0:
                    position_in_range = (current_price - nearest['support']) / range_size
                    features['position_in_sr_range'] = round(max(0, min(1, position_in_range)), 4)
                else:
                    features['position_in_sr_range'] = 0.5
            else:
                features['position_in_sr_range'] = 0.5
            
            logger.debug(f"S/R Features: {features}")
            
        except Exception as e:
            logger.error(f"Error calculating S/R features: {str(e)}")
            # قيم افتراضية في حالة الخطأ
            features = {
                'distance_to_support_pct': 5.0,
                'distance_to_resistance_pct': 5.0,
                'nearest_support_strength': 0.5,
                'nearest_resistance_strength': 0.5,
                'position_in_sr_range': 0.5
            }
        
        return features
    
    def calculate_dynamic_sl_tp(self, 
                               signal: str, 
                               entry_price: float, 
                               levels: Dict,
                               symbol: str,
                               risk_reward_ratio: float = 2.0) -> Dict:
        """
        حساب SL/TP الديناميكي بناءً على مستويات الدعم والمقاومة
        
        Args:
            signal: BUY/SELL
            entry_price: سعر الدخول
            levels: مستويات الدعم والمقاومة
            symbol: رمز الزوج
            risk_reward_ratio: نسبة المخاطرة للربح
            
        Returns:
            dict: قيم SL/TP مع الطريقة المستخدمة
        """
        try:
            result = {
                'sl': None,
                'tp': None,
                'sl_distance': None,
                'tp_distance': None,
                'sl_method': None,
                'tp_method': None,
                'risk_reward': risk_reward_ratio
            }
            
            if signal == 'BUY' or signal == 'STRONG_BUY':
                # للشراء: SL تحت أقرب دعم قوي
                if levels['support']:
                    # البحث عن دعم قوي (قوة > 0.7)
                    strong_supports = [s for s in levels['support'] if s['strength'] > 0.7]
                    if strong_supports:
                        sl_level = strong_supports[0]['level']
                        buffer = self._calculate_buffer(symbol, sl_level)
                        result['sl'] = sl_level - buffer
                        result['sl_distance'] = entry_price - result['sl']
                        result['sl_method'] = f"strong_support_{strong_supports[0]['type']}"
                    else:
                        # استخدام أقرب دعم مع buffer أكبر
                        sl_level = levels['support'][0]['level']
                        buffer = self._calculate_buffer(symbol, sl_level) * 2
                        result['sl'] = sl_level - buffer
                        result['sl_distance'] = entry_price - result['sl']
                        result['sl_method'] = f"nearest_support_{levels['support'][0]['type']}"
                
                # TP عند أقرب مقاومة قوية أو بناءً على R:R
                if levels['resistance']:
                    # البحث عن مقاومة قوية
                    strong_resistances = [r for r in levels['resistance'] if r['strength'] > 0.7]
                    if strong_resistances:
                        tp_level = strong_resistances[0]['level']
                        buffer = self._calculate_buffer(symbol, tp_level)
                        potential_tp = tp_level - buffer
                        
                        # التحقق من نسبة R:R
                        if result['sl_distance'] and result['sl_distance'] > 0:
                            min_tp = entry_price + (result['sl_distance'] * risk_reward_ratio)
                            if potential_tp >= min_tp:
                                result['tp'] = potential_tp
                                result['tp_method'] = f"strong_resistance_{strong_resistances[0]['type']}"
                            else:
                                result['tp'] = min_tp
                                result['tp_method'] = "risk_reward_based"
                        else:
                            result['tp'] = potential_tp
                            result['tp_method'] = f"resistance_{strong_resistances[0]['type']}"
                    
                    if result['tp']:
                        result['tp_distance'] = result['tp'] - entry_price
            
            elif signal == 'SELL' or signal == 'STRONG_SELL':
                # للبيع: SL فوق أقرب مقاومة قوية
                if levels['resistance']:
                    # البحث عن مقاومة قوية
                    strong_resistances = [r for r in levels['resistance'] if r['strength'] > 0.7]
                    if strong_resistances:
                        sl_level = strong_resistances[0]['level']
                        buffer = self._calculate_buffer(symbol, sl_level)
                        result['sl'] = sl_level + buffer
                        result['sl_distance'] = result['sl'] - entry_price
                        result['sl_method'] = f"strong_resistance_{strong_resistances[0]['type']}"
                    else:
                        # استخدام أقرب مقاومة مع buffer أكبر
                        sl_level = levels['resistance'][0]['level']
                        buffer = self._calculate_buffer(symbol, sl_level) * 2
                        result['sl'] = sl_level + buffer
                        result['sl_distance'] = result['sl'] - entry_price
                        result['sl_method'] = f"nearest_resistance_{levels['resistance'][0]['type']}"
                
                # TP عند أقرب دعم قوي أو بناءً على R:R
                if levels['support']:
                    # البحث عن دعم قوي
                    strong_supports = [s for s in levels['support'] if s['strength'] > 0.7]
                    if strong_supports:
                        tp_level = strong_supports[0]['level']
                        buffer = self._calculate_buffer(symbol, tp_level)
                        potential_tp = tp_level + buffer
                        
                        # التحقق من نسبة R:R
                        if result['sl_distance'] and result['sl_distance'] > 0:
                            min_tp = entry_price - (result['sl_distance'] * risk_reward_ratio)
                            if potential_tp <= min_tp:
                                result['tp'] = potential_tp
                                result['tp_method'] = f"strong_support_{strong_supports[0]['type']}"
                            else:
                                result['tp'] = min_tp
                                result['tp_method'] = "risk_reward_based"
                        else:
                            result['tp'] = potential_tp
                            result['tp_method'] = f"support_{strong_supports[0]['type']}"
                    
                    if result['tp']:
                        result['tp_distance'] = entry_price - result['tp']
            
            # إضافة معلومات إضافية
            if result['sl_distance'] and result['tp_distance'] and result['sl_distance'] > 0:
                result['actual_risk_reward'] = result['tp_distance'] / result['sl_distance']
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating dynamic SL/TP: {str(e)}")
            return {
                'sl': None,
                'tp': None,
                'sl_distance': None,
                'tp_distance': None,
                'sl_method': 'error',
                'tp_method': 'error',
                'risk_reward': risk_reward_ratio
            }
    
    def _calculate_buffer(self, symbol: str, price: float) -> float:
        """حساب المسافة الآمنة (buffer) بناءً على نوع الأداة"""
        # تحديد نوع الأداة
        symbol_upper = symbol.upper()
        
        if 'JPY' in symbol_upper:
            # أزواج الين
            return 0.05  # 5 نقاط
        elif 'XAU' in symbol_upper or 'GOLD' in symbol_upper:
            # الذهب
            return 0.50  # 50 سنت
        elif any(idx in symbol_upper for idx in ['US30', 'NAS100', 'SP500', 'DAX']):
            # المؤشرات
            return price * 0.0005  # 0.05%
        elif 'BTC' in symbol_upper or 'ETH' in symbol_upper:
            # العملات الرقمية
            return price * 0.002  # 0.2%
        elif 'USD' in symbol_upper or 'EUR' in symbol_upper or 'GBP' in symbol_upper:
            # أزواج العملات الرئيسية
            return 0.0005  # 5 نقاط
        else:
            # افتراضي
            return price * 0.0002  # 0.02%
    
    def _empty_levels(self) -> Dict:
        """إرجاع مستويات فارغة في حالة الخطأ"""
        return {
            'support': [],
            'resistance': [],
            'current_price': 0,
            'timestamp': datetime.now().isoformat(),
            'nearest': {
                'support': None,
                'resistance': None,
                'support_distance': None,
                'resistance_distance': None,
                'support_distance_pct': None,
                'resistance_distance_pct': None
            },
            'features': {
                'distance_to_support_pct': 5.0,
                'distance_to_resistance_pct': 5.0,
                'nearest_support_strength': 0.5,
                'nearest_resistance_strength': 0.5,
                'position_in_sr_range': 0.5
            }
        }
    
    def save_levels(self, levels: Dict, symbol: str, timeframe: str):
        """حفظ المستويات في ملف JSON"""
        try:
            filename = f"sr_levels/{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # إنشاء المجلد إذا لم يكن موجوداً
            import os
            os.makedirs('sr_levels', exist_ok=True)
            
            # حفظ البيانات
            with open(filename, 'w') as f:
                json.dump(levels, f, indent=2)
            
            logger.info(f"✅ Saved S/R levels to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving S/R levels: {str(e)}")


# دالة مساعدة للاستخدام المباشر
def calculate_support_resistance(df: pd.DataFrame, symbol: str) -> Dict:
    """
    دالة سريعة لحساب مستويات الدعم والمقاومة
    
    Args:
        df: DataFrame يحتوي على OHLCV
        symbol: رمز الزوج
        
    Returns:
        dict: مستويات الدعم والمقاومة مع الميزات
    """
    calculator = SupportResistanceCalculator()
    return calculator.calculate_all_levels(df, symbol)


if __name__ == "__main__":
    # مثال للاختبار
    print("🔍 Support & Resistance Calculator Test")
    
    # بيانات تجريبية
    test_data = {
        'open': np.random.rand(200) * 0.01 + 1.1000,
        'high': np.random.rand(200) * 0.01 + 1.1100,
        'low': np.random.rand(200) * 0.01 + 1.0900,
        'close': np.random.rand(200) * 0.01 + 1.1000,
        'volume': np.random.randint(1000, 10000, 200)
    }
    
    df = pd.DataFrame(test_data)
    
    # حساب المستويات
    calculator = SupportResistanceCalculator()
    levels = calculator.calculate_all_levels(df, "EURUSD")
    
    # عرض النتائج
    print(f"\n📊 Current Price: {levels['current_price']:.5f}")
    print(f"\n🟢 Support Levels ({len(levels['support'])}):")
    for s in levels['support'][:3]:
        print(f"   {s['level']:.5f} - {s['type']} (strength: {s['strength']:.2f})")
    
    print(f"\n🔴 Resistance Levels ({len(levels['resistance'])}):")
    for r in levels['resistance'][:3]:
        print(f"   {r['level']:.5f} - {r['type']} (strength: {r['strength']:.2f})")
    
    print(f"\n📏 Nearest Levels:")
    print(f"   Support: {levels['nearest']['support']:.5f} ({levels['nearest']['support_distance_pct']:.2f}%)")
    print(f"   Resistance: {levels['nearest']['resistance']:.5f} ({levels['nearest']['resistance_distance_pct']:.2f}%)")
    
    print(f"\n🎯 Features for ML Model:")
    for key, value in levels['features'].items():
        print(f"   {key}: {value}")
    
    # اختبار حساب SL/TP
    sl_tp = calculator.calculate_dynamic_sl_tp('BUY', levels['current_price'], levels, "EURUSD")
    print(f"\n💰 Dynamic SL/TP for BUY:")
    print(f"   SL: {sl_tp['sl']:.5f} ({sl_tp['sl_method']})")
    print(f"   TP: {sl_tp['tp']:.5f} ({sl_tp['tp_method']})")
    print(f"   Risk:Reward = 1:{sl_tp.get('actual_risk_reward', 0):.2f}")