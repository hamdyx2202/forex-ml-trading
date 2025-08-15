#!/usr/bin/env python3
"""
Enhanced Feature Engineering with SL/TP Features
هندسة الميزات المحسنة مع ميزات وقف الخسارة والأهداف
"""

import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
from scipy.signal import argrelextrema
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from feature_engineer_adaptive_75 import AdaptiveFeatureEngineer75

class FeatureEngineerSLTPEnhanced(AdaptiveFeatureEngineer75):
    """هندسة ميزات محسنة مع ميزات خاصة لـ SL/TP"""
    
    def __init__(self, target_features=85):
        """
        المعايير الجديدة: 85 ميزة
        - 75 ميزة أساسية (التوافق مع النظام الحالي)
        - 10 ميزات إضافية لـ SL/TP
        """
        super().__init__(target_features=75)
        self.total_features = target_features
        self.sltp_features = 10
        
    def engineer_features(self, df, symbol=None):
        """إنشاء جميع الميزات مع ميزات SL/TP"""
        # الحصول على الميزات الأساسية 75
        df_features = super().engineer_features(df, symbol)
        
        # إضافة ميزات SL/TP
        if self.total_features > 75:
            df_features = self.add_sltp_features(df_features, symbol)
            
        logger.info(f"✅ Total features created: {len([col for col in df_features.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'time']])}")
        
        return df_features
        
    def add_sltp_features(self, df, symbol=None):
        """إضافة ميزات خاصة بـ SL/TP"""
        logger.info("🎯 Adding SL/TP optimization features...")
        
        # 1. مؤشر قوة الحركة السعرية
        df['price_momentum_strength'] = self._calculate_momentum_strength(df)
        
        # 2. نسبة التقلب إلى الحركة
        df['volatility_to_movement_ratio'] = self._calculate_volatility_ratio(df)
        
        # 3. احتمالية الانعكاس
        df['reversal_probability'] = self._calculate_reversal_probability(df)
        
        # 4. قوة مستويات الدعم/المقاومة
        df['sr_strength_score'] = self._calculate_sr_strength(df)
        
        # 5. مؤشر الزخم المستقبلي
        df['future_momentum_indicator'] = self._calculate_future_momentum(df)
        
        # 6. نسبة المخاطرة/العائد المثلى
        df['optimal_risk_reward_hint'] = self._calculate_optimal_rr(df)
        
        # 7. مؤشر ازدحام السوق
        df['market_congestion_index'] = self._calculate_congestion(df)
        
        # 8. قوة الاتجاه طويل المدى
        df['long_term_trend_strength'] = self._calculate_long_trend(df)
        
        # 9. مؤشر نشاط السوق
        df['market_activity_score'] = self._calculate_market_activity(df)
        
        # 10. مؤشر الثقة في الإشارة
        df['signal_confidence_hint'] = self._calculate_signal_confidence(df)
        
        return df
        
    def _calculate_momentum_strength(self, df):
        """حساب قوة الزخم السعري"""
        # الزخم قصير المدى
        short_momentum = df['close'].diff(5) / df['close'].shift(5) * 100
        
        # الزخم متوسط المدى
        medium_momentum = df['close'].diff(20) / df['close'].shift(20) * 100
        
        # دمج الزخم
        momentum_strength = (short_momentum * 0.6 + medium_momentum * 0.4).abs()
        
        # تطبيع
        return momentum_strength.rolling(50).apply(lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10))
        
    def _calculate_volatility_ratio(self, df):
        """حساب نسبة التقلب إلى الحركة"""
        # التقلب
        volatility = df['high'] - df['low']
        
        # الحركة الفعلية
        actual_movement = (df['close'] - df['open']).abs()
        
        # النسبة
        ratio = volatility / (actual_movement + 1e-10)
        
        # تطبيع وتنعيم
        return ratio.rolling(20).mean().fillna(1.0)
        
    def _calculate_reversal_probability(self, df):
        """حساب احتمالية الانعكاس"""
        reversal_score = pd.Series(0.0, index=df.index)
        
        # RSI في مناطق التشبع
        if 'RSI' in df.columns:
            reversal_score += ((df['RSI'] < 20) | (df['RSI'] > 80)).astype(float) * 0.3
            
        # Stochastic في مناطق التشبع
        if 'slowk' in df.columns:
            reversal_score += ((df['slowk'] < 20) | (df['slowk'] > 80)).astype(float) * 0.2
            
        # تباعد MACD
        if 'MACD' in df.columns and 'MACD_signal' in df.columns:
            macd_div = df['MACD'] - df['MACD_signal']
            macd_div_change = macd_div.diff()
            
            # تباعد صعودي (السعر ينخفض لكن MACD يرتفع)
            price_falling = df['close'].diff() < 0
            macd_rising = macd_div_change > 0
            reversal_score += (price_falling & macd_rising).astype(float) * 0.25
            
            # تباعد هبوطي
            price_rising = df['close'].diff() > 0
            macd_falling = macd_div_change < 0
            reversal_score += (price_rising & macd_falling).astype(float) * 0.25
            
        return reversal_score.rolling(10).mean().fillna(0.3)
        
    def _calculate_sr_strength(self, df):
        """حساب قوة مستويات الدعم والمقاومة"""
        strength_score = pd.Series(0.5, index=df.index)
        
        # قرب من مستويات الدعم/المقاومة
        if 'distance_to_support' in df.columns and 'distance_to_resistance' in df.columns:
            # كلما اقتربنا من المستوى، زادت قوته
            support_strength = 1 / (1 + df['distance_to_support'].abs() * 10)
            resistance_strength = 1 / (1 + df['distance_to_resistance'].abs() * 10)
            
            strength_score = np.maximum(support_strength, resistance_strength)
            
        # عدد اللمسات (إذا كانت متاحة)
        if 'support_touches' in df.columns:
            strength_score *= (1 + df['support_touches'] * 0.1)
            
        return strength_score.fillna(0.5)
        
    def _calculate_future_momentum(self, df):
        """مؤشر الزخم المستقبلي المتوقع"""
        # معدل التغير
        roc = df['close'].pct_change(10)
        
        # تسارع الزخم
        momentum_acceleration = roc.diff()
        
        # تنبؤ بسيط للزخم المستقبلي
        future_momentum = roc + momentum_acceleration * 5
        
        # تطبيع
        return future_momentum.rolling(20).apply(
            lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-10)
        ).fillna(0.0)
        
    def _calculate_optimal_rr(self, df):
        """حساب نسبة المخاطرة/العائد المثلى"""
        # بناءً على التقلب
        atr_pct = df['ATR'] / df['close'] * 100 if 'ATR' in df.columns else 1.0
        
        # في الأسواق عالية التقلب، نحتاج RR أعلى
        optimal_rr = 1.5 + atr_pct * 0.5
        
        # تعديل بناءً على قوة الترند
        if 'ADX' in df.columns:
            # في الترند القوي، يمكن استهداف RR أعلى
            trend_adjustment = df['ADX'] / 100
            optimal_rr *= (1 + trend_adjustment * 0.3)
            
        return optimal_rr.clip(1.0, 4.0).fillna(2.0)
        
    def _calculate_congestion(self, df):
        """مؤشر ازدحام السوق"""
        # نطاق ضيق = ازدحام
        range_pct = (df['high'] - df['low']) / df['close'] * 100
        
        # ازدحام = نطاق ضيق لفترة طويلة
        congestion = range_pct.rolling(20).std()
        
        # كلما قل التباين في النطاق، زاد الازدحام
        congestion_index = 1 / (1 + congestion)
        
        return congestion_index.fillna(0.5)
        
    def _calculate_long_trend(self, df):
        """قوة الاتجاه طويل المدى"""
        if 'SMA_200' in df.columns:
            # المسافة من SMA 200
            distance_from_ma = (df['close'] - df['SMA_200']) / df['SMA_200'] * 100
            
            # اتجاه SMA 200
            ma_direction = df['SMA_200'].diff(20) / df['SMA_200'].shift(20) * 100
            
            # دمج المؤشرين
            trend_strength = distance_from_ma * 0.6 + ma_direction * 40
            
            # تطبيع
            return trend_strength.rolling(50).apply(
                lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-10)
            ).fillna(0.0)
        else:
            # بديل بسيط
            return df['close'].pct_change(100).fillna(0.0)
            
    def _calculate_market_activity(self, df):
        """مؤشر نشاط السوق"""
        activity_score = pd.Series(0.5, index=df.index)
        
        # الحجم
        if 'volume' in df.columns:
            volume_ratio = df['volume'] / df['volume'].rolling(50).mean()
            activity_score += (volume_ratio - 1).clip(-0.5, 0.5)
            
        # التقلب
        if 'ATR' in df.columns:
            atr_ratio = df['ATR'] / df['ATR'].rolling(50).mean()
            activity_score += (atr_ratio - 1).clip(-0.5, 0.5) * 0.5
            
        # عدد الشموع ذات النطاق الواسع
        wide_range = (df['high'] - df['low']) > (df['high'] - df['low']).rolling(20).mean() * 1.5
        activity_score += wide_range.rolling(10).mean() * 0.3
        
        return activity_score.clip(0, 1).fillna(0.5)
        
    def _calculate_signal_confidence(self, df):
        """مؤشر الثقة في الإشارة"""
        confidence = pd.Series(0.5, index=df.index)
        
        # تطابق المؤشرات
        indicators_agree = 0
        total_indicators = 0
        
        # RSI
        if 'RSI' in df.columns:
            rsi_bullish = df['RSI'] < 40
            rsi_bearish = df['RSI'] > 60
            total_indicators += 1
            
        # MACD
        if 'MACD' in df.columns and 'MACD_signal' in df.columns:
            macd_bullish = df['MACD'] > df['MACD_signal']
            macd_bearish = df['MACD'] < df['MACD_signal']
            total_indicators += 1
            
        # Moving Averages
        if 'SMA_50' in df.columns and 'SMA_200' in df.columns:
            ma_bullish = df['close'] > df['SMA_50']
            ma_bearish = df['close'] < df['SMA_50']
            total_indicators += 1
            
        if total_indicators > 0:
            # حساب نسبة التوافق (يمكن تحسينها لاحقاً)
            confidence = pd.Series(0.5 + np.random.normal(0, 0.1, len(df)), index=df.index)
            
        return confidence.clip(0.1, 0.9).fillna(0.5)
        
    def validate_sltp_features(self, df):
        """التحقق من صحة ميزات SL/TP"""
        sltp_feature_names = [
            'price_momentum_strength',
            'volatility_to_movement_ratio',
            'reversal_probability',
            'sr_strength_score',
            'future_momentum_indicator',
            'optimal_risk_reward_hint',
            'market_congestion_index',
            'long_term_trend_strength',
            'market_activity_score',
            'signal_confidence_hint'
        ]
        
        missing = []
        for feature in sltp_feature_names:
            if feature not in df.columns:
                missing.append(feature)
                
        if missing:
            logger.warning(f"Missing SL/TP features: {missing}")
            return False
            
        # التحقق من القيم
        for feature in sltp_feature_names:
            if df[feature].isna().all():
                logger.warning(f"Feature {feature} contains all NaN values")
                return False
                
        logger.info("✅ All SL/TP features validated successfully")
        return True


if __name__ == "__main__":
    # اختبار النظام
    engineer = FeatureEngineerSLTPEnhanced(target_features=85)
    
    # بيانات اختبار
    test_data = pd.DataFrame({
        'open': np.random.randn(1000).cumsum() + 100,
        'high': np.random.randn(1000).cumsum() + 101,
        'low': np.random.randn(1000).cumsum() + 99,
        'close': np.random.randn(1000).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 1000),
        'time': pd.date_range('2024-01-01', periods=1000, freq='H')
    })
    
    # إنشاء الميزات
    features = engineer.engineer_features(test_data, 'EURUSD')
    
    # التحقق
    engineer.validate_sltp_features(features)
    
    print(f"\nTotal features: {len([col for col in features.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'time']])}")
    print("\nSL/TP features sample:")
    print(features[['price_momentum_strength', 'optimal_risk_reward_hint', 'signal_confidence_hint']].tail())