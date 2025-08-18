import pandas as pd
import numpy as np
import logging
from datetime import datetime
import MetaTrader5 as mt5
from abc import ABC, abstractmethod
import talib

class Hypothesis(ABC):
    """الفئة الأساسية لجميع الفرضيات"""
    def __init__(self, name, description, weight=1.0):
        self.name = name
        self.description = description
        self.weight = weight
        self.signals = []
        
    @abstractmethod
    def evaluate(self, df, features):
        """تقييم الفرضية وإرجاع إشارة (-1: بيع، 0: حياد، 1: شراء)"""
        pass
        
    def get_confidence(self):
        """حساب مستوى الثقة للفرضية"""
        if not self.signals:
            return 0.0
        return np.mean(np.abs(self.signals))
        
class TrendFollowingHypothesis(Hypothesis):
    """فرضية اتباع الاتجاه"""
    def __init__(self):
        super().__init__(
            "Trend Following",
            "السوق يميل للاستمرار في اتجاهه الحالي",
            weight=1.5
        )
        
    def evaluate(self, df, features):
        signals = []
        
        # تحليل المتوسطات المتحركة
        if 'sma_50' in features.columns and 'sma_200' in features.columns:
            golden_cross = features['sma_50'] > features['sma_200']
            death_cross = features['sma_50'] < features['sma_200']
            
            signal = np.where(golden_cross, 1, np.where(death_cross, -1, 0))
            signals.append(signal[-1])
            
        # تحليل ADX للاتجاه القوي
        if 'adx_14' in features.columns:
            strong_trend = features['adx_14'] > 25
            if 'plus_di_14' in features.columns and 'minus_di_14' in features.columns:
                uptrend = features['plus_di_14'] > features['minus_di_14']
                signal = np.where(strong_trend & uptrend, 1, 
                                np.where(strong_trend & ~uptrend, -1, 0))
                signals.append(signal[-1])
                
        # تحليل MACD
        if 'macd' in features.columns and 'macd_signal' in features.columns:
            macd_bullish = features['macd'] > features['macd_signal']
            macd_bearish = features['macd'] < features['macd_signal']
            signal = np.where(macd_bullish, 1, np.where(macd_bearish, -1, 0))
            signals.append(signal[-1])
            
        self.signals = signals
        return np.mean(signals) if signals else 0
        
class MeanReversionHypothesis(Hypothesis):
    """فرضية العودة للمتوسط"""
    def __init__(self):
        super().__init__(
            "Mean Reversion",
            "الأسعار تميل للعودة إلى متوسطها",
            weight=1.2
        )
        
    def evaluate(self, df, features):
        signals = []
        
        # تحليل Bollinger Bands
        if all(col in features.columns for col in ['bb_upper', 'bb_lower', 'bb_position']):
            oversold = features['bb_position'] < 0.2
            overbought = features['bb_position'] > 0.8
            signal = np.where(oversold, 1, np.where(overbought, -1, 0))
            signals.append(signal[-1])
            
        # تحليل RSI
        if 'rsi_14' in features.columns:
            rsi_oversold = features['rsi_14'] < 30
            rsi_overbought = features['rsi_14'] > 70
            signal = np.where(rsi_oversold, 1, np.where(rsi_overbought, -1, 0))
            signals.append(signal[-1])
            
        # تحليل CCI
        if 'cci_14' in features.columns:
            cci_oversold = features['cci_14'] < -100
            cci_overbought = features['cci_14'] > 100
            signal = np.where(cci_oversold, 1, np.where(cci_overbought, -1, 0))
            signals.append(signal[-1])
            
        self.signals = signals
        return np.mean(signals) if signals else 0
        
class MomentumHypothesis(Hypothesis):
    """فرضية الزخم"""
    def __init__(self):
        super().__init__(
            "Momentum",
            "الأصول ذات الأداء القوي تستمر في الأداء الجيد",
            weight=1.3
        )
        
    def evaluate(self, df, features):
        signals = []
        
        # تحليل معدل التغيير
        if 'roc_10' in features.columns and 'roc_20' in features.columns:
            strong_momentum = (features['roc_10'] > 0) & (features['roc_20'] > 0)
            weak_momentum = (features['roc_10'] < 0) & (features['roc_20'] < 0)
            signal = np.where(strong_momentum, 1, np.where(weak_momentum, -1, 0))
            signals.append(signal[-1])
            
        # تحليل MFI
        if 'mfi_14' in features.columns:
            mfi_strong = features['mfi_14'] > 50
            mfi_weak = features['mfi_14'] < 50
            signal = np.where(mfi_strong, 1, np.where(mfi_weak, -1, 0))
            signals.append(signal[-1])
            
        # تحليل الزخم المخصص
        if 'momentum_oscillator' in features.columns:
            signal = np.where(features['momentum_oscillator'] > 50, 1,
                            np.where(features['momentum_oscillator'] < 50, -1, 0))
            signals.append(signal[-1])
            
        self.signals = signals
        return np.mean(signals) if signals else 0
        
class VolatilityBreakoutHypothesis(Hypothesis):
    """فرضية اختراق التذبذب"""
    def __init__(self):
        super().__init__(
            "Volatility Breakout",
            "اختراقات التذبذب تؤدي إلى حركات سعرية قوية",
            weight=1.4
        )
        
    def evaluate(self, df, features):
        signals = []
        
        # تحليل ATR
        if 'atr_14' in features.columns and 'atr_28' in features.columns:
            volatility_expansion = features['atr_14'] > features['atr_28']
            
            # دمج مع اتجاه السعر
            if 'price_change_5' in features.columns:
                bullish_breakout = volatility_expansion & (features['price_change_5'] > 0)
                bearish_breakout = volatility_expansion & (features['price_change_5'] < 0)
                signal = np.where(bullish_breakout, 1, np.where(bearish_breakout, -1, 0))
                signals.append(signal[-1])
                
        # تحليل عرض Bollinger Band
        if 'bb_width' in features.columns:
            # البحث عن الضغط (squeeze)
            bb_squeeze = features['bb_width'] < features['bb_width'].rolling(20).mean()
            if len(df) > 1:
                price_breakout = df['close'].iloc[-1] > df['close'].iloc[-2]
                signal = np.where(bb_squeeze & price_breakout, 1,
                                np.where(bb_squeeze & ~price_breakout, -1, 0))
                signals.append(signal[-1])
                
        self.signals = signals
        return np.mean(signals) if signals else 0
        
class SeasonalityHypothesis(Hypothesis):
    """فرضية الموسمية"""
    def __init__(self):
        super().__init__(
            "Seasonality",
            "الأسواق تظهر أنماطاً موسمية متكررة",
            weight=1.1
        )
        
    def evaluate(self, df, features):
        signals = []
        
        # تحليل الجلسات التداولية
        if all(col in features.columns for col in ['is_london_session', 'is_ny_session']):
            # الجلسات النشطة عادة ما تشهد حركات أقوى
            active_session = features['is_london_session'] | features['is_ny_session']
            
            # دمج مع الاتجاه
            if 'trend_strength' in features.columns:
                signal = np.where(active_session & (features['trend_strength'] > 0), 1,
                                np.where(active_session & (features['trend_strength'] < 0), -1, 0))
                signals.append(signal[-1])
                
        # تحليل أيام الأسبوع
        if 'day_of_week' in features.columns:
            # الإثنين والجمعة قد يكون لهما سلوك مختلف
            mid_week = features['day_of_week'].isin([1, 2, 3])  # الثلاثاء-الخميس
            
            if 'momentum_oscillator' in features.columns:
                signal = np.where(mid_week & (features['momentum_oscillator'] > 50), 1,
                                np.where(mid_week & (features['momentum_oscillator'] < 50), -1, 0))
                signals.append(signal[-1])
                
        self.signals = signals
        return np.mean(signals) if signals else 0
        
class SupportResistanceHypothesis(Hypothesis):
    """فرضية الدعم والمقاومة"""
    def __init__(self):
        super().__init__(
            "Support Resistance",
            "الأسعار تحترم مستويات الدعم والمقاومة",
            weight=1.3
        )
        
    def evaluate(self, df, features):
        signals = []
        
        # تحليل مستويات البيفوت
        if all(col in features.columns for col in ['pivot', 'r1', 's1']):
            current_price = df['close'].iloc[-1]
            
            # الموقع النسبي للسعر
            near_support = abs(current_price - features['s1'].iloc[-1]) < (features['atr_14'].iloc[-1] * 0.5)
            near_resistance = abs(current_price - features['r1'].iloc[-1]) < (features['atr_14'].iloc[-1] * 0.5)
            
            signal = np.where(near_support, 1, np.where(near_resistance, -1, 0))
            signals.append(signal)
            
        # تحليل القمم والقيعان
        if 'rolling_max_20' in features.columns and 'rolling_min_20' in features.columns:
            near_high = df['close'].iloc[-1] > features['rolling_max_20'].iloc[-1] * 0.98
            near_low = df['close'].iloc[-1] < features['rolling_min_20'].iloc[-1] * 1.02
            
            signal = np.where(near_low, 1, np.where(near_high, -1, 0))
            signals.append(signal)
            
        self.signals = signals
        return np.mean(signals) if signals else 0
        
class MarketStructureHypothesis(Hypothesis):
    """فرضية هيكل السوق"""
    def __init__(self):
        super().__init__(
            "Market Structure",
            "هيكل السوق يحدد الاتجاه المستقبلي",
            weight=1.4
        )
        
    def evaluate(self, df, features):
        signals = []
        
        # تحليل Higher Highs و Lower Lows
        if len(df) >= 20:
            highs = df['high'].rolling(5).max()
            lows = df['low'].rolling(5).min()
            
            # اتجاه صاعد: قمم وقيعان أعلى
            hh = highs.iloc[-1] > highs.iloc[-10]
            hl = lows.iloc[-1] > lows.iloc[-10]
            uptrend = hh and hl
            
            # اتجاه هابط: قمم وقيعان أقل
            lh = highs.iloc[-1] < highs.iloc[-10]
            ll = lows.iloc[-1] < lows.iloc[-10]
            downtrend = lh and ll
            
            signal = 1 if uptrend else (-1 if downtrend else 0)
            signals.append(signal)
            
        # تحليل الفجوات السعرية
        if len(df) > 1:
            gap_up = df['open'].iloc[-1] > df['close'].iloc[-2] * 1.001
            gap_down = df['open'].iloc[-1] < df['close'].iloc[-2] * 0.999
            
            signal = 1 if gap_up else (-1 if gap_down else 0)
            signals.append(signal)
            
        self.signals = signals
        return np.mean(signals) if signals else 0
        
class VolumeAnalysisHypothesis(Hypothesis):
    """فرضية تحليل الحجم"""
    def __init__(self):
        super().__init__(
            "Volume Analysis",
            "الحجم يؤكد حركة السعر",
            weight=1.2
        )
        
    def evaluate(self, df, features):
        signals = []
        
        # تحليل OBV
        if 'obv' in features.columns:
            obv_rising = features['obv'].iloc[-1] > features['obv'].iloc[-5]
            price_rising = df['close'].iloc[-1] > df['close'].iloc[-5]
            
            # التوافق بين السعر والحجم
            bullish_divergence = obv_rising and not price_rising
            bearish_divergence = not obv_rising and price_rising
            
            signal = np.where(bullish_divergence, 1, np.where(bearish_divergence, -1, 0))
            signals.append(signal)
            
        # تحليل نسبة الحجم
        if 'volume_ratio' in features.columns:
            high_volume = features['volume_ratio'] > 1.5
            
            if 'price_change_1' in features.columns:
                bullish_volume = high_volume & (features['price_change_1'] > 0)
                bearish_volume = high_volume & (features['price_change_1'] < 0)
                
                signal = np.where(bullish_volume, 1, np.where(bearish_volume, -1, 0))
                signals.append(signal[-1])
                
        self.signals = signals
        return np.mean(signals) if signals else 0
        
class PatternRecognitionHypothesis(Hypothesis):
    """فرضية التعرف على الأنماط"""
    def __init__(self):
        super().__init__(
            "Pattern Recognition",
            "الأنماط الفنية تتنبأ بالحركات المستقبلية",
            weight=1.5
        )
        
    def evaluate(self, df, features):
        signals = []
        
        # جمع إشارات الأنماط الشمعية
        pattern_columns = [col for col in features.columns if col.startswith('pattern_')]
        
        if pattern_columns:
            bullish_patterns = 0
            bearish_patterns = 0
            
            for col in pattern_columns:
                pattern_value = features[col].iloc[-1]
                if pattern_value > 0:
                    bullish_patterns += 1
                elif pattern_value < 0:
                    bearish_patterns += 1
                    
            # إشارة بناءً على عدد الأنماط
            if bullish_patterns > bearish_patterns + 2:
                signals.append(1)
            elif bearish_patterns > bullish_patterns + 2:
                signals.append(-1)
            else:
                signals.append(0)
                
        # تحليل أنماط السعر
        if len(df) >= 20:
            # نمط المثلث
            high_range = df['high'].iloc[-20:].max() - df['high'].iloc[-20:].min()
            low_range = df['low'].iloc[-20:].max() - df['low'].iloc[-20:].min()
            
            converging = high_range < df['high'].iloc[-40:-20].max() - df['high'].iloc[-40:-20].min()
            
            if converging:
                # اختراق المثلث
                breakout_up = df['close'].iloc[-1] > df['high'].iloc[-20:].max()
                breakout_down = df['close'].iloc[-1] < df['low'].iloc[-20:].min()
                
                signal = 1 if breakout_up else (-1 if breakout_down else 0)
                signals.append(signal)
                
        self.signals = signals
        return np.mean(signals) if signals else 0
        
class CorrelationHypothesis(Hypothesis):
    """فرضية الارتباط"""
    def __init__(self):
        super().__init__(
            "Correlation Analysis",
            "الأزواج المترابطة تتحرك معاً",
            weight=1.1
        )
        
    def evaluate(self, df, features):
        signals = []
        
        # تحليل قوة العملة
        symbol = df.index.name if hasattr(df.index, 'name') else 'UNKNOWN'
        
        # استخراج العملات من الرمز
        if len(symbol) >= 6:
            base_currency = symbol[:3]
            quote_currency = symbol[3:6]
            
            # تحليل بناءً على اتجاه الدولار
            if 'USD' in [base_currency, quote_currency]:
                # افتراض: إذا كان الدولار قوياً
                usd_strength = features['trend_strength'].iloc[-1] if 'trend_strength' in features.columns else 0
                
                if base_currency == 'USD':
                    signal = 1 if usd_strength > 0 else -1
                else:
                    signal = -1 if usd_strength > 0 else 1
                    
                signals.append(signal)
                
        self.signals = signals
        return np.mean(signals) if signals else 0
        
class HypothesisEngine:
    """محرك إدارة وتقييم الفرضيات"""
    def __init__(self):
        self.hypotheses = [
            TrendFollowingHypothesis(),
            MeanReversionHypothesis(),
            MomentumHypothesis(),
            VolatilityBreakoutHypothesis(),
            SeasonalityHypothesis(),
            SupportResistanceHypothesis(),
            MarketStructureHypothesis(),
            VolumeAnalysisHypothesis(),
            PatternRecognitionHypothesis(),
            CorrelationHypothesis()
        ]
        
        logging.info(f"Initialized Hypothesis Engine with {len(self.hypotheses)} hypotheses")
        
    def evaluate_all(self, df, features):
        """تقييم جميع الفرضيات"""
        results = {}
        weighted_signals = []
        
        for hypothesis in self.hypotheses:
            try:
                signal = hypothesis.evaluate(df, features)
                confidence = hypothesis.get_confidence()
                
                results[hypothesis.name] = {
                    'signal': signal,
                    'confidence': confidence,
                    'weight': hypothesis.weight,
                    'weighted_signal': signal * hypothesis.weight
                }
                
                weighted_signals.append(signal * hypothesis.weight)
                
            except Exception as e:
                logging.error(f"Error evaluating {hypothesis.name}: {str(e)}")
                results[hypothesis.name] = {
                    'signal': 0,
                    'confidence': 0,
                    'weight': hypothesis.weight,
                    'weighted_signal': 0
                }
                
        # حساب الإشارة النهائية
        final_signal = np.mean(weighted_signals)
        
        # تحديد القرار
        if final_signal > 0.3:
            decision = 'BUY'
            action = 0
        elif final_signal < -0.3:
            decision = 'SELL'
            action = 1
        else:
            decision = 'HOLD'
            action = 2
            
        return {
            'decision': decision,
            'action': action,
            'final_signal': final_signal,
            'hypothesis_results': results,
            'confidence': abs(final_signal)
        }
        
    def get_hypothesis_summary(self, results):
        """الحصول على ملخص نتائج الفرضيات"""
        summary = []
        
        for name, data in results['hypothesis_results'].items():
            if data['signal'] != 0:
                direction = 'Bullish' if data['signal'] > 0 else 'Bearish'
                summary.append(f"{name}: {direction} (Confidence: {data['confidence']:.1%})")
                
        return summary
        
    def integrate_with_ml_predictions(self, hypothesis_results, ml_predictions, ml_probabilities):
        """دمج نتائج الفرضيات مع تنبؤات ML"""
        # وزن نتائج الفرضيات
        hypothesis_weight = 0.3
        ml_weight = 0.7
        
        # تحويل قرار ML إلى إشارة
        ml_signal = 0
        if ml_predictions == 0:  # Buy
            ml_signal = 1
        elif ml_predictions == 1:  # Sell
            ml_signal = -1
        else:  # Hold
            ml_signal = 0
            
        # حساب الثقة من احتماليات ML
        ml_confidence = np.max(ml_probabilities)
        
        # دمج الإشارات
        combined_signal = (hypothesis_results['final_signal'] * hypothesis_weight + 
                          ml_signal * ml_confidence * ml_weight)
        
        # القرار النهائي
        if combined_signal > 0.2:
            final_decision = 'BUY'
            final_action = 0
        elif combined_signal < -0.2:
            final_decision = 'SELL'
            final_action = 1
        else:
            final_decision = 'HOLD'
            final_action = 2
            
        return {
            'final_decision': final_decision,
            'final_action': final_action,
            'combined_signal': combined_signal,
            'hypothesis_contribution': hypothesis_results['final_signal'] * hypothesis_weight,
            'ml_contribution': ml_signal * ml_confidence * ml_weight,
            'total_confidence': (hypothesis_results['confidence'] * hypothesis_weight + 
                               ml_confidence * ml_weight)
        }