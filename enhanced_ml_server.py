#!/usr/bin/env python3
"""
🚀 Enhanced ML Server - نظام التداول الذكي المتكامل
📊 يجمع بين: تحليل السوق + إدارة المخاطر + 6 نماذج ML
💰 نظام قادر على تحقيق الربح المستدام
"""

import os
import sys
import json
import logging
import threading
import sqlite3
import joblib
import numpy as np
import pandas as pd
import warnings
import time
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS

# إخفاء تحذيرات الأداء
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', message='DataFrame is highly fragmented')

# ML Libraries
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, f_classif

# Import our custom systems
from market_analysis_engine import MarketAnalysisEngine
from risk_management_system import RiskManagementSystem

# Optional libraries
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except:
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('enhanced_ml_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class EnhancedMLTradingSystem:
    """النظام المتكامل للتداول الذكي"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        # استخدام القاعدة الرئيسية الكبيرة
        self.db_path = './data/forex_ml.db'  # 910 MB - القاعدة الرئيسية
        # القاعدة الثانوية للاختبار: './data/forex_data.db'
        self.models_dir = './trained_models'
        
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize subsystems
        self.market_analyzer = MarketAnalysisEngine(self.db_path)
        self.risk_manager = RiskManagementSystem(initial_balance=10000)
        
        # Trading parameters - محدثة للربحية
        self.min_confidence = 0.70  # رفع من 0.55 إلى 0.70
        self.min_market_score = 40  # رفع من 20 إلى 40
        self.max_daily_trades = 10  # حد أقصى للصفقات اليومية
        self.trade_cooldown = {}    # فترة انتظار بين الصفقات
        
        # Performance tracking
        self.performance_tracker = {
            'predictions': [],
            'trades': [],
            'daily_stats': {},
            'model_performance': {}  # أداء كل نموذج
        }
        
        # Feature selectors
        self.feature_selectors = {}
        
        # Position management
        self.open_positions = {}
        self.trailing_stop_levels = {}
        
        # Load existing models
        self.load_existing_models()
        
        # عد النماذج
        total_models = sum(len(models) for models in self.models.values())
        enhanced_models = sum(1 for key, models in self.models.items() 
                            for model_name in models.keys() 
                            if any(m in model_name for m in ['neural', 'lightgbm', 'xgboost']))
        
        logger.info(f"✅ Enhanced ML Trading System initialized")
        logger.info(f"   📊 Market Analyzer: Ready")
        logger.info(f"   💰 Risk Manager: Balance ${self.risk_manager.current_balance}")
        logger.info(f"   🤖 ML Models: {total_models} loaded ({len(self.models)} pairs)")
        if enhanced_models > 0:
            logger.info(f"   💪 Using advanced models (6-model ensemble)")
    
    def calculate_enhanced_features(self, df, market_context=None):
        """حساب الميزات المحسنة مع سياق السوق"""
        # Get base features (200+)
        from advanced_ml_server import AdvancedMLSystem
        base_system = AdvancedMLSystem()
        features = base_system.calculate_advanced_features(df)
        
        # Add market context features if available
        if market_context:
            # Trend features
            trend = market_context.get('trend', {})
            features['trend_score'] = trend.get('overall_score', 0)
            features['trend_alignment'] = 1 if trend.get('alignment') else 0
            features['trend_strength'] = trend.get('strength', 0)
            
            # Support/Resistance features
            sr = market_context.get('support_resistance', {})
            if sr.get('nearest_support'):
                features['distance_to_support'] = (df['close'].iloc[-1] - sr['nearest_support']['price']) / df['close'].iloc[-1]
                features['support_strength'] = sr['nearest_support']['strength']
            
            if sr.get('nearest_resistance'):
                features['distance_to_resistance'] = (sr['nearest_resistance']['price'] - df['close'].iloc[-1]) / df['close'].iloc[-1]
                features['resistance_strength'] = sr['nearest_resistance']['strength']
            
            # Volume features
            volume = market_context.get('volume', {})
            features['volume_ratio'] = volume.get('volume_ratio', 1)
            features['volume_signal_score'] = self._score_volume_signal(volume.get('volume_signal'))
            
            # Session features
            session = market_context.get('session', {})
            features['session_quality_score'] = self._score_session_quality(session.get('session_quality'))
            features['is_news_time'] = 1 if session.get('is_news_time') else 0
            
            # Momentum features
            momentum = market_context.get('momentum', {})
            features['momentum_score'] = momentum.get('score', 0)
            features['rsi_normalized'] = momentum.get('rsi', 50) / 100
            
            # Pattern features
            patterns = market_context.get('patterns', [])
            features['bullish_patterns'] = sum(1 for p in patterns if p['direction'] == 'bullish')
            features['bearish_patterns'] = sum(1 for p in patterns if p['direction'] == 'bearish')
            
            # Volatility features
            volatility = market_context.get('volatility', {})
            features['volatility_level_score'] = self._score_volatility_level(volatility.get('volatility_level'))
            
            # Overall market score
            features['market_score'] = market_context.get('score', 0)
        
        # Ensure all values are scalar (not arrays or lists)
        for col in features.columns:
            try:
                if len(features) > 0 and isinstance(features[col].iloc[-1], (list, np.ndarray)):
                    features[col] = features[col].apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) and len(x) > 0 else 0)
            except IndexError:
                continue
        
        return features
    
    def _score_volume_signal(self, signal):
        """تحويل إشارة الحجم لقيمة رقمية"""
        scores = {
            'BULLISH_CONFIRMATION': 2,
            'BEARISH_CONFIRMATION': -2,
            'BULLISH_DIVERGENCE': -1,
            'BEARISH_DIVERGENCE': 1,
            'NEUTRAL': 0
        }
        return scores.get(signal, 0)
    
    def _score_session_quality(self, quality):
        """تحويل جودة الجلسة لقيمة رقمية"""
        scores = {
            'EXCELLENT': 3,
            'GOOD': 2,
            'MODERATE': 1,
            'LOW': 0
        }
        return scores.get(quality, 1)
    
    def _score_volatility_level(self, level):
        """تحويل مستوى التقلب لقيمة رقمية"""
        scores = {
            'VERY_HIGH': -2,
            'HIGH': -1,
            'NORMAL': 0,
            'LOW': 1,
            'VERY_LOW': -1
        }
        return scores.get(level, 0)
    
    def calculate_trading_metrics(self, y_true, y_pred, prices, lot_sizes=None):
        """حساب مقاييس التداول بدلاً من accuracy"""
        if lot_sizes is None:
            lot_sizes = np.ones(len(y_true))
        
        # حساب العوائد
        returns = []
        wins = []
        losses = []
        
        for i in range(len(y_true) - 1):
            if y_pred[i] == 1:  # Buy signal
                ret = prices[i+1] - prices[i]
            else:  # Sell signal
                ret = prices[i] - prices[i+1]
            
            returns.append(ret * lot_sizes[i])
            if ret > 0:
                wins.append(ret * lot_sizes[i])
            elif ret < 0:
                losses.append(abs(ret * lot_sizes[i]))
        
        # حساب المقاييس
        total_return = sum(returns) if returns else 0
        num_trades = len(returns)
        num_wins = len(wins)
        num_losses = len(losses)
        
        # Profit Factor
        profit_factor = sum(wins) / sum(losses) if losses and sum(losses) > 0 else float('inf') if wins else 0
        
        # Win Rate
        win_rate = num_wins / num_trades if num_trades > 0 else 0
        
        # Expected Return
        expected_return = total_return / num_trades if num_trades > 0 else 0
        
        # Risk-Reward Ratio
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        risk_reward = avg_win / avg_loss if avg_loss > 0 else float('inf')
        
        # Maximum Drawdown
        if returns:
            cumulative = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = np.where(running_max > 0, (cumulative - running_max) / running_max, 0)
            max_drawdown = abs(np.min(drawdown))
        else:
            max_drawdown = 0
        
        return {
            'total_return': total_return,
            'profit_factor': profit_factor,
            'win_rate': win_rate,
            'expected_return': expected_return,
            'risk_reward_ratio': risk_reward,
            'max_drawdown': max_drawdown,
            'num_trades': num_trades
        }
    
    def calculate_kelly_position(self, symbol, win_rate, avg_win, avg_loss):
        """حساب حجم الصفقة باستخدام معادلة Kelly"""
        if avg_win <= 0 or win_rate <= 0:
            return 0.005  # الحد الأدنى 0.5%
        
        # Kelly formula
        loss_rate = 1 - win_rate
        kelly_percentage = (win_rate * avg_win - loss_rate * avg_loss) / avg_win
        
        # تطبيق الحدود (0.5% - 2%)
        kelly_percentage = max(0.005, min(0.02, kelly_percentage))
        
        # حساب حجم اللوت
        balance = self.risk_manager.current_balance
        risk_amount = balance * kelly_percentage
        
        return risk_amount
    
    def update_stop_loss(self, symbol, position_info):
        """نظام Trailing Stop Loss"""
        current_price = position_info['current_price']
        entry_price = position_info['entry_price']
        current_sl = position_info['stop_loss']
        is_buy = position_info['direction'] == 'BUY'
        
        # حساب الربح الحالي
        if is_buy:
            profit_pct = (current_price - entry_price) / entry_price
            
            # Trailing rules for BUY
            if profit_pct > 0.03:  # ربح > 3%
                new_sl = entry_price * 1.02  # SL عند +2%
            elif profit_pct > 0.02:  # ربح > 2%
                new_sl = entry_price * 1.01  # SL عند +1%
            elif profit_pct > 0.01:  # ربح > 1%
                new_sl = entry_price  # SL عند نقطة الدخول
            else:
                return current_sl  # لا تغيير
            
            # تحديث فقط إذا كان SL الجديد أفضل
            if new_sl > current_sl:
                return new_sl
        else:
            profit_pct = (entry_price - current_price) / entry_price
            
            # Trailing rules for SELL
            if profit_pct > 0.03:  # ربح > 3%
                new_sl = entry_price * 0.98  # SL عند -2%
            elif profit_pct > 0.02:  # ربح > 2%
                new_sl = entry_price * 0.99  # SL عند -1%
            elif profit_pct > 0.01:  # ربح > 1%
                new_sl = entry_price  # SL عند نقطة الدخول
            else:
                return current_sl  # لا تغيير
            
            # تحديث فقط إذا كان SL الجديد أفضل
            if new_sl < current_sl:
                return new_sl
        
        return current_sl
    
    def check_entry_conditions(self, market_context, prediction_confidence):
        """فحص شروط الدخول المحسنة - يجب تحقق 3 من 4 شروط"""
        conditions_met = 0
        conditions_details = []
        
        # 1. Volume أعلى من المتوسط بـ 50%
        volume_ratio = market_context.get('volume', {}).get('volume_ratio', 1)
        if volume_ratio > 1.5:
            conditions_met += 1
            conditions_details.append("High volume")
        
        # 2. RSI بين 30-70
        rsi = market_context.get('momentum', {}).get('rsi', 50)
        if 30 <= rsi <= 70:
            conditions_met += 1
            conditions_details.append("Good RSI")
        
        # 3. Trend alignment
        trend_alignment = market_context.get('trend', {}).get('alignment', False)
        if trend_alignment:
            conditions_met += 1
            conditions_details.append("Trend aligned")
        
        # 4. No news time
        is_news_time = market_context.get('session', {}).get('is_news_time', False)
        if not is_news_time:
            conditions_met += 1
            conditions_details.append("No news")
        
        # حفظ عدد الشروط المحققة
        self.performance_tracker['last_conditions_met'] = conditions_met
        
        # يجب تحقق 3 من 4 شروط على الأقل
        if conditions_met >= 3:
            logger.info(f"   ✅ Entry conditions met ({conditions_met}/4): {', '.join(conditions_details)}")
            return True
        else:
            logger.info(f"   ❌ Entry conditions not met ({conditions_met}/4): Missing - {', '.join([c for c in ['High volume', 'Good RSI', 'Trend aligned', 'No news'] if c not in conditions_details])}")
            return False
    
    def calculate_model_weights(self):
        """حساب أوزان النماذج بناءً على الأداء"""
        if 'model_weights' not in self.performance_tracker:
            self.performance_tracker['model_weights'] = {}
            
        for key, performances in self.performance_tracker['model_performance'].items():
            if isinstance(performances, list) and len(performances) > 0:
                # حساب متوسط الأداء لكل نموذج
                model_scores = {}
                
                for perf in performances[-100:]:  # آخر 100 صفقة
                    model = perf.get('model')
                    profit = perf.get('profit', 0)
                    
                    if model not in model_scores:
                        model_scores[model] = []
                    model_scores[model].append(profit)
                
                # حساب الوزن لكل نموذج
                weights = {}
                for model, profits in model_scores.items():
                    avg_profit = np.mean(profits)
                    # النموذج الخاسر وزنه 0.5، الرابح 1.5
                    weight = 1.5 if avg_profit > 0 else 0.5
                    weights[model] = weight
                
                self.performance_tracker['model_weights'][key] = weights
    
    def predict_with_weighted_ensemble(self, symbol, timeframe, df):
        """التنبؤ مع تصويت مرجح بناءً على أداء النماذج"""
        try:
            # تحليل السوق
            start_time = time.time()
            market_context = self.market_analyzer.analyze_complete_market_context(
                symbol, df.reset_index().to_dict('records'), timeframe
            )
            
            if time.time() - start_time > 5:
                logger.warning(f"Market analysis took too long: {time.time() - start_time:.1f}s")
                return self._simple_prediction(df)
            
            if not market_context:
                logger.warning("Failed to analyze market context")
                return self._simple_prediction(df)
            
            # فحص شروط السوق
            market_score = market_context['score']
            logger.info(f"   📊 Market Score: {market_score}")
            
            # رفض الإشارات الضعيفة
            if abs(market_score) < self.min_market_score:
                entry_reason = f"Weak market score: {market_score} (min required: ±{self.min_market_score})"
                logger.info(f"   ❌ {entry_reason}")
                return {
                    'action': 2,
                    'direction': 'HOLD',
                    'confidence': 0.0,
                    'market_context': market_context,
                    'reason': 'Weak market conditions',
                    'entry_reason': entry_reason
                }
            
            # حساب الميزات
            features = self.calculate_enhanced_features(df, market_context)
            if features.empty:
                return self._simple_prediction(df)
            
            # تطبيق feature selection و scaling
            key = f"{symbol}_{timeframe}"
            
            if key not in self.models:
                logger.warning(f"No models for {key}")
                return self._simple_prediction(df)
            
            # Feature selection إذا كان متاحًا
            X = features.values[-1:]
            if key in self.feature_selectors:
                try:
                    X = self.feature_selectors[key].transform(X)
                except:
                    pass  # استخدم الميزات الأصلية إذا فشل
            
            # Scaling
            if key in self.scalers:
                X_scaled = self.scalers[key].transform(X)
            else:
                X_scaled = X
            
            # التنبؤ من كل نموذج
            predictions = []
            confidences = []
            weights = []
            model_names = []
            
            # الحصول على أوزان النماذج
            model_weights = self.performance_tracker.get('model_weights', {}).get(key, {})
            
            for model_name, model in self.models[key].items():
                try:
                    pred = model.predict(X_scaled)[0]
                    prob = model.predict_proba(X_scaled)[0]
                    confidence = max(prob)
                    
                    predictions.append(pred)
                    confidences.append(confidence)
                    
                    # الوزن الافتراضي 1، أو من الأداء السابق
                    weight = model_weights.get(model_name, 1.0)
                    weights.append(weight)
                    
                    model_names.append(model_name)
                    logger.info(f"      {model_name}: {'BUY' if pred == 1 else 'SELL'} ({confidence:.2%}) - Weight: {weight:.2f}")
                    
                except Exception as e:
                    logger.error(f"      ❌ {model_name}: {e}")
            
            if not predictions:
                return self._simple_prediction(df)
            
            # تصويت مرجح
            weighted_sum = sum(p * w for p, w in zip(predictions, weights))
            total_weight = sum(weights)
            weighted_avg = weighted_sum / total_weight
            
            # القرار النهائي
            if weighted_avg > 0.5:
                direction = 'BUY'
                action = 0
            else:
                direction = 'SELL'
                action = 1
            
            # حساب الثقة المرجحة
            weighted_confidence = sum(c * w for c, w in zip(confidences, weights)) / total_weight
            
            # فحص شروط الدخول الإضافية
            if weighted_confidence >= self.min_confidence:
                if self.check_entry_conditions(market_context, weighted_confidence):
                    entry_reason = f"Strong signal: {direction} - Confidence: {weighted_confidence:.1%}, Market Score: {market_score}, Models agree: {sum(1 for p in predictions if p == (1 if direction == 'BUY' else 0))}/{len(predictions)}"
                    logger.info(f"   ✅ {entry_reason}")
                    
                    return {
                        'action': action,
                        'direction': direction,
                        'confidence': weighted_confidence,
                        'market_context': market_context,
                        'models_used': model_names,
                        'weighted_score': weighted_avg,
                        'buy_votes': sum(1 for p in predictions if p == 1),
                        'total_votes': len(predictions),
                        'entry_reason': entry_reason
                    }
                else:
                    entry_reason = f"Entry conditions not met (only {self.performance_tracker.get('last_conditions_met', 0)}/4 conditions satisfied)"
                    logger.info(f"   ❌ {entry_reason}")
                    return {
                        'action': 2,
                        'direction': 'HOLD',
                        'confidence': weighted_confidence,
                        'market_context': market_context,
                        'reason': 'Entry conditions not met',
                        'entry_reason': entry_reason
                    }
            else:
                entry_reason = f"Low confidence: {weighted_confidence:.1%} < {self.min_confidence:.1%} (min required)"
                logger.info(f"   ❌ {entry_reason}")
                return {
                    'action': 2,
                    'direction': 'HOLD',
                    'confidence': weighted_confidence,
                    'market_context': market_context,
                    'reason': 'Low confidence',
                    'entry_reason': entry_reason
                }
                
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return self._simple_prediction(df)
    
    def train_enhanced_models(self, symbol, timeframe):
        """تدريب النماذج مع الميزات المحسنة"""
        try:
            logger.info(f"🤖 Training enhanced models for {symbol} {timeframe}...")
            
            # Load data
            conn = sqlite3.connect(self.db_path)
            
            # Try different symbol formats
            possible_symbols = [
                symbol,
                f"{symbol}m",
                f"{symbol}.ecn",
                symbol.lower(),
                f"{symbol.lower()}m"
            ]
            
            df = None
            for sym in possible_symbols:
                # استخدام parameterized query لتجنب SQL injection
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM price_data 
                    WHERE symbol = ? OR symbol LIKE ?
                    ORDER BY time ASC
                    LIMIT 30000
                """, (sym, f"{sym}%"))
                
                data = cursor.fetchall()
                if data:
                    columns = [desc[0] for desc in cursor.description]
                    df = pd.DataFrame(data, columns=columns)
                    logger.info(f"Found {len(df)} records with symbol: {sym}")
                    break
            
            conn.close()
            
            if df is None or len(df) < 2000:
                logger.warning(f"Not enough data for {symbol}")
                return False
            
            # Prepare data
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            df = df.sort_index()
            
            for col in ['open', 'high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna()
            
            # Create training dataset with market context
            logger.info("   📊 Analyzing market context for training data...")
            
            X_list = []
            y_list = []
            market_scores = []
            prices_list = []  # لحساب المقاييس التداولية
            
            # Process data in chunks
            chunk_size = 500
            for i in range(1000, len(df) - 100, 50):  # Skip 50 candles for efficiency
                # Get chunk of data
                chunk = df.iloc[i-chunk_size:i]
                
                # Analyze market context
                market_context = self.market_analyzer.analyze_complete_market_context(
                    symbol, chunk.reset_index().to_dict('records'), timeframe
                )
                
                if market_context:
                    # Calculate features with context
                    features = self.calculate_enhanced_features(chunk, market_context)
                    
                    if not features.empty and len(features) > 0:
                        # Get the last feature row and ensure it's 1D
                        feature_values = features.iloc[-1].values
                        
                        # Ensure consistent shape
                        if isinstance(feature_values, np.ndarray) and feature_values.ndim == 1:
                            X_list.append(feature_values)
                            
                            # Target: price movement in next candle
                            future_return = (df['close'].iloc[i+1] - df['close'].iloc[i]) / df['close'].iloc[i]
                            y_list.append(1 if future_return > 0 else 0)
                            
                            # Store market score for filtering
                            market_scores.append(market_context['score'])
                            
                            # Store price for metrics calculation
                            prices_list.append(df['close'].iloc[i])
            
            if len(X_list) < 100:  # خفضنا من 500 إلى 100
                logger.warning(f"Not enough training samples: {len(X_list)} < 100")
                return False
            
            logger.info(f"   ✅ Prepared {len(X_list)} training samples")
            
            # Check feature consistency
            if len(X_list) > 0:
                feature_counts = [len(x) for x in X_list]
                if len(set(feature_counts)) > 1:
                    logger.warning(f"Inconsistent feature sizes: {set(feature_counts)}")
                    # Keep only samples with the most common feature count
                    most_common_size = max(set(feature_counts), key=feature_counts.count)
                    valid_indices = [i for i, x in enumerate(X_list) if len(x) == most_common_size]
                    
                    X_list = [X_list[i] for i in valid_indices]
                    y_list = [y_list[i] for i in valid_indices]
                    market_scores = [market_scores[i] for i in valid_indices]
                    
                    logger.info(f"   📊 Filtered to {len(X_list)} samples with {most_common_size} features")
            
            # Convert to arrays
            try:
                X = np.array(X_list)
                y = np.array(y_list)
                market_scores = np.array(market_scores)
                prices = np.array(prices_list)
            except Exception as e:
                logger.error(f"Error converting to arrays: {e}")
                return False
            
            logger.info(f"   📊 Created {len(X)} training samples")
            
            # Filter high-quality samples (market score > 40 or < -40)
            quality_mask = np.abs(market_scores) > 40  # رفع المعيار من 20 إلى 40
            X_quality = X[quality_mask]
            y_quality = y[quality_mask]
            prices_quality = prices[quality_mask]
            
            logger.info(f"   ✨ {len(X_quality)} high-quality samples")
            
            # Feature Selection - اختيار أفضل 50 feature
            selector = SelectKBest(f_classif, k=min(50, X_quality.shape[1]))
            X_selected = selector.fit_transform(X_quality, y_quality)
            
            # حفظ selector
            key = f"{symbol}_{timeframe}"
            self.feature_selectors[key] = selector
            
            # Time Series Split بدلاً من random split
            tscv = TimeSeriesSplit(n_splits=5)
            
            best_models = {}
            model_performances = []
            
            for fold, (train_idx, test_idx) in enumerate(tscv.split(X_selected)):
                X_train, X_test = X_selected[train_idx], X_selected[test_idx]
                y_train, y_test = y_quality[train_idx], y_quality[test_idx]
                prices_train = prices_quality[train_idx]
                prices_test = prices_quality[test_idx]
                
                # Scale
                scaler = RobustScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Save scaler for last fold
                if fold == 4:  # آخر fold
                    key = f"{symbol}_{timeframe}"
                    self.scalers[key] = scaler
                
                # التدريب على هذا fold
                logger.info(f"   📊 Training fold {fold+1}/5...")
                
                # 1. Random Forest
                try:
                    rf = RandomForestClassifier(
                        n_estimators=200,
                        max_depth=15,
                        min_samples_split=5,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=-1
                    )
                    rf.fit(X_train_scaled, y_train)
                    
                    # تقييم بمقاييس التداول
                    y_pred = rf.predict(X_test_scaled)
                    metrics = self.calculate_trading_metrics(y_test, y_pred, prices_test)
                    
                    if metrics['total_return'] > 0 and metrics['profit_factor'] > 1:
                        model_performances.append({
                            'model': 'random_forest',
                            'fold': fold,
                            'metrics': metrics,
                            'instance': rf
                        })
                        logger.info(f"   ✅ RF: Return={metrics['total_return']:.2f}, PF={metrics['profit_factor']:.2f}")
                    else:
                        logger.info(f"   ❌ RF: Unprofitable (Return={metrics['total_return']:.2f})")
                        
                except Exception as e:
                    logger.error(f"   ❌ Random Forest: {e}")
            
                # 2. Gradient Boosting
                try:
                    gb = GradientBoostingClassifier(
                        n_estimators=150,
                        learning_rate=0.05,
                        max_depth=6,
                        min_samples_split=5,
                        random_state=42
                    )
                    gb.fit(X_train_scaled, y_train)
                    
                    y_pred = gb.predict(X_test_scaled)
                    metrics = self.calculate_trading_metrics(y_test, y_pred, prices_test)
                    
                    if metrics['total_return'] > 0 and metrics['profit_factor'] > 1:
                        model_performances.append({
                            'model': 'gradient_boosting',
                            'fold': fold,
                            'metrics': metrics,
                            'instance': gb
                        })
                        logger.info(f"   ✅ GB: Return={metrics['total_return']:.2f}, PF={metrics['profit_factor']:.2f}")
                    else:
                        logger.info(f"   ❌ GB: Unprofitable")
                        
                except Exception as e:
                    logger.error(f"   ❌ Gradient Boosting: {e}")
            
            # 3. Extra Trees
            try:
                et = ExtraTreesClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    random_state=42,
                    n_jobs=-1
                )
                et.fit(X_train_scaled, y_train)
                y_pred = et.predict(X_test_scaled)
                metrics = self.calculate_trading_metrics(y_test, y_pred, prices_test)
                
                if metrics['total_return'] > 0 and metrics['profit_factor'] > 1:
                    model_performances.append({
                        'model': 'extra_trees',
                        'fold': fold,
                        'metrics': metrics,
                        'instance': et
                    })
                    logger.info(f"   ✅ Extra Trees: Return={metrics['total_return']:.2f}, PF={metrics['profit_factor']:.2f}")
                else:
                    logger.info(f"   ❌ Extra Trees: Unprofitable")
            except Exception as e:
                logger.error(f"   ❌ Extra Trees: {e}")
            
            # 4. Neural Network
            try:
                nn = MLPClassifier(
                    hidden_layer_sizes=(200, 100, 50),
                    activation='relu',
                    solver='adam',
                    alpha=0.001,
                    batch_size=32,
                    learning_rate='adaptive',
                    max_iter=1000,
                    random_state=42
                )
                nn.fit(X_train_scaled, y_train)
                y_pred = nn.predict(X_test_scaled)
                metrics = self.calculate_trading_metrics(y_test, y_pred, prices_test)
                
                if metrics['total_return'] > 0 and metrics['profit_factor'] > 1:
                    model_performances.append({
                        'model': 'neural_network',
                        'fold': fold,
                        'metrics': metrics,
                        'instance': nn
                    })
                    logger.info(f"   ✅ Neural Network: Return={metrics['total_return']:.2f}, PF={metrics['profit_factor']:.2f}")
                else:
                    logger.info(f"   ❌ Neural Network: Unprofitable")
            except Exception as e:
                logger.error(f"   ❌ Neural Network: {e}")
            
            # 5. LightGBM
            if LIGHTGBM_AVAILABLE:
                try:
                    lgbm = lgb.LGBMClassifier(
                        n_estimators=200,
                        num_leaves=50,
                        learning_rate=0.05,
                        feature_fraction=0.8,
                        bagging_fraction=0.8,
                        bagging_freq=5,
                        random_state=42,
                        verbosity=-1
                    )
                    lgbm.fit(X_train_scaled, y_train)
                    y_pred = lgbm.predict(X_test_scaled)
                    metrics = self.calculate_trading_metrics(y_test, y_pred, prices_test)
                    
                    if metrics['total_return'] > 0 and metrics['profit_factor'] > 1:
                        model_performances.append({
                            'model': 'lightgbm',
                            'fold': fold,
                            'metrics': metrics,
                            'instance': lgbm
                        })
                        logger.info(f"   ✅ LightGBM: Return={metrics['total_return']:.2f}, PF={metrics['profit_factor']:.2f}")
                    else:
                        logger.info(f"   ❌ LightGBM: Unprofitable")
                except Exception as e:
                    logger.error(f"   ❌ LightGBM: {e}")
            
            # 6. XGBoost
            if XGBOOST_AVAILABLE:
                try:
                    xgb_model = xgb.XGBClassifier(
                        n_estimators=200,
                        max_depth=6,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42,
                        use_label_encoder=False
                    )
                    xgb_model.fit(X_train_scaled, y_train)
                    y_pred = xgb_model.predict(X_test_scaled)
                    metrics = self.calculate_trading_metrics(y_test, y_pred, prices_test)
                    
                    if metrics['total_return'] > 0 and metrics['profit_factor'] > 1:
                        model_performances.append({
                            'model': 'xgboost',
                            'fold': fold,
                            'metrics': metrics,
                            'instance': xgb_model
                        })
                        logger.info(f"   ✅ XGBoost: Return={metrics['total_return']:.2f}, PF={metrics['profit_factor']:.2f}")
                    else:
                        logger.info(f"   ❌ XGBoost: Unprofitable")
                except Exception as e:
                    logger.error(f"   ❌ XGBoost: {e}")
            
            # Save models
            self.models[key] = models
            
            # Save to disk
            logger.info(f"   💾 Saving {len(models)} models to disk...")
            saved_count = 0
            
            for model_name, model in models.items():
                try:
                    model_path = os.path.join(self.models_dir, f"{symbol}_{timeframe}_{model_name}_enhanced.pkl")
                    joblib.dump(model, model_path)
                    saved_count += 1
                    logger.info(f"      ✅ Saved {model_name}")
                except Exception as e:
                    logger.error(f"      ❌ Failed to save {model_name}: {e}")
            
            try:
                scaler_path = os.path.join(self.models_dir, f"{symbol}_{timeframe}_scaler_enhanced.pkl")
                joblib.dump(scaler, scaler_path)
                logger.info(f"      ✅ Saved scaler")
            except Exception as e:
                logger.error(f"      ❌ Failed to save scaler: {e}")
            
            logger.info(f"   ✅ Successfully saved {saved_count}/{len(models)} models for {symbol} {timeframe}!")
            
            # Calculate ensemble accuracy
            ensemble_preds = []
            for model in models.values():
                ensemble_preds.append(model.predict(X_test_scaled))
            
            ensemble_pred = np.round(np.mean(ensemble_preds, axis=0))
            ensemble_acc = accuracy_score(y_test, ensemble_pred)
            
            logger.info(f"   🎯 Ensemble Accuracy: {ensemble_acc:.4f}")
            logger.info(f"✅ Trained {len(models)} enhanced models for {symbol} {timeframe}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Training error: {str(e)}")
            return False
    
    def predict_with_context(self, symbol, timeframe, df):
        """التنبؤ مع تحليل السياق الكامل"""
        try:
            # 1. Analyze market context - with timeout protection
            start_time = time.time()
            market_context = self.market_analyzer.analyze_complete_market_context(
                symbol, df.reset_index().to_dict('records'), timeframe
            )
            
            # Check if analysis took too long
            if time.time() - start_time > 5:  # 5 seconds max for analysis
                logger.warning(f"Market analysis took too long: {time.time() - start_time:.1f}s")
                return self._simple_prediction(df)
            
            if not market_context:
                logger.warning("Failed to analyze market context")
                return self._simple_prediction(df)
            
            # 2. Check if market conditions are favorable
            market_score = market_context['score']
            market_strength = market_context['strength']
            
            logger.info(f"   📊 Market Analysis: Score={market_score}, Strength={market_strength}")
            
            # Log important market conditions
            if market_context['session']['is_news_time']:
                logger.info("   📰 News time detected")
            
            if market_context['volatility']['volatility_level'] == 'VERY_HIGH':
                logger.warning("   ⚠️ Very high volatility")
            
            # 3. Calculate enhanced features
            features = self.calculate_enhanced_features(df, market_context)
            
            if features.empty:
                return self._simple_prediction(df)
            
            # 4. Get ML predictions
            key = f"{symbol}_{timeframe}"
            
            # Train if no model exists
            if key not in self.models:
                logger.warning(f"   ⚠️ No models found for {key}")
                logger.info(f"   🤖 Available models: {list(self.models.keys())}")
                return self._simple_prediction(df)
            
            # Get predictions from all models
            X = features.values[-1:]
            
            # Scale
            if key in self.scalers:
                X_scaled = self.scalers[key].transform(X)
            else:
                X_scaled = X
            
            predictions = []
            confidences = []
            model_names = []
            
            for model_name, model in self.models[key].items():
                try:
                    pred = model.predict(X_scaled)[0]
                    predictions.append(pred)
                    
                    if hasattr(model, 'predict_proba'):
                        prob = model.predict_proba(X_scaled)[0]
                        confidences.append(max(prob))
                    else:
                        confidences.append(0.6)
                    
                    model_names.append(model_name)
                    
                except Exception as e:
                    logger.error(f"Prediction error {model_name}: {e}")
            
            if not predictions:
                return self._simple_prediction(df)
            
            # 5. Ensemble voting with market context weight
            buy_votes = sum(1 for p in predictions if p == 1)
            total_votes = len(predictions)
            buy_ratio = buy_votes / total_votes
            
            # Adjust confidence based on market context
            base_confidence = np.mean(confidences)
            
            # Market score adjustment
            if abs(market_score) >= 70:
                confidence_multiplier = 1.3
            elif abs(market_score) >= 50:
                confidence_multiplier = 1.15
            elif abs(market_score) >= 30:
                confidence_multiplier = 1.0
            else:
                confidence_multiplier = 0.8
            
            # Session quality adjustment
            session_quality = market_context['session']['session_quality']
            if session_quality == 'EXCELLENT':
                confidence_multiplier *= 1.1
            elif session_quality == 'LOW':
                confidence_multiplier *= 0.7
            
            # News time adjustment - خفيف جداً
            if market_context['session']['is_news_time']:
                confidence_multiplier *= 0.9  # بدلاً من 0.6
            
            final_confidence = min(0.95, base_confidence * confidence_multiplier)
            
            # Determine action - خفضنا المتطلبات
            if buy_ratio > 0.55 and market_score > 5:  # كان 0.6 و 20
                action = 0  # BUY
                direction = 'BUY'
            elif buy_ratio < 0.45 and market_score < -5:  # كان 0.4 و -20
                action = 1  # SELL
                direction = 'SELL'
            else:
                action = 2  # HOLD
                direction = 'HOLD'
            
            # Override if confidence too low - رفعنا إلى 0.70
            if final_confidence < 0.70:
                action = 2  # HOLD
                direction = 'HOLD'
                logger.info(f"   ⚠️ HOLD due to low confidence: {final_confidence:.1%}")
            
            # لا نتحقق من market score مرة أخرى - أزلنا الشرط المكرر
            
            logger.info(f"   🎯 Ensemble: {buy_votes}/{total_votes} buy votes")
            logger.info(f"   📊 Final: {direction} with {final_confidence:.1%} confidence")
            
            # Return prediction with context
            return {
                'action': action,
                'direction': direction,
                'confidence': final_confidence,
                'market_context': market_context,
                'buy_votes': buy_votes,
                'total_votes': total_votes,
                'models_used': model_names
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return self._simple_prediction(df)
    
    def calculate_dynamic_sl_tp(self, symbol, direction, entry_price, market_context):
        """حساب SL/TP ذكي بناءً على سياق السوق"""
        pip_value = 0.01 if 'JPY' in symbol else 0.0001
        
        # Handle None market_context with default values
        if market_context is None:
            # Use fixed percentages when no market context available
            logger.warning(f"No market context for {symbol}, using default SL/TP")
            sl_percentage = 0.002  # 0.2% default stop loss
            tp0_percentage = 0.0015  # 0.15% default TP0 (50% of TP1)
            tp1_percentage = 0.003  # 0.3% default TP1
            tp2_percentage = 0.005  # 0.5% default TP2
            
            if direction == 'BUY':
                sl_price = entry_price * (1 - sl_percentage)
                tp0_price = entry_price * (1 + tp0_percentage)
                tp1_price = entry_price * (1 + tp1_percentage)
                tp2_price = entry_price * (1 + tp2_percentage)
            else:  # SELL
                sl_price = entry_price * (1 + sl_percentage)
                tp0_price = entry_price * (1 - tp0_percentage)
                tp1_price = entry_price * (1 - tp1_percentage)
                tp2_price = entry_price * (1 - tp2_percentage)
            
            return {
                'sl_price': float(sl_price),
                'tp0_price': float(tp0_price),
                'tp1_price': float(tp1_price),
                'tp2_price': float(tp2_price),
                'sl_pips': float(sl_percentage * 10000),
                'tp0_pips': float(tp0_percentage * 10000),
                'tp1_pips': float(tp1_percentage * 10000),
                'tp2_pips': float(tp2_percentage * 10000),
                'risk_reward_ratio': float(tp1_percentage / sl_percentage),
                'based_on': 'default_percentages',
                'partial_close_info': {
                    'tp0_percentage': 50,
                    'tp1_percentage': 30,
                    'tp2_percentage': 20
                }
            }
        
        # Get ATR for volatility-based stops
        atr = market_context['volatility']['atr']
        volatility_level = market_context['volatility']['volatility_level']
        
        # Base SL distance
        if volatility_level == 'VERY_HIGH':
            sl_atr_multiplier = 3.0
        elif volatility_level == 'HIGH':
            sl_atr_multiplier = 2.5
        elif volatility_level == 'LOW':
            sl_atr_multiplier = 1.5
        else:
            sl_atr_multiplier = 2.0
        
        base_sl_distance = atr * sl_atr_multiplier
        
        # Adjust SL based on support/resistance
        sr_levels = market_context['support_resistance']
        
        if direction == 'BUY':
            # For BUY, place SL below support
            if sr_levels.get('nearest_support'):
                support_price = sr_levels['nearest_support']['price']
                support_distance = entry_price - support_price
                
                # Use support as SL if it's reasonable
                if 0.5 * base_sl_distance < support_distance < 2 * base_sl_distance:
                    sl_distance = support_distance + (5 * pip_value)  # Buffer below support
                else:
                    sl_distance = base_sl_distance
            else:
                sl_distance = base_sl_distance
            
            sl_price = entry_price - sl_distance
            
            # TP based on resistance and risk/reward
            if sr_levels.get('nearest_resistance'):
                resistance_price = sr_levels['nearest_resistance']['price']
                resistance_distance = resistance_price - entry_price
                
                # Dynamic risk/reward based on market strength
                market_score = abs(market_context['score'])
                if market_score >= 70:
                    min_rr_ratio = 2.5
                elif market_score >= 50:
                    min_rr_ratio = 2.0
                else:
                    min_rr_ratio = 1.5
                
                # Use resistance as TP if it provides good R/R
                if resistance_distance >= sl_distance * min_rr_ratio:
                    tp1_distance = resistance_distance * 0.8  # Take profit before resistance
                    tp2_distance = resistance_distance * 1.2  # Extended target
                else:
                    tp1_distance = sl_distance * min_rr_ratio
                    tp2_distance = sl_distance * (min_rr_ratio + 1)
            else:
                # No resistance, use ATR-based targets
                tp1_distance = sl_distance * 2.0
                tp2_distance = sl_distance * 3.0
            
            # TP0 for partial close at 50% of TP1
            tp0_distance = tp1_distance * 0.5
            tp0_price = entry_price + tp0_distance
            tp1_price = entry_price + tp1_distance
            tp2_price = entry_price + tp2_distance
            
        else:  # SELL
            # For SELL, place SL above resistance
            if sr_levels.get('nearest_resistance'):
                resistance_price = sr_levels['nearest_resistance']['price']
                resistance_distance = resistance_price - entry_price
                
                # Use resistance as SL if it's reasonable
                if 0.5 * base_sl_distance < resistance_distance < 2 * base_sl_distance:
                    sl_distance = resistance_distance + (5 * pip_value)  # Buffer above resistance
                else:
                    sl_distance = base_sl_distance
            else:
                sl_distance = base_sl_distance
            
            sl_price = entry_price + sl_distance
            
            # TP based on support and risk/reward
            if sr_levels.get('nearest_support'):
                support_price = sr_levels['nearest_support']['price']
                support_distance = entry_price - support_price
                
                # Dynamic risk/reward based on market strength
                market_score = abs(market_context['score'])
                if market_score >= 70:
                    min_rr_ratio = 2.5
                elif market_score >= 50:
                    min_rr_ratio = 2.0
                else:
                    min_rr_ratio = 1.5
                
                # Use support as TP if it provides good R/R
                if support_distance >= sl_distance * min_rr_ratio:
                    tp1_distance = support_distance * 0.8  # Take profit before support
                    tp2_distance = support_distance * 1.2  # Extended target
                else:
                    tp1_distance = sl_distance * min_rr_ratio
                    tp2_distance = sl_distance * (min_rr_ratio + 1)
            else:
                # No support, use ATR-based targets
                tp1_distance = sl_distance * 2.0
                tp2_distance = sl_distance * 3.0
            
            # TP0 for partial close at 50% of TP1
            tp0_distance = tp1_distance * 0.5
            tp0_price = entry_price - tp0_distance
            tp1_price = entry_price - tp1_distance
            tp2_price = entry_price - tp2_distance
        
        # Final adjustments for session quality
        session_quality = market_context['session']['session_quality']
        if session_quality == 'LOW':
            # Tighter stops in low quality sessions
            sl_distance *= 0.8
            tp0_distance *= 0.8
            tp1_distance *= 0.8
            tp2_distance *= 0.8
        
        return {
            'sl_price': float(sl_price),
            'tp0_price': float(tp0_price),  # TP0 للإغلاق الجزئي عند 50%
            'tp1_price': float(tp1_price),
            'tp2_price': float(tp2_price),
            'sl_pips': float(sl_distance / pip_value),
            'tp0_pips': float(tp0_distance / pip_value),
            'tp1_pips': float(tp1_distance / pip_value),
            'tp2_pips': float(tp2_distance / pip_value),
            'risk_reward_ratio': float(tp1_distance / sl_distance),
            'partial_close_info': {
                'tp0_percentage': 50,  # إغلاق 50% من الصفقة عند TP0
                'tp1_percentage': 30,  # إغلاق 30% إضافية عند TP1
                'tp2_percentage': 20   # إغلاق الـ 20% المتبقية عند TP2
            }
        }
    
    def _simple_prediction(self, df):
        """Simple fallback prediction - استراتيجية بسيطة نشطة"""
        try:
            # التحقق من وجود بيانات كافية
            if len(df) < 20:
                return {
                    'action': 2,
                    'direction': 'HOLD',
                    'confidence': 0.0,
                    'market_context': None,
                    'buy_votes': 0,
                    'total_votes': 0,
                    'models_used': ['insufficient_data']
                }
            
            latest = df.iloc[-1]
            close = latest['close']
            
            # حساب المتوسطات
            sma10 = df['close'].rolling(10).mean().iloc[-1]
            sma20 = df['close'].rolling(20).mean().iloc[-1]
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] != 0 else 0
            rsi = 100 - (100 / (1 + rs))
            
            # Momentum
            momentum = (close - df['close'].iloc[-10]) / df['close'].iloc[-10] * 100
            
            logger.info(f"   📊 Simple Analysis: SMA10={sma10:.5f}, SMA20={sma20:.5f}, RSI={rsi:.1f}, Momentum={momentum:.2f}%")
            
            # قرار الشراء
            if close > sma10 and sma10 > sma20 and rsi < 70 and momentum > 0:
                confidence = 0.60 if rsi < 60 else 0.58
                return {
                    'action': 0,
                    'direction': 'BUY',
                    'confidence': confidence,
                    'market_context': None,
                    'buy_votes': 1,
                    'total_votes': 1,
                    'models_used': ['simple_strategy']
                }
            # قرار البيع
            elif close < sma10 and sma10 < sma20 and rsi > 30 and momentum < 0:
                confidence = 0.60 if rsi > 40 else 0.58
                return {
                    'action': 1,
                    'direction': 'SELL',
                    'confidence': confidence,
                    'market_context': None,
                    'buy_votes': 0,
                    'total_votes': 1,
                    'models_used': ['simple_strategy']
                }
            else:
                # حتى في HOLD، نعطي اتجاه محتمل
                if close > sma20:
                    return {
                        'action': 2,
                        'direction': 'HOLD',
                        'confidence': 0.52,
                        'market_context': None,
                        'buy_votes': 1,
                        'total_votes': 2,
                        'models_used': ['simple_strategy']
                    }
                else:
                    return {
                        'action': 2,
                        'direction': 'HOLD',
                        'confidence': 0.52,
                        'market_context': None,
                        'buy_votes': 0,
                        'total_votes': 2,
                        'models_used': ['simple_strategy']
                    }
        except:
            return {
                'action': 2,
                'direction': 'HOLD',
                'confidence': 0.0,
                'market_context': None,
                'buy_votes': 0,
                'total_votes': 0,
                'models_used': []
            }
    
    def train_and_save_from_candles(self, symbol, timeframe, candles):
        """تدريب وحفظ النماذج من الشموع المستلمة"""
        try:
            logger.info(f"\n🤖 Auto-training started for {symbol} {timeframe}")
            
            # تحويل إلى DataFrame
            df = pd.DataFrame(candles)
            for col in ['open', 'high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            df = df.sort_index()
            
            # حساب الميزات الأساسية
            features = pd.DataFrame(index=df.index)
            
            # Returns and price movements
            features['returns'] = df['close'].pct_change()
            features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # Moving averages
            for period in [5, 10, 20, 50]:
                features[f'sma_{period}'] = df['close'].rolling(period).mean()
                features[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['close'].ewm(span=12).mean()
            exp2 = df['close'].ewm(span=26).mean()
            features['macd'] = exp1 - exp2
            features['macd_signal'] = features['macd'].ewm(span=9).mean()
            
            # Bollinger Bands
            sma = df['close'].rolling(20).mean()
            std = df['close'].rolling(20).std()
            features['bb_upper'] = sma + (std * 2)
            features['bb_lower'] = sma - (std * 2)
            features['bb_width'] = features['bb_upper'] - features['bb_lower']
            
            # Volume features
            features['volume_sma'] = df['volume'].rolling(20).mean()
            features['volume_ratio'] = df['volume'] / features['volume_sma'].replace(0, 1)
            
            # Target (التنبؤ بالحركة التالية)
            features['target'] = (df['close'].shift(-1) > df['close']).astype(int)
            
            # حذف NaN
            features = features.dropna()
            
            if len(features) < 100:
                logger.warning(f"Not enough data after feature calculation: {len(features)}")
                return
            
            # تحضير البيانات للتدريب
            X = features.drop('target', axis=1).values
            y = features['target'].values
            
            # Time Series Split بدلاً من random split
            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.feature_selection import SelectKBest, f_classif
            tscv = TimeSeriesSplit(n_splits=3)  # 3 splits for auto-training
            
            # Feature Selection - اختيار أفضل 50 feature
            selector = SelectKBest(f_classif, k=min(50, X.shape[1]))
            X_selected = selector.fit_transform(X, y)
            
            # حفظ selector
            key = f"{symbol}_{timeframe}"
            self.feature_selectors[key] = selector
            
            # Use last split for testing
            for train_idx, test_idx in tscv.split(X_selected):
                X_train, X_test = X_selected[train_idx], X_selected[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
            
            # Scaling
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # تدريب النماذج
            models = {}
            
            # Random Forest
            try:
                rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
                rf.fit(X_train_scaled, y_train)
                accuracy = rf.score(X_test_scaled, y_test)
                models['random_forest'] = rf
                logger.info(f"   ✅ Random Forest: {accuracy:.2%}")
            except Exception as e:
                logger.error(f"   ❌ Random Forest failed: {e}")
            
            # Gradient Boosting
            try:
                gb = GradientBoostingClassifier(n_estimators=50, max_depth=5, random_state=42)
                gb.fit(X_train_scaled, y_train)
                accuracy = gb.score(X_test_scaled, y_test)
                models['gradient_boosting'] = gb
                logger.info(f"   ✅ Gradient Boosting: {accuracy:.2%}")
            except Exception as e:
                logger.error(f"   ❌ Gradient Boosting failed: {e}")
            
            # Extra Trees
            try:
                et = ExtraTreesClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
                et.fit(X_train_scaled, y_train)
                accuracy = et.score(X_test_scaled, y_test)
                models['extra_trees'] = et
                logger.info(f"   ✅ Extra Trees: {accuracy:.2%}")
            except Exception as e:
                logger.error(f"   ❌ Extra Trees failed: {e}")
            
            # حفظ النماذج
            if models:
                model_key = f"{symbol}_{timeframe}"
                self.models[model_key] = models
                self.scalers[model_key] = scaler
                
                # حفظ على القرص بأسماء مختلفة لتجنب الكتابة فوق النماذج القوية
                logger.info(f"   💾 Saving {len(models)} auto-trained models...")
                for model_name, model in models.items():
                    # حفظ بـ _auto بدلاً من _enhanced
                    model_path = os.path.join(self.models_dir, f"{symbol}_{timeframe}_{model_name}_auto.pkl")
                    joblib.dump(model, model_path)
                    logger.info(f"      ✅ Saved {model_name} (auto-trained)")
                
                # حفظ Scaler
                scaler_path = os.path.join(self.models_dir, f"{symbol}_{timeframe}_scaler_auto.pkl")
                joblib.dump(scaler, scaler_path)
                logger.info(f"      ✅ Saved scaler (auto-trained)")
                
                logger.info(f"   ✅ Auto-training completed! {len(models)} models saved for {symbol} {timeframe}")
            else:
                logger.error(f"   ❌ No models were trained successfully")
                
        except Exception as e:
            logger.error(f"Auto-training failed: {e}")
            import traceback
            traceback.print_exc()
    
    def load_existing_models(self):
        """Load pre-trained models - يفضل النماذج القوية (enhanced) على الضعيفة (auto)"""
        if not os.path.exists(self.models_dir):
            return
        
        # أولاً: حمل النماذج القوية (6 نماذج من التدريب الكامل)
        enhanced_count = 0
        for file in os.listdir(self.models_dir):
            if file.endswith('_enhanced.pkl') and 'scaler' not in file:
                try:
                    parts = file.replace('.pkl', '').split('_')
                    if len(parts) >= 4:
                        symbol = parts[0]
                        timeframe = parts[1]
                        model_type = '_'.join(parts[2:-1])  # Handle model names with underscores
                        
                        key = f"{symbol}_{timeframe}"
                        if key not in self.models:
                            self.models[key] = {}
                        
                        model = joblib.load(os.path.join(self.models_dir, file))
                        self.models[key][model_type] = model
                        enhanced_count += 1
                        
                except Exception as e:
                    logger.error(f"Error loading {file}: {e}")
        
        # ثانياً: حمل النماذج التلقائية فقط إذا لم توجد نماذج قوية
        auto_count = 0
        for file in os.listdir(self.models_dir):
            if file.endswith('_auto.pkl') and 'scaler' not in file:
                try:
                    parts = file.replace('.pkl', '').split('_')
                    if len(parts) >= 4:
                        symbol = parts[0]
                        timeframe = parts[1]
                        model_type = '_'.join(parts[2:-1])
                        
                        key = f"{symbol}_{timeframe}"
                        # حمل النموذج التلقائي فقط إذا لم يوجد نموذج قوي
                        if key not in self.models:
                            self.models[key] = {}
                        
                        if model_type not in self.models[key]:
                            model = joblib.load(os.path.join(self.models_dir, file))
                            self.models[key][model_type] = model
                            auto_count += 1
                        
                except Exception as e:
                    logger.error(f"Error loading {file}: {e}")
        
        # حمل Scalers (يفضل enhanced على auto)
        for file in os.listdir(self.models_dir):
            if file.endswith('_scaler_enhanced.pkl'):
                try:
                    parts = file.replace('.pkl', '').split('_')
                    if len(parts) >= 3:
                        symbol = parts[0]
                        timeframe = parts[1]
                        
                        scaler = joblib.load(os.path.join(self.models_dir, file))
                        self.scalers[f"{symbol}_{timeframe}"] = scaler
                        
                except Exception as e:
                    logger.error(f"Error loading {file}: {e}")
    
    def update_from_trade_result(self, trade_info):
        """تحديث النظام بناءً على نتائج التداول"""
        try:
            # Update risk manager
            if trade_info['status'] == 'closed':
                self.risk_manager.close_trade(
                    trade_info['id'],
                    trade_info['exit_price']
                )
            elif trade_info['status'] == 'opened':
                self.risk_manager.register_trade(trade_info)
            
            # Track performance
            self.performance_tracker['trades'].append(trade_info)
            
            # Log the update
            logger.info(f"📊 Trade update: {trade_info['symbol']} - {trade_info['status']}")
            
        except Exception as e:
            logger.error(f"Error updating from trade: {e}")

# Initialize system
system = EnhancedMLTradingSystem()

# Flask routes
@app.route('/status', methods=['GET'])
def status():
    """Server status with enhanced information"""
    risk_report = system.risk_manager.get_risk_report()
    
    return jsonify({
        'status': 'running',
        'version': '3.0-enhanced',
        'features': '200+ with market context',
        'models': '6 ML models + market analysis',
        'total_loaded': len(system.models),
        'risk_management': {
            'current_balance': risk_report['current_balance'],
            'total_pl': risk_report['total_pl'],
            'open_trades': risk_report['open_trades'],
            'risk_status': risk_report['risk_status']
        },
        'ml_libraries': {
            'lightgbm': LIGHTGBM_AVAILABLE,
            'xgboost': XGBOOST_AVAILABLE
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Enhanced prediction endpoint with full market analysis"""
    try:
        # Parse JSON
        raw_data = request.get_data(as_text=True)
        
        try:
            data = json.loads(raw_data)
        except json.JSONDecodeError:
            return jsonify({'error': 'Invalid JSON', 'action': 'NONE', 'confidence': 0})
        
        symbol = data.get('symbol', 'UNKNOWN')
        timeframe = data.get('timeframe', 'M15')
        candles = data.get('candles', [])
        account_info = data.get('account_info', {})
        
        # Clean symbol
        clean_symbol = symbol.replace('m', '').replace('.ecn', '').replace('_pro', '')
        
        logger.info(f"\n📊 Request: {symbol} ({clean_symbol}) {timeframe} - {len(candles)} candles")
        
        # Update risk manager balance if provided
        if account_info.get('balance'):
            system.risk_manager.update_balance(account_info['balance'])
        
        # تدريب تلقائي وحفظ النماذج كل 100 طلب أو إذا لم توجد نماذج
        model_key = f"{clean_symbol}_{timeframe}"
        
        # عداد الطلبات للتدريب الدوري
        if not hasattr(system, 'request_counter'):
            system.request_counter = {}
        
        if model_key not in system.request_counter:
            system.request_counter[model_key] = 0
        
        system.request_counter[model_key] += 1
        
        # تدريب إذا: لا توجد نماذج، أو كل 100 طلب
        should_train = (model_key not in system.models) or (system.request_counter[model_key] % 100 == 0)
        
        if should_train and len(candles) >= 500:
            logger.info(f"   🤖 Auto-training triggered for {clean_symbol} {timeframe}")
            
            # تدريب في خيط منفصل لعدم تأخير الاستجابة
            train_thread = threading.Thread(
                target=system.train_and_save_from_candles,
                args=(clean_symbol, timeframe, candles)
            )
            train_thread.daemon = True
            train_thread.start()
        
        if len(candles) < 200:
            return jsonify({
                'symbol': symbol,
                'action': 'NONE',
                'confidence': 0,
                'error': 'Need at least 200 candles for analysis'
            })
        
        # Convert to DataFrame
        df = pd.DataFrame(candles)
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
        
        df = df.dropna()
        
        # Get enhanced prediction with weighted ensemble voting
        prediction_result = system.predict_with_weighted_ensemble(clean_symbol, timeframe, df)
        
        # Extract results
        action = prediction_result['direction']
        confidence = prediction_result['confidence']
        market_context = prediction_result.get('market_context')
        
        logger.info(f"   📊 Prediction: {action} with {confidence:.1%} confidence")
        logger.info(f"   📊 Buy votes: {prediction_result.get('buy_votes', 0)}/{prediction_result.get('total_votes', 0)}")
        
        # Determine signal with risk management
        current_price = float(df['close'].iloc[-1])
        
        if action != 'HOLD' and confidence >= 0.70:  # رفعنا إلى 0.70
            # Check risk management approval
            sl_tp_info = system.calculate_dynamic_sl_tp(
                clean_symbol, action, current_price, market_context
            )
            
            # Validate trade with risk manager
            lot_size, risk_message = system.risk_manager.calculate_position_size(
                clean_symbol,
                current_price,
                sl_tp_info['sl_price'],
                market_context,
                confidence
            )
            
            if lot_size > 0:
                # Validate complete trade setup
                validation = system.risk_manager.validate_trade_setup(
                    clean_symbol,
                    action,
                    current_price,
                    sl_tp_info['sl_price'],
                    sl_tp_info['tp1_price'],
                    lot_size,
                    market_context
                )
                
                if not validation['is_valid']:
                    logger.warning(f"Trade validation failed: {validation['errors']}")
                    action = 'NONE'
                    confidence = 0
                else:
                    # Add risk management info
                    sl_tp_info['lot_size'] = lot_size
                    sl_tp_info['risk_message'] = risk_message
                    sl_tp_info['validation'] = validation
            else:
                logger.warning(f"Position sizing failed: {risk_message}")
                action = 'NONE'
                confidence = 0
        else:
            logger.info(f"   ⚠️ No trade: action={action}, confidence={confidence:.1%}")
            if action == 'HOLD':
                logger.info(f"      Reason: Model voted HOLD")
            elif confidence < 0.70:
                logger.info(f"      Reason: Low confidence ({confidence:.1%} < 70%)")
            
            action = 'NONE'
            sl_tp_info = {
                'sl_price': current_price,
                'tp1_price': current_price,
                'tp2_price': current_price,
                'sl_pips': 0,
                'tp1_pips': 0,
                'tp2_pips': 0,
                'risk_reward_ratio': 0,
                'lot_size': 0
            }
        
        # Prepare response
        response = {
            'symbol': symbol,
            'timeframe': timeframe,
            'action': action,
            'confidence': float(confidence),
            'current_price': current_price,
            'sl_price': sl_tp_info['sl_price'],
            'tp0_price': sl_tp_info.get('tp0_price', sl_tp_info['tp1_price']),  # TP0 للإغلاق الجزئي
            'tp1_price': sl_tp_info['tp1_price'],
            'tp2_price': sl_tp_info['tp2_price'],
            'sl_pips': sl_tp_info['sl_pips'],
            'tp0_pips': sl_tp_info.get('tp0_pips', 0),
            'tp1_pips': sl_tp_info['tp1_pips'],
            'tp2_pips': sl_tp_info['tp2_pips'],
            'lot_size': sl_tp_info.get('lot_size', 0),
            'risk_reward_ratio': sl_tp_info['risk_reward_ratio'],
            'partial_close_info': sl_tp_info.get('partial_close_info', {}),
            'models_used': len(prediction_result['models_used']),
            'entry_reason': prediction_result.get('entry_reason', 'Signal generated'),
            'market_analysis': {
                'score': market_context['score'] if market_context else 0,
                'trend': market_context['trend']['overall'] if market_context else 'UNKNOWN',
                'session_quality': market_context['session']['session_quality'] if market_context else 'UNKNOWN',
                'volatility': market_context['volatility']['volatility_level'] if market_context else 'UNKNOWN'
            } if market_context else {},
            'risk_management': {
                'validation': sl_tp_info.get('validation', {}),
                'risk_status': system.risk_manager.get_risk_report()['risk_status']
            }
        }
        
        # Log summary
        if action != 'NONE':
            logger.info(f"   ✅ Signal: {action} @ {current_price}")
            logger.info(f"   🎯 SL: {sl_tp_info['sl_pips']:.1f} pips, TP: {sl_tp_info['tp1_pips']:.1f} pips")
            logger.info(f"   💰 Lot size: {sl_tp_info.get('lot_size', 0):.2f}")
        else:
            logger.info(f"   ⏸️ No trade signal")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'action': 'NONE',
            'confidence': 0
        })

@app.route('/train', methods=['POST'])
def train():
    """Train enhanced models endpoint"""
    try:
        data = request.json
        symbol = data.get('symbol', '').replace('m', '').replace('.ecn', '')
        timeframe = data.get('timeframe', 'M15')
        
        logger.info(f"\n🎯 Training request: {symbol} {timeframe}")
        
        success = system.train_enhanced_models(symbol, timeframe)
        
        return jsonify({
            'success': success,
            'message': f'Enhanced training completed for {symbol} {timeframe}' if success else 'Training failed'
        })
    except Exception as e:
        logger.error(f"Training error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/update_trade', methods=['POST'])
def update_trade():
    """Update system with trade results"""
    try:
        trade_info = request.json
        system.update_from_trade_result(trade_info)
        
        return jsonify({
            'success': True,
            'message': 'Trade update processed'
        })
    except Exception as e:
        logger.error(f"Trade update error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/risk_report', methods=['GET'])
def risk_report():
    """Get current risk management report"""
    report = system.risk_manager.get_risk_report()
    return jsonify(report)

@app.route('/performance', methods=['GET'])
def performance():
    """Get system performance metrics"""
    trades = system.performance_tracker['trades']
    
    if not trades:
        return jsonify({
            'total_trades': 0,
            'message': 'No trades recorded yet'
        })
    
    # Calculate metrics
    closed_trades = [t for t in trades if t.get('status') == 'closed']
    winning_trades = [t for t in closed_trades if t.get('profit', 0) > 0]
    losing_trades = [t for t in closed_trades if t.get('profit', 0) < 0]
    
    total_profit = sum(t.get('profit', 0) for t in closed_trades)
    
    return jsonify({
        'total_trades': len(trades),
        'closed_trades': len(closed_trades),
        'open_trades': len(trades) - len(closed_trades),
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'win_rate': (len(winning_trades) / len(closed_trades) * 100) if closed_trades else 0,
        'total_profit': total_profit,
        'average_win': sum(t['profit'] for t in winning_trades) / len(winning_trades) if winning_trades else 0,
        'average_loss': sum(t['profit'] for t in losing_trades) / len(losing_trades) if losing_trades else 0,
        'current_balance': system.risk_manager.current_balance,
        'risk_status': system.risk_manager.get_risk_report()['risk_status']
    })

@app.route('/models', methods=['GET'])
def models():
    """List available models"""
    models_info = {}
    
    for key, models in system.models.items():
        models_info[key] = {
            'models': list(models.keys()),
            'count': len(models),
            'enhanced': True
        }
    
    return jsonify({
        'total_pairs': len(system.models),
        'models': models_info,
        'risk_manager': 'active',
        'market_analyzer': 'active'
    })

@app.route('/backtest', methods=['POST'])
def backtest():
    """Run backtest on historical data"""
    try:
        data = request.json
        symbol = data.get('symbol', '').replace('m', '').replace('.ecn', '')
        timeframe = data.get('timeframe', 'M15')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        initial_balance = data.get('initial_balance', 10000)
        
        logger.info(f"\n📊 Backtest request: {symbol} {timeframe} from {start_date} to {end_date}")
        
        # Load historical data
        conn = sqlite3.connect(system.db_path)
        cursor = conn.cursor()
        
        # استخدام parameterized query لتجنب SQL injection
        query = """
            SELECT * FROM price_data 
            WHERE (symbol = ? OR symbol LIKE ?) 
            AND time >= ? AND time <= ?
            ORDER BY time ASC
        """
        
        cursor.execute(query, (symbol, f"{symbol}%", start_date, end_date))
        data = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(data, columns=columns)
        conn.close()
        
        if len(df) < 100:
            return jsonify({
                'success': False,
                'error': 'Not enough data for backtest'
            })
        
        # Prepare data
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        df = df.sort_index()
        
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Run backtest
        trades = []
        balance = initial_balance
        position = None
        
        # Sliding window backtest
        window_size = 100  # Need 100 candles for indicators
        
        for i in range(window_size, len(df)):
            # Get window of data
            window_df = df.iloc[i-window_size:i+1]
            
            # Get prediction
            prediction = system.predict_with_weighted_ensemble(symbol, timeframe, window_df.copy())
            
            current_price = window_df.iloc[-1]['close']
            current_time = window_df.index[-1]
            
            # Check if we should open a position
            if position is None and prediction['direction'] != 'HOLD' and prediction['confidence'] >= 0.70:
                # Calculate position size (1% risk)
                risk_amount = balance * 0.01
                
                # Get SL/TP
                sl_tp = system.calculate_dynamic_sl_tp(
                    symbol, 
                    prediction['direction'], 
                    current_price, 
                    prediction['market_context']
                )
                
                # Calculate lot size
                sl_pips = abs(sl_tp['sl_pips'])
                pip_value = 0.01 if 'JPY' in symbol else 0.0001
                lot_size = risk_amount / (sl_pips * 10)  # Assuming $10 per pip per lot
                lot_size = min(lot_size, balance / (current_price * 100000))  # Max leverage check
                
                position = {
                    'open_time': current_time,
                    'open_price': current_price,
                    'direction': prediction['direction'],
                    'lot_size': lot_size,
                    'sl_price': sl_tp['sl_price'],
                    'tp1_price': sl_tp['tp1_price'],
                    'confidence': prediction['confidence'],
                    'market_score': prediction['market_context']['score'] if prediction['market_context'] else 0
                }
                
            # Check if we should close position
            elif position is not None:
                if position['direction'] == 'BUY':
                    # Check SL/TP for BUY
                    if current_price <= position['sl_price']:
                        # Stop loss hit
                        profit = (position['sl_price'] - position['open_price']) * position['lot_size'] * 100000
                        balance += profit
                        
                        trades.append({
                            'open_time': position['open_time'],
                            'close_time': current_time,
                            'direction': position['direction'],
                            'open_price': position['open_price'],
                            'close_price': position['sl_price'],
                            'lot_size': position['lot_size'],
                            'profit': profit,
                            'pips': (position['sl_price'] - position['open_price']) / pip_value,
                            'exit_reason': 'SL',
                            'confidence': position['confidence'],
                            'market_score': position['market_score']
                        })
                        position = None
                        
                    elif current_price >= position['tp1_price']:
                        # Take profit hit
                        profit = (position['tp1_price'] - position['open_price']) * position['lot_size'] * 100000
                        balance += profit
                        
                        trades.append({
                            'open_time': position['open_time'],
                            'close_time': current_time,
                            'direction': position['direction'],
                            'open_price': position['open_price'],
                            'close_price': position['tp1_price'],
                            'lot_size': position['lot_size'],
                            'profit': profit,
                            'pips': (position['tp1_price'] - position['open_price']) / pip_value,
                            'exit_reason': 'TP',
                            'confidence': position['confidence'],
                            'market_score': position['market_score']
                        })
                        position = None
                        
                else:  # SELL
                    # Check SL/TP for SELL
                    if current_price >= position['sl_price']:
                        # Stop loss hit
                        profit = (position['open_price'] - position['sl_price']) * position['lot_size'] * 100000
                        balance += profit
                        
                        trades.append({
                            'open_time': position['open_time'],
                            'close_time': current_time,
                            'direction': position['direction'],
                            'open_price': position['open_price'],
                            'close_price': position['sl_price'],
                            'lot_size': position['lot_size'],
                            'profit': profit,
                            'pips': (position['open_price'] - position['sl_price']) / pip_value,
                            'exit_reason': 'SL',
                            'confidence': position['confidence'],
                            'market_score': position['market_score']
                        })
                        position = None
                        
                    elif current_price <= position['tp1_price']:
                        # Take profit hit
                        profit = (position['open_price'] - position['tp1_price']) * position['lot_size'] * 100000
                        balance += profit
                        
                        trades.append({
                            'open_time': position['open_time'],
                            'close_time': current_time,
                            'direction': position['direction'],
                            'open_price': position['open_price'],
                            'close_price': position['tp1_price'],
                            'lot_size': position['lot_size'],
                            'profit': profit,
                            'pips': (position['open_price'] - position['tp1_price']) / pip_value,
                            'exit_reason': 'TP',
                            'confidence': position['confidence'],
                            'market_score': position['market_score']
                        })
                        position = None
        
        # Close any open position at end
        if position is not None:
            current_price = df.iloc[-1]['close']
            if position['direction'] == 'BUY':
                profit = (current_price - position['open_price']) * position['lot_size'] * 100000
                pips = (current_price - position['open_price']) / pip_value
            else:
                profit = (position['open_price'] - current_price) * position['lot_size'] * 100000
                pips = (position['open_price'] - current_price) / pip_value
            
            balance += profit
            trades.append({
                'open_time': position['open_time'],
                'close_time': df.index[-1],
                'direction': position['direction'],
                'open_price': position['open_price'],
                'close_price': current_price,
                'lot_size': position['lot_size'],
                'profit': profit,
                'pips': pips,
                'exit_reason': 'END',
                'confidence': position['confidence'],
                'market_score': position['market_score']
            })
        
        # Calculate metrics
        if trades:
            winning_trades = [t for t in trades if t['profit'] > 0]
            losing_trades = [t for t in trades if t['profit'] < 0]
            
            total_profit = sum(t['profit'] for t in trades)
            gross_profit = sum(t['profit'] for t in winning_trades) if winning_trades else 0
            gross_loss = abs(sum(t['profit'] for t in losing_trades)) if losing_trades else 0
            
            # Calculate maximum drawdown
            equity_curve = [initial_balance]
            for trade in trades:
                equity_curve.append(equity_curve[-1] + trade['profit'])
            
            peak = equity_curve[0]
            max_drawdown = 0
            for equity in equity_curve:
                if equity > peak:
                    peak = equity
                drawdown = (peak - equity) / peak * 100
                max_drawdown = max(max_drawdown, drawdown)
            
            results = {
                'success': True,
                'initial_balance': initial_balance,
                'final_balance': balance,
                'total_return': (balance - initial_balance) / initial_balance * 100,
                'total_trades': len(trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': len(winning_trades) / len(trades) * 100 if trades else 0,
                'profit_factor': gross_profit / gross_loss if gross_loss > 0 else float('inf'),
                'average_win': gross_profit / len(winning_trades) if winning_trades else 0,
                'average_loss': gross_loss / len(losing_trades) if losing_trades else 0,
                'expected_return': total_profit / len(trades) if trades else 0,
                'max_drawdown': max_drawdown,
                'total_pips': sum(t['pips'] for t in trades),
                'trades': trades[-20:]  # Last 20 trades for review
            }
            
            logger.info(f"✅ Backtest complete: {len(trades)} trades, {results['total_return']:.2f}% return")
            
        else:
            results = {
                'success': True,
                'initial_balance': initial_balance,
                'final_balance': balance,
                'total_return': 0,
                'total_trades': 0,
                'message': 'No trades generated in backtest period'
            }
            
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Backtest error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    logger.info("\n" + "="*70)
    logger.info("🚀 ENHANCED ML TRADING SERVER")
    logger.info("📊 Market Analysis + Risk Management + 6 ML Models")
    logger.info("🌐 Server: http://0.0.0.0:5000")
    logger.info("="*70 + "\n")
    
    # Initial setup message
    logger.info("🔧 System Components:")
    logger.info("   ✅ Market Analysis Engine: Analyzes context before trading")
    logger.info("   ✅ Risk Management System: Protects capital and optimizes position sizes")
    logger.info("   ✅ ML Models: 6 models with ensemble voting")
    logger.info("   ✅ Dynamic SL/TP: Based on market structure and volatility")
    logger.info("   ✅ Trade Validation: Multi-layer checks before execution")
    logger.info("")
    
    # Train initial models for major pairs
    logger.info("🤖 Training initial enhanced models...")
    initial_pairs = ['EURUSD', 'GBPUSD', 'USDJPY']
    
    for symbol in initial_pairs:
        try:
            logger.info(f"\n🎯 Training {symbol}...")
            success = system.train_enhanced_models(symbol, 'M15')
            if success:
                logger.info(f"   ✅ {symbol} ready for trading")
            else:
                logger.info(f"   ⚠️ {symbol} training skipped (insufficient data)")
        except Exception as e:
            logger.error(f"   ❌ {symbol} training failed: {e}")
    
    logger.info("\n🎆 Server ready for intelligent trading!\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)