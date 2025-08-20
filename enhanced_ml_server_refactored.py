#!/usr/bin/env python3
"""
🚀 Enhanced ML Server - نظام التداول الذكي المتكامل (محدث)
📊 يستخدم Profit-based metrics بدلاً من Accuracy
💰 نظام قادر على تحقيق الربح المستدام مع إدارة مخاطر متقدمة
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

# Flask app
app = Flask(__name__)
CORS(app)

class EnhancedMLTradingSystem:
    """النظام المتكامل للتداول الذكي - محدث بمقاييس الربحية"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}  # لحفظ feature selectors
        
        # استخدام القاعدة الرئيسية الكبيرة
        self.db_path = './data/forex_ml.db'
        self.models_dir = './trained_models'
        
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize subsystems
        self.market_analyzer = MarketAnalysisEngine(self.db_path)
        self.risk_manager = RiskManagementSystem(initial_balance=10000)
        
        # Trading parameters - محدثة
        self.min_confidence = 0.70  # رفع من 0.55 إلى 0.70
        self.min_market_score = 40  # رفع من 20 إلى 40
        self.max_daily_trades = 10
        self.trade_cooldown = {}
        
        # Performance tracking
        self.performance_tracker = {
            'predictions': [],
            'trades': [],
            'daily_stats': {},
            'model_performance': {},  # أداء كل نموذج
            'model_weights': {}  # أوزان النماذج للتصويت
        }
        
        # Position management
        self.open_positions = {}
        self.trailing_stop_levels = {}
        
        # Load existing models
        self.load_existing_models()
        
        logger.info(f"✅ Enhanced ML Trading System initialized (Profit-based)")
        logger.info(f"   📊 Min Confidence: {self.min_confidence}")
        logger.info(f"   📊 Min Market Score: {self.min_market_score}")
    
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
    
    def calculate_kelly_position(self, symbol, confidence=0.7):
        """حساب حجم الصفقة باستخدام معادلة Kelly محسنة"""
        # الحصول على أداء النموذج لهذا الرمز
        key = f"{symbol}_performance"
        if key in self.performance_tracker['model_performance']:
            perf = self.performance_tracker['model_performance'][key]
            win_rate = perf.get('win_rate', 0.5)
            avg_win = perf.get('avg_win', 1)
            avg_loss = perf.get('avg_loss', 1)
        else:
            # قيم افتراضية محافظة
            win_rate = 0.5
            avg_win = 1
            avg_loss = 1
        
        if avg_win <= 0 or win_rate <= 0:
            return 0.005  # الحد الأدنى 0.5%
        
        # Kelly formula
        loss_rate = 1 - win_rate
        kelly_percentage = (win_rate * avg_win - loss_rate * avg_loss) / avg_win
        
        # ضبط بناءً على الثقة
        kelly_percentage *= confidence
        
        # تطبيق الحدود (0.5% - 2%)
        kelly_percentage = max(0.005, min(0.02, kelly_percentage))
        
        return kelly_percentage
    
    def update_trailing_stop(self, symbol, current_price, entry_price, current_sl, direction):
        """نظام Trailing Stop Loss محدث"""
        profit_pct = 0
        
        if direction == 'BUY':
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
                logger.info(f"📈 Trailing SL for {symbol}: {current_sl:.5f} → {new_sl:.5f} (Profit: {profit_pct:.1%})")
                return new_sl
                
        else:  # SELL
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
                logger.info(f"📈 Trailing SL for {symbol}: {current_sl:.5f} → {new_sl:.5f} (Profit: {profit_pct:.1%})")
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
        
        # يجب تحقق 3 من 4 شروط على الأقل
        if conditions_met >= 3:
            logger.info(f"   ✅ Entry conditions met ({conditions_met}/4): {', '.join(conditions_details)}")
            return True
        else:
            logger.info(f"   ❌ Entry conditions not met ({conditions_met}/4)")
            return False
    
    def calculate_model_weights(self):
        """حساب أوزان النماذج بناءً على الأداء"""
        for key, performances in self.performance_tracker['model_performance'].items():
            if isinstance(performances, list) and len(performances) > 0:
                # حساب متوسط الأداء لكل نموذج
                model_scores = {}
                
                for perf in performances[-100:]:  # آخر 100 صفقة
                    model = perf['model']
                    profit = perf['profit']
                    
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
                
                self.model_weights[key] = weights
    
    def train_enhanced_models(self, symbol, timeframe):
        """تدريب النماذج مع Time Series Split ومقاييس الربحية"""
        try:
            logger.info(f"🤖 Training enhanced models for {symbol} {timeframe}...")
            
            # Load data
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Try different symbol formats
            possible_symbols = [symbol, f"{symbol}m", f"{symbol}.ecn", symbol.lower()]
            
            df = None
            for sym in possible_symbols:
                cursor.execute("""
                    SELECT * FROM price_data 
                    WHERE symbol = ? OR symbol LIKE ?
                    ORDER BY time ASC
                    LIMIT 50000
                """, (sym, f"{sym}%"))
                
                data = cursor.fetchall()
                if data:
                    columns = [desc[0] for desc in cursor.description]
                    df = pd.DataFrame(data, columns=columns)
                    logger.info(f"Found {len(df)} records with symbol: {sym}")
                    break
            
            conn.close()
            
            if df is None or len(df) < 5000:
                logger.warning(f"Not enough data for {symbol}")
                return False
            
            # Prepare data
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            df = df.sort_index()
            
            # تحويل الأعمدة لأرقام
            for col in ['open', 'high', 'low', 'close', 'tick_volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna()
            
            # Create features with market context
            logger.info("   📊 Creating features with market context...")
            
            X_list = []
            y_list = []
            prices_list = []
            
            # Process data
            window_size = 200
            skip_size = 20  # معالجة كل 20 شمعة لتسريع التدريب
            
            for i in range(window_size, len(df) - 1, skip_size):
                window_data = df.iloc[i-window_size:i]
                
                # تحليل السوق
                market_context = self.market_analyzer.analyze_complete_market_context(
                    symbol, window_data.reset_index().to_dict('records'), timeframe
                )
                
                if market_context and abs(market_context['score']) > 20:
                    # حساب الميزات
                    features = self.calculate_enhanced_features(window_data, market_context)
                    
                    if not features.empty:
                        X_list.append(features.iloc[-1].values)
                        
                        # التنبؤ: هل السعر سيرتفع؟
                        future_return = df['close'].iloc[i] - df['close'].iloc[i-1]
                        y_list.append(1 if future_return > 0 else 0)
                        
                        # السعر للمقاييس
                        prices_list.append(df['close'].iloc[i-1])
            
            if len(X_list) < 1000:
                logger.warning(f"Not enough quality samples: {len(X_list)}")
                return False
            
            # Convert to arrays
            X = np.array(X_list)
            y = np.array(y_list)
            prices = np.array(prices_list)
            
            logger.info(f"   📊 Created {len(X)} training samples")
            
            # Feature Selection - أفضل 50 ميزة
            selector = SelectKBest(f_classif, k=min(50, X.shape[1]))
            X_selected = selector.fit_transform(X, y)
            
            # حفظ selector
            key = f"{symbol}_{timeframe}"
            self.feature_selectors[key] = selector
            
            # Time Series Split
            tscv = TimeSeriesSplit(n_splits=5)
            best_models = {}
            
            logger.info("   🔄 Training with Time Series Cross-Validation...")
            
            for fold, (train_idx, test_idx) in enumerate(tscv.split(X_selected)):
                logger.info(f"   📊 Fold {fold+1}/5...")
                
                X_train = X_selected[train_idx]
                X_test = X_selected[test_idx]
                y_train = y[train_idx]
                y_test = y[test_idx]
                prices_test = prices[test_idx]
                
                # Scale
                scaler = RobustScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # حفظ آخر scaler
                if fold == 4:
                    self.scalers[key] = scaler
                
                # تدريب النماذج
                fold_models = {}
                
                # 1. Random Forest
                try:
                    rf = RandomForestClassifier(
                        n_estimators=200, max_depth=15, min_samples_split=5,
                        random_state=42, n_jobs=-1
                    )
                    rf.fit(X_train_scaled, y_train)
                    y_pred = rf.predict(X_test_scaled)
                    
                    metrics = self.calculate_trading_metrics(y_test, y_pred, prices_test)
                    
                    if metrics['profit_factor'] > 1.2 and metrics['total_return'] > 0:
                        fold_models['random_forest'] = {
                            'model': rf,
                            'metrics': metrics
                        }
                        logger.info(f"      ✅ RF: PF={metrics['profit_factor']:.2f}, Return={metrics['total_return']:.2f}")
                except Exception as e:
                    logger.error(f"      ❌ RF: {e}")
                
                # 2. Gradient Boosting
                try:
                    gb = GradientBoostingClassifier(
                        n_estimators=150, learning_rate=0.05, max_depth=6,
                        random_state=42
                    )
                    gb.fit(X_train_scaled, y_train)
                    y_pred = gb.predict(X_test_scaled)
                    
                    metrics = self.calculate_trading_metrics(y_test, y_pred, prices_test)
                    
                    if metrics['profit_factor'] > 1.2 and metrics['total_return'] > 0:
                        fold_models['gradient_boosting'] = {
                            'model': gb,
                            'metrics': metrics
                        }
                        logger.info(f"      ✅ GB: PF={metrics['profit_factor']:.2f}, Return={metrics['total_return']:.2f}")
                except Exception as e:
                    logger.error(f"      ❌ GB: {e}")
                
                # 3. Extra Trees
                try:
                    et = ExtraTreesClassifier(
                        n_estimators=200, max_depth=15, min_samples_split=5,
                        random_state=42, n_jobs=-1
                    )
                    et.fit(X_train_scaled, y_train)
                    y_pred = et.predict(X_test_scaled)
                    
                    metrics = self.calculate_trading_metrics(y_test, y_pred, prices_test)
                    
                    if metrics['profit_factor'] > 1.2 and metrics['total_return'] > 0:
                        fold_models['extra_trees'] = {
                            'model': et,
                            'metrics': metrics
                        }
                        logger.info(f"      ✅ ET: PF={metrics['profit_factor']:.2f}, Return={metrics['total_return']:.2f}")
                except Exception as e:
                    logger.error(f"      ❌ ET: {e}")
                
                # 4. Neural Network
                try:
                    nn = MLPClassifier(
                        hidden_layer_sizes=(100, 50, 25),
                        activation='relu', solver='adam',
                        max_iter=500, random_state=42
                    )
                    nn.fit(X_train_scaled, y_train)
                    y_pred = nn.predict(X_test_scaled)
                    
                    metrics = self.calculate_trading_metrics(y_test, y_pred, prices_test)
                    
                    if metrics['profit_factor'] > 1.2 and metrics['total_return'] > 0:
                        fold_models['neural_network'] = {
                            'model': nn,
                            'metrics': metrics
                        }
                        logger.info(f"      ✅ NN: PF={metrics['profit_factor']:.2f}, Return={metrics['total_return']:.2f}")
                except Exception as e:
                    logger.error(f"      ❌ NN: {e}")
                
                # 5. LightGBM
                if LIGHTGBM_AVAILABLE:
                    try:
                        lgbm = lgb.LGBMClassifier(
                            n_estimators=200, num_leaves=50, learning_rate=0.05,
                            random_state=42, verbosity=-1
                        )
                        lgbm.fit(X_train_scaled, y_train)
                        y_pred = lgbm.predict(X_test_scaled)
                        
                        metrics = self.calculate_trading_metrics(y_test, y_pred, prices_test)
                        
                        if metrics['profit_factor'] > 1.2 and metrics['total_return'] > 0:
                            fold_models['lightgbm'] = {
                                'model': lgbm,
                                'metrics': metrics
                            }
                            logger.info(f"      ✅ LGBM: PF={metrics['profit_factor']:.2f}, Return={metrics['total_return']:.2f}")
                    except Exception as e:
                        logger.error(f"      ❌ LGBM: {e}")
                
                # 6. XGBoost
                if XGBOOST_AVAILABLE:
                    try:
                        xgb_model = xgb.XGBClassifier(
                            n_estimators=200, max_depth=6, learning_rate=0.05,
                            random_state=42, use_label_encoder=False
                        )
                        xgb_model.fit(X_train_scaled, y_train)
                        y_pred = xgb_model.predict(X_test_scaled)
                        
                        metrics = self.calculate_trading_metrics(y_test, y_pred, prices_test)
                        
                        if metrics['profit_factor'] > 1.2 and metrics['total_return'] > 0:
                            fold_models['xgboost'] = {
                                'model': xgb_model,
                                'metrics': metrics
                            }
                            logger.info(f"      ✅ XGB: PF={metrics['profit_factor']:.2f}, Return={metrics['total_return']:.2f}")
                    except Exception as e:
                        logger.error(f"      ❌ XGB: {e}")
                
                # تحديث أفضل النماذج
                for model_name, model_data in fold_models.items():
                    if model_name not in best_models or \
                       model_data['metrics']['profit_factor'] > best_models[model_name]['metrics']['profit_factor']:
                        best_models[model_name] = model_data
            
            # حفظ أفضل النماذج
            if best_models:
                self.models[key] = {}
                
                logger.info(f"   💾 Saving best models...")
                for model_name, model_data in best_models.items():
                    self.models[key][model_name] = model_data['model']
                    
                    # حفظ على القرص
                    model_path = os.path.join(self.models_dir, f"{symbol}_{timeframe}_{model_name}_enhanced.pkl")
                    joblib.dump(model_data['model'], model_path)
                    
                    logger.info(f"      ✅ {model_name}: PF={model_data['metrics']['profit_factor']:.2f}")
                
                # حفظ scaler و selector
                scaler_path = os.path.join(self.models_dir, f"{symbol}_{timeframe}_scaler_enhanced.pkl")
                joblib.dump(self.scalers[key], scaler_path)
                
                selector_path = os.path.join(self.models_dir, f"{symbol}_{timeframe}_selector_enhanced.pkl")
                joblib.dump(self.feature_selectors[key], selector_path)
                
                logger.info(f"✅ Successfully trained {len(best_models)} profitable models for {symbol} {timeframe}!")
                return True
            else:
                logger.warning(f"❌ No profitable models found for {symbol} {timeframe}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Training error: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def calculate_enhanced_features(self, df, market_context=None):
        """حساب الميزات المحسنة مع سياق السوق"""
        features = pd.DataFrame(index=[df.index[-1]])
        
        # Basic price features
        features['returns'] = df['close'].pct_change().iloc[-1]
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1)).iloc[-1]
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            features[f'sma_{period}'] = df['close'].rolling(period).mean().iloc[-1]
            features[f'sma_{period}_ratio'] = df['close'].iloc[-1] / features[f'sma_{period}']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] != 0 else 0
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        sma20 = df['close'].rolling(20).mean().iloc[-1]
        std20 = df['close'].rolling(20).std().iloc[-1]
        features['bb_upper'] = sma20 + 2 * std20
        features['bb_lower'] = sma20 - 2 * std20
        features['bb_position'] = (df['close'].iloc[-1] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        
        # Volume features
        features['volume_ratio'] = df['tick_volume'].iloc[-1] / df['tick_volume'].rolling(20).mean().iloc[-1]
        
        # Market context features
        if market_context:
            features['market_score'] = market_context.get('score', 0)
            features['trend_strength'] = market_context.get('trend', {}).get('strength', 0)
            features['support_distance'] = market_context.get('support_resistance', {}).get('distance_to_support', 0)
            features['resistance_distance'] = market_context.get('support_resistance', {}).get('distance_to_resistance', 0)
            features['volume_signal'] = 1 if market_context.get('volume', {}).get('volume_signal') == 'BULLISH_CONFIRMATION' else -1
            features['is_news_time'] = 1 if market_context.get('session', {}).get('is_news_time') else 0
        
        return features
    
    def predict_with_weighted_ensemble(self, symbol, timeframe, df):
        """التنبؤ مع تصويت مرجح بناءً على أداء النماذج"""
        try:
            # تحليل السوق
            market_context = self.market_analyzer.analyze_complete_market_context(
                symbol, df.reset_index().to_dict('records'), timeframe
            )
            
            if not market_context:
                logger.warning("Failed to analyze market context")
                return self._simple_fallback_prediction(df)
            
            # فحص شروط السوق
            market_score = market_context['score']
            
            logger.info(f"   📊 Market Score: {market_score}")
            
            # رفض الإشارات الضعيفة
            if abs(market_score) < self.min_market_score:
                logger.info(f"   ❌ Weak market score: {market_score} < {self.min_market_score}")
                return {
                    'action': 2,
                    'direction': 'HOLD',
                    'confidence': 0.0,
                    'reason': 'Weak market conditions'
                }
            
            # حساب الميزات
            features = self.calculate_enhanced_features(df, market_context)
            
            if features.empty:
                return self._simple_fallback_prediction(df)
            
            # تطبيق feature selection و scaling
            key = f"{symbol}_{timeframe}"
            
            if key not in self.models or key not in self.feature_selectors:
                logger.warning(f"No models or selectors for {key}")
                return self._simple_fallback_prediction(df)
            
            # Feature selection
            X = features.values.reshape(1, -1)
            X_selected = self.feature_selectors[key].transform(X)
            
            # Scaling
            X_scaled = self.scalers[key].transform(X_selected)
            
            # التنبؤ من كل نموذج
            predictions = []
            weights = []
            model_names = []
            
            # الحصول على أوزان النماذج
            model_weights = self.model_weights.get(key, {})
            
            for model_name, model in self.models[key].items():
                try:
                    pred = model.predict(X_scaled)[0]
                    prob = model.predict_proba(X_scaled)[0]
                    
                    predictions.append(pred)
                    
                    # الوزن الافتراضي 1، أو من الأداء السابق
                    weight = model_weights.get(model_name, 1.0)
                    weights.append(weight)
                    
                    model_names.append(model_name)
                    
                    logger.info(f"      {model_name}: {'BUY' if pred == 1 else 'SELL'} ({max(prob):.2%}) - Weight: {weight:.2f}")
                    
                except Exception as e:
                    logger.error(f"      ❌ {model_name}: {e}")
            
            if not predictions:
                return self._simple_fallback_prediction(df)
            
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
            
            # حساب الثقة
            confidence = abs(weighted_avg - 0.5) * 2  # تحويل لنسبة 0-1
            
            # فحص شروط الدخول الإضافية
            if confidence >= self.min_confidence:
                if self.check_entry_conditions(market_context, confidence):
                    logger.info(f"   ✅ Signal: {direction} with {confidence:.1%} confidence")
                    
                    return {
                        'action': action,
                        'direction': direction,
                        'confidence': confidence,
                        'market_context': market_context,
                        'models_used': model_names,
                        'weighted_score': weighted_avg
                    }
                else:
                    return {
                        'action': 2,
                        'direction': 'HOLD',
                        'confidence': confidence,
                        'reason': 'Entry conditions not met'
                    }
            else:
                logger.info(f"   ❌ Low confidence: {confidence:.1%} < {self.min_confidence:.1%}")
                return {
                    'action': 2,
                    'direction': 'HOLD',
                    'confidence': confidence,
                    'reason': 'Low confidence'
                }
                
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._simple_fallback_prediction(df)
    
    def _simple_fallback_prediction(self, df):
        """استراتيجية احتياطية بسيطة"""
        try:
            if len(df) < 20:
                return {
                    'action': 2,
                    'direction': 'HOLD',
                    'confidence': 0.0,
                    'reason': 'Insufficient data'
                }
            
            # حساب مؤشرات بسيطة
            sma10 = df['close'].rolling(10).mean().iloc[-1]
            sma20 = df['close'].rolling(20).mean().iloc[-1]
            current_price = df['close'].iloc[-1]
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] != 0 else 0
            rsi = 100 - (100 / (1 + rs))
            
            # قرار بسيط
            if sma10 > sma20 and current_price > sma10 and 30 < rsi < 70:
                return {
                    'action': 0,
                    'direction': 'BUY',
                    'confidence': 0.6,
                    'reason': 'Simple strategy'
                }
            elif sma10 < sma20 and current_price < sma10 and 30 < rsi < 70:
                return {
                    'action': 1,
                    'direction': 'SELL',
                    'confidence': 0.6,
                    'reason': 'Simple strategy'
                }
            else:
                return {
                    'action': 2,
                    'direction': 'HOLD',
                    'confidence': 0.0,
                    'reason': 'No clear signal'
                }
                
        except Exception as e:
            logger.error(f"Fallback prediction error: {e}")
            return {
                'action': 2,
                'direction': 'HOLD',
                'confidence': 0.0,
                'reason': 'Error in prediction'
            }
    
    def calculate_dynamic_sl_tp_with_partial(self, symbol, direction, entry_price, market_context):
        """حساب SL/TP مع إضافة TP0 للإغلاق الجزئي"""
        pip_value = 0.01 if 'JPY' in symbol else 0.0001
        
        # Handle None market_context
        if market_context is None:
            logger.warning(f"No market context for {symbol}, using default SL/TP")
            sl_percentage = 0.002  # 0.2%
            tp0_percentage = 0.0015  # 0.15% - نصف TP1
            tp1_percentage = 0.003  # 0.3%
            tp2_percentage = 0.005  # 0.5%
            
            if direction == 'BUY':
                return {
                    'sl_price': entry_price * (1 - sl_percentage),
                    'tp0_price': entry_price * (1 + tp0_percentage),  # إغلاق 50%
                    'tp1_price': entry_price * (1 + tp1_percentage),
                    'tp2_price': entry_price * (1 + tp2_percentage),
                    'partial_close_at_tp0': 0.5  # إغلاق 50% عند TP0
                }
            else:
                return {
                    'sl_price': entry_price * (1 + sl_percentage),
                    'tp0_price': entry_price * (1 - tp0_percentage),
                    'tp1_price': entry_price * (1 - tp1_percentage),
                    'tp2_price': entry_price * (1 - tp2_percentage),
                    'partial_close_at_tp0': 0.5
                }
        
        # حساب متقدم بناءً على السوق
        atr = market_context['volatility']['atr']
        volatility_level = market_context['volatility']['volatility_level']
        
        # تعديل المضاعفات بناءً على التقلب
        if volatility_level == 'VERY_HIGH':
            sl_multiplier = 3.0
            tp_multiplier = 4.0
        elif volatility_level == 'HIGH':
            sl_multiplier = 2.5
            tp_multiplier = 3.5
        else:
            sl_multiplier = 2.0
            tp_multiplier = 3.0
        
        sl_distance = atr * sl_multiplier
        
        # مستويات الدعم والمقاومة
        sr_levels = market_context.get('support_resistance', {})
        
        if direction == 'BUY':
            # ضبط SL بناءً على الدعم
            if sr_levels.get('nearest_support'):
                support_price = sr_levels['nearest_support']['price']
                support_distance = entry_price - support_price
                
                if 0.5 * sl_distance < support_distance < 2 * sl_distance:
                    sl_distance = support_distance + (5 * pip_value)
            
            sl_price = entry_price - sl_distance
            
            # حساب TPs
            tp0_price = entry_price + (sl_distance * 0.75)  # TP0 عند 75% من المخاطرة
            tp1_price = entry_price + (sl_distance * 1.5)   # TP1 عند 1.5x
            tp2_price = entry_price + (sl_distance * 2.5)   # TP2 عند 2.5x
            
        else:  # SELL
            # ضبط SL بناءً على المقاومة
            if sr_levels.get('nearest_resistance'):
                resistance_price = sr_levels['nearest_resistance']['price']
                resistance_distance = resistance_price - entry_price
                
                if 0.5 * sl_distance < resistance_distance < 2 * sl_distance:
                    sl_distance = resistance_distance + (5 * pip_value)
            
            sl_price = entry_price + sl_distance
            
            # حساب TPs
            tp0_price = entry_price - (sl_distance * 0.75)
            tp1_price = entry_price - (sl_distance * 1.5)
            tp2_price = entry_price - (sl_distance * 2.5)
        
        return {
            'sl_price': float(sl_price),
            'tp0_price': float(tp0_price),
            'tp1_price': float(tp1_price),
            'tp2_price': float(tp2_price),
            'sl_pips': float(sl_distance / pip_value),
            'tp0_pips': float(abs(tp0_price - entry_price) / pip_value),
            'tp1_pips': float(abs(tp1_price - entry_price) / pip_value),
            'tp2_pips': float(abs(tp2_price - entry_price) / pip_value),
            'risk_reward_ratio': 1.5,
            'partial_close_at_tp0': 0.5
        }
    
    def load_existing_models(self):
        """تحميل النماذج الموجودة"""
        if not os.path.exists(self.models_dir):
            return
        
        loaded_count = 0
        for file in os.listdir(self.models_dir):
            if file.endswith('_enhanced.pkl') and 'scaler' not in file and 'selector' not in file:
                try:
                    parts = file.replace('_enhanced.pkl', '').split('_')
                    if len(parts) >= 3:
                        symbol = parts[0]
                        timeframe = parts[1]
                        model_type = '_'.join(parts[2:])
                        
                        key = f"{symbol}_{timeframe}"
                        if key not in self.models:
                            self.models[key] = {}
                        
                        model_path = os.path.join(self.models_dir, file)
                        self.models[key][model_type] = joblib.load(model_path)
                        
                        # تحميل scaler
                        scaler_file = f"{symbol}_{timeframe}_scaler_enhanced.pkl"
                        scaler_path = os.path.join(self.models_dir, scaler_file)
                        if os.path.exists(scaler_path):
                            self.scalers[key] = joblib.load(scaler_path)
                        
                        # تحميل selector
                        selector_file = f"{symbol}_{timeframe}_selector_enhanced.pkl"
                        selector_path = os.path.join(self.models_dir, selector_file)
                        if os.path.exists(selector_path):
                            self.feature_selectors[key] = joblib.load(selector_path)
                        
                        loaded_count += 1
                        
                except Exception as e:
                    logger.error(f"Failed to load {file}: {e}")
        
        logger.info(f"   📂 Loaded {loaded_count} existing models")
        
        # حساب أوزان النماذج الأولية
        self.calculate_model_weights()
    
    def log_trade(self, symbol, action, entry_price, sl, tp, confidence, market_context, entry_reason):
        """تسجيل تفاصيل كل صفقة"""
        trade_log = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'action': action,
            'entry_price': entry_price,
            'stop_loss': sl,
            'take_profit': tp,
            'confidence': confidence,
            'entry_reason': entry_reason,
            'market_conditions': {
                'score': market_context.get('score', 0) if market_context else 0,
                'volatility': market_context.get('volatility', {}).get('volatility_level', 'UNKNOWN') if market_context else 'UNKNOWN',
                'trend': market_context.get('trend', {}).get('direction', 'UNKNOWN') if market_context else 'UNKNOWN',
                'session': market_context.get('session', {}).get('name', 'UNKNOWN') if market_context else 'UNKNOWN'
            }
        }
        
        self.performance_tracker['trades'].append(trade_log)
        
        # حفظ في ملف
        with open('trade_log.json', 'a') as f:
            f.write(json.dumps(trade_log) + '\n')
        
        logger.info(f"   📝 Trade logged: {symbol} {action} @ {entry_price:.5f}")
    
    def backtest(self, symbol, timeframe, start_date, end_date):
        """نظام Backtesting للاستراتيجية"""
        try:
            logger.info(f"🔄 Running backtest for {symbol} {timeframe} from {start_date} to {end_date}")
            
            # تحميل البيانات
            conn = sqlite3.connect(self.db_path)
            query = """
                SELECT * FROM price_data 
                WHERE symbol = ? AND time >= ? AND time <= ?
                ORDER BY time ASC
            """
            
            df = pd.read_sql_query(query, conn, params=(symbol, start_date, end_date))
            conn.close()
            
            if df.empty or len(df) < 1000:
                return {
                    'error': 'Insufficient data for backtesting'
                }
            
            # إعداد البيانات
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            
            # متغيرات الbacktest
            initial_balance = 10000
            balance = initial_balance
            trades = []
            equity_curve = [initial_balance]
            
            # المرور عبر البيانات
            window_size = 200
            for i in range(window_size, len(df) - 1):
                window_data = df.iloc[i-window_size:i+1]
                
                # الحصول على إشارة
                prediction = self.predict_with_weighted_ensemble(symbol, timeframe, window_data)
                
                if prediction['action'] != 2:  # ليس HOLD
                    # حساب حجم الصفقة
                    risk_pct = self.calculate_kelly_position(symbol, prediction['confidence'])
                    position_size = balance * risk_pct
                    
                    # محاكاة الصفقة
                    entry_price = df['close'].iloc[i]
                    
                    # البحث عن نتيجة الصفقة
                    if prediction['action'] == 0:  # BUY
                        # البحث عن أول سعر يحقق TP أو SL
                        for j in range(i+1, min(i+100, len(df))):
                            high = df['high'].iloc[j]
                            low = df['low'].iloc[j]
                            
                            # افتراض SL عند -1% و TP عند +1.5%
                            sl_price = entry_price * 0.99
                            tp_price = entry_price * 1.015
                            
                            if low <= sl_price:
                                # خسارة
                                loss = position_size * 0.01
                                balance -= loss
                                trades.append({
                                    'entry_time': df.index[i],
                                    'exit_time': df.index[j],
                                    'type': 'BUY',
                                    'result': 'LOSS',
                                    'pnl': -loss
                                })
                                break
                            elif high >= tp_price:
                                # ربح
                                profit = position_size * 0.015
                                balance += profit
                                trades.append({
                                    'entry_time': df.index[i],
                                    'exit_time': df.index[j],
                                    'type': 'BUY',
                                    'result': 'WIN',
                                    'pnl': profit
                                })
                                break
                    
                    else:  # SELL
                        for j in range(i+1, min(i+100, len(df))):
                            high = df['high'].iloc[j]
                            low = df['low'].iloc[j]
                            
                            sl_price = entry_price * 1.01
                            tp_price = entry_price * 0.985
                            
                            if high >= sl_price:
                                # خسارة
                                loss = position_size * 0.01
                                balance -= loss
                                trades.append({
                                    'entry_time': df.index[i],
                                    'exit_time': df.index[j],
                                    'type': 'SELL',
                                    'result': 'LOSS',
                                    'pnl': -loss
                                })
                                break
                            elif low <= tp_price:
                                # ربح
                                profit = position_size * 0.015
                                balance += profit
                                trades.append({
                                    'entry_time': df.index[i],
                                    'exit_time': df.index[j],
                                    'type': 'SELL',
                                    'result': 'WIN',
                                    'pnl': profit
                                })
                                break
                
                equity_curve.append(balance)
            
            # حساب النتائج
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t['result'] == 'WIN'])
            losing_trades = total_trades - winning_trades
            
            total_profit = sum([t['pnl'] for t in trades if t['pnl'] > 0])
            total_loss = abs(sum([t['pnl'] for t in trades if t['pnl'] < 0]))
            
            # المقاييس
            results = {
                'initial_balance': initial_balance,
                'final_balance': balance,
                'total_return': (balance - initial_balance) / initial_balance * 100,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': winning_trades / total_trades * 100 if total_trades > 0 else 0,
                'profit_factor': total_profit / total_loss if total_loss > 0 else float('inf'),
                'max_drawdown': self._calculate_max_drawdown(equity_curve),
                'trades': trades[-20:]  # آخر 20 صفقة
            }
            
            logger.info(f"✅ Backtest completed:")
            logger.info(f"   Return: {results['total_return']:.2f}%")
            logger.info(f"   Win Rate: {results['win_rate']:.1f}%")
            logger.info(f"   Profit Factor: {results['profit_factor']:.2f}")
            logger.info(f"   Max Drawdown: {results['max_drawdown']:.2f}%")
            
            return results
            
        except Exception as e:
            logger.error(f"Backtest error: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}
    
    def _calculate_max_drawdown(self, equity_curve):
        """حساب أقصى انخفاض"""
        peak = equity_curve[0]
        max_dd = 0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            else:
                dd = (peak - value) / peak * 100
                if dd > max_dd:
                    max_dd = dd
        
        return max_dd


# Global system instance
system = EnhancedMLTradingSystem()

@app.route('/status', methods=['GET'])
def status():
    """فحص حالة السيرفر"""
    return jsonify({
        'status': 'running',
        'version': '3.0',
        'type': 'enhanced_profit_based',
        'models_loaded': sum(len(models) for models in system.models.values()),
        'min_confidence': system.min_confidence,
        'min_market_score': system.min_market_score
    })

@app.route('/predict', methods=['POST'])
def predict():
    """التنبؤ مع النظام المحسن"""
    try:
        data = request.json
        symbol = data.get('symbol', '').replace('m', '').upper()
        clean_symbol = data.get('clean_symbol', symbol)
        timeframe = data.get('timeframe', 'M15')
        candles = data.get('candles', [])
        
        if not candles or len(candles) < 50:
            return jsonify({
                'action': 2,
                'direction': 'NONE',
                'confidence': 0.0,
                'error': 'Insufficient data'
            })
        
        # تحويل للDataFrame
        df = pd.DataFrame(candles)
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # التنبؤ
        logger.info(f"\n📊 Request: {symbol} ({clean_symbol}) {timeframe} - {len(candles)} candles")
        
        prediction_result = system.predict_with_weighted_ensemble(clean_symbol, timeframe, df)
        
        # معالجة النتيجة
        action = prediction_result.get('direction', 'HOLD')
        confidence = prediction_result.get('confidence', 0)
        market_context = prediction_result.get('market_context')
        
        logger.info(f"   📊 Prediction: {action} with {confidence:.1%} confidence")
        
        # إذا كانت إشارة قوية
        if action != 'HOLD' and confidence >= system.min_confidence:
            current_price = float(df['close'].iloc[-1])
            
            # حساب SL/TP مع الإغلاق الجزئي
            sl_tp_info = system.calculate_dynamic_sl_tp_with_partial(
                clean_symbol, action, current_price, market_context
            )
            
            # حساب حجم الصفقة بKelly
            kelly_pct = system.calculate_kelly_position(clean_symbol, confidence)
            lot_size = system.risk_manager.calculate_position_size(
                clean_symbol,
                current_price,
                sl_tp_info['sl_price'],
                market_context,
                confidence
            )[0]
            
            # تسجيل الصفقة
            entry_reason = f"Market score: {market_context['score'] if market_context else 'N/A'}, Models agree"
            system.log_trade(
                clean_symbol, action, current_price,
                sl_tp_info['sl_price'], sl_tp_info['tp1_price'],
                confidence, market_context, entry_reason
            )
            
            logger.info(f"   ✅ Signal: {action} @ {current_price:.5f}")
            logger.info(f"   🎯 SL: {sl_tp_info['sl_pips']:.0f} pips, TP0: {sl_tp_info['tp0_pips']:.0f} pips (50%), TP1: {sl_tp_info['tp1_pips']:.0f} pips")
            logger.info(f"   💰 Position size: {kelly_pct:.1%} of capital")
            
            return jsonify({
                'action': 0 if action == 'BUY' else 1,
                'direction': action,
                'confidence': float(confidence),
                'sl_price': float(sl_tp_info['sl_price']),
                'tp0_price': float(sl_tp_info['tp0_price']),
                'tp1_price': float(sl_tp_info['tp1_price']),
                'tp2_price': float(sl_tp_info['tp2_price']),
                'partial_close': sl_tp_info['partial_close_at_tp0'],
                'lot_size': float(lot_size),
                'position_size_pct': float(kelly_pct),
                'market_score': market_context['score'] if market_context else 0,
                'entry_reason': entry_reason
            })
        else:
            reason = prediction_result.get('reason', 'Low confidence or weak signal')
            logger.info(f"   ⏸️ HOLD: {reason}")
            
            return jsonify({
                'action': 2,
                'direction': 'HOLD',
                'confidence': float(confidence),
                'reason': reason
            })
            
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'action': 2,
            'direction': 'NONE',
            'confidence': 0.0,
            'error': str(e)
        })

@app.route('/update_position', methods=['POST'])
def update_position():
    """تحديث trailing stop للصفقة المفتوحة"""
    try:
        data = request.json
        symbol = data.get('symbol', '').replace('m', '').upper()
        current_price = float(data.get('current_price', 0))
        entry_price = float(data.get('entry_price', 0))
        current_sl = float(data.get('stop_loss', 0))
        direction = data.get('direction', 'BUY')
        
        # حساب trailing stop جديد
        new_sl = system.update_trailing_stop(symbol, current_price, entry_price, current_sl, direction)
        
        return jsonify({
            'new_stop_loss': float(new_sl),
            'updated': new_sl != current_sl
        })
        
    except Exception as e:
        logger.error(f"Error updating position: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/backtest', methods=['POST'])
def backtest():
    """تشغيل backtest على فترة محددة"""
    try:
        data = request.json
        symbol = data.get('symbol', 'EURUSD')
        timeframe = data.get('timeframe', 'M15')
        start_date = data.get('start_date', '2024-01-01')
        end_date = data.get('end_date', '2024-12-31')
        
        results = system.backtest(symbol, timeframe, start_date, end_date)
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Backtest error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/train', methods=['POST'])
def train():
    """تدريب النماذج لزوج معين"""
    try:
        data = request.json
        symbol = data.get('symbol', 'EURUSD')
        timeframe = data.get('timeframe', 'M15')
        
        success = system.train_enhanced_models(symbol, timeframe)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': f'Models trained successfully for {symbol} {timeframe}'
            })
        else:
            return jsonify({
                'status': 'failed',
                'message': f'Failed to train models for {symbol} {timeframe}'
            })
            
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/performance', methods=['GET'])
def get_performance():
    """الحصول على تقرير الأداء"""
    try:
        # حساب إحصائيات الأداء
        trades = system.performance_tracker['trades']
        
        if not trades:
            return jsonify({
                'message': 'No trades yet',
                'total_trades': 0
            })
        
        # تحليل الصفقات
        total_trades = len(trades)
        recent_trades = trades[-20:]  # آخر 20 صفقة
        
        return jsonify({
            'total_trades': total_trades,
            'recent_trades': recent_trades,
            'models_performance': system.performance_tracker.get('model_performance', {}),
            'model_weights': system.model_weights
        })
        
    except Exception as e:
        logger.error(f"Performance error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("🚀 Starting Enhanced ML Server (Profit-based)...")
    app.run(host='0.0.0.0', port=5000, debug=False)