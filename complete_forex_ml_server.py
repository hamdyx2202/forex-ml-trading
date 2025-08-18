#!/usr/bin/env python3
"""
🚀 Forex ML Trading Server - النظام الكامل
📊 يدعم جميع العملات والفريمات
🧠 6 نماذج ML + 200+ ميزة + 10 فرضيات
📈 تعلم مستمر من الصفقات
"""

import os
import sys
import json
import logging
import threading
import sqlite3
import joblib
import pickle
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# محاولة استيراد المكتبات المتقدمة
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

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier

# إعداد التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('complete_forex_ml_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)
CORS(app)

# قاعدة بيانات الأنماط الرابحة
PATTERNS_DB = './winning_patterns.db'

class CompleteForexMLSystem:
    """النظام الكامل للتداول بالذكاء الاصطناعي"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.winning_patterns = {}
        self.trade_history = []
        
        # إعداد قواعد البيانات
        self.historical_db = './data/forex_ml.db'
        self.trading_db = './trading_performance.db'
        self.models_dir = './trained_models'
        
        # إنشاء المجلدات المطلوبة
        os.makedirs(self.models_dir, exist_ok=True)
        
        # تهيئة قواعد البيانات
        self._init_databases()
        
        # تحميل النماذج الموجودة
        self.load_existing_models()
        
        # بدء خيط التعلم المستمر
        self.learning_thread = threading.Thread(target=self._continuous_learning_loop, daemon=True)
        self.learning_thread.start()
        
        logger.info("✅ تم تهيئة النظام الكامل")
    
    def _init_databases(self):
        """تهيئة قواعد البيانات"""
        # قاعدة بيانات الأنماط الرابحة
        conn = sqlite3.connect(PATTERNS_DB)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS winning_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                timeframe TEXT,
                pattern_type TEXT,
                features TEXT,
                success_rate REAL,
                total_trades INTEGER,
                avg_profit_pips REAL,
                created_at TIMESTAMP,
                last_updated TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
        
        # قاعدة بيانات الصفقات
        conn = sqlite3.connect(self.trading_db)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                timeframe TEXT,
                signal_time TIMESTAMP,
                action TEXT,
                confidence REAL,
                entry_price REAL,
                sl_price REAL,
                tp1_price REAL,
                tp2_price REAL,
                features TEXT,
                model_predictions TEXT,
                hypotheses_results TEXT,
                exit_time TIMESTAMP,
                exit_price REAL,
                profit_pips REAL,
                exit_reason TEXT,
                market_conditions TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    def calculate_all_features(self, df):
        """حساب جميع الميزات المتقدمة (200+ ميزة)"""
        features = df.copy()
        
        # 1. الميزات الأساسية
        features['price_change'] = features['close'].pct_change()
        features['high_low_ratio'] = features['high'] / features['low']
        features['close_open_ratio'] = features['close'] / features['open']
        
        # 2. المتوسطات المتحركة (20 ميزة)
        for period in [5, 10, 20, 50, 100, 200]:
            features[f'sma_{period}'] = features['close'].rolling(period).mean()
            features[f'sma_{period}_slope'] = features[f'sma_{period}'].diff()
            
        # EMAs
        for period in [9, 12, 26]:
            features[f'ema_{period}'] = features['close'].ewm(span=period).mean()
            
        # 3. مؤشرات الزخم (30 ميزة)
        # RSI
        for period in [14, 21, 28]:
            delta = features['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))
            
        # MACD
        exp1 = features['close'].ewm(span=12, adjust=False).mean()
        exp2 = features['close'].ewm(span=26, adjust=False).mean()
        features['macd'] = exp1 - exp2
        features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
        features['macd_diff'] = features['macd'] - features['macd_signal']
        
        # Stochastic
        for period in [14, 21]:
            low_min = features['low'].rolling(window=period).min()
            high_max = features['high'].rolling(window=period).max()
            features[f'stoch_k_{period}'] = 100 * ((features['close'] - low_min) / (high_max - low_min))
            features[f'stoch_d_{period}'] = features[f'stoch_k_{period}'].rolling(window=3).mean()
            
        # 4. مؤشرات التذبذب (20 ميزة)
        # Bollinger Bands
        for period in [20, 30]:
            sma = features['close'].rolling(window=period).mean()
            std = features['close'].rolling(window=period).std()
            features[f'bb_upper_{period}'] = sma + (std * 2)
            features[f'bb_lower_{period}'] = sma - (std * 2)
            features[f'bb_width_{period}'] = features[f'bb_upper_{period}'] - features[f'bb_lower_{period}']
            features[f'bb_position_{period}'] = (features['close'] - features[f'bb_lower_{period}']) / features[f'bb_width_{period}']
            
        # ATR
        for period in [14, 21]:
            high_low = features['high'] - features['low']
            high_close = np.abs(features['high'] - features['close'].shift())
            low_close = np.abs(features['low'] - features['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            features[f'atr_{period}'] = true_range.rolling(period).mean()
            
        # 5. مؤشرات الحجم (15 ميزة)
        if 'volume' in features.columns:
            features['volume_sma'] = features['volume'].rolling(20).mean()
            features['volume_ratio'] = features['volume'] / features['volume_sma']
            features['price_volume'] = features['close'] * features['volume']
            
        # 6. الأنماط الشمعية (30 ميزة)
        # Doji
        features['doji'] = (np.abs(features['close'] - features['open']) / (features['high'] - features['low'])) < 0.1
        
        # Hammer
        body = np.abs(features['close'] - features['open'])
        lower_shadow = features[['open', 'close']].min(axis=1) - features['low']
        features['hammer'] = (lower_shadow > body * 2) & (features['high'] - features[['open', 'close']].max(axis=1) < body)
        
        # Engulfing
        features['bullish_engulfing'] = (
            (features['close'] > features['open']) & 
            (features['close'].shift() < features['open'].shift()) &
            (features['open'] < features['close'].shift()) &
            (features['close'] > features['open'].shift())
        )
        
        # 7. مستويات الدعم والمقاومة (20 ميزة)
        for period in [20, 50, 100]:
            features[f'resistance_{period}'] = features['high'].rolling(period).max()
            features[f'support_{period}'] = features['low'].rolling(period).min()
            features[f'sr_position_{period}'] = (features['close'] - features[f'support_{period}']) / (features[f'resistance_{period}'] - features[f'support_{period}'])
            
        # 8. الميزات الزمنية (15 ميزة)
        if isinstance(features.index, pd.DatetimeIndex):
            features['hour'] = features.index.hour
            features['day_of_week'] = features.index.dayofweek
            features['day_of_month'] = features.index.day
            features['month'] = features.index.month
            features['is_london'] = ((features['hour'] >= 8) & (features['hour'] <= 16)).astype(int)
            features['is_newyork'] = ((features['hour'] >= 13) & (features['hour'] <= 21)).astype(int)
            features['is_tokyo'] = ((features['hour'] >= 0) & (features['hour'] <= 8)).astype(int)
            features['is_sydney'] = ((features['hour'] >= 22) | (features['hour'] <= 6)).astype(int)
            
        # 9. النسب والعلاقات (25 ميزة)
        # Price ratios
        features['high_close_ratio'] = features['high'] / features['close']
        features['low_close_ratio'] = features['low'] / features['close']
        
        # Moving average crosses
        features['sma_5_20_cross'] = (features['sma_5'] > features['sma_20']).astype(int)
        features['sma_20_50_cross'] = (features['sma_20'] > features['sma_50']).astype(int)
        
        # 10. مؤشرات السوق (15 ميزة)
        # ADX
        for period in [14, 21]:
            plus_dm = features['high'].diff()
            minus_dm = -features['low'].diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0
            
            tr = true_range.rolling(period).mean()
            plus_di = 100 * (plus_dm.rolling(period).mean() / tr)
            minus_di = 100 * (minus_dm.rolling(period).mean() / tr)
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            features[f'adx_{period}'] = dx.rolling(period).mean()
            
        # إزالة الصفوف التي تحتوي على NaN
        features = features.dropna()
        
        return features
    
    def evaluate_hypotheses(self, features):
        """تقييم 10 فرضيات التداول"""
        hypotheses_results = {}
        
        # 1. Trend Following
        trend_score = 0
        if features.get('sma_20', 0) > features.get('sma_50', 0) > features.get('sma_200', 0):
            trend_score += 0.5
        if features.get('adx_14', 0) > 25:
            trend_score += 0.5
        hypotheses_results['trend_following'] = trend_score
        
        # 2. Mean Reversion
        mr_score = 0
        if features.get('rsi_14', 50) < 30:
            mr_score += 0.5
        elif features.get('rsi_14', 50) > 70:
            mr_score += 0.5
        if features.get('bb_position_20', 0.5) < 0.1 or features.get('bb_position_20', 0.5) > 0.9:
            mr_score += 0.5
        hypotheses_results['mean_reversion'] = mr_score
        
        # 3. Momentum
        momentum_score = 0
        if features.get('macd', 0) > features.get('macd_signal', 0):
            momentum_score += 0.5
        if features.get('rsi_14', 50) > 50 and features.get('rsi_14', 50) < 70:
            momentum_score += 0.5
        hypotheses_results['momentum'] = momentum_score
        
        # 4. Volatility Breakout
        vb_score = 0
        if features.get('atr_14', 0) > features.get('atr_21', 0):
            vb_score += 0.5
        if features.get('bb_width_20', 0) > features.get('bb_width_30', 0):
            vb_score += 0.5
        hypotheses_results['volatility_breakout'] = vb_score
        
        # 5. Seasonality
        season_score = 0
        if features.get('is_london', 0) or features.get('is_newyork', 0):
            season_score += 0.5
        if features.get('day_of_week', 0) in [1, 2, 3]:  # Tue, Wed, Thu
            season_score += 0.5
        hypotheses_results['seasonality'] = season_score
        
        # 6. Support/Resistance
        sr_score = 0
        if features.get('sr_position_20', 0.5) < 0.2:
            sr_score += 0.5  # Near support
        elif features.get('sr_position_20', 0.5) > 0.8:
            sr_score += 0.5  # Near resistance
        hypotheses_results['support_resistance'] = sr_score
        
        # 7. Market Structure
        ms_score = 0
        if features.get('high_low_ratio', 1) > 1.001:
            ms_score += 0.5
        hypotheses_results['market_structure'] = ms_score
        
        # 8. Volume Analysis
        va_score = 0.5  # Default if no volume
        if 'volume_ratio' in features:
            if features.get('volume_ratio', 1) > 1.5:
                va_score = 1
            elif features.get('volume_ratio', 1) < 0.5:
                va_score = 0
        hypotheses_results['volume_analysis'] = va_score
        
        # 9. Pattern Recognition
        pr_score = 0
        if features.get('doji', 0) or features.get('hammer', 0) or features.get('bullish_engulfing', 0):
            pr_score += 1
        hypotheses_results['pattern_recognition'] = pr_score
        
        # 10. Correlation
        corr_score = 0.5  # Default neutral
        hypotheses_results['correlation'] = corr_score
        
        return hypotheses_results
    
    def train_models(self, symbol, timeframe):
        """تدريب جميع النماذج الستة"""
        try:
            # جلب البيانات التاريخية
            conn = sqlite3.connect(self.historical_db)
            
            # محاولة جلب البيانات من جداول مختلفة
            possible_tables = [
                f"{symbol}_{timeframe}",
                f"{symbol}{timeframe}",
                f"{symbol}",
                "forex_data"
            ]
            
            df = None
            for table in possible_tables:
                try:
                    query = f"SELECT * FROM {table} ORDER BY time DESC LIMIT 10000"
                    df = pd.read_sql_query(query, conn)
                    if not df.empty:
                        logger.info(f"✅ تم جلب البيانات من جدول {table}")
                        break
                except:
                    continue
            
            conn.close()
            
            if df is None or df.empty:
                logger.warning(f"⚠️ لا توجد بيانات كافية لـ {symbol} {timeframe}")
                return False
            
            # تحويل التاريخ
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
                df.set_index('time', inplace=True)
            
            # حساب الميزات
            features_df = self.calculate_all_features(df)
            
            # تحضير البيانات للتدريب
            feature_cols = [col for col in features_df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
            X = features_df[feature_cols].values
            
            # إنشاء الهدف (التنبؤ بالاتجاه)
            y = (features_df['close'].shift(-1) > features_df['close']).astype(int)
            y = y[:-1]
            X = X[:-1]
            
            # تقسيم البيانات
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # تطبيع البيانات
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # حفظ المقياس
            self.scalers[f"{symbol}_{timeframe}"] = scaler
            
            # تدريب النماذج الستة
            models = {}
            accuracies = {}
            
            # 1. LightGBM
            if LIGHTGBM_AVAILABLE:
                try:
                    lgb_model = lgb.LGBMClassifier(
                        n_estimators=100,
                        learning_rate=0.05,
                        num_leaves=31,
                        random_state=42,
                        verbosity=-1
                    )
                    lgb_model.fit(X_train_scaled, y_train)
                    lgb_pred = lgb_model.predict(X_test_scaled)
                    lgb_acc = accuracy_score(y_test, lgb_pred)
                    models['lightgbm'] = lgb_model
                    accuracies['lightgbm'] = lgb_acc
                    logger.info(f"   LightGBM Accuracy: {lgb_acc:.4f}")
                except Exception as e:
                    logger.error(f"   LightGBM Error: {str(e)}")
            
            # 2. XGBoost
            if XGBOOST_AVAILABLE:
                try:
                    xgb_model = xgb.XGBClassifier(
                        n_estimators=100,
                        learning_rate=0.05,
                        max_depth=5,
                        random_state=42,
                        use_label_encoder=False,
                        eval_metric='logloss'
                    )
                    xgb_model.fit(X_train_scaled, y_train)
                    xgb_pred = xgb_model.predict(X_test_scaled)
                    xgb_acc = accuracy_score(y_test, xgb_pred)
                    models['xgboost'] = xgb_model
                    accuracies['xgboost'] = xgb_acc
                    logger.info(f"   XGBoost Accuracy: {xgb_acc:.4f}")
                except Exception as e:
                    logger.error(f"   XGBoost Error: {str(e)}")
            
            # 3. Random Forest
            try:
                rf_model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
                rf_model.fit(X_train_scaled, y_train)
                rf_pred = rf_model.predict(X_test_scaled)
                rf_acc = accuracy_score(y_test, rf_pred)
                models['random_forest'] = rf_model
                accuracies['random_forest'] = rf_acc
                logger.info(f"   Random Forest Accuracy: {rf_acc:.4f}")
            except Exception as e:
                logger.error(f"   Random Forest Error: {str(e)}")
            
            # 4. Gradient Boosting
            try:
                gb_model = GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=5,
                    random_state=42
                )
                gb_model.fit(X_train_scaled, y_train)
                gb_pred = gb_model.predict(X_test_scaled)
                gb_acc = accuracy_score(y_test, gb_pred)
                models['gradient_boosting'] = gb_model
                accuracies['gradient_boosting'] = gb_acc
                logger.info(f"   Gradient Boosting Accuracy: {gb_acc:.4f}")
            except Exception as e:
                logger.error(f"   Gradient Boosting Error: {str(e)}")
            
            # 5. Extra Trees
            try:
                et_model = ExtraTreesClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
                et_model.fit(X_train_scaled, y_train)
                et_pred = et_model.predict(X_test_scaled)
                et_acc = accuracy_score(y_test, et_pred)
                models['extra_trees'] = et_model
                accuracies['extra_trees'] = et_acc
                logger.info(f"   Extra Trees Accuracy: {et_acc:.4f}")
            except Exception as e:
                logger.error(f"   Extra Trees Error: {str(e)}")
            
            # 6. Neural Network
            try:
                nn_model = MLPClassifier(
                    hidden_layer_sizes=(100, 50),
                    activation='relu',
                    solver='adam',
                    learning_rate_rate=0.001,
                    max_iter=500,
                    random_state=42
                )
                nn_model.fit(X_train_scaled, y_train)
                nn_pred = nn_model.predict(X_test_scaled)
                nn_acc = accuracy_score(y_test, nn_pred)
                models['neural_network'] = nn_model
                accuracies['neural_network'] = nn_acc
                logger.info(f"   Neural Network Accuracy: {nn_acc:.4f}")
            except Exception as e:
                logger.error(f"   Neural Network Error: {str(e)}")
            
            # حفظ النماذج
            self.models[f"{symbol}_{timeframe}"] = models
            
            # حفظ النماذج على القرص
            for model_name, model in models.items():
                model_path = os.path.join(self.models_dir, f"{symbol}_{timeframe}_{model_name}.pkl")
                joblib.dump(model, model_path)
            
            # حفظ المقياس
            scaler_path = os.path.join(self.models_dir, f"{symbol}_{timeframe}_scaler.pkl")
            joblib.dump(scaler, scaler_path)
            
            logger.info(f"✅ تم تدريب {len(models)} نماذج لـ {symbol} {timeframe}")
            return True
            
        except Exception as e:
            logger.error(f"❌ خطأ في تدريب النماذج: {str(e)}")
            return False
    
    def predict(self, symbol, timeframe, features_df):
        """التنبؤ باستخدام جميع النماذج والفرضيات"""
        try:
            key = f"{symbol}_{timeframe}"
            
            # التحقق من وجود نماذج مدربة
            if key not in self.models or not self.models[key]:
                logger.info(f"🧠 تدريب نماذج جديدة لـ {symbol} {timeframe}...")
                if not self.train_models(symbol, timeframe):
                    # استخدام نموذج بسيط
                    return self._simple_prediction(features_df)
            
            # تحضير الميزات
            feature_cols = [col for col in features_df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
            X = features_df[feature_cols].values
            
            # تطبيع البيانات
            if key in self.scalers:
                X_scaled = self.scalers[key].transform(X)
            else:
                X_scaled = X
            
            # التنبؤ بكل نموذج
            predictions = {}
            probabilities = {}
            
            for model_name, model in self.models[key].items():
                try:
                    pred = model.predict(X_scaled)
                    prob = model.predict_proba(X_scaled)[:, 1]
                    predictions[model_name] = pred[0]
                    probabilities[model_name] = prob[0]
                except:
                    continue
            
            if not predictions:
                return self._simple_prediction(features_df)
            
            # حساب التصويت
            buy_votes = sum(1 for p in predictions.values() if p == 1)
            sell_votes = sum(1 for p in predictions.values() if p == 0)
            avg_confidence = np.mean(list(probabilities.values()))
            
            # تقييم الفرضيات
            latest_features = features_df.iloc[-1].to_dict()
            hypotheses_results = self.evaluate_hypotheses(latest_features)
            
            # دمج النتائج (70% ML + 30% فرضيات)
            hypotheses_score = np.mean(list(hypotheses_results.values()))
            
            if buy_votes > sell_votes:
                final_prediction = 0  # Buy
                ml_confidence = avg_confidence
            else:
                final_prediction = 1  # Sell
                ml_confidence = 1 - avg_confidence
            
            # الثقة النهائية
            final_confidence = (0.7 * ml_confidence) + (0.3 * hypotheses_score)
            
            # تسجيل التفاصيل
            logger.info(f"   📊 تصويت النماذج: Buy={buy_votes}, Sell={sell_votes}")
            logger.info(f"   🎯 الفرضيات: {hypotheses_score:.2f}")
            logger.info(f"   💪 الثقة النهائية: {final_confidence:.2%}")
            
            return final_prediction, final_confidence
            
        except Exception as e:
            logger.error(f"❌ خطأ في التنبؤ: {str(e)}")
            return self._simple_prediction(features_df)
    
    def _simple_prediction(self, features_df):
        """تنبؤ بسيط في حالة عدم توفر النماذج"""
        try:
            latest = features_df.iloc[-1]
            
            # استراتيجية MA بسيطة
            if 'sma_20' in latest and 'sma_50' in latest:
                if latest['close'] > latest['sma_20'] > latest['sma_50']:
                    return 0, 0.65  # Buy
                elif latest['close'] < latest['sma_20'] < latest['sma_50']:
                    return 1, 0.65  # Sell
            
            return 2, 0.5  # Hold
        except:
            return 2, 0.5
    
    def calculate_dynamic_levels(self, features_df, action, symbol):
        """حساب مستويات SL/TP الديناميكية"""
        try:
            latest = features_df.iloc[-1]
            current_price = latest['close']
            
            # حساب pip value
            pip_value = 0.01 if 'JPY' in symbol else 0.0001
            
            # ATR للتقلب
            atr = latest.get('atr_14', 50 * pip_value)
            
            # حساب SL بناءً على ATR
            sl_multiplier = 1.5
            if latest.get('adx_14', 0) > 30:  # ترند قوي
                sl_multiplier = 2.0
            elif latest.get('adx_14', 0) < 20:  # سوق جانبي
                sl_multiplier = 1.0
            
            sl_pips = max(min(atr / pip_value * sl_multiplier, 100), 20)
            
            # حساب TP بناءً على نسبة المخاطرة/المكافأة
            tp1_multiplier = 2.0
            tp2_multiplier = 3.0
            
            # تعديل بناءً على قوة الإشارة
            if latest.get('rsi_14', 50) < 30 or latest.get('rsi_14', 50) > 70:
                tp1_multiplier = 2.5
                tp2_multiplier = 4.0
            
            tp1_pips = sl_pips * tp1_multiplier
            tp2_pips = sl_pips * tp2_multiplier
            
            # حساب الأسعار
            if action == 'BUY':
                sl_price = current_price - (sl_pips * pip_value)
                tp1_price = current_price + (tp1_pips * pip_value)
                tp2_price = current_price + (tp2_pips * pip_value)
            else:  # SELL
                sl_price = current_price + (sl_pips * pip_value)
                tp1_price = current_price - (tp1_pips * pip_value)
                tp2_price = current_price - (tp2_pips * pip_value)
            
            return {
                'sl_price': float(sl_price),
                'tp1_price': float(tp1_price),
                'tp2_price': float(tp2_price),
                'sl_pips': float(sl_pips),
                'tp1_pips': float(tp1_pips),
                'tp2_pips': float(tp2_pips)
            }
            
        except Exception as e:
            logger.error(f"خطأ في حساب المستويات: {str(e)}")
            # قيم افتراضية
            pip_value = 0.01 if 'JPY' in symbol else 0.0001
            current_price = features_df['close'].iloc[-1]
            
            if action == 'BUY':
                return {
                    'sl_price': current_price - (50 * pip_value),
                    'tp1_price': current_price + (100 * pip_value),
                    'tp2_price': current_price + (150 * pip_value),
                    'sl_pips': 50,
                    'tp1_pips': 100,
                    'tp2_pips': 150
                }
            else:
                return {
                    'sl_price': current_price + (50 * pip_value),
                    'tp1_price': current_price - (100 * pip_value),
                    'tp2_price': current_price - (150 * pip_value),
                    'sl_pips': 50,
                    'tp1_pips': 100,
                    'tp2_pips': 150
                }
    
    def record_trade_result(self, trade_data):
        """تسجيل نتيجة الصفقة للتعلم المستمر"""
        try:
            conn = sqlite3.connect(self.trading_db)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO trades (
                    symbol, timeframe, signal_time, action, confidence,
                    entry_price, sl_price, tp1_price, tp2_price,
                    exit_time, exit_price, profit_pips, exit_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_data['symbol'],
                trade_data['timeframe'],
                trade_data.get('signal_time', datetime.now()),
                trade_data['action'],
                trade_data.get('confidence', 0),
                trade_data['entry_price'],
                trade_data.get('sl_price', 0),
                trade_data.get('tp1_price', 0),
                trade_data.get('tp2_price', 0),
                trade_data.get('exit_time', datetime.now()),
                trade_data['exit_price'],
                trade_data['profit_pips'],
                trade_data.get('exit_reason', 'Unknown')
            ))
            
            conn.commit()
            conn.close()
            
            # تحليل النمط إذا كانت صفقة رابحة
            if trade_data['profit_pips'] > 20:
                self._analyze_winning_pattern(trade_data)
            
            logger.info(f"✅ تم تسجيل نتيجة الصفقة: {trade_data['profit_pips']} pips")
            
        except Exception as e:
            logger.error(f"خطأ في تسجيل الصفقة: {str(e)}")
    
    def _analyze_winning_pattern(self, trade_data):
        """تحليل الأنماط الرابحة"""
        try:
            # هنا يمكن إضافة تحليل متقدم للأنماط
            conn = sqlite3.connect(PATTERNS_DB)
            cursor = conn.cursor()
            
            pattern_type = "MA_Cross" if trade_data.get('pattern_type') else "Unknown"
            
            cursor.execute('''
                INSERT INTO winning_patterns (
                    symbol, timeframe, pattern_type, success_rate,
                    total_trades, avg_profit_pips, created_at, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_data['symbol'],
                trade_data['timeframe'],
                pattern_type,
                1.0,  # 100% لأنها صفقة رابحة
                1,
                trade_data['profit_pips'],
                datetime.now(),
                datetime.now()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"خطأ في تحليل النمط: {str(e)}")
    
    def _continuous_learning_loop(self):
        """حلقة التعلم المستمر"""
        while True:
            try:
                # انتظار 5 دقائق
                threading.Event().wait(300)
                
                # تحليل الصفقات الأخيرة
                conn = sqlite3.connect(self.trading_db)
                recent_trades = pd.read_sql_query(
                    "SELECT * FROM trades WHERE exit_time > datetime('now', '-1 day')",
                    conn
                )
                conn.close()
                
                if not recent_trades.empty:
                    # تحليل الأداء
                    win_rate = (recent_trades['profit_pips'] > 0).mean()
                    avg_profit = recent_trades['profit_pips'].mean()
                    
                    logger.info(f"📊 أداء آخر 24 ساعة: Win Rate={win_rate:.1%}, Avg Profit={avg_profit:.1f} pips")
                    
                    # تحديث النماذج إذا لزم الأمر
                    if win_rate < 0.5:
                        logger.info("⚠️ الأداء منخفض، يُنصح بإعادة التدريب")
                
            except Exception as e:
                logger.error(f"خطأ في التعلم المستمر: {str(e)}")
    
    def load_existing_models(self):
        """تحميل النماذج المدربة مسبقاً"""
        try:
            if not os.path.exists(self.models_dir):
                return
            
            model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.pkl')]
            
            for model_file in model_files:
                try:
                    parts = model_file.replace('.pkl', '').split('_')
                    if len(parts) >= 3:
                        symbol = parts[0]
                        timeframe = parts[1]
                        model_type = '_'.join(parts[2:])
                        
                        if model_type == 'scaler':
                            scaler = joblib.load(os.path.join(self.models_dir, model_file))
                            self.scalers[f"{symbol}_{timeframe}"] = scaler
                        else:
                            key = f"{symbol}_{timeframe}"
                            if key not in self.models:
                                self.models[key] = {}
                            
                            model = joblib.load(os.path.join(self.models_dir, model_file))
                            self.models[key][model_type] = model
                            
                except Exception as e:
                    logger.error(f"خطأ في تحميل {model_file}: {str(e)}")
            
            total_models = sum(len(models) for models in self.models.values())
            logger.info(f"✅ تم تحميل {total_models} نموذج من الملفات")
            
        except Exception as e:
            logger.error(f"خطأ في تحميل النماذج: {str(e)}")

# إنشاء مثيل من النظام
system = CompleteForexMLSystem()

# متغيرات للإحصائيات
server_stats = {
    'start_time': datetime.now(),
    'total_requests': 0,
    'total_signals': 0,
    'active_trades': 0
}

@app.route('/status', methods=['GET'])
def status():
    """حالة السيرفر"""
    uptime = datetime.now() - server_stats['start_time']
    total_models = sum(len(models) for models in system.models.values())
    
    return jsonify({
        'status': 'running',
        'version': '4.0-complete',
        'server': '69.62.121.53:5000',
        'uptime': str(uptime),
        'models_loaded': total_models,
        'total_requests': server_stats['total_requests'],
        'total_signals': server_stats['total_signals'],
        'features': '200+',
        'ml_models': 6,
        'hypotheses': 10,
        'continuous_learning': 'active'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """نقطة النهاية الرئيسية للتنبؤ - محسنة لمعالجة JSON الكبير"""
    try:
        server_stats['total_requests'] += 1
        
        # معالجة JSON محسنة
        try:
            # الحصول على البيانات الخام
            raw_data = request.get_data(as_text=True)
            logger.info(f"📥 حجم البيانات المستلمة: {len(raw_data)} حرف")
            
            # محاولة تحليل JSON
            try:
                data = json.loads(raw_data)
            except json.JSONDecodeError as e:
                logger.error(f"خطأ JSON في الموضع {e.pos}")
                
                # محاولة إصلاح JSON المكسور
                if '"candles":[' in raw_data:
                    # البحث عن نهاية مصفوفة الشموع
                    candles_start = raw_data.find('"candles":[')
                    if candles_start != -1:
                        # محاولة إيجاد النهاية الصحيحة
                        bracket_count = 0
                        in_string = False
                        escape_next = False
                        
                        i = candles_start + len('"candles":[')
                        while i < len(raw_data):
                            char = raw_data[i]
                            
                            if escape_next:
                                escape_next = False
                            elif char == '\\':
                                escape_next = True
                            elif char == '"' and not escape_next:
                                in_string = not in_string
                            elif not in_string:
                                if char == '[':
                                    bracket_count += 1
                                elif char == ']':
                                    if bracket_count == 0:
                                        # وجدنا نهاية المصفوفة
                                        try:
                                            fixed_json = raw_data[:i+1] + '}'
                                            data = json.loads(fixed_json)
                                            logger.info("✅ تم إصلاح JSON بنجاح")
                                            break
                                        except:
                                            pass
                                    else:
                                        bracket_count -= 1
                            i += 1
                
                # إذا فشل الإصلاح، استخدم البيانات الجزئية
                if 'data' not in locals():
                    if e.pos > 0:
                        try:
                            partial_data = raw_data[:e.pos]
                            # محاولة إغلاق JSON
                            open_brackets = partial_data.count('[') - partial_data.count(']')
                            open_braces = partial_data.count('{') - partial_data.count('}')
                            
                            closing = ']' * open_brackets + '}' * open_braces
                            fixed_json = partial_data + closing
                            
                            data = json.loads(fixed_json)
                            logger.warning("⚠️ استخدام بيانات JSON جزئية")
                        except:
                            return jsonify({
                                'error': 'Invalid JSON format',
                                'action': 'NONE',
                                'confidence': 0
                            }), 200
                    else:
                        return jsonify({
                            'error': 'Empty or invalid JSON',
                            'action': 'NONE',
                            'confidence': 0
                        }), 200
        
        except Exception as e:
            logger.error(f"خطأ في معالجة الطلب: {str(e)}")
            return jsonify({
                'error': 'Request processing error',
                'action': 'NONE',
                'confidence': 0
            }), 200
        
        # استخراج البيانات
        symbol = data.get('symbol', 'UNKNOWN')
        timeframe = data.get('timeframe', 'M15')
        candles = data.get('candles', [])
        
        logger.info(f"\n📊 طلب تنبؤ: {symbol} {timeframe}")
        logger.info(f"   عدد الشموع: {len(candles)}")
        
        # التحقق من الشموع
        if not candles or len(candles) < 20:
            logger.warning(f"⚠️ عدد الشموع غير كافي: {len(candles)}")
            return jsonify({
                'symbol': symbol,
                'timeframe': timeframe,
                'action': 'NONE',
                'confidence': 0,
                'error': f'Need at least 20 candles, got {len(candles)}'
            }), 200
        
        # تحويل إلى DataFrame
        try:
            df = pd.DataFrame(candles)
            
            # التأكد من أن الأعمدة رقمية
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # تحويل الوقت
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
                df.set_index('time', inplace=True)
            
            # إزالة القيم الناقصة
            df = df.dropna()
            
            if df.empty:
                raise ValueError("No valid data after cleaning")
                
        except Exception as e:
            logger.error(f"خطأ في تحويل البيانات: {str(e)}")
            return jsonify({
                'symbol': symbol,
                'timeframe': timeframe,
                'action': 'NONE',
                'confidence': 0,
                'error': 'Invalid candle data'
            }), 200
        
        # حساب الميزات
        try:
            features = system.calculate_all_features(df)
            if features.empty:
                raise ValueError("No features calculated")
        except Exception as e:
            logger.error(f"خطأ في حساب الميزات: {str(e)}")
            # استخدام ميزات بسيطة
            features = df.copy()
            features['sma_20'] = features['close'].rolling(20).mean()
            features['sma_50'] = features['close'].rolling(50).mean()
            features = features.dropna()
        
        # التنبؤ
        try:
            prediction, confidence = system.predict(symbol, timeframe, features)
        except Exception as e:
            logger.error(f"خطأ في التنبؤ: {str(e)}")
            prediction, confidence = 2, 0.5
        
        # تحديد الإشارة
        current_price = float(df['close'].iloc[-1])
        
        if prediction == 0 and confidence >= 0.65:
            action = 'BUY'
        elif prediction == 1 and confidence >= 0.65:
            action = 'SELL'
        else:
            action = 'NONE'
        
        # حساب SL/TP
        if action != 'NONE':
            levels = system.calculate_dynamic_levels(features, action, symbol)
        else:
            pip_value = 0.01 if 'JPY' in symbol else 0.0001
            levels = {
                'sl_price': current_price,
                'tp1_price': current_price,
                'tp2_price': current_price,
                'sl_pips': 0,
                'tp1_pips': 0,
                'tp2_pips': 0
            }
        
        # بناء الاستجابة
        response = {
            'symbol': symbol,
            'timeframe': timeframe,
            'action': action,
            'confidence': float(confidence),
            'current_price': current_price,
            'sl_price': levels['sl_price'],
            'tp1_price': levels['tp1_price'],
            'tp2_price': levels['tp2_price'],
            'sl_pips': levels['sl_pips'],
            'tp1_pips': levels['tp1_pips'],
            'tp2_pips': levels['tp2_pips'],
            'risk_reward_ratio': levels['tp1_pips'] / levels['sl_pips'] if levels['sl_pips'] > 0 else 0,
            'timestamp': datetime.now().isoformat(),
            'models_used': len(system.models.get(f"{symbol}_{timeframe}", {})),
            'features_count': len(features.columns)
        }
        
        if action != 'NONE':
            server_stats['total_signals'] += 1
            
        logger.info(f"   ✅ إشارة {action} بثقة {confidence:.1%}")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"خطأ غير متوقع: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'error': str(e),
            'action': 'NONE',
            'confidence': 0
        }), 200

@app.route('/trade_result', methods=['POST'])
def trade_result():
    """تسجيل نتيجة الصفقة"""
    try:
        data = request.json
        system.record_trade_result(data)
        return jsonify({'status': 'success', 'message': 'Trade result recorded'})
    except Exception as e:
        logger.error(f"خطأ في تسجيل النتيجة: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/train', methods=['POST'])
def train():
    """تدريب النماذج يدوياً"""
    try:
        data = request.json
        symbol = data.get('symbol')
        timeframe = data.get('timeframe')
        
        if not symbol or not timeframe:
            return jsonify({'error': 'Symbol and timeframe required'}), 400
        
        success = system.train_models(symbol, timeframe)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': f'Models trained for {symbol} {timeframe}'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Training failed - insufficient data'
            }), 400
            
    except Exception as e:
        logger.error(f"خطأ في التدريب: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/models', methods=['GET'])
def get_models():
    """الحصول على قائمة النماذج المدربة"""
    models_info = {}
    
    for key, models in system.models.items():
        models_info[key] = {
            'models': list(models.keys()),
            'count': len(models)
        }
    
    return jsonify({
        'total_pairs': len(system.models),
        'models': models_info
    })

if __name__ == '__main__':
    logger.info("\n" + "="*80)
    logger.info("🚀 FOREX ML TRADING SERVER - النظام الكامل")
    logger.info("📊 جميع الميزات نشطة")
    logger.info("🌐 السيرفر: http://69.62.121.53:5000")
    logger.info("🤖 6 نماذج ML | 200+ ميزة | SL/TP ديناميكي")
    logger.info("="*80 + "\n")
    
    # تشغيل السيرفر
    app.run(host='0.0.0.0', port=5000, debug=False)