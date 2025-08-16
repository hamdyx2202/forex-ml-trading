#!/usr/bin/env python3
"""
Ultimate Model Training System - نظام التدريب الشامل المتكامل
يدمج جميع الميزات المتقدمة: دعم/مقاومة، ATR متعدد، أهداف متنوعة، تعلم مستمر
"""

import os
import sys
import json
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

# إضافة المسار للمشروع
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from loguru import logger
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import VotingClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import optuna

# استيراد الأنظمة المتقدمة
from src.feature_engineer_fixed_v2 import FeatureEngineer
from support_resistance import SupportResistanceCalculator
from src.dynamic_sl_tp_system import DynamicSLTPSystem
from src.advanced_learner import AdvancedHistoricalLearner
from src.continuous_learner import ContinuousLearner

# إعداد التسجيل
logger.remove()
logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")
logger.add("logs/ultimate_training.log", rotation="1 day", retention="30 days")

class UltimateModelTrainer:
    """نظام التدريب الشامل المتكامل"""
    
    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.sr_calculator = SupportResistanceCalculator()
        self.sltp_system = DynamicSLTPSystem()
        self.advanced_learner = AdvancedHistoricalLearner()
        self.continuous_learner = ContinuousLearner()
        
        # إعدادات التدريب
        self.min_data_points = 5000
        self.test_size = 0.2
        self.validation_split = 0.1
        self.random_state = 42
        
        # إعدادات الأهداف المتعددة
        self.target_configs = [
            {'name': 'target_5m', 'minutes': 5, 'min_pips': 5},
            {'name': 'target_15m', 'minutes': 15, 'min_pips': 10},
            {'name': 'target_30m', 'minutes': 30, 'min_pips': 15},
            {'name': 'target_1h', 'minutes': 60, 'min_pips': 20},
            {'name': 'target_4h', 'minutes': 240, 'min_pips': 40},
        ]
        
        # مستويات SL/TP مختلفة
        self.sltp_strategies = [
            {'name': 'conservative', 'risk_reward': 1.5, 'atr_multiplier': 1.0},
            {'name': 'balanced', 'risk_reward': 2.0, 'atr_multiplier': 1.5},
            {'name': 'aggressive', 'risk_reward': 3.0, 'atr_multiplier': 2.0},
            {'name': 'scalping', 'risk_reward': 1.0, 'atr_multiplier': 0.5},
            {'name': 'swing', 'risk_reward': 4.0, 'atr_multiplier': 3.0},
        ]
        
        # إعدادات النماذج
        self.model_configs = {
            'lightgbm': {
                'objective': 'multiclass',
                'num_class': 3,
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 64,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1
            },
            'xgboost': {
                'objective': 'multi:softprob',
                'num_class': 3,
                'max_depth': 8,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'eval_metric': 'mlogloss',
                'use_label_encoder': False
            },
            'catboost': {
                'loss_function': 'MultiClass',
                'classes_count': 3,
                'depth': 8,
                'learning_rate': 0.05,
                'l2_leaf_reg': 5,
                'random_state': 42,
                'verbose': False
            }
        }
        
    def get_all_symbols_from_db(self):
        """الحصول على جميع العملات من قاعدة البيانات"""
        try:
            conn = sqlite3.connect("data/forex_data.db")
            query = """
                SELECT DISTINCT symbol, timeframe, COUNT(*) as count
                FROM price_data
                GROUP BY symbol, timeframe
                HAVING count >= ?
                ORDER BY symbol, timeframe
            """
            df = pd.read_sql_query(query, conn, params=(self.min_data_points,))
            conn.close()
            
            logger.info(f"✅ تم العثور على {len(df)} مجموعة بيانات متاحة")
            return df
            
        except Exception as e:
            logger.error(f"❌ خطأ في قراءة العملات: {e}")
            return pd.DataFrame()
    
    def load_data_with_sr_levels(self, symbol, timeframe, limit=100000):
        """تحميل البيانات مع مستويات الدعم والمقاومة"""
        try:
            conn = sqlite3.connect("data/forex_data.db")
            query = """
                SELECT * FROM price_data 
                WHERE symbol = ? AND timeframe = ?
                ORDER BY time DESC
                LIMIT ?
            """
            df = pd.read_sql_query(query, conn, params=(symbol, timeframe, limit))
            conn.close()
            
            if df.empty:
                return None
                
            df = df.sort_values('time').reset_index(drop=True)
            
            # تحويل الوقت
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # حساب مستويات الدعم والمقاومة
            sr_levels = self.sr_calculator.calculate_all_levels(df, symbol)
            
            # إضافة مستويات الدعم والمقاومة كميزات
            df = self._add_sr_features(df, sr_levels)
            
            logger.info(f"✅ تم تحميل {len(df)} سجل لـ {symbol} {timeframe} مع مستويات S/R")
            return df
            
        except Exception as e:
            logger.error(f"❌ خطأ في تحميل البيانات: {e}")
            return None
    
    def _add_sr_features(self, df, sr_levels):
        """إضافة ميزات الدعم والمقاومة"""
        # أقرب مستوى دعم ومقاومة
        df['nearest_support'] = 0.0
        df['nearest_resistance'] = 0.0
        df['distance_to_support'] = 0.0
        df['distance_to_resistance'] = 0.0
        df['sr_strength'] = 0.0
        
        all_levels = []
        
        # جمع جميع المستويات
        for method, levels in sr_levels.items():
            for level in levels:
                all_levels.append({
                    'price': level['price'],
                    'strength': level['strength'],
                    'type': level['type']
                })
        
        # لكل شمعة، حساب أقرب مستويات
        for idx in range(len(df)):
            current_price = df.loc[idx, 'close']
            
            supports = [l for l in all_levels if l['price'] < current_price]
            resistances = [l for l in all_levels if l['price'] > current_price]
            
            if supports:
                nearest_sup = max(supports, key=lambda x: x['price'])
                df.loc[idx, 'nearest_support'] = nearest_sup['price']
                df.loc[idx, 'distance_to_support'] = (current_price - nearest_sup['price']) / current_price
                df.loc[idx, 'sr_strength'] = nearest_sup['strength']
            
            if resistances:
                nearest_res = min(resistances, key=lambda x: x['price'])
                df.loc[idx, 'nearest_resistance'] = nearest_res['price']
                df.loc[idx, 'distance_to_resistance'] = (nearest_res['price'] - current_price) / current_price
        
        return df
    
    def create_multiple_targets(self, df, symbol):
        """إنشاء أهداف متعددة بأوقات وأحجام مختلفة"""
        targets = {}
        
        # الحصول على معلومات العملة
        symbol_info = self.sltp_system.get_symbol_info(symbol)
        pip_value = symbol_info['pip_size']
        
        for config in self.target_configs:
            target_name = config['name']
            minutes = config['minutes']
            min_pips = config['min_pips']
            
            # حساب الهدف
            target = []
            for i in range(len(df) - minutes):
                future_prices = df.iloc[i:i+minutes]['close'].values
                current_price = df.iloc[i]['close']
                
                max_price = future_prices.max()
                min_price = future_prices.min()
                
                # حساب التغير بالنقاط
                max_change_pips = (max_price - current_price) / pip_value
                min_change_pips = (current_price - min_price) / pip_value
                
                if max_change_pips >= min_pips:
                    target.append(2)  # صعود قوي
                elif min_change_pips >= min_pips:
                    target.append(0)  # هبوط قوي
                else:
                    target.append(1)  # محايد
            
            # ملء القيم الأخيرة
            target.extend([1] * minutes)
            targets[target_name] = target
        
        return targets
    
    def create_advanced_features(self, df, symbol):
        """إنشاء ميزات متقدمة شاملة"""
        # الميزات الأساسية من FeatureEngineer
        features = self.feature_engineer.create_features(df)
        
        # إضافة ميزات الوقت
        features['hour'] = df['time'].dt.hour
        features['day_of_week'] = df['time'].dt.dayofweek
        features['day_of_month'] = df['time'].dt.day
        features['is_london_session'] = ((features['hour'] >= 8) & (features['hour'] <= 16)).astype(int)
        features['is_ny_session'] = ((features['hour'] >= 13) & (features['hour'] <= 22)).astype(int)
        features['is_asian_session'] = ((features['hour'] >= 0) & (features['hour'] <= 8)).astype(int)
        
        # ميزات التقلب المتقدمة
        for period in [5, 10, 20, 50]:
            features[f'volatility_{period}'] = df['close'].pct_change().rolling(period).std()
            features[f'range_{period}'] = (df['high'] - df['low']).rolling(period).mean()
            features[f'true_range_{period}'] = self._calculate_true_range(df).rolling(period).mean()
        
        # ميزات الحجم المتقدمة
        features['volume_sma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        features['volume_trend'] = df['volume'].rolling(10).mean() - df['volume'].rolling(30).mean()
        
        # ميزات السعر المتقدمة
        features['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 0.0001)
        features['body_size'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 0.0001)
        features['upper_shadow'] = (df['high'] - df[['close', 'open']].max(axis=1)) / (df['high'] - df['low'] + 0.0001)
        features['lower_shadow'] = (df[['close', 'open']].min(axis=1) - df['low']) / (df['high'] - df['low'] + 0.0001)
        
        # ميزات الاتجاه
        for period in [10, 20, 50, 100]:
            # ميل المتوسط المتحرك
            sma = df['close'].rolling(period).mean()
            features[f'sma_{period}_slope'] = (sma - sma.shift(5)) / 5
            
            # المسافة من المتوسط
            features[f'distance_from_sma_{period}'] = (df['close'] - sma) / sma
        
        # نماذج الشموع
        features['doji'] = (features['body_size'] < 0.1).astype(int)
        features['hammer'] = ((features['lower_shadow'] > 2 * features['body_size']) & 
                             (features['upper_shadow'] < features['body_size'])).astype(int)
        features['shooting_star'] = ((features['upper_shadow'] > 2 * features['body_size']) & 
                                    (features['lower_shadow'] < features['body_size'])).astype(int)
        
        # دمج ميزات S/R من DataFrame الأصلي
        sr_columns = ['nearest_support', 'nearest_resistance', 'distance_to_support', 
                     'distance_to_resistance', 'sr_strength']
        for col in sr_columns:
            if col in df.columns:
                features[col] = df[col]
        
        return features
    
    def _calculate_true_range(self, df):
        """حساب True Range"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        
        return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    def simulate_trading_patterns(self, df, features, symbol):
        """محاكاة أنماط تداول مختلفة لكل يوم"""
        patterns = []
        
        # معلومات العملة
        symbol_info = self.sltp_system.get_symbol_info(symbol)
        
        # لكل صف في البيانات
        for idx in range(len(df)):
            row_patterns = []
            
            # لكل استراتيجية SL/TP
            for strategy in self.sltp_strategies:
                # حساب SL/TP ديناميكي
                sl_tp = self.sltp_system.calculate_sl_tp(
                    symbol=symbol,
                    entry_price=df.iloc[idx]['close'],
                    position_type='BUY',  # سنحاكي الشراء والبيع
                    lot_size=0.1,
                    df=df.iloc[:idx+1]  # البيانات حتى هذه النقطة
                )
                
                # إضافة النمط
                pattern = {
                    'strategy': strategy['name'],
                    'sl': sl_tp['sl'],
                    'tp': sl_tp['tp'],
                    'risk_reward': sl_tp.get('risk_reward_ratio', strategy['risk_reward']),
                    'confidence': self._calculate_pattern_confidence(features.iloc[idx])
                }
                row_patterns.append(pattern)
            
            patterns.append(row_patterns)
        
        return patterns
    
    def _calculate_pattern_confidence(self, features):
        """حساب ثقة النمط بناءً على المؤشرات"""
        confidence = 0.5  # قيمة أساسية
        
        # RSI
        if 'rsi_14' in features:
            if features['rsi_14'] < 30:
                confidence += 0.1  # oversold
            elif features['rsi_14'] > 70:
                confidence += 0.1  # overbought
        
        # Bollinger Bands
        if 'bb_position_20' in features:
            if features['bb_position_20'] < 0.2 or features['bb_position_20'] > 0.8:
                confidence += 0.1
        
        # Volume
        if 'volume_sma_ratio' in features and features['volume_sma_ratio'] > 1.5:
            confidence += 0.1
        
        # ADX
        if 'adx_14' in features and features['adx_14'] > 25:
            confidence += 0.1
        
        return min(confidence, 0.95)
    
    def train_ensemble_model(self, X_train, y_train, X_val, y_val, target_name):
        """تدريب نموذج مجمع متقدم"""
        logger.info(f"🎯 تدريب نموذج مجمع لـ {target_name}")
        
        # تحسين معاملات LightGBM باستخدام Optuna
        best_lgb_params = self.optimize_lightgbm(X_train, y_train, X_val, y_val)
        
        # إنشاء النماذج
        models = []
        
        # LightGBM مع المعاملات المحسنة
        lgb_model = lgb.LGBMClassifier(**best_lgb_params, n_estimators=500)
        models.append(('lightgbm', lgb_model))
        
        # XGBoost
        xgb_model = xgb.XGBClassifier(**self.model_configs['xgboost'], n_estimators=500)
        models.append(('xgboost', xgb_model))
        
        # CatBoost
        cat_model = CatBoostClassifier(**self.model_configs['catboost'], iterations=500)
        models.append(('catboost', cat_model))
        
        # نموذج مجمع
        ensemble = VotingClassifier(models, voting='soft', n_jobs=-1)
        ensemble.fit(X_train, y_train)
        
        # تقييم الأداء
        train_score = ensemble.score(X_train, y_train)
        val_score = ensemble.score(X_val, y_val)
        
        logger.info(f"✅ دقة التدريب: {train_score:.4f}")
        logger.info(f"✅ دقة التحقق: {val_score:.4f}")
        
        return ensemble, {'train_score': train_score, 'val_score': val_score}
    
    def optimize_lightgbm(self, X_train, y_train, X_val, y_val, n_trials=50):
        """تحسين معاملات LightGBM باستخدام Optuna"""
        def objective(trial):
            params = {
                'objective': 'multiclass',
                'num_class': 3,
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'verbosity': -1
            }
            
            model = lgb.LGBMClassifier(**params, n_estimators=100)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                     eval_metric='multi_logloss', callbacks=[lgb.early_stopping(50)])
            
            return model.score(X_val, y_val)
        
        study = optuna.create_study(direction='maximize', study_name='lightgbm_optimization')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        best_params = study.best_params
        best_params.update(self.model_configs['lightgbm'])
        
        return best_params
    
    def train_symbol(self, symbol, timeframe):
        """تدريب نماذج لعملة واحدة"""
        logger.info(f"\n{'='*80}")
        logger.info(f"🚀 بدء تدريب {symbol} {timeframe}")
        logger.info(f"{'='*80}")
        
        # تحميل البيانات مع S/R
        df = self.load_data_with_sr_levels(symbol, timeframe)
        if df is None or len(df) < self.min_data_points:
            logger.warning(f"⚠️ بيانات غير كافية لـ {symbol} {timeframe}")
            return None
        
        # إنشاء الميزات المتقدمة
        features = self.create_advanced_features(df, symbol)
        
        # إنشاء أهداف متعددة
        targets = self.create_multiple_targets(df, symbol)
        
        # محاكاة أنماط التداول
        patterns = self.simulate_trading_patterns(df, features, symbol)
        
        # التعلم من الأنماط التاريخية
        self.advanced_learner.analyze_historical_opportunities(
            df, features, targets, patterns, symbol, timeframe
        )
        
        # تنظيف البيانات
        features = features.fillna(0)
        features = features.replace([np.inf, -np.inf], 0)
        
        # النتائج
        results = {
            'symbol': symbol,
            'timeframe': timeframe,
            'total_samples': len(features),
            'models': {},
            'patterns_learned': len(patterns),
            'timestamp': datetime.now()
        }
        
        # تدريب نموذج لكل هدف
        for target_name, target_values in targets.items():
            logger.info(f"\n📊 تدريب نموذج {target_name}")
            
            # تحضير البيانات
            X = features.values
            y = np.array(target_values)
            
            # تقسيم البيانات
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=self.test_size, shuffle=False
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=self.validation_split, shuffle=False
            )
            
            # معايرة البيانات
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
            
            # تدريب النموذج المجمع
            model, scores = self.train_ensemble_model(
                X_train_scaled, y_train, X_val_scaled, y_val, target_name
            )
            
            # تقييم على بيانات الاختبار
            y_pred = model.predict(X_test_scaled)
            test_accuracy = accuracy_score(y_test, y_pred)
            
            # حساب مقاييس متقدمة
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            logger.info(f"📈 دقة الاختبار: {test_accuracy:.4f}")
            logger.info(f"📊 Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            
            # حفظ النموذج
            model_dir = Path(f"models/{symbol}_{timeframe}/{target_name}")
            model_dir.mkdir(parents=True, exist_ok=True)
            
            model_data = {
                'model': model,
                'scaler': scaler,
                'feature_names': list(features.columns),
                'scores': {
                    'train': scores['train_score'],
                    'validation': scores['val_score'],
                    'test': test_accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                },
                'target_config': next(c for c in self.target_configs if c['name'] == target_name),
                'timestamp': datetime.now()
            }
            
            joblib.dump(model_data, model_dir / 'model.pkl')
            
            results['models'][target_name] = model_data['scores']
            
            # تحديث نظام التعلم المستمر
            self.continuous_learner.learn_from_predictions(
                y_test, y_pred, features.iloc[-len(y_test):], symbol, timeframe
            )
        
        # حفظ تقرير شامل
        self._save_training_report(results, symbol, timeframe)
        
        return results
    
    def _save_training_report(self, results, symbol, timeframe):
        """حفظ تقرير التدريب الشامل"""
        report_dir = Path(f"models/{symbol}_{timeframe}")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report = {
            'training_summary': results,
            'feature_importance': self._get_feature_importance(symbol, timeframe),
            'pattern_analysis': self.advanced_learner.get_pattern_summary(symbol, timeframe),
            'recommendations': self.continuous_learner.get_improvement_suggestions()
        }
        
        with open(report_dir / 'training_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str, ensure_ascii=False)
        
        logger.info(f"📄 تم حفظ تقرير التدريب في {report_dir}")
    
    def _get_feature_importance(self, symbol, timeframe):
        """استخراج أهمية الميزات من النماذج"""
        importance_dict = {}
        
        for target_name in self.target_configs:
            model_path = Path(f"models/{symbol}_{timeframe}/{target_name['name']}/model.pkl")
            if model_path.exists():
                model_data = joblib.load(model_path)
                model = model_data['model']
                
                # الحصول على أهمية الميزات من LightGBM
                if hasattr(model, 'estimators_'):
                    lgb_model = next((est for name, est in model.estimators_ if name == 'lightgbm'), None)
                    if lgb_model and hasattr(lgb_model, 'feature_importances_'):
                        importance = lgb_model.feature_importances_
                        feature_names = model_data['feature_names']
                        
                        importance_dict[target_name['name']] = dict(
                            zip(feature_names, importance.tolist())
                        )
        
        return importance_dict
    
    def train_all_symbols(self):
        """تدريب جميع العملات المتاحة"""
        logger.info("\n" + "="*100)
        logger.info("🚀 بدء تدريب النماذج الشاملة لجميع العملات")
        logger.info("="*100)
        
        # الحصول على جميع العملات
        available_data = self.get_all_symbols_from_db()
        
        if available_data.empty:
            logger.error("❌ لا توجد بيانات متاحة للتدريب")
            return
        
        successful_trainings = []
        failed_trainings = []
        
        # تدريب كل عملة
        for idx, row in available_data.iterrows():
            symbol = row['symbol']
            timeframe = row['timeframe']
            
            try:
                logger.info(f"\n📊 معالجة {idx+1}/{len(available_data)}: {symbol} {timeframe}")
                
                results = self.train_symbol(symbol, timeframe)
                
                if results:
                    successful_trainings.append({
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'models': len(results['models']),
                        'best_accuracy': max(m['test'] for m in results['models'].values())
                    })
                    logger.info(f"✅ نجح تدريب {symbol} {timeframe}")
                else:
                    failed_trainings.append({'symbol': symbol, 'timeframe': timeframe})
                    
            except Exception as e:
                logger.error(f"❌ خطأ في تدريب {symbol} {timeframe}: {e}")
                failed_trainings.append({'symbol': symbol, 'timeframe': timeframe, 'error': str(e)})
        
        # طباعة الملخص النهائي
        self._print_final_summary(successful_trainings, failed_trainings)
        
        # حفظ ملخص شامل
        self._save_overall_summary(successful_trainings, failed_trainings)
    
    def _print_final_summary(self, successful, failed):
        """طباعة الملخص النهائي"""
        logger.info("\n" + "="*100)
        logger.info("📊 الملخص النهائي للتدريب الشامل")
        logger.info("="*100)
        
        logger.info(f"\n✅ نجح التدريب: {len(successful)} عملة/فريم")
        logger.info(f"❌ فشل التدريب: {len(failed)} عملة/فريم")
        
        if successful:
            logger.info("\n🏆 أفضل النماذج:")
            sorted_models = sorted(successful, key=lambda x: x['best_accuracy'], reverse=True)[:10]
            for model in sorted_models:
                logger.info(f"  • {model['symbol']} {model['timeframe']}: "
                          f"دقة {model['best_accuracy']:.4f} ({model['models']} نماذج)")
        
        if failed:
            logger.info("\n⚠️ العملات التي فشلت:")
            for fail in failed[:10]:
                error_msg = fail.get('error', 'بيانات غير كافية')
                logger.info(f"  • {fail['symbol']} {fail['timeframe']}: {error_msg}")
    
    def _save_overall_summary(self, successful, failed):
        """حفظ ملخص شامل للتدريب"""
        summary = {
            'training_date': datetime.now().isoformat(),
            'total_attempted': len(successful) + len(failed),
            'successful': len(successful),
            'failed': len(failed),
            'success_rate': len(successful) / (len(successful) + len(failed)) if successful or failed else 0,
            'successful_models': successful,
            'failed_models': failed,
            'configuration': {
                'min_data_points': self.min_data_points,
                'target_configs': self.target_configs,
                'sltp_strategies': self.sltp_strategies,
                'features_used': 'Advanced with S/R, ATR, Time, Patterns'
            }
        }
        
        summary_path = Path("models/training_summary_ultimate.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n💾 تم حفظ الملخص الشامل في: {summary_path}")

def main():
    """الدالة الرئيسية"""
    trainer = UltimateModelTrainer()
    trainer.train_all_symbols()

if __name__ == "__main__":
    main()