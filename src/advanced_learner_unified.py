#!/usr/bin/env python3
"""
Advanced Learner with Unified Standards
نظام التعلم المتقدم مع المعايير الموحدة
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import sqlite3
from loguru import logger
import json
import time

# استيراد المعايير الموحدة
from unified_standards import (
    STANDARD_FEATURES, 
    get_model_filename,
    ensure_standard_features,
    TRAINING_STANDARDS,
    SAVING_STANDARDS
)

# استيراد أدوات التدريب
from feature_engineer_adaptive import AdaptiveFeatureEngineer
from src.model_trainer import ModelTrainer

class UnifiedAdvancedLearner:
    """نظام التعلم المتقدم مع المعايير الموحدة"""
    
    def __init__(self):
        self.feature_engineer = AdaptiveFeatureEngineer(target_features=STANDARD_FEATURES)
        self.trainer = ModelTrainer()
        self.db_path = "data/forex_data.db"
        self.models_dir = Path(SAVING_STANDARDS['models_dir'])
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # سجل الأداء
        self.performance_log = self.load_performance_log()
        logger.info(f"🚀 Unified Advanced Learner initialized")
        logger.info(f"📊 Standard features: {STANDARD_FEATURES}")
        
    def load_performance_log(self):
        """تحميل سجل الأداء"""
        log_file = self.models_dir / "performance_log.json"
        if log_file.exists():
            with open(log_file, 'r') as f:
                return json.load(f)
        return {"models": {}, "last_update": None}
    
    def save_performance_log(self):
        """حفظ سجل الأداء"""
        log_file = self.models_dir / "performance_log.json"
        self.performance_log['last_update'] = datetime.now().isoformat()
        with open(log_file, 'w') as f:
            json.dump(self.performance_log, f, indent=2)
    
    def get_recent_data(self, symbol, timeframe, days=30):
        """الحصول على البيانات الحديثة"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # حساب التاريخ
            end_date = datetime.now()
            start_date = end_date - pd.Timedelta(days=days)
            
            query = """
            SELECT * FROM forex_data 
            WHERE symbol = ? AND timeframe = ? 
            AND datetime >= ?
            ORDER BY datetime
            """
            
            df = pd.read_sql_query(
                query, 
                conn, 
                params=(symbol, timeframe, start_date.isoformat())
            )
            conn.close()
            
            if len(df) < 100:
                logger.warning(f"Not enough data for {symbol} {timeframe}: {len(df)} rows")
                return None
                
            # تحويل الوقت
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None
    
    def evaluate_model_performance(self, symbol, timeframe):
        """تقييم أداء النموذج الحالي"""
        model_key = f"{symbol}_{timeframe}"
        
        # تحميل النموذج الحالي
        model_file = self.models_dir / get_model_filename(symbol, timeframe)
        if not model_file.exists():
            logger.info(f"No model found for {model_key}")
            return None
            
        try:
            model_data = joblib.load(model_file)
            current_accuracy = model_data.get('metrics', {}).get('accuracy', 0)
            
            # الحصول على البيانات الحديثة
            df = self.get_recent_data(symbol, timeframe)
            if df is None:
                return None
            
            # إنشاء الميزات
            df_features = self.feature_engineer.create_features(df)
            
            # إزالة الصفوف بدون هدف
            df_features = df_features.dropna(subset=['target_binary'])
            
            if len(df_features) < 50:
                return None
            
            # تحضير البيانات
            feature_cols = [col for col in df_features.columns 
                          if col not in ['target', 'target_binary', 'target_3class', 
                                       'future_return', 'time', 'open', 'high', 
                                       'low', 'close', 'volume', 'spread', 'datetime']]
            
            # ضمان 70 ميزة
            df_features, feature_cols = ensure_standard_features(df_features, feature_cols)
            
            X = df_features[feature_cols].values
            y = df_features['target_binary'].values
            
            # تقييم النموذج
            model = model_data['model']
            scaler = model_data['scaler']
            
            X_scaled = scaler.transform(X)
            accuracy = model.score(X_scaled, y)
            
            logger.info(f"{model_key} - Current: {current_accuracy:.2%}, Recent: {accuracy:.2%}")
            
            return {
                'current_accuracy': current_accuracy,
                'recent_accuracy': accuracy,
                'needs_update': accuracy < current_accuracy - 0.05  # انخفاض 5%
            }
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return None
    
    def update_model(self, symbol, timeframe):
        """تحديث النموذج إذا لزم الأمر"""
        logger.info(f"🔄 Updating model for {symbol} {timeframe}")
        
        try:
            # الحصول على البيانات الكاملة
            conn = sqlite3.connect(self.db_path)
            query = """
            SELECT * FROM forex_data 
            WHERE symbol = ? AND timeframe = ?
            ORDER BY datetime
            """
            df = pd.read_sql_query(query, conn, params=(symbol, timeframe))
            conn.close()
            
            if len(df) < 1000:
                logger.warning(f"Not enough data for training: {len(df)} rows")
                return False
            
            # تحويل الوقت
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            
            # إنشاء الميزات
            logger.info("Creating features...")
            df_features = self.feature_engineer.create_features(df)
            
            # إزالة NaN
            df_features = df_features.dropna(subset=['target_binary'])
            
            # تحضير البيانات
            feature_cols = [col for col in df_features.columns 
                          if col not in ['target', 'target_binary', 'target_3class', 
                                       'future_return', 'time', 'open', 'high', 
                                       'low', 'close', 'volume', 'spread', 'datetime']]
            
            # ضمان 70 ميزة
            df_features, feature_cols = ensure_standard_features(df_features, feature_cols)
            
            X = df_features[feature_cols].values
            y = df_features['target_binary'].values
            
            logger.info(f"Training data shape: {X.shape}")
            
            # التدريب
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import RobustScaler
            from sklearn.ensemble import VotingClassifier
            import lightgbm as lgb
            import xgboost as xgb
            from catboost import CatBoostClassifier
            from sklearn.ensemble import RandomForestClassifier
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=TRAINING_STANDARDS['test_size'], 
                random_state=TRAINING_STANDARDS['random_state']
            )
            
            # التطبيع
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # بناء النماذج
            models = []
            
            # LightGBM
            lgb_model = lgb.LGBMClassifier(
                **TRAINING_STANDARDS['models']['lightgbm'],
                random_state=42,
                verbosity=-1
            )
            lgb_model.fit(X_train_scaled, y_train)
            models.append(('lightgbm', lgb_model))
            
            # XGBoost
            xgb_model = xgb.XGBClassifier(
                **TRAINING_STANDARDS['models']['xgboost'],
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            xgb_model.fit(X_train_scaled, y_train)
            models.append(('xgboost', xgb_model))
            
            # Random Forest
            rf_model = RandomForestClassifier(
                **TRAINING_STANDARDS['models']['random_forest'],
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X_train_scaled, y_train)
            models.append(('random_forest', rf_model))
            
            # Ensemble
            ensemble = VotingClassifier(
                estimators=models,
                voting='soft',
                n_jobs=-1
            )
            ensemble.fit(X_train_scaled, y_train)
            
            # التقييم
            accuracy = ensemble.score(X_test_scaled, y_test)
            logger.info(f"✅ New model accuracy: {accuracy:.2%}")
            
            # حفظ النموذج
            model_data = {
                'model': ensemble,
                'scaler': scaler,
                'feature_names': feature_cols,
                'n_features': STANDARD_FEATURES,
                'metrics': {
                    'accuracy': float(accuracy),
                    'training_samples': len(X_train),
                    'test_samples': len(X_test),
                    'update_date': datetime.now().isoformat()
                },
                'metadata': {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'learner': 'UnifiedAdvancedLearner',
                    'standards_version': '1.0'
                }
            }
            
            # حفظ بالاسم القياسي (بدون timestamp)
            filename = get_model_filename(symbol, timeframe)
            filepath = self.models_dir / filename
            
            # نسخة احتياطية
            if filepath.exists():
                backup_dir = Path(SAVING_STANDARDS['backup_dir'])
                backup_dir.mkdir(parents=True, exist_ok=True)
                backup_file = backup_dir / f"{filename}.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                joblib.dump(joblib.load(filepath), backup_file)
            
            # حفظ النموذج الجديد
            joblib.dump(model_data, filepath, compress=SAVING_STANDARDS['compression'])
            logger.info(f"✅ Model saved: {filepath}")
            
            # تحديث سجل الأداء
            model_key = f"{symbol}_{timeframe}"
            self.performance_log['models'][model_key] = {
                'accuracy': float(accuracy),
                'last_update': datetime.now().isoformat(),
                'training_samples': len(X_train)
            }
            self.save_performance_log()
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_continuous_learning(self):
        """تشغيل التعلم المستمر"""
        logger.info("🚀 Starting Unified Advanced Learning...")
        
        symbols = ['EURUSDm', 'GBPUSDm', 'USDJPYm', 'USDCHFm', 
                  'AUDUSDm', 'USDCADm', 'NZDUSDm', 'EURJPYm']
        timeframes = ['PERIOD_M5', 'PERIOD_M15', 'PERIOD_H1', 'PERIOD_H4']
        
        while True:
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"🔄 Learning cycle started at {datetime.now()}")
                
                updated_count = 0
                
                for symbol in symbols:
                    for timeframe in timeframes:
                        # تقييم الأداء
                        evaluation = self.evaluate_model_performance(symbol, timeframe)
                        
                        if evaluation:
                            if evaluation['needs_update']:
                                logger.warning(f"⚠️ {symbol} {timeframe} needs update")
                                logger.info(f"   Current: {evaluation['current_accuracy']:.2%}")
                                logger.info(f"   Recent: {evaluation['recent_accuracy']:.2%}")
                                
                                # تحديث النموذج
                                if self.update_model(symbol, timeframe):
                                    updated_count += 1
                            else:
                                logger.info(f"✅ {symbol} {timeframe} performing well")
                
                logger.info(f"\n📊 Updated {updated_count} models")
                logger.info(f"💤 Sleeping for 1 hour...")
                
                # الانتظار ساعة واحدة
                time.sleep(3600)
                
            except KeyboardInterrupt:
                logger.info("🛑 Learning stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in learning cycle: {e}")
                time.sleep(300)  # 5 دقائق في حالة الخطأ

if __name__ == "__main__":
    learner = UnifiedAdvancedLearner()
    learner.run_continuous_learning()