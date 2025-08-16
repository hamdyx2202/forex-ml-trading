#!/usr/bin/env python3
"""
Retrain Models with Unified Feature Engineering
إعادة تدريب النماذج باستخدام هندسة الميزات الموحدة
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Import unified feature engineer
from feature_engineering_unified import UnifiedFeatureEngineer

class UnifiedModelTrainer:
    """مدرب النماذج الموحد"""
    
    def __init__(self):
        self.feature_engineer = UnifiedFeatureEngineer()
        self.models = {}
        self.scalers = {}
        self.feature_info = {}
        
    def load_data(self, symbol: str, timeframe: str, db_path: str = 'forex_data.db'):
        """تحميل البيانات من قاعدة البيانات"""
        import sqlite3
        
        conn = sqlite3.connect(db_path)
        query = """
            SELECT time, open, high, low, close, volume
            FROM forex_data
            WHERE symbol = ? AND timeframe = ?
            ORDER BY time
        """
        
        df = pd.read_sql_query(query, conn, params=(symbol, timeframe))
        conn.close()
        
        if len(df) < 1000:
            logger.warning(f"Not enough data for {symbol} {timeframe}: {len(df)} rows")
            return None
            
        # تحويل الوقت
        df['datetime'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('datetime', inplace=True)
        
        return df
    
    def prepare_data(self, df: pd.DataFrame, lookahead: int = 5, threshold: float = 0.001):
        """تحضير البيانات للتدريب"""
        # إنشاء الميزات باستخدام الموحد
        X, feature_names = self.feature_engineer.create_features(df)
        
        # إنشاء التسميات
        future_returns = df['close'].shift(-lookahead) / df['close'] - 1
        y = np.where(future_returns > threshold, 1, 0)  # 1 for UP, 0 for DOWN
        
        # إزالة الصفوف الأخيرة بدون تسميات
        X = X[:-lookahead]
        y = y[:-lookahead]
        
        # إزالة NaN
        mask = ~np.isnan(y)
        X = X[mask]
        y = y[mask]
        
        logger.info(f"Prepared {len(X)} samples with {len(feature_names)} features")
        
        return X, y, feature_names
    
    def create_ensemble_model(self):
        """إنشاء نموذج ensemble"""
        models = []
        
        # LightGBM
        lgb_model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=7,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )
        models.append(('lgb', lgb_model))
        
        # XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=7,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        models.append(('xgb', xgb_model))
        
        # CatBoost
        cat_model = CatBoostClassifier(
            iterations=100,
            learning_rate=0.05,
            depth=7,
            random_state=42,
            verbose=False
        )
        models.append(('cat', cat_model))
        
        # Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        models.append(('rf', rf_model))
        
        # Voting Classifier
        ensemble = VotingClassifier(
            estimators=models,
            voting='soft',
            n_jobs=-1
        )
        
        return ensemble
    
    def train_model(self, symbol: str, timeframe: str, df: pd.DataFrame):
        """تدريب النموذج مع حفظ كل المعلومات"""
        logger.info(f"Training model for {symbol} {timeframe}")
        
        # تحضير البيانات
        X, y, feature_names = self.prepare_data(df)
        
        if len(X) < 100:
            logger.error(f"Not enough samples for {symbol} {timeframe}")
            return None
        
        # تقسيم البيانات
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # التطبيع
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # إنشاء وتدريب النموذج
        model = self.create_ensemble_model()
        model.fit(X_train_scaled, y_train)
        
        # التقييم
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # حساب الدقة للثقة العالية
        y_proba = model.predict_proba(X_test_scaled)
        high_conf_mask = np.max(y_proba, axis=1) > 0.7
        if high_conf_mask.sum() > 0:
            high_conf_accuracy = accuracy_score(
                y_test[high_conf_mask], 
                y_pred[high_conf_mask]
            )
        else:
            high_conf_accuracy = 0
        
        logger.info(f"Model accuracy: {accuracy:.4f}")
        logger.info(f"High confidence accuracy: {high_conf_accuracy:.4f}")
        
        # حفظ النموذج مع كل المعلومات
        model_key = f"{symbol}_{timeframe}"
        
        # حزمة النموذج الكاملة
        model_package = {
            'model': model,
            'scaler': scaler,
            'feature_names': feature_names,
            'n_features': len(feature_names),
            'feature_version': UnifiedFeatureEngineer.VERSION,
            'feature_config': UnifiedFeatureEngineer.INDICATORS_CONFIG,
            'metrics': {
                'accuracy': float(accuracy),
                'high_confidence_accuracy': float(high_conf_accuracy),
                'high_confidence_trades': int(high_conf_mask.sum()),
                'total_test_samples': len(y_test),
                'class_distribution': {
                    'train': dict(zip(*np.unique(y_train, return_counts=True))),
                    'test': dict(zip(*np.unique(y_test, return_counts=True)))
                }
            },
            'training_metadata': {
                'symbol': symbol,
                'timeframe': timeframe,
                'training_date': datetime.now().isoformat(),
                'n_samples': len(X),
                'model_version': '2.0',
                'unified_features': True
            }
        }
        
        # حفظ النموذج
        os.makedirs('models/unified', exist_ok=True)
        model_filename = f'models/unified/{model_key}_unified_v2.pkl'
        joblib.dump(model_package, model_filename)
        logger.info(f"✅ Model saved: {model_filename}")
        
        # حفظ معلومات النموذج في JSON للمراجعة
        config_filename = f'models/unified/{model_key}_config.json'
        config_data = {
            'feature_names': feature_names,
            'n_features': len(feature_names),
            'feature_version': UnifiedFeatureEngineer.VERSION,
            'feature_config': UnifiedFeatureEngineer.INDICATORS_CONFIG,
            'metrics': model_package['metrics'],
            'metadata': model_package['training_metadata']
        }
        
        with open(config_filename, 'w') as f:
            json.dump(config_data, f, indent=2)
        logger.info(f"📋 Config saved: {config_filename}")
        
        return model_package
    
    def train_all_models(self, symbols: list, timeframes: list):
        """تدريب جميع النماذج"""
        logger.info(f"Training {len(symbols) * len(timeframes)} models...")
        
        successful = 0
        failed = 0
        
        for symbol in symbols:
            for timeframe in timeframes:
                try:
                    # تحميل البيانات
                    df = self.load_data(symbol, timeframe)
                    if df is None:
                        failed += 1
                        continue
                    
                    # تدريب النموذج
                    model_package = self.train_model(symbol, timeframe, df)
                    if model_package:
                        successful += 1
                        self.models[f"{symbol}_{timeframe}"] = model_package
                    else:
                        failed += 1
                        
                except Exception as e:
                    logger.error(f"Error training {symbol} {timeframe}: {e}")
                    failed += 1
        
        logger.info(f"✅ Training completed: {successful} successful, {failed} failed")
        
        # حفظ ملخص التدريب
        summary = {
            'training_date': datetime.now().isoformat(),
            'total_models': successful + failed,
            'successful': successful,
            'failed': failed,
            'feature_version': UnifiedFeatureEngineer.VERSION,
            'models': {}
        }
        
        for key, package in self.models.items():
            summary['models'][key] = {
                'accuracy': package['metrics']['accuracy'],
                'high_conf_accuracy': package['metrics']['high_confidence_accuracy'],
                'n_features': package['n_features']
            }
        
        with open('models/unified/training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary

# للاستخدام المباشر
if __name__ == "__main__":
    # الأزواج والأطر الزمنية
    symbols = ['EURUSDm', 'GBPUSDm', 'USDJPYm', 'USDCHFm', 
               'AUDUSDm', 'USDCADm', 'NZDUSDm', 'XAUUSDm',
               'EURJPYm', 'GBPJPYm']  # أضف EURJPYm و GBPJPYm
    
    timeframes = ['PERIOD_M5', 'PERIOD_M15', 'PERIOD_H1', 'PERIOD_H4']
    
    # إنشاء المدرب
    trainer = UnifiedModelTrainer()
    
    # تدريب جميع النماذج
    summary = trainer.train_all_models(symbols, timeframes)
    
    print("\n" + "="*60)
    print("🎯 Training Summary")
    print("="*60)
    print(f"✅ Successful: {summary['successful']}")
    print(f"❌ Failed: {summary['failed']}")
    print(f"📊 Feature Version: {summary['feature_version']}")
    print("\nModel Performance:")
    
    for model_key, metrics in summary['models'].items():
        print(f"\n{model_key}:")
        print(f"  • Accuracy: {metrics['accuracy']:.2%}")
        print(f"  • High Conf Accuracy: {metrics['high_conf_accuracy']:.2%}")
        print(f"  • Features: {metrics['n_features']}")