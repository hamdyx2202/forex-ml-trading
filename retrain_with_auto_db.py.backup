#!/usr/bin/env python3
"""
Retrain Models with Auto Database Discovery
إعادة تدريب النماذج مع اكتشاف قاعدة البيانات تلقائياً
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Import unified feature engineer
from feature_engineering_unified import UnifiedFeatureEngineer

class AutoDBModelTrainer:
    """مدرب النماذج مع اكتشاف قاعدة البيانات تلقائياً"""
    
    def __init__(self):
        self.feature_engineer = UnifiedFeatureEngineer()
        self.models = {}
        self.scalers = {}
        self.feature_info = {}
        self.db_path = None
        self.table_name = None
        self.find_database()
        
    def find_database(self):
        """البحث عن قاعدة البيانات تلقائياً"""
        logger.info("🔍 Searching for database...")
        
        # البحث في المسارات المحتملة
        possible_paths = [
            'forex_data.db',
            'data/forex_data.db',
            '../forex_data.db',
            'forex_ml_data.db',
            'mt5_data.db',
            'trading_data.db',
            '../../forex_data.db',
            '../data/forex_data.db'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                if self.validate_database(path):
                    self.db_path = path
                    logger.info(f"✅ Found database: {path}")
                    return
        
        # البحث العميق
        logger.info("Searching in subdirectories...")
        for root, dirs, files in os.walk('.', followlinks=True):
            # تجنب المجلدات غير المرغوبة
            if 'venv' in root or '.git' in root or '__pycache__' in root:
                continue
                
            for file in files:
                if file.endswith('.db'):
                    full_path = os.path.join(root, file)
                    if self.validate_database(full_path):
                        self.db_path = full_path
                        logger.info(f"✅ Found database: {full_path}")
                        return
        
        logger.error("❌ No suitable database found!")
        
    def validate_database(self, db_path):
        """التحقق من صحة قاعدة البيانات"""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # الحصول على قائمة الجداول
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            for table in tables:
                table_name = table[0]
                
                # التحقق من الأعمدة
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                col_names = [col[1].lower() for col in columns]
                
                # التحقق من وجود أعمدة OHLCV
                required_cols = ['open', 'high', 'low', 'close']
                if all(col in col_names for col in required_cols):
                    # التحقق من وجود بيانات
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    count = cursor.fetchone()[0]
                    
                    if count > 0:
                        self.table_name = table_name
                        
                        # طباعة معلومات القاعدة
                        cursor.execute(f"SELECT DISTINCT symbol FROM {table_name} LIMIT 10")
                        symbols = [row[0] for row in cursor.fetchall()]
                        
                        cursor.execute(f"SELECT DISTINCT timeframe FROM {table_name} LIMIT 10")
                        timeframes = [row[0] for row in cursor.fetchall()]
                        
                        logger.info(f"📊 Table: {table_name}")
                        logger.info(f"   Records: {count}")
                        logger.info(f"   Symbols: {symbols[:5]}...")
                        logger.info(f"   Timeframes: {timeframes[:5]}...")
                        
                        conn.close()
                        return True
                        
            conn.close()
            return False
            
        except Exception as e:
            logger.error(f"Error validating {db_path}: {e}")
            return False
    
    def load_data(self, symbol: str, timeframe: str):
        """تحميل البيانات من قاعدة البيانات"""
        if not self.db_path or not self.table_name:
            logger.error("No database configured!")
            return None
            
        try:
            conn = sqlite3.connect(self.db_path)
            
            # بناء الاستعلام
            query = f"""
                SELECT time, open, high, low, close, volume
                FROM {self.table_name}
                WHERE symbol = ? AND timeframe = ?
                ORDER BY time
            """
            
            df = pd.read_sql_query(query, conn, params=(symbol, timeframe))
            conn.close()
            
            if len(df) < 500:  # خفض الحد الأدنى
                logger.warning(f"Not enough data for {symbol} {timeframe}: {len(df)} rows")
                return None
                
            # تحويل الوقت
            df['datetime'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('datetime', inplace=True)
            
            # التأكد من وجود عمود volume
            if 'volume' not in df.columns:
                df['volume'] = 100  # قيمة افتراضية
                
            logger.info(f"Loaded {len(df)} rows for {symbol} {timeframe}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data for {symbol} {timeframe}: {e}")
            return None
    
    def prepare_data(self, df: pd.DataFrame, lookahead: int = 5, threshold: float = 0.001):
        """تحضير البيانات للتدريب"""
        # إنشاء الميزات
        X, feature_names = self.feature_engineer.create_features(df)
        
        # إنشاء التسميات
        future_returns = df['close'].shift(-lookahead) / df['close'] - 1
        y = np.where(future_returns > threshold, 1, 0)
        
        # إزالة الصفوف الأخيرة
        X = X[:-lookahead]
        y = y[:-lookahead]
        
        # إزالة NaN
        mask = ~np.isnan(y)
        X = X[mask]
        y = y[mask]
        
        logger.info(f"Prepared {len(X)} samples with {len(feature_names)} features")
        
        return X, y, feature_names
    
    def create_ensemble_model(self):
        """إنشاء نموذج ensemble بسيط"""
        models = []
        
        # LightGBM
        lgb_model = lgb.LGBMClassifier(
            n_estimators=50,  # تقليل للسرعة
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )
        models.append(('lgb', lgb_model))
        
        # Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=50,
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
        """تدريب النموذج"""
        logger.info(f"Training model for {symbol} {timeframe}")
        
        # تحضير البيانات
        X, y, feature_names = self.prepare_data(df)
        
        if len(X) < 100:
            logger.error(f"Not enough samples for {symbol} {timeframe}")
            return None
        
        # تقسيم البيانات
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # التطبيع
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # التدريب
        model = self.create_ensemble_model()
        model.fit(X_train_scaled, y_train)
        
        # التقييم
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Model accuracy: {accuracy:.4f}")
        
        # حفظ النموذج
        model_key = f"{symbol}_{timeframe}"
        
        model_package = {
            'model': model,
            'scaler': scaler,
            'feature_names': feature_names,
            'n_features': len(feature_names),
            'feature_version': UnifiedFeatureEngineer.VERSION,
            'metrics': {
                'accuracy': float(accuracy),
                'samples': len(X)
            },
            'training_metadata': {
                'symbol': symbol,
                'timeframe': timeframe,
                'training_date': datetime.now().isoformat(),
                'db_path': self.db_path,
                'table_name': self.table_name
            }
        }
        
        # إنشاء مجلد النماذج
        os.makedirs('models/unified', exist_ok=True)
        
        # حفظ النموذج
        model_filename = f'models/unified/{model_key}_unified_v2.pkl'
        joblib.dump(model_package, model_filename)
        logger.info(f"✅ Model saved: {model_filename}")
        
        return model_package
    
    def train_all_models(self, symbols=None, timeframes=None):
        """تدريب جميع النماذج المتاحة"""
        if not self.db_path:
            logger.error("No database found!")
            return None
            
        # إذا لم يتم تحديد الرموز، احصل عليها من قاعدة البيانات
        if symbols is None or timeframes is None:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if symbols is None:
                cursor.execute(f"SELECT DISTINCT symbol FROM {self.table_name}")
                symbols = [row[0] for row in cursor.fetchall()]
                # تصفية الرموز المطلوبة
                wanted_symbols = ['EURUSDm', 'GBPUSDm', 'USDJPYm', 'EURJPYm', 'GBPJPYm']
                symbols = [s for s in symbols if any(w in s for w in wanted_symbols)]
                
            if timeframes is None:
                cursor.execute(f"SELECT DISTINCT timeframe FROM {self.table_name}")
                timeframes = [row[0] for row in cursor.fetchall()]
                # تصفية الأطر الزمنية المطلوبة
                wanted_tf = ['M5', 'M15', 'H1', 'H4']
                timeframes = [tf for tf in timeframes if any(w in tf for w in wanted_tf)]
                
            conn.close()
        
        logger.info(f"Training models for {len(symbols)} symbols and {len(timeframes)} timeframes")
        
        successful = 0
        failed = 0
        
        for symbol in symbols:
            for timeframe in timeframes:
                try:
                    # تحميل البيانات
                    df = self.load_data(symbol, timeframe)
                    if df is not None:
                        # تدريب النموذج
                        model_package = self.train_model(symbol, timeframe, df)
                        if model_package:
                            successful += 1
                        else:
                            failed += 1
                    else:
                        failed += 1
                        
                except Exception as e:
                    logger.error(f"Error training {symbol} {timeframe}: {e}")
                    failed += 1
        
        logger.info(f"✅ Training completed: {successful} successful, {failed} failed")
        
        # حفظ الملخص
        if successful > 0:
            summary = {
                'training_date': datetime.now().isoformat(),
                'successful': successful,
                'failed': failed,
                'db_path': self.db_path,
                'table_name': self.table_name,
                'feature_version': UnifiedFeatureEngineer.VERSION
            }
            
            with open('models/unified/training_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
                
        return successful, failed

# للاستخدام المباشر
if __name__ == "__main__":
    print("🚀 Starting Auto-DB Model Training...")
    
    # إنشاء المدرب
    trainer = AutoDBModelTrainer()
    
    if trainer.db_path:
        # تدريب النماذج
        successful, failed = trainer.train_all_models()
        
        print(f"\n✅ Training Summary:")
        print(f"   Successful: {successful}")
        print(f"   Failed: {failed}")
        print(f"   Database: {trainer.db_path}")
        print(f"   Table: {trainer.table_name}")
    else:
        print("❌ No database found! Please check your data files.")