#!/usr/bin/env python3
"""
🚀 Quick Train and Save - تدريب سريع وحفظ مباشر
📊 يتخطى كل الفحوصات ويحفظ النماذج فوراً
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import joblib
import logging
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# استيراد النظام المحسن
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from enhanced_ml_server import EnhancedMLTradingSystem

def force_save_models():
    """حفظ نماذج بشكل مباشر"""
    logger.info("="*60)
    logger.info("🚀 Quick Model Training & Saving")
    logger.info("="*60)
    
    # إنشاء النظام
    system = EnhancedMLTradingSystem()
    
    # البيانات المتاحة
    conn = sqlite3.connect('./data/forex_ml.db')
    
    # جلب الأزواج المتاحة
    query = """
    SELECT DISTINCT symbol, timeframe, COUNT(*) as count
    FROM price_data
    GROUP BY symbol, timeframe
    HAVING count > 1000
    ORDER BY count DESC
    LIMIT 20
    """
    
    available = pd.read_sql_query(query, conn)
    logger.info(f"\n📊 Found {len(available)} symbol/timeframe combinations")
    
    models_saved = 0
    
    for _, row in available.iterrows():
        symbol = row['symbol']
        timeframe = row['timeframe']
        count = row['count']
        
        # تنظيف الرمز
        clean_symbol = symbol.replace('m', '').replace('.ecn', '').upper()
        
        logger.info(f"\n{'='*40}")
        logger.info(f"Processing {clean_symbol} {timeframe} ({count:,} records)")
        
        try:
            # جلب البيانات
            data_query = f"""
            SELECT time, open, high, low, close, volume
            FROM price_data
            WHERE symbol = '{symbol}' AND timeframe = '{timeframe}'
            ORDER BY time
            LIMIT 5000
            """
            
            df = pd.read_sql_query(data_query, conn)
            
            if len(df) < 500:
                logger.warning(f"Skipping - insufficient data: {len(df)}")
                continue
            
            # تحضير البيانات
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            
            # حساب ميزات بسيطة
            features = pd.DataFrame(index=df.index)
            
            # Returns
            features['returns'] = df['close'].pct_change()
            
            # Moving averages
            for period in [5, 10, 20]:
                features[f'sma_{period}'] = df['close'].rolling(period).mean()
                features[f'ratio_sma_{period}'] = df['close'] / features[f'sma_{period}']
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            sma = df['close'].rolling(20).mean()
            std = df['close'].rolling(20).std()
            features['bb_upper'] = sma + (std * 2)
            features['bb_lower'] = sma - (std * 2)
            features['bb_position'] = (df['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
            
            # Volume
            features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
            
            # Target
            features['target'] = (df['close'].shift(-1) > df['close']).astype(int)
            
            # حذف NaN
            features = features.dropna()
            
            if len(features) < 200:
                logger.warning(f"Skipping - insufficient features: {len(features)}")
                continue
            
            # تحضير X و y
            X = features.drop('target', axis=1)
            y = features['target']
            
            # تقسيم البيانات
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scaling
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # إنشاء المجلد
            os.makedirs('./trained_models', exist_ok=True)
            
            # تدريب وحفظ النماذج
            models = {
                'random_forest_enhanced': RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42),
                'gradient_boosting_enhanced': GradientBoostingClassifier(n_estimators=50, max_depth=5, random_state=42),
                'extra_trees_enhanced': ExtraTreesClassifier(n_estimators=50, max_depth=10, random_state=42)
            }
            
            for model_name, model in models.items():
                try:
                    # تدريب
                    model.fit(X_train_scaled, y_train)
                    
                    # حفظ
                    model_path = f'./trained_models/{clean_symbol}_{timeframe}_{model_name}.pkl'
                    joblib.dump(model, model_path)
                    
                    # تقييم
                    score = model.score(X_test_scaled, y_test)
                    logger.info(f"   ✅ {model_name}: {score:.2%} accuracy")
                    models_saved += 1
                    
                except Exception as e:
                    logger.error(f"   ❌ Failed {model_name}: {e}")
            
            # حفظ Scaler
            scaler_path = f'./trained_models/{clean_symbol}_{timeframe}_scaler_enhanced.pkl'
            joblib.dump(scaler, scaler_path)
            
        except Exception as e:
            logger.error(f"Error processing {symbol} {timeframe}: {e}")
            continue
        
        # توقف بعد 5 أزواج للاختبار
        if models_saved >= 15:
            logger.info("\n✅ Saved 15 models for testing - stopping here")
            break
    
    conn.close()
    
    # التقرير النهائي
    logger.info("\n" + "="*60)
    logger.info("🏁 Training Complete")
    logger.info(f"✅ Models saved: {models_saved}")
    
    # عرض النماذج المحفوظة
    if os.path.exists('./trained_models'):
        all_files = os.listdir('./trained_models')
        model_files = [f for f in all_files if f.endswith('.pkl') and 'scaler' not in f]
        scaler_files = [f for f in all_files if 'scaler' in f]
        
        logger.info(f"\n📁 Files in trained_models/:")
        logger.info(f"   🤖 Models: {len(model_files)}")
        logger.info(f"   📏 Scalers: {len(scaler_files)}")
        
        if model_files:
            logger.info("\n✅ Sample models:")
            for model in model_files[:5]:
                logger.info(f"   - {model}")

if __name__ == "__main__":
    force_save_models()