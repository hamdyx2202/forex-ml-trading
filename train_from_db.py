#!/usr/bin/env python3
"""
Train models directly from database data
تدريب النماذج مباشرة من البيانات المحفوظة
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.feature_engineer import FeatureEngineer
from src.model_trainer import ModelTrainer
from loguru import logger
import pandas as pd
import sqlite3
from pathlib import Path

# Setup logging
logger.remove()
logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")
logger.add("logs/train_from_db.log", rotation="1 day", retention="30 days")

def get_data_from_db(symbol: str, timeframe: str, limit: int = 10000):
    """جلب البيانات من قاعدة البيانات"""
    try:
        conn = sqlite3.connect("data/forex_ml.db")
        
        query = """
            SELECT time, open, high, low, close, volume, spread
            FROM price_data
            WHERE symbol = ? AND timeframe = ?
            ORDER BY time DESC
            LIMIT ?
        """
        
        df = pd.read_sql_query(query, conn, params=(symbol, timeframe, limit))
        conn.close()
        
        if not df.empty:
            # تحويل للتنسيق المطلوب
            df = df.sort_values('time')
            df['datetime'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('datetime', inplace=True)
            
            # إضافة الأعمدة المطلوبة
            df['tick_volume'] = df['volume']
            df['real_volume'] = 0
            
            logger.info(f"Loaded {len(df)} bars for {symbol} {timeframe}")
            return df
        else:
            logger.warning(f"No data found for {symbol} {timeframe}")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Database error: {e}")
        return pd.DataFrame()

def get_available_pairs():
    """الحصول على الأزواج المتاحة في قاعدة البيانات"""
    try:
        conn = sqlite3.connect("data/forex_ml.db")
        cursor = conn.execute("""
            SELECT DISTINCT symbol, timeframe, COUNT(*) as count
            FROM price_data
            GROUP BY symbol, timeframe
            HAVING count > 1000
            ORDER BY symbol, timeframe
        """)
        
        pairs = []
        for row in cursor:
            pairs.append({
                'symbol': row[0],
                'timeframe': row[1],
                'count': row[2]
            })
        
        conn.close()
        return pairs
        
    except Exception as e:
        logger.error(f"Database error: {e}")
        return []

def train_single_model(symbol: str, timeframe: str):
    """تدريب نموذج واحد"""
    logger.info(f"Training model for {symbol} {timeframe}")
    
    # جلب البيانات
    df = get_data_from_db(symbol, timeframe)
    
    if df.empty:
        logger.error(f"No data for {symbol} {timeframe}")
        return False
    
    try:
        # إنشاء المؤشرات
        engineer = FeatureEngineer()
        df_features = engineer.create_features(
            df, 
            target_config={'lookahead': 5, 'threshold': 0.001}
        )
        
        if df_features.empty:
            logger.warning(f"No features created for {symbol} {timeframe}")
            return False
        
        logger.info(f"Created {len(df_features.columns)} features")
        
        # تدريب النماذج
        trainer = ModelTrainer()
        models = trainer.train_all_models(df_features)
        
        # حفظ النماذج
        trainer.save_models(models, symbol, timeframe)
        
        logger.info(f"✅ Model training completed for {symbol} {timeframe}")
        return True
        
    except Exception as e:
        logger.error(f"Training error for {symbol} {timeframe}: {e}")
        return False

def main():
    """التدريب الرئيسي"""
    logger.info("="*60)
    logger.info("🚀 Starting model training from database")
    logger.info("="*60)
    
    # إنشاء المجلدات المطلوبة
    Path("models").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    # الحصول على الأزواج المتاحة
    pairs = get_available_pairs()
    
    if not pairs:
        logger.error("No data found in database!")
        logger.info("Please run data loader first")
        return
    
    logger.info(f"Found {len(pairs)} symbol/timeframe combinations:")
    for pair in pairs:
        logger.info(f"  • {pair['symbol']} {pair['timeframe']}: {pair['count']} bars")
    
    # تدريب النماذج
    success_count = 0
    total_count = len(pairs)
    
    for i, pair in enumerate(pairs):
        logger.info(f"\nProgress: {i+1}/{total_count}")
        
        if train_single_model(pair['symbol'], pair['timeframe']):
            success_count += 1
    
    # الملخص
    logger.info("\n" + "="*60)
    logger.info("📊 Training Summary:")
    logger.info(f"Total pairs: {total_count}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {total_count - success_count}")
    
    if success_count > 0:
        logger.info("\n✅ Training completed successfully!")
        logger.info("Models saved in: models/")
        logger.info("\nNext steps:")
        logger.info("1. Run advanced learner: python src/advanced_learner.py")
        logger.info("2. Start the server: python src/mt5_bridge_server_linux.py")
        logger.info("3. Use ForexMLBot.mq5 in MT5 to receive signals")
    else:
        logger.error("\n❌ No models were trained successfully")
    
    logger.info("="*60)

if __name__ == "__main__":
    main()