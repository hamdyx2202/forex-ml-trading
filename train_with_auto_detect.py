#!/usr/bin/env python3
"""
Advanced training script with auto-detection of symbols
سكريبت تدريب متقدم مع اكتشاف تلقائي للرموز
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# إضافة المسار
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the trainer
from train_advanced_complete_full_features import UltimateAdvancedTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)

class SmartTrainer:
    """مدرب ذكي يكتشف الرموز تلقائياً"""
    
    def __init__(self):
        self.trainer = UltimateAdvancedTrainer()
        self.db_path = "data/forex_ml.db"
        
    def get_available_symbols(self, min_records=1000):
        """الحصول على الرموز المتاحة مع بيانات كافية"""
        try:
            conn = sqlite3.connect(self.db_path)
            query = """
                SELECT symbol, timeframe, COUNT(*) as count,
                       MIN(time) as start_time, MAX(time) as end_time
                FROM price_data
                GROUP BY symbol, timeframe
                HAVING count >= ?
                ORDER BY count DESC
            """
            df = pd.read_sql_query(query, conn, params=(min_records,))
            conn.close()
            
            logger.info(f"Found {len(df)} symbol/timeframe combinations with >= {min_records} records")
            return df
            
        except Exception as e:
            logger.error(f"Error getting symbols: {e}")
            return pd.DataFrame()
    
    def auto_train_best_symbols(self, max_symbols=5):
        """تدريب أفضل الرموز تلقائياً"""
        logger.info("="*80)
        logger.info("🚀 Smart Auto-Training System")
        logger.info("="*80)
        
        # الحصول على الرموز المتاحة
        available = self.get_available_symbols(min_records=2000)
        
        if available.empty:
            logger.error("❌ No symbols found with enough data!")
            return
        
        # عرض الرموز المتاحة
        logger.info("\n📊 Top symbols by data availability:")
        for idx, row in available.head(10).iterrows():
            logger.info(f"  {idx+1}. {row['symbol']} {row['timeframe']} - {row['count']:,} records")
        
        # فلترة الرموز الرئيسية
        major_pairs = available[available['symbol'].str.contains('USD|EUR|GBP|JPY', na=False)]
        
        if not major_pairs.empty:
            logger.info(f"\n🎯 Found {len(major_pairs)} major currency pairs")
            symbols_to_train = major_pairs.head(max_symbols)
        else:
            symbols_to_train = available.head(max_symbols)
        
        # تعديل إعدادات التدريب للأداء الأمثل
        self.trainer.use_all_features = True   # استخدام جميع الميزات
        self.trainer.use_all_models = False    # نموذج واحد للسرعة
        self.trainer.max_workers = 2           # معالجة متوازية محدودة
        
        # التدريب
        results = {}
        for idx, row in symbols_to_train.iterrows():
            symbol = row['symbol']
            timeframe = row['timeframe']
            count = row['count']
            
            logger.info(f"\n{'='*60}")
            logger.info(f"📈 Training {idx+1}/{len(symbols_to_train)}: {symbol} {timeframe}")
            logger.info(f"📊 Data points: {count:,}")
            logger.info(f"{'='*60}")
            
            try:
                # تعديل الدالة لتتعامل مع البيانات الموجودة
                result = self._train_with_exact_symbol(symbol, timeframe)
                
                if result:
                    results[f"{symbol}_{timeframe}"] = result
                    logger.info(f"✅ Success! Best accuracy: {result.get('best_accuracy', 0):.2%}")
                else:
                    logger.warning(f"⚠️ No result returned for {symbol} {timeframe}")
                    
            except Exception as e:
                logger.error(f"❌ Error training {symbol}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # ملخص النتائج
        self._print_summary(results)
        return results
    
    def _train_with_exact_symbol(self, symbol, timeframe):
        """تدريب باستخدام الرمز المحدد بالضبط"""
        # تعديل مؤقت لدالة load_data_advanced
        original_load = self.trainer.load_data_advanced
        
        def custom_load(sym, tf, limit=100000):
            """تحميل مخصص يستخدم الرمز الصحيح"""
            try:
                conn = sqlite3.connect(self.db_path)
                query = """
                    SELECT * FROM price_data 
                    WHERE symbol = ? AND timeframe = ?
                    ORDER BY time ASC
                    LIMIT ?
                """
                df = pd.read_sql_query(query, conn, params=(symbol, tf, limit))
                conn.close()
                
                if len(df) < self.trainer.min_data_points:
                    logger.warning(f"⚠️ Insufficient data: {len(df)} records")
                    return None
                
                # معالجة البيانات
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df = df.set_index('time')
                
                # إزالة القيم المفقودة
                df = df.dropna()
                
                # التحقق من الأعمدة المطلوبة
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                for col in required_cols:
                    if col not in df.columns:
                        logger.error(f"Missing required column: {col}")
                        return None
                
                logger.info(f"✅ Loaded {len(df)} records for {symbol} {tf}")
                return df
                
            except Exception as e:
                logger.error(f"❌ Error loading data: {e}")
                return None
        
        # استبدال الدالة مؤقتاً
        self.trainer.load_data_advanced = custom_load
        
        try:
            # التدريب
            result = self.trainer.train_symbol(symbol, timeframe)
            return result
        finally:
            # إعادة الدالة الأصلية
            self.trainer.load_data_advanced = original_load
    
    def _print_summary(self, results):
        """طباعة ملخص النتائج"""
        logger.info("\n" + "="*80)
        logger.info("📊 TRAINING SUMMARY")
        logger.info("="*80)
        
        if not results:
            logger.warning("❌ No successful training results!")
            return
        
        # ترتيب النتائج حسب الدقة
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1].get('best_accuracy', 0),
            reverse=True
        )
        
        logger.info("\n🏆 Results by accuracy:")
        for rank, (key, result) in enumerate(sorted_results, 1):
            accuracy = result.get('best_accuracy', 0)
            strategy = result.get('best_strategy', 'Unknown')
            logger.info(f"  {rank}. {key}: {accuracy:.2%} ({strategy})")
        
        # أفضل نموذج
        if sorted_results:
            best_key, best_result = sorted_results[0]
            logger.info(f"\n🥇 Best model: {best_key}")
            logger.info(f"   Accuracy: {best_result.get('best_accuracy', 0):.2%}")
            logger.info(f"   Strategy: {best_result.get('best_strategy', 'Unknown')}")
            
            # عرض تفاصيل الاستراتيجيات
            if 'strategies' in best_result:
                logger.info("\n📈 Strategy breakdown:")
                for strat_name, strat_results in best_result['strategies'].items():
                    acc = strat_results.get('accuracy', 0)
                    logger.info(f"   - {strat_name}: {acc:.2%}")

def main():
    """الدالة الرئيسية"""
    trainer = SmartTrainer()
    
    # تدريب تلقائي لأفضل الرموز
    results = trainer.auto_train_best_symbols(max_symbols=3)
    
    if results:
        logger.info("\n✅ Training completed successfully!")
        logger.info(f"📁 Models saved in: models/")
    else:
        logger.error("\n❌ Training failed!")

if __name__ == "__main__":
    main()