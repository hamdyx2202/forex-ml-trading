#!/usr/bin/env python3
"""
🚀 تدريب جميع الأزواج بالميزات المتقدمة
📊 يستخدم نفس نظام optimized_forex_server
"""

import os
import sys
import time
import logging
from optimized_forex_server import OptimizedForexSystem

# إعداد التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("\n" + "="*80)
    logger.info("🚀 بدء التدريب المتقدم لجميع الأزواج")
    logger.info("="*80)
    
    # إنشاء النظام
    system = OptimizedForexSystem()
    
    # الأزواج الرئيسية للتدريب
    major_pairs = [
        'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD',
        'USDCAD', 'NZDUSD', 'EURJPY', 'GBPJPY', 'EURAUD',
        'EURGBP', 'AUDNZD', 'XAUUSD', 'XAGUSD'
    ]
    
    # الفريمات الزمنية
    timeframes = ['M15', 'M30', 'H1', 'H4', 'D1']
    
    total_trained = 0
    failed_count = 0
    
    # تدريب كل زوج
    for symbol in system.available_pairs:
        # تحقق من الأزواج المهمة أولاً
        if any(major in symbol.upper() for major in major_pairs):
            logger.info(f"\n{'='*60}")
            logger.info(f"📊 تدريب {symbol}")
            
            for timeframe in timeframes:
                try:
                    start_time = time.time()
                    logger.info(f"   ⏰ {timeframe}...")
                    
                    success = system.train_model(symbol, timeframe)
                    
                    if success:
                        elapsed = time.time() - start_time
                        logger.info(f"   ✅ تم في {elapsed:.1f} ثانية")
                        total_trained += 1
                    else:
                        logger.warning(f"   ⚠️ فشل التدريب")
                        failed_count += 1
                        
                except Exception as e:
                    logger.error(f"   ❌ خطأ: {str(e)}")
                    failed_count += 1
                
                # راحة بين النماذج
                time.sleep(0.5)
    
    # الملخص
    logger.info("\n" + "="*80)
    logger.info("📊 ملخص التدريب:")
    logger.info(f"✅ نماذج ناجحة: {total_trained}")
    logger.info(f"❌ نماذج فاشلة: {failed_count}")
    logger.info(f"📁 النماذج محفوظة في: ./trained_models/")
    logger.info("="*80)

if __name__ == "__main__":
    main()