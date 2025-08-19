#!/usr/bin/env python3
"""
🚀 تدريب جميع الـ 36 زوج بالنظام المتقدم
📊 200+ ميزة | 6 نماذج | كل الفريمات
"""

import os
import sys
import time
import sqlite3
import pandas as pd
import logging

# إضافة مسار السيرفر
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from advanced_ml_server import AdvancedMLSystem

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def get_all_available_pairs():
    """جلب جميع الأزواج المتاحة من قاعدة البيانات"""
    conn = sqlite3.connect('./data/forex_ml.db')
    query = """
    SELECT symbol, COUNT(*) as count 
    FROM price_data 
    GROUP BY symbol 
    HAVING count > 1000
    ORDER BY count DESC
    """
    pairs = pd.read_sql_query(query, conn)
    conn.close()
    return pairs

def main():
    logger.info("\n" + "="*80)
    logger.info("🚀 بدء تدريب جميع الأزواج المتاحة")
    logger.info("="*80)
    
    # إنشاء النظام
    system = AdvancedMLSystem()
    
    # جلب جميع الأزواج
    pairs_df = get_all_available_pairs()
    logger.info(f"\n📊 عدد الأزواج المتاحة: {len(pairs_df)}")
    logger.info(f"📈 إجمالي السجلات: {pairs_df['count'].sum():,}")
    
    # الفريمات الزمنية
    timeframes = ['M15', 'M30', 'H1', 'H4', 'D1']
    
    total_models = 0
    failed_models = 0
    start_time = time.time()
    
    # تدريب كل زوج
    for idx, row in pairs_df.iterrows():
        symbol = row['symbol']
        count = row['count']
        
        # تنظيف اسم الرمز
        clean_symbol = symbol.replace('m', '').replace('.ecn', '').replace('_pro', '')
        
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 [{idx+1}/{len(pairs_df)}] تدريب {symbol} ({count:,} سجل)")
        
        for timeframe in timeframes:
            try:
                logger.info(f"   ⏰ {timeframe}...")
                start = time.time()
                
                # محاولة التدريب
                success = system.train_all_models(clean_symbol, timeframe)
                
                if success:
                    elapsed = time.time() - start
                    logger.info(f"   ✅ تم في {elapsed:.1f} ثانية")
                    total_models += 1
                else:
                    logger.warning(f"   ⚠️ فشل التدريب")
                    failed_models += 1
                    
                # راحة بين النماذج لتجنب الحمل الزائد
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"   ❌ خطأ: {str(e)}")
                failed_models += 1
                
        # تقرير مؤقت كل 5 أزواج
        if (idx + 1) % 5 == 0:
            elapsed_total = time.time() - start_time
            logger.info(f"\n📊 تقرير مؤقت:")
            logger.info(f"   - تم تدريب: {total_models} نموذج")
            logger.info(f"   - فشل: {failed_models} نموذج")
            logger.info(f"   - الوقت المنقضي: {elapsed_total/60:.1f} دقيقة")
            logger.info(f"   - متوسط الوقت لكل زوج: {elapsed_total/(idx+1):.1f} ثانية")
    
    # التقرير النهائي
    total_time = time.time() - start_time
    logger.info("\n" + "="*80)
    logger.info("📊 التقرير النهائي:")
    logger.info(f"✅ عدد النماذج المدربة: {total_models}")
    logger.info(f"❌ عدد النماذج الفاشلة: {failed_models}")
    logger.info(f"⏰ الوقت الإجمالي: {total_time/60:.1f} دقيقة")
    logger.info(f"📁 النماذج محفوظة في: ./trained_models/")
    logger.info(f"🎯 نسبة النجاح: {(total_models/(total_models+failed_models)*100):.1f}%")
    logger.info("="*80)

if __name__ == "__main__":
    main()