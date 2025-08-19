#!/usr/bin/env python3
"""
🔄 Full Data Merger - دمج كامل للبيانات
📊 ينسخ كل البيانات بدون حدود
"""

import sqlite3
import pandas as pd
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def merge_all_data():
    """دمج كل البيانات من الجداول المعالجة"""
    db_path = './data/forex_ml.db'
    
    logger.info("="*60)
    logger.info("🔄 Full Data Merger - No Limits!")
    logger.info("="*60)
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 1. حذف وإعادة إنشاء جدول price_data
        logger.info("🗑️ Recreating price_data table...")
        cursor.execute("DROP TABLE IF EXISTS price_data")
        
        cursor.execute("""
        CREATE TABLE price_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            time TEXT NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL DEFAULT 0,
            spread INTEGER DEFAULT 0
        )
        """)
        
        # 2. جلب الجداول المعالجة
        cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name LIKE '%_processed'
        ORDER BY name
        """)
        tables = [t[0] for t in cursor.fetchall()]
        
        logger.info(f"\n📁 Found {len(tables)} processed tables")
        
        total_records = 0
        
        # 3. معالجة كل جدول بالكامل
        for table_name in tables:
            logger.info(f"\n🔄 Processing {table_name}...")
            
            try:
                # عد السجلات أولاً
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                table_count = cursor.fetchone()[0]
                logger.info(f"   📊 Table has {table_count:,} total records")
                
                # نسخ البيانات بدون LIMIT
                insert_query = f"""
                INSERT INTO price_data (symbol, timeframe, time, open, high, low, close, volume, spread)
                SELECT 
                    symbol,
                    CASE 
                        WHEN timeframe = 'PERIOD_M1' THEN 'M1'
                        WHEN timeframe = 'PERIOD_M5' THEN 'M5'
                        WHEN timeframe = 'PERIOD_M15' THEN 'M15'
                        WHEN timeframe = 'PERIOD_M30' THEN 'M30'
                        WHEN timeframe = 'PERIOD_H1' THEN 'H1'
                        WHEN timeframe = 'PERIOD_H4' THEN 'H4'
                        WHEN timeframe = 'PERIOD_D1' THEN 'D1'
                        ELSE timeframe
                    END as timeframe,
                    CASE
                        WHEN typeof(time) = 'integer' THEN datetime(time, 'unixepoch')
                        ELSE time
                    END as time,
                    open, high, low, close, 
                    COALESCE(volume, 0) as volume,
                    COALESCE(spread, 0) as spread
                FROM {table_name}
                WHERE open IS NOT NULL 
                AND high IS NOT NULL 
                AND low IS NOT NULL 
                AND close IS NOT NULL
                """
                
                cursor.execute(insert_query)
                records_added = cursor.rowcount
                
                if records_added > 0:
                    logger.info(f"   ✅ Copied {records_added:,} records successfully")
                    total_records += records_added
                else:
                    logger.warning(f"   ⚠️ No valid records found")
                
                # حفظ كل 5 جداول
                if len(tables) > 5 and tables.index(table_name) % 5 == 0:
                    conn.commit()
                    logger.info("   💾 Progress saved...")
                
            except Exception as e:
                logger.error(f"   ❌ Error: {e}")
                
                # محاولة بديلة - تجاهل الأخطاء
                try:
                    logger.info("   🔄 Trying with error handling...")
                    
                    # نسخ مع تجاهل السجلات الخاطئة
                    cursor.execute(f"""
                    INSERT OR IGNORE INTO price_data 
                    SELECT 
                        NULL as id,
                        symbol,
                        CASE 
                            WHEN timeframe LIKE 'PERIOD_%' THEN REPLACE(timeframe, 'PERIOD_', '')
                            ELSE timeframe
                        END as timeframe,
                        CAST(time as TEXT) as time,
                        CAST(open as REAL) as open,
                        CAST(high as REAL) as high,
                        CAST(low as REAL) as low,
                        CAST(close as REAL) as close,
                        CAST(COALESCE(volume, 0) as REAL) as volume,
                        CAST(COALESCE(spread, 0) as INTEGER) as spread
                    FROM {table_name}
                    WHERE open > 0 AND high > 0 AND low > 0 AND close > 0
                    """)
                    
                    alt_records = cursor.rowcount
                    if alt_records > 0:
                        logger.info(f"   ✅ Alternative method: {alt_records:,} records")
                        total_records += alt_records
                        
                except Exception as e2:
                    logger.error(f"   ❌ Alternative also failed: {e2}")
        
        # 4. حفظ نهائي
        conn.commit()
        
        # 5. إنشاء الفهارس
        logger.info("\n🔧 Creating indexes...")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol_timeframe ON price_data(symbol, timeframe)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_time ON price_data(time)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol_timeframe_time ON price_data(symbol, timeframe, time)")
        
        # 6. الإحصائيات النهائية
        logger.info("\n" + "="*60)
        logger.info("📊 Final Statistics:")
        
        cursor.execute("SELECT COUNT(*) FROM price_data")
        final_count = cursor.fetchone()[0]
        logger.info(f"   Total records merged: {final_count:,}")
        
        # البيانات حسب الإطار الزمني
        logger.info("\n📈 Data by timeframe:")
        cursor.execute("""
        SELECT timeframe, COUNT(*) as count, COUNT(DISTINCT symbol) as symbols
        FROM price_data
        GROUP BY timeframe
        ORDER BY 
            CASE timeframe
                WHEN 'M1' THEN 1
                WHEN 'M5' THEN 2
                WHEN 'M15' THEN 3
                WHEN 'M30' THEN 4
                WHEN 'H1' THEN 5
                WHEN 'H4' THEN 6
                WHEN 'D1' THEN 7
                ELSE 8
            END
        """)
        
        for tf, count, symbols in cursor.fetchall():
            logger.info(f"   {tf}: {count:,} records ({symbols} symbols)")
        
        # الأزواج الجاهزة للتدريب مع متطلبات مخفضة
        for min_candles in [2000, 1500, 1000, 500]:
            logger.info(f"\n🎯 Pairs with >{min_candles} M15 candles:")
            cursor.execute(f"""
            SELECT symbol, COUNT(*) as count
            FROM price_data
            WHERE timeframe = 'M15'
            GROUP BY symbol
            HAVING count > {min_candles}
            ORDER BY count DESC
            """)
            
            ready_pairs = cursor.fetchall()
            if ready_pairs:
                for symbol, count in ready_pairs[:10]:  # أول 10 فقط
                    logger.info(f"   ✅ {symbol}: {count:,} candles")
                if len(ready_pairs) > 10:
                    logger.info(f"   ... and {len(ready_pairs) - 10} more pairs")
                logger.info(f"\n✅ Total: {len(ready_pairs)} pairs ready with >{min_candles} candles!")
                break
        
        # تحليل أفضل الأزواج
        logger.info("\n🏆 Top 10 pairs by M15 data:")
        cursor.execute("""
        SELECT symbol, COUNT(*) as count
        FROM price_data
        WHERE timeframe = 'M15'
        GROUP BY symbol
        ORDER BY count DESC
        LIMIT 10
        """)
        
        for i, (symbol, count) in enumerate(cursor.fetchall(), 1):
            status = "✅" if count > 2000 else "⚠️"
            logger.info(f"   {i}. {status} {symbol}: {count:,} candles")
        
        conn.close()
        
        logger.info("\n" + "="*60)
        logger.info("✅ Full data merge completed successfully!")
        
        if final_count > 0:
            logger.info(f"\n🚀 Ready for training with {final_count:,} total records!")
            logger.info("\nNext steps:")
            logger.info("1. python3 train_all_pairs_enhanced.py")
            logger.info("   OR")
            logger.info("2. python3 train_with_available_data.py")
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    merge_all_data()