#!/usr/bin/env python3
"""
🔄 Direct Data Merger - دمج مباشر للبيانات
📊 ينسخ البيانات من الجداول المعالجة مباشرة
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

def merge_direct_data():
    """دمج البيانات المباشرة من الجداول المعالجة"""
    db_path = './data/forex_ml.db'
    
    logger.info("="*60)
    logger.info("🔄 Direct Data Merger")
    logger.info("="*60)
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 1. حذف جدول price_data القديم إذا كان موجوداً
        logger.info("🗑️ Dropping old price_data table if exists...")
        cursor.execute("DROP TABLE IF EXISTS price_data")
        
        # 2. إنشاء جدول price_data جديد
        logger.info("📊 Creating new price_data table...")
        cursor.execute("""
        CREATE TABLE price_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            time DATETIME NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL DEFAULT 0,
            spread INTEGER DEFAULT 0,
            UNIQUE(symbol, timeframe, time)
        )
        """)
        
        # 3. جلب قائمة الجداول المعالجة
        cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name LIKE '%_processed'
        """)
        processed_tables = cursor.fetchall()
        
        logger.info(f"\n📁 Found {len(processed_tables)} processed tables")
        
        total_records = 0
        
        # 4. نسخ البيانات من كل جدول
        for table_tuple in processed_tables:
            table_name = table_tuple[0]
            logger.info(f"\n🔄 Processing {table_name}...")
            
            try:
                # نسخ البيانات مباشرة مع تحويل الوقت
                insert_query = f"""
                INSERT INTO price_data (symbol, timeframe, time, open, high, low, close, volume, spread)
                SELECT 
                    symbol,
                    CASE 
                        WHEN timeframe = 'PERIOD_M5' THEN 'M5'
                        WHEN timeframe = 'PERIOD_M15' THEN 'M15'
                        WHEN timeframe = 'PERIOD_M30' THEN 'M30'
                        WHEN timeframe = 'PERIOD_H1' THEN 'H1'
                        WHEN timeframe = 'PERIOD_H4' THEN 'H4'
                        WHEN timeframe = 'PERIOD_D1' THEN 'D1'
                        ELSE timeframe
                    END as timeframe,
                    datetime(time, 'unixepoch') as time,
                    open, high, low, close, volume, spread
                FROM {table_name}
                WHERE open IS NOT NULL AND high IS NOT NULL AND low IS NOT NULL AND close IS NOT NULL
                """
                
                cursor.execute(insert_query)
                records_added = cursor.rowcount
                
                if records_added > 0:
                    logger.info(f"   ✅ Added {records_added:,} records")
                    total_records += records_added
                else:
                    logger.warning(f"   ⚠️ No valid records found")
                
            except Exception as e:
                logger.error(f"   ❌ Error: {e}")
                # محاولة بديلة - نسخ سجل بسجل
                try:
                    logger.info(f"   🔄 Trying record-by-record copy...")
                    cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
                    sample = cursor.fetchall()
                    
                    if sample:
                        # تحديد مواقع الأعمدة
                        cursor.execute(f"PRAGMA table_info({table_name})")
                        columns = cursor.fetchall()
                        col_names = [col[1] for col in columns]
                        
                        # البحث عن مواقع الأعمدة المطلوبة
                        idx_map = {}
                        for i, name in enumerate(col_names):
                            idx_map[name] = i
                        
                        # نسخ البيانات
                        cursor.execute(f"SELECT * FROM {table_name}")
                        rows = cursor.fetchall()
                        
                        records_added = 0
                        for row in rows:
                            try:
                                symbol = row[idx_map.get('symbol', 2)]
                                timeframe = row[idx_map.get('timeframe', 3)]
                                time_val = row[idx_map.get('time', 0)]
                                open_val = row[idx_map.get('open', 4)]
                                high_val = row[idx_map.get('high', 5)]
                                low_val = row[idx_map.get('low', 6)]
                                close_val = row[idx_map.get('close', 7)]
                                volume = row[idx_map.get('volume', 8)] if 'volume' in idx_map else 0
                                spread = row[idx_map.get('spread', 9)] if 'spread' in idx_map else 0
                                
                                # تحويل الإطار الزمني
                                tf_map = {
                                    'PERIOD_M5': 'M5',
                                    'PERIOD_M15': 'M15',
                                    'PERIOD_M30': 'M30',
                                    'PERIOD_H1': 'H1',
                                    'PERIOD_H4': 'H4',
                                    'PERIOD_D1': 'D1'
                                }
                                timeframe = tf_map.get(timeframe, timeframe)
                                
                                # تحويل الوقت
                                if isinstance(time_val, (int, float)):
                                    time_str = datetime.fromtimestamp(time_val).strftime('%Y-%m-%d %H:%M:%S')
                                else:
                                    time_str = str(time_val)
                                
                                # إدراج السجل
                                cursor.execute("""
                                INSERT OR IGNORE INTO price_data 
                                (symbol, timeframe, time, open, high, low, close, volume, spread)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """, (symbol, timeframe, time_str, open_val, high_val, 
                                      low_val, close_val, volume, spread))
                                
                                records_added += 1
                                
                            except Exception as row_error:
                                continue
                        
                        if records_added > 0:
                            logger.info(f"   ✅ Added {records_added:,} records (record-by-record)")
                            total_records += records_added
                        
                except Exception as e2:
                    logger.error(f"   ❌ Alternative method also failed: {e2}")
            
            # حفظ التقدم
            conn.commit()
        
        # 5. إنشاء الفهارس
        logger.info("\n🔧 Creating indexes...")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol_timeframe ON price_data(symbol, timeframe)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_time ON price_data(time)")
        
        # 6. الإحصائيات النهائية
        logger.info("\n" + "="*60)
        logger.info("📊 Final Statistics:")
        
        cursor.execute("SELECT COUNT(*) FROM price_data")
        final_count = cursor.fetchone()[0]
        logger.info(f"   Total records: {final_count:,}")
        
        # البيانات حسب الإطار الزمني
        logger.info("\n📈 Data by timeframe:")
        cursor.execute("""
        SELECT timeframe, COUNT(*) as count, COUNT(DISTINCT symbol) as symbols
        FROM price_data
        GROUP BY timeframe
        ORDER BY timeframe
        """)
        
        for tf, count, symbols in cursor.fetchall():
            logger.info(f"   {tf}: {count:,} records ({symbols} symbols)")
        
        # الأزواج الجاهزة للتدريب
        logger.info("\n🎯 Training-ready pairs (M15 >2000):")
        cursor.execute("""
        SELECT symbol, COUNT(*) as count
        FROM price_data
        WHERE timeframe = 'M15'
        GROUP BY symbol
        HAVING count > 2000
        ORDER BY count DESC
        """)
        
        ready_pairs = cursor.fetchall()
        if ready_pairs:
            for symbol, count in ready_pairs:
                logger.info(f"   ✅ {symbol}: {count:,} candles")
            logger.info(f"\n✅ {len(ready_pairs)} pairs ready for training!")
            logger.info("\n🚀 You can now run: python3 train_all_pairs_enhanced.py")
        else:
            logger.info("   ⚠️ No pairs with sufficient M15 data")
            
            # عرض ما هو متاح
            cursor.execute("""
            SELECT symbol, COUNT(*) as count
            FROM price_data
            WHERE timeframe = 'M15'
            GROUP BY symbol
            ORDER BY count DESC
            LIMIT 10
            """)
            
            available = cursor.fetchall()
            if available:
                logger.info("\n📊 Available M15 data:")
                for symbol, count in available:
                    status = "✅" if count > 2000 else "⚠️"
                    logger.info(f"   {status} {symbol}: {count:,} candles (need 2000+)")
        
        conn.close()
        
        logger.info("\n✅ Data merge completed!")
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    merge_direct_data()