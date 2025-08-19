#!/usr/bin/env python3
"""
🔄 دمج جداول الأزواج المنفصلة في جدول موحد
📊 ينشئ جدول price_data من الجداول المعالجة
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

def merge_tables_to_price_data():
    """دمج جميع جداول الأزواج في جدول موحد"""
    db_path = './data/forex_ml.db'
    
    logger.info("="*60)
    logger.info("🔄 Merging Separate Tables into Unified price_data")
    logger.info("="*60)
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 1. إنشاء جدول price_data إذا لم يكن موجوداً
        logger.info("\n📊 Creating unified price_data table...")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS price_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            time TEXT NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume INTEGER DEFAULT 0,
            spread REAL DEFAULT 0,
            UNIQUE(symbol, timeframe, time)
        )
        """)
        
        # 2. جلب قائمة الجداول المعالجة
        cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name LIKE '%_processed'
        """)
        processed_tables = cursor.fetchall()
        
        logger.info(f"\n📁 Found {len(processed_tables)} processed tables")
        
        total_records = 0
        
        # 3. معالجة كل جدول
        for table_tuple in processed_tables:
            table_name = table_tuple[0]
            # استخراج اسم الزوج
            symbol = table_name.replace('_processed', '')
            
            logger.info(f"\n🔄 Processing {symbol}...")
            
            # جلب أسماء الأعمدة
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns_info = cursor.fetchall()
            column_names = [col[1] for col in columns_info]
            
            # تحديد تحويل الأطر الزمنية
            timeframe_mapping = {
                'PERIOD_M5': 'M5',
                'PERIOD_M15': 'M15',
                'PERIOD_M30': 'M30',
                'PERIOD_H1': 'H1',
                'PERIOD_H4': 'H4',
                'PERIOD_D1': 'D1'
            }
            
            # معالجة كل إطار زمني
            records_added = 0
            for period_col, tf in timeframe_mapping.items():
                if period_col in column_names:
                    try:
                        # جلب البيانات
                        query = f"""
                        SELECT time, {period_col} 
                        FROM {table_name} 
                        WHERE {period_col} IS NOT NULL
                        """
                        cursor.execute(query)
                        rows = cursor.fetchall()
                        
                        if rows:
                            logger.info(f"   {tf}: {len(rows)} records found")
                            
                            # معالجة كل سجل
                            for row in rows:
                                time_val = row[0]
                                price_data = row[1]
                                
                                # تحويل التوقيت إلى تنسيق قابل للقراءة
                                try:
                                    if isinstance(time_val, (int, float)):
                                        time_str = datetime.fromtimestamp(time_val).strftime('%Y-%m-%d %H:%M:%S')
                                    else:
                                        time_str = str(time_val)
                                except:
                                    time_str = str(time_val)
                                
                                # استخراج OHLC من البيانات
                                # افتراض أن البيانات مخزنة كـ "open,high,low,close,volume"
                                try:
                                    if isinstance(price_data, str):
                                        parts = price_data.split(',')
                                        if len(parts) >= 4:
                                            open_price = float(parts[0])
                                            high_price = float(parts[1])
                                            low_price = float(parts[2])
                                            close_price = float(parts[3])
                                            volume = int(parts[4]) if len(parts) > 4 else 0
                                            
                                            # إدراج في جدول price_data
                                            cursor.execute("""
                                            INSERT OR REPLACE INTO price_data 
                                            (symbol, timeframe, time, open, high, low, close, volume)
                                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                            """, (symbol, tf, time_str, open_price, high_price, 
                                                  low_price, close_price, volume))
                                            records_added += 1
                                except Exception as e:
                                    # محاولة تحليل مختلفة
                                    continue
                            
                    except Exception as e:
                        logger.error(f"   Error processing {tf}: {e}")
            
            if records_added > 0:
                logger.info(f"   ✅ Added {records_added} records")
                total_records += records_added
                conn.commit()
        
        # 4. إنشاء فهارس لتحسين الأداء
        logger.info("\n🔧 Creating indexes...")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol_timeframe ON price_data(symbol, timeframe)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_time ON price_data(time)")
        
        # 5. إحصائيات نهائية
        cursor.execute("SELECT COUNT(*) FROM price_data")
        final_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT symbol) FROM price_data")
        unique_symbols = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT timeframe) FROM price_data")
        unique_timeframes = cursor.fetchone()[0]
        
        logger.info("\n" + "="*60)
        logger.info("✅ Data Merge Complete!")
        logger.info(f"📊 Total records in price_data: {final_count:,}")
        logger.info(f"💹 Unique symbols: {unique_symbols}")
        logger.info(f"⏰ Unique timeframes: {unique_timeframes}")
        
        # عرض ملخص البيانات المتاحة للتدريب
        logger.info("\n🎯 Training-ready pairs (M15 with >2000 candles):")
        query = """
        SELECT symbol, COUNT(*) as count 
        FROM price_data 
        WHERE timeframe = 'M15'
        GROUP BY symbol 
        HAVING count > 2000
        ORDER BY count DESC
        """
        df = pd.read_sql_query(query, conn)
        
        if not df.empty:
            for _, row in df.iterrows():
                logger.info(f"   ✅ {row['symbol']}: {row['count']:,} candles")
        else:
            logger.info("   ⚠️ No pairs with sufficient M15 data yet")
        
        conn.close()
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def check_table_structure():
    """فحص هيكل الجداول المعالجة"""
    db_path = './data/forex_ml.db'
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # فحص جدول واحد كمثال
        cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name LIKE '%_processed'
        LIMIT 1
        """)
        
        table = cursor.fetchone()
        if table:
            table_name = table[0]
            logger.info(f"\n🔍 Examining table structure: {table_name}")
            
            # عرض بعض البيانات
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
            rows = cursor.fetchall()
            
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            logger.info("\nColumns:")
            for col in columns:
                logger.info(f"   - {col[1]} ({col[2]})")
            
            logger.info("\nSample data:")
            for i, row in enumerate(rows):
                logger.info(f"   Row {i+1}: {row[:3]}...")  # أول 3 أعمدة فقط
        
        conn.close()
        
    except Exception as e:
        logger.error(f"Error checking structure: {e}")

if __name__ == "__main__":
    # فحص الهيكل أولاً
    check_table_structure()
    
    # ثم دمج البيانات
    merge_tables_to_price_data()