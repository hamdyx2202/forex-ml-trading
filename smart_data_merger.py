#!/usr/bin/env python3
"""
🔄 Smart Data Merger - دمج ذكي للبيانات
📊 يحلل هيكل البيانات ويدمجها بذكاء
"""

import sqlite3
import pandas as pd
import logging
from datetime import datetime
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

class SmartDataMerger:
    def __init__(self, db_path='./data/forex_ml.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        
    def analyze_data_structure(self):
        """تحليل هيكل البيانات الموجودة"""
        logger.info("🔍 Analyzing database structure...")
        
        # 1. جلب الجداول المعالجة
        self.cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name LIKE '%_processed'
        """)
        tables = [t[0] for t in self.cursor.fetchall()]
        
        if not tables:
            logger.error("No processed tables found!")
            return None
        
        # 2. تحليل جدول واحد كعينة
        sample_table = tables[0]
        logger.info(f"\n📊 Analyzing sample table: {sample_table}")
        
        # جلب بعض البيانات
        self.cursor.execute(f"SELECT * FROM {sample_table} LIMIT 10")
        sample_data = self.cursor.fetchall()
        
        # جلب أسماء الأعمدة
        self.cursor.execute(f"PRAGMA table_info({sample_table})")
        columns = [(col[1], col[2]) for col in self.cursor.fetchall()]
        
        logger.info(f"\nColumns found: {len(columns)}")
        for col_name, col_type in columns[:10]:  # أول 10 أعمدة
            logger.info(f"   - {col_name} ({col_type})")
        
        # 3. تحديد نوع البيانات
        timeframe_columns = []
        for col_name, _ in columns:
            if 'PERIOD_' in col_name:
                timeframe_columns.append(col_name)
        
        logger.info(f"\nTimeframe columns: {timeframe_columns}")
        
        # 4. فحص نوع البيانات في أعمدة الأطر الزمنية
        if sample_data and timeframe_columns:
            sample_row = sample_data[0]
            time_col_index = next((i for i, (name, _) in enumerate(columns) if name == 'time'), None)
            
            if time_col_index is not None:
                logger.info(f"\nSample time value: {sample_row[time_col_index]}")
            
            # فحص بيانات OHLC
            for tf_col in timeframe_columns[:1]:  # فحص عمود واحد
                col_index = next((i for i, (name, _) in enumerate(columns) if name == tf_col), None)
                if col_index is not None and col_index < len(sample_row):
                    value = sample_row[col_index]
                    logger.info(f"\nSample {tf_col} value type: {type(value)}")
                    if value:
                        logger.info(f"Sample value: {str(value)[:100]}...")
        
        return {
            'tables': tables,
            'columns': columns,
            'timeframe_columns': timeframe_columns,
            'sample_data': sample_data[:3] if sample_data else []
        }
    
    def create_unified_table(self):
        """إنشاء جدول موحد للبيانات"""
        logger.info("\n📊 Creating unified price_data table...")
        
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS price_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            time DATETIME NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume INTEGER DEFAULT 0,
            spread REAL DEFAULT 0,
            UNIQUE(symbol, timeframe, time)
        )
        """)
        
        # إنشاء فهارس
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol_timeframe ON price_data(symbol, timeframe)")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_time ON price_data(time)")
        
        self.conn.commit()
        logger.info("✅ Table created successfully")
    
    def parse_ohlc_data(self, data_value):
        """تحليل بيانات OHLC من قيم مختلفة"""
        if data_value is None or data_value == '':
            return None
        
        try:
            # إذا كانت البيانات نصية
            if isinstance(data_value, str):
                # محاولة تحليل JSON
                try:
                    data = json.loads(data_value)
                    if isinstance(data, dict):
                        return {
                            'open': float(data.get('open', 0)),
                            'high': float(data.get('high', 0)),
                            'low': float(data.get('low', 0)),
                            'close': float(data.get('close', 0)),
                            'volume': int(data.get('volume', 0))
                        }
                except:
                    pass
                
                # محاولة تحليل CSV
                parts = data_value.split(',')
                if len(parts) >= 4:
                    return {
                        'open': float(parts[0]),
                        'high': float(parts[1]),
                        'low': float(parts[2]),
                        'close': float(parts[3]),
                        'volume': int(parts[4]) if len(parts) > 4 else 0
                    }
            
            # إذا كانت البيانات رقمية (ربما close price فقط)
            elif isinstance(data_value, (int, float)):
                price = float(data_value)
                return {
                    'open': price,
                    'high': price,
                    'low': price,
                    'close': price,
                    'volume': 0
                }
            
        except Exception as e:
            pass
        
        return None
    
    def merge_data(self):
        """دمج البيانات من الجداول المنفصلة"""
        # تحليل الهيكل أولاً
        structure = self.analyze_data_structure()
        if not structure:
            return
        
        # إنشاء الجدول الموحد
        self.create_unified_table()
        
        # تحويل الأطر الزمنية
        timeframe_mapping = {
            'PERIOD_M5': 'M5',
            'PERIOD_M15': 'M15',
            'PERIOD_M30': 'M30',
            'PERIOD_H1': 'H1',
            'PERIOD_H4': 'H4',
            'PERIOD_D1': 'D1'
        }
        
        total_records = 0
        
        # معالجة كل جدول
        for table_name in structure['tables']:
            symbol = table_name.replace('_processed', '')
            logger.info(f"\n🔄 Processing {symbol}...")
            
            # جلب البيانات
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql_query(query, self.conn)
            
            if df.empty:
                logger.warning(f"   ⚠️ No data in {table_name}")
                continue
            
            records_added = 0
            
            # معالجة كل إطار زمني
            for period_col, tf in timeframe_mapping.items():
                if period_col in df.columns and 'time' in df.columns:
                    # فلترة السجلات غير الفارغة
                    valid_data = df[df[period_col].notna()]
                    
                    if len(valid_data) == 0:
                        continue
                    
                    logger.info(f"   {tf}: Processing {len(valid_data)} records...")
                    
                    for _, row in valid_data.iterrows():
                        try:
                            # تحويل الوقت
                            time_val = row['time']
                            if isinstance(time_val, (int, float)):
                                time_str = datetime.fromtimestamp(time_val).strftime('%Y-%m-%d %H:%M:%S')
                            else:
                                time_str = str(time_val)
                            
                            # تحليل بيانات OHLC
                            ohlc = self.parse_ohlc_data(row[period_col])
                            
                            if ohlc:
                                # إدراج في الجدول الموحد
                                self.cursor.execute("""
                                INSERT OR REPLACE INTO price_data 
                                (symbol, timeframe, time, open, high, low, close, volume)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                """, (
                                    symbol, tf, time_str,
                                    ohlc['open'], ohlc['high'], ohlc['low'], ohlc['close'],
                                    ohlc['volume']
                                ))
                                records_added += 1
                                
                        except Exception as e:
                            continue
                    
                    # حفظ كل 1000 سجل
                    if records_added % 1000 == 0:
                        self.conn.commit()
            
            if records_added > 0:
                logger.info(f"   ✅ Added {records_added} records")
                total_records += records_added
                self.conn.commit()
        
        # إحصائيات نهائية
        self.show_final_stats()
        
        return total_records
    
    def show_final_stats(self):
        """عرض الإحصائيات النهائية"""
        logger.info("\n" + "="*60)
        logger.info("📊 Final Statistics:")
        
        # إجمالي السجلات
        self.cursor.execute("SELECT COUNT(*) FROM price_data")
        total = self.cursor.fetchone()[0]
        logger.info(f"   Total records: {total:,}")
        
        # البيانات حسب الإطار الزمني
        logger.info("\n📈 Data by timeframe:")
        self.cursor.execute("""
        SELECT timeframe, COUNT(*) as count, COUNT(DISTINCT symbol) as symbols
        FROM price_data
        GROUP BY timeframe
        ORDER BY timeframe
        """)
        
        for tf, count, symbols in self.cursor.fetchall():
            logger.info(f"   {tf}: {count:,} records ({symbols} symbols)")
        
        # أفضل الأزواج للتدريب
        logger.info("\n🎯 Training-ready pairs (M15 >2000):")
        self.cursor.execute("""
        SELECT symbol, COUNT(*) as count
        FROM price_data
        WHERE timeframe = 'M15'
        GROUP BY symbol
        HAVING count > 2000
        ORDER BY count DESC
        """)
        
        ready_pairs = self.cursor.fetchall()
        if ready_pairs:
            for symbol, count in ready_pairs:
                logger.info(f"   ✅ {symbol}: {count:,} candles")
            logger.info(f"\n✅ {len(ready_pairs)} pairs ready for training!")
        else:
            logger.info("   ⚠️ No pairs with sufficient M15 data")
            
            # عرض ما هو متاح
            self.cursor.execute("""
            SELECT symbol, COUNT(*) as count
            FROM price_data
            WHERE timeframe = 'M15'
            GROUP BY symbol
            ORDER BY count DESC
            LIMIT 10
            """)
            
            available = self.cursor.fetchall()
            if available:
                logger.info("\n📊 Available M15 data:")
                for symbol, count in available:
                    logger.info(f"   {symbol}: {count:,} candles")
    
    def close(self):
        """إغلاق الاتصال"""
        self.conn.close()

def main():
    logger.info("="*60)
    logger.info("🔄 Smart Data Merger")
    logger.info("="*60)
    
    merger = SmartDataMerger()
    
    try:
        # دمج البيانات
        total = merger.merge_data()
        
        if total and total > 0:
            logger.info(f"\n✅ Successfully merged {total:,} records!")
            logger.info("\n🚀 You can now run: python3 train_all_pairs_enhanced.py")
        else:
            logger.info("\n⚠️ No data was merged. Please check the data format.")
            
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        merger.close()

if __name__ == "__main__":
    main()