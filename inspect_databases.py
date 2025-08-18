#!/usr/bin/env python3
"""
🔍 فحص قواعد البيانات المتاحة
📊 عرض الجداول والبيانات
"""

import sqlite3
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def inspect_database(db_path):
    """فحص قاعدة بيانات واحدة"""
    logger.info(f"\n{'='*80}")
    logger.info(f"📊 Inspecting: {db_path}")
    logger.info(f"{'='*80}")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # الحصول على قائمة الجداول
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = cursor.fetchall()
        
        if not tables:
            logger.warning("❌ No tables found in this database")
            conn.close()
            return
            
        logger.info(f"✅ Found {len(tables)} tables:")
        
        table_info = []
        
        for table in tables:
            table_name = table[0]
            
            try:
                # عدد الصفوف
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                
                # الحصول على أسماء الأعمدة
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                column_names = [col[1] for col in columns]
                
                # أول صف كعينة
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 1")
                sample = cursor.fetchone()
                
                logger.info(f"\n  📌 Table: {table_name}")
                logger.info(f"     Rows: {count:,}")
                logger.info(f"     Columns: {column_names}")
                
                if sample and count > 0:
                    logger.info(f"     Sample: {dict(zip(column_names, sample))}")
                    
                table_info.append({
                    'table': table_name,
                    'rows': count,
                    'columns': column_names
                })
                
            except Exception as e:
                logger.error(f"     Error reading {table_name}: {str(e)}")
                
        conn.close()
        
        return table_info
        
    except Exception as e:
        logger.error(f"❌ Error opening database: {str(e)}")
        return None

def find_best_database():
    """إيجاد أفضل قاعدة بيانات للتدريب"""
    databases = [
        './forex_data.db',
        './trading_performance.db',
        './data/forex_ml.db',
        './data/forex_data.db'
    ]
    
    best_db = None
    max_tables = 0
    max_rows = 0
    
    all_info = {}
    
    for db_path in databases:
        info = inspect_database(db_path)
        if info:
            all_info[db_path] = info
            
            total_tables = len(info)
            total_rows = sum(t['rows'] for t in info)
            
            if total_tables > max_tables or (total_tables == max_tables and total_rows > max_rows):
                best_db = db_path
                max_tables = total_tables
                max_rows = total_rows
                
    logger.info(f"\n{'='*80}")
    logger.info(f"🏆 Best database for training: {best_db}")
    logger.info(f"   Tables: {max_tables}")
    logger.info(f"   Total rows: {max_rows:,}")
    logger.info(f"{'='*80}")
    
    # إنشاء سكريبت محدث
    if best_db:
        create_updated_config(best_db, all_info[best_db])
        
    return best_db, all_info

def create_updated_config(db_path, table_info):
    """إنشاء ملف تكوين محدث"""
    logger.info("\n📝 Creating updated configuration...")
    
    # استخراج الرموز المتاحة
    symbols = set()
    timeframes = set()
    
    for table in table_info:
        table_name = table['table']
        if '_' in table_name and table['rows'] > 100:
            parts = table_name.split('_')
            if len(parts) >= 2:
                symbol = parts[0]
                tf = parts[1]
                
                # التحقق من صحة الرمز
                if any(currency in symbol.upper() for currency in ['EUR', 'USD', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD']):
                    symbols.add(symbol)
                    timeframes.add(tf)
    
    config = f'''#!/usr/bin/env python3
"""
📊 Database Configuration
🔧 Auto-generated based on available data
"""

# Best database path
DATABASE_PATH = '{db_path}'

# Available symbols
SYMBOLS = {sorted(list(symbols))}

# Available timeframes
TIMEFRAMES = {sorted(list(timeframes))}

# Total tables: {len(table_info)}
# Total data rows: {sum(t['rows'] for t in table_info):,}

print(f"✅ Using database: {DATABASE_PATH}")
print(f"📊 Symbols: {len(SYMBOLS)}")
print(f"⏰ Timeframes: {len(TIMEFRAMES)}")
'''
    
    with open('database_config.py', 'w', encoding='utf-8') as f:
        f.write(config)
        
    logger.info("✅ Created database_config.py")
    
    # تحديث النظام المتقدم
    update_complete_system(db_path)

def update_complete_system(db_path):
    """تحديث النظام المتقدم لاستخدام قاعدة البيانات الصحيحة"""
    
    update_script = f'''#!/usr/bin/env python3
"""
🚀 Run Complete Advanced System with correct database
"""

import sys
import os

# Import configuration
from database_config import DATABASE_PATH, SYMBOLS, TIMEFRAMES

# Import the complete system
from complete_advanced_system import CompleteAdvancedSystem

class ConfiguredAdvancedSystem(CompleteAdvancedSystem):
    def __init__(self):
        super().__init__()
        # Use discovered symbols and timeframes
        self.symbols = SYMBOLS
        self.timeframes = {{tf: self.timeframes.get(tf, 60) for tf in TIMEFRAMES if tf in self.timeframes}}
        
    def train_symbol(self, symbol, timeframe, data_path=None):
        # Use the correct database path
        return super().train_symbol(symbol, timeframe, DATABASE_PATH)

def main():
    print("🚀 Starting Complete Advanced System with discovered data")
    print(f"📊 Database: {{DATABASE_PATH}}")
    print(f"🎯 Symbols: {{len(SYMBOLS)}}")
    print(f"⏰ Timeframes: {{len(TIMEFRAMES)}}")
    print("="*80)
    
    system = ConfiguredAdvancedSystem()
    system.train_all_symbols()

if __name__ == "__main__":
    main()
'''
    
    with open('run_configured_system.py', 'w', encoding='utf-8') as f:
        f.write(update_script)
        
    logger.info("✅ Created run_configured_system.py")

def main():
    logger.info("🔍 Database Inspector")
    logger.info("="*80)
    
    best_db, all_info = find_best_database()
    
    logger.info("\n✅ Analysis complete!")
    logger.info("\n📌 Next steps:")
    logger.info("1. Check the database configuration: cat database_config.py")
    logger.info("2. Run the configured system: python3 run_configured_system.py")
    
if __name__ == "__main__":
    main()