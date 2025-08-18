#!/usr/bin/env python3
"""
🔍 فحص وإصلاح قاعدة البيانات
📊 التحقق من الجداول المتاحة واستخدامها
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def check_database():
    """فحص قاعدة البيانات المتاحة"""
    logger.info("🔍 Checking available databases...")
    
    # البحث عن قواعد البيانات
    db_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.db') or file.endswith('.sqlite'):
                db_files.append(os.path.join(root, file))
    
    if not db_files:
        logger.error("❌ No database files found!")
        return None
    
    logger.info(f"✅ Found {len(db_files)} database files:")
    for db in db_files:
        logger.info(f"  - {db}")
        
    # فحص الجداول في كل قاعدة بيانات
    available_tables = {}
    
    for db_path in db_files:
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # الحصول على قائمة الجداول
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            if tables:
                logger.info(f"\n📊 Database: {db_path}")
                logger.info(f"   Tables found: {len(tables)}")
                
                table_list = []
                for table in tables:
                    table_name = table[0]
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    count = cursor.fetchone()[0]
                    logger.info(f"   - {table_name}: {count} records")
                    table_list.append((table_name, count))
                
                available_tables[db_path] = table_list
                
            conn.close()
            
        except Exception as e:
            logger.error(f"Error checking {db_path}: {str(e)}")
            
    return available_tables

def create_sample_data():
    """إنشاء بيانات تجريبية للتدريب"""
    logger.info("\n📊 Creating sample data for training...")
    
    # إنشاء مجلد البيانات
    os.makedirs('data', exist_ok=True)
    
    # قاعدة البيانات
    db_path = 'data/forex_data.db'
    conn = sqlite3.connect(db_path)
    
    # الرموز الأساسية للتدريب
    symbols = ['EURUSDm', 'GBPUSDm', 'USDJPYm', 'AUDUSDm', 'USDCADm']
    timeframes = ['M5', 'M15', 'M30', 'H1', 'H4']
    
    created_tables = 0
    
    for symbol in symbols:
        for timeframe in timeframes:
            table_name = f"{symbol}_{timeframe}"
            
            try:
                # إنشاء بيانات تجريبية
                num_candles = 5000
                
                # بداية عشوائية للسعر
                base_price = np.random.uniform(0.5, 2.0) if 'JPY' not in symbol else np.random.uniform(100, 150)
                
                # توليد الأسعار
                dates = pd.date_range(end=datetime.now(), periods=num_candles, freq='5min')
                
                # حركة عشوائية
                returns = np.random.normal(0, 0.0002, num_candles)
                prices = base_price * np.exp(np.cumsum(returns))
                
                # إنشاء OHLC
                df = pd.DataFrame({
                    'timestamp': dates,
                    'open': prices * (1 + np.random.uniform(-0.0005, 0.0005, num_candles)),
                    'high': prices * (1 + np.random.uniform(0, 0.001, num_candles)),
                    'low': prices * (1 - np.random.uniform(0, 0.001, num_candles)),
                    'close': prices,
                    'tick_volume': np.random.randint(100, 1000, num_candles),
                    'spread': np.random.randint(1, 5, num_candles),
                    'real_volume': 0
                })
                
                # التأكد من أن high هو الأعلى و low هو الأدنى
                df['high'] = df[['open', 'high', 'close']].max(axis=1)
                df['low'] = df[['open', 'low', 'close']].min(axis=1)
                
                # حفظ في قاعدة البيانات
                df.to_sql(table_name, conn, if_exists='replace', index=False)
                created_tables += 1
                
                logger.info(f"✅ Created {table_name} with {len(df)} records")
                
            except Exception as e:
                logger.error(f"❌ Error creating {table_name}: {str(e)}")
    
    conn.close()
    
    logger.info(f"\n✅ Created {created_tables} tables in {db_path}")
    
    return db_path

def update_training_script():
    """تحديث سكريبت التدريب لاستخدام البيانات المتاحة"""
    logger.info("\n📝 Creating updated training script...")
    
    script_content = '''#!/usr/bin/env python3
"""
🚀 Complete Advanced System - Updated
✨ يستخدم البيانات المتاحة فقط
📊 جميع الميزات والنماذج المتقدمة
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# إضافة المسار
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the complete system
from complete_advanced_system import CompleteAdvancedSystem

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

class UpdatedAdvancedSystem(CompleteAdvancedSystem):
    """نسخة محدثة تستخدم البيانات المتاحة"""
    
    def __init__(self):
        super().__init__()
        
        # تحديث الرموز بناءً على البيانات المتاحة
        self.symbols = self.get_available_symbols()
        
    def get_available_symbols(self):
        """الحصول على الرموز المتاحة من قاعدة البيانات"""
        db_path = 'data/forex_data.db'
        
        if not os.path.exists(db_path):
            logger.warning("Database not found, using default symbols")
            return ['EURUSDm', 'GBPUSDm', 'USDJPYm', 'AUDUSDm', 'USDCADm']
            
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # الحصول على الجداول
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            # استخراج الرموز الفريدة
            symbols = set()
            for table in tables:
                table_name = table[0]
                # استخراج اسم الرمز
                if '_' in table_name:
                    symbol = table_name.split('_')[0]
                    symbols.add(symbol)
                    
            conn.close()
            
            symbol_list = sorted(list(symbols))
            logger.info(f"Found {len(symbol_list)} symbols: {symbol_list}")
            
            return symbol_list if symbol_list else ['EURUSDm']
            
        except Exception as e:
            logger.error(f"Error getting symbols: {str(e)}")
            return ['EURUSDm']

def main():
    # إنشاء النظام المحدث
    system = UpdatedAdvancedSystem()
    
    logger.info("="*100)
    logger.info("🚀 Starting Complete Advanced Training with Available Data")
    logger.info(f"📊 Symbols: {len(system.symbols)}")
    logger.info(f"⏰ Timeframes: {len(system.timeframes)}")
    logger.info(f"🤖 Models: {len(system.model_configs)}")
    logger.info(f"🎯 Strategies: {len(system.target_configs)}")
    logger.info("="*100)
    
    # بدء التدريب
    system.train_all_symbols()

if __name__ == "__main__":
    main()
'''
    
    with open('train_with_available_data.py', 'w', encoding='utf-8') as f:
        f.write(script_content)
        
    logger.info("✅ Created train_with_available_data.py")
    
def main():
    logger.info("="*80)
    logger.info("🔧 Database Check and Fix Tool")
    logger.info("="*80)
    
    # فحص قواعد البيانات المتاحة
    available_tables = check_database()
    
    if not available_tables or all(len(tables) == 0 for tables in available_tables.values()):
        logger.info("\n⚠️ No data found, creating sample data...")
        db_path = create_sample_data()
        
    # إنشاء سكريبت محدث
    update_training_script()
    
    logger.info("\n✅ Setup complete!")
    logger.info("\n📌 Next steps:")
    logger.info("1. Run: python3 train_with_available_data.py")
    logger.info("2. Or run: python3 complete_advanced_system.py")
    
if __name__ == "__main__":
    main()