#!/usr/bin/env python3
"""
🔍 البحث عن قواعد البيانات وتدريب جميع النماذج
📊 للملايين من البيانات المخزنة
"""

import os
import sys
import sqlite3
import glob
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def find_databases():
    """البحث عن جميع قواعد البيانات في النظام"""
    logger.info("🔍 البحث عن قواعد البيانات...")
    
    # المسارات المحتملة
    search_paths = [
        "./",
        "./data/",
        "../",
        "/home/forex-ml-trading/",
        "/root/forex-ml-trading/",
        "/home/",
        "/root/",
        "/var/lib/",
        "/opt/"
    ]
    
    all_dbs = []
    
    for path in search_paths:
        if os.path.exists(path):
            # البحث عن ملفات .db
            pattern = os.path.join(path, "**/*.db")
            dbs = glob.glob(pattern, recursive=True)
            all_dbs.extend(dbs)
    
    # إزالة المكرر
    all_dbs = list(set(all_dbs))
    
    # فحص كل قاعدة بيانات
    valid_dbs = []
    for db_path in all_dbs:
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # الحصول على الجداول
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            if tables:
                # حساب إجمالي السجلات
                total_records = 0
                table_info = []
                
                for table in tables:
                    table_name = table[0]
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                        count = cursor.fetchone()[0]
                        if count > 0:
                            total_records += count
                            table_info.append(f"{table_name} ({count:,} records)")
                    except:
                        continue
                
                if total_records > 1000:  # فقط القواعد التي تحتوي على بيانات كافية
                    valid_dbs.append({
                        'path': db_path,
                        'size': os.path.getsize(db_path) / (1024*1024),  # MB
                        'tables': len(tables),
                        'records': total_records,
                        'table_info': table_info[:5]  # أول 5 جداول
                    })
            
            conn.close()
            
        except Exception as e:
            continue
    
    return sorted(valid_dbs, key=lambda x: x['records'], reverse=True)

def train_from_database(db_path):
    """تدريب النماذج من قاعدة بيانات محددة"""
    logger.info(f"\n📊 تدريب من: {db_path}")
    
    try:
        # استيراد النظام
        from complete_forex_ml_server import CompleteForexMLSystem
        system = CompleteForexMLSystem()
        
        # تحديث مسار قاعدة البيانات
        system.historical_db = db_path
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # الحصول على الجداول
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        trained_count = 0
        
        for table in tables:
            table_name = table[0]
            
            # تحليل اسم الجدول
            symbol = None
            timeframe = None
            
            # محاولة استخراج الرمز والإطار الزمني
            if '_' in table_name:
                parts = table_name.split('_')
                if len(parts) >= 2:
                    symbol = parts[0]
                    timeframe = parts[1]
            elif 'm' in table_name.lower():
                # مثل EURUSDm أو USDJPYm
                idx = table_name.lower().find('m')
                if idx > 0:
                    symbol = table_name[:idx+1]
                    timeframe = 'M15'  # افتراضي
            
            if symbol:
                try:
                    # التحقق من عدد السجلات
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    count = cursor.fetchone()[0]
                    
                    if count > 1000:
                        logger.info(f"\n🎯 تدريب {symbol} {timeframe or 'ALL'} ({count:,} سجل)")
                        
                        # تدريب لأطر زمنية مختلفة إذا لم يكن محدد
                        if not timeframe:
                            timeframes = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1']
                        else:
                            timeframes = [timeframe]
                        
                        for tf in timeframes:
                            try:
                                success = system.train_models(symbol, tf)
                                if success:
                                    trained_count += 1
                                    logger.info(f"   ✅ {symbol} {tf} - تم التدريب")
                            except:
                                continue
                                
                except Exception as e:
                    logger.error(f"   ❌ خطأ في {table_name}: {str(e)}")
        
        conn.close()
        
        logger.info(f"\n✅ تم تدريب {trained_count} نموذج من {db_path}")
        return trained_count
        
    except Exception as e:
        logger.error(f"❌ خطأ في التدريب: {str(e)}")
        return 0

def update_server_config():
    """تحديث إعدادات السيرفر لاستخدام أفضل قاعدة بيانات"""
    config = """
# تحديث مسار قاعدة البيانات في complete_forex_ml_server.py
# ابحث عن السطر:
# self.historical_db = './data/forex_ml.db'
# واستبدله بـ:
# self.historical_db = '{best_db_path}'
"""
    
    return config

def main():
    logger.info("\n" + "="*80)
    logger.info("🔍 البحث عن قواعد البيانات وتدريب النماذج")
    logger.info("="*80)
    
    # البحث عن قواعد البيانات
    databases = find_databases()
    
    if not databases:
        logger.error("❌ لم يتم العثور على أي قواعد بيانات!")
        return
    
    logger.info(f"\n📊 تم العثور على {len(databases)} قاعدة بيانات:")
    
    for i, db in enumerate(databases[:5], 1):  # عرض أول 5 فقط
        logger.info(f"\n{i}. {db['path']}")
        logger.info(f"   📏 الحجم: {db['size']:.1f} MB")
        logger.info(f"   📊 السجلات: {db['records']:,}")
        logger.info(f"   📋 الجداول: {db['tables']}")
        if db['table_info']:
            logger.info(f"   🔍 عينة: {', '.join(db['table_info'][:3])}")
    
    # اختيار أفضل قاعدة بيانات (الأكبر)
    best_db = databases[0]
    logger.info(f"\n🎯 أفضل قاعدة بيانات: {best_db['path']}")
    logger.info(f"   مع {best_db['records']:,} سجل")
    
    # السؤال عن التدريب
    logger.info("\n🤖 بدء التدريب التلقائي...")
    
    total_trained = 0
    
    # تدريب من أكبر 3 قواعد بيانات
    for db in databases[:3]:
        trained = train_from_database(db['path'])
        total_trained += trained
    
    # عرض الملخص
    logger.info("\n" + "="*80)
    logger.info("📊 ملخص التدريب:")
    logger.info(f"✅ تم تدريب {total_trained} نموذج")
    logger.info(f"📁 النماذج محفوظة في: ./trained_models/")
    
    # نصائح للتحديث
    logger.info("\n💡 لتحديث السيرفر:")
    logger.info(f"1. عدّل complete_forex_ml_server.py")
    logger.info(f"2. غيّر self.historical_db = '{best_db['path']}'")
    logger.info(f"3. أعد تشغيل السيرفر")
    
    # إنشاء ملف إعدادات
    with open('database_config.txt', 'w') as f:
        f.write(f"BEST_DATABASE={best_db['path']}\n")
        f.write(f"TOTAL_RECORDS={best_db['records']}\n")
        f.write(f"DATABASE_SIZE={best_db['size']:.1f}MB\n")
    
    logger.info("\n✅ تم حفظ الإعدادات في database_config.txt")

if __name__ == "__main__":
    main()