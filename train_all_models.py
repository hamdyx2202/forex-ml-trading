#!/usr/bin/env python3
"""
🚀 تدريب جميع النماذج من البيانات الموجودة
📊 7.8 مليون سجل للتدريب
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, classification_report

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Import the complete system
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from complete_forex_ml_server import CompleteForexMLSystem

def get_available_pairs():
    """الحصول على أزواج العملات المتاحة في قاعدة البيانات"""
    db_path = './data/forex_ml.db'
    if not os.path.exists(db_path):
        logger.error(f"❌ قاعدة البيانات غير موجودة: {db_path}")
        return []
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # الحصول على جميع الجداول
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        pairs = []
        for table in tables:
            table_name = table[0]
            # تحليل اسم الجدول للحصول على الزوج والإطار الزمني
            if '_' in table_name:
                parts = table_name.split('_')
                if len(parts) == 2:
                    symbol = parts[0]
                    timeframe = parts[1]
                    
                    # التحقق من عدد السجلات
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    count = cursor.fetchone()[0]
                    
                    if count > 1000:  # فقط الجداول التي تحتوي على بيانات كافية
                        pairs.append({
                            'table': table_name,
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'records': count
                        })
        
        conn.close()
        return pairs
        
    except Exception as e:
        logger.error(f"❌ خطأ في قراءة قاعدة البيانات: {str(e)}")
        return []

def train_pair(system, pair_info):
    """تدريب نماذج لزوج عملات واحد"""
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"🎯 تدريب {pair_info['symbol']} {pair_info['timeframe']}")
        logger.info(f"📊 عدد السجلات: {pair_info['records']:,}")
        
        # تدريب النماذج
        success = system.train_models(pair_info['symbol'], pair_info['timeframe'])
        
        if success:
            logger.info(f"✅ تم التدريب بنجاح")
            return True
        else:
            logger.info(f"❌ فشل التدريب")
            return False
            
    except Exception as e:
        logger.error(f"❌ خطأ في التدريب: {str(e)}")
        return False

def main():
    """البرنامج الرئيسي"""
    logger.info("\n" + "="*80)
    logger.info("🚀 بدء تدريب جميع النماذج")
    logger.info("📊 قاعدة البيانات: ./data/forex_ml.db")
    logger.info("="*80)
    
    # إنشاء مثيل من النظام
    system = CompleteForexMLSystem()
    
    # الحصول على الأزواج المتاحة
    pairs = get_available_pairs()
    
    if not pairs:
        logger.error("❌ لم يتم العثور على أي بيانات للتدريب")
        return
    
    logger.info(f"\n📊 تم العثور على {len(pairs)} زوج عملات:")
    for pair in pairs:
        logger.info(f"   - {pair['symbol']} {pair['timeframe']}: {pair['records']:,} سجل")
    
    # تدريب كل زوج
    successful = 0
    failed = 0
    
    for i, pair in enumerate(pairs, 1):
        logger.info(f"\n[{i}/{len(pairs)}] معالجة {pair['symbol']} {pair['timeframe']}...")
        
        if train_pair(system, pair):
            successful += 1
        else:
            failed += 1
    
    # ملخص النتائج
    logger.info("\n" + "="*80)
    logger.info("📊 ملخص التدريب:")
    logger.info(f"✅ نجح: {successful}")
    logger.info(f"❌ فشل: {failed}")
    logger.info(f"📁 النماذج محفوظة في: ./trained_models/")
    logger.info("="*80)
    
    # عرض النماذج المدربة
    models_dir = './trained_models'
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
        logger.info(f"\n📁 النماذج المحفوظة ({len(model_files)} ملف):")
        
        # تنظيم حسب الزوج
        models_by_pair = {}
        for file in model_files:
            parts = file.replace('.pkl', '').split('_')
            if len(parts) >= 2:
                pair_key = f"{parts[0]}_{parts[1]}"
                if pair_key not in models_by_pair:
                    models_by_pair[pair_key] = []
                models_by_pair[pair_key].append(file)
        
        for pair, files in models_by_pair.items():
            logger.info(f"\n   {pair}:")
            for file in files:
                logger.info(f"      - {file}")

if __name__ == "__main__":
    main()