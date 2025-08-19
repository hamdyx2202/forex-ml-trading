#!/usr/bin/env python3
"""
🚀 تدريب بالبيانات المتاحة
📊 يستخدم البيانات الموجودة حتى لو كانت أقل من 2000 شمعة
"""

import os
import sys
import sqlite3
import pandas as pd
import logging

# إضافة مسار السيرفر
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from enhanced_ml_server import EnhancedMLTradingSystem

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def get_all_available_pairs(min_candles=500):  # خفضنا الحد الأدنى
    """جلب جميع الأزواج المتاحة من قاعدة البيانات"""
    db_path = './data/forex_ml.db'
    
    if not os.path.exists(db_path):
        logger.error(f"❌ Database not found at: {db_path}")
        return pd.DataFrame()
    
    try:
        conn = sqlite3.connect(db_path)
        
        # البحث في جدول price_data أولاً
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='price_data'")
        
        if cursor.fetchone():
            # جدول price_data موجود
            query = f"""
            SELECT symbol, COUNT(*) as count 
            FROM price_data 
            WHERE timeframe = 'M15'
            GROUP BY symbol 
            HAVING count > {min_candles}
            ORDER BY count DESC
            """
            pairs = pd.read_sql_query(query, conn)
            
            if not pairs.empty:
                logger.info(f"✅ Found {len(pairs)} pairs in price_data with >{min_candles} M15 candles")
                conn.close()
                return pairs
        
        # البحث في الجداول المعالجة مباشرة
        logger.info("📊 Checking processed tables directly...")
        
        cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name LIKE '%_processed'
        """)
        tables = cursor.fetchall()
        
        pairs_data = []
        
        for table_tuple in tables:
            table_name = table_tuple[0]
            symbol = table_name.replace('_processed', '')
            
            # عد السجلات M15
            try:
                query = f"""
                SELECT COUNT(*) as count
                FROM {table_name}
                WHERE timeframe = 'PERIOD_M15'
                """
                cursor.execute(query)
                count = cursor.fetchone()[0]
                
                if count > min_candles:
                    pairs_data.append({
                        'symbol': symbol,
                        'count': count,
                        'table': table_name
                    })
                    logger.info(f"   ✅ {symbol}: {count:,} M15 candles")
                
            except Exception as e:
                continue
        
        conn.close()
        
        if pairs_data:
            return pd.DataFrame(pairs_data)
        else:
            logger.warning(f"⚠️ No pairs found with >{min_candles} M15 candles")
            return pd.DataFrame()
        
    except Exception as e:
        logger.error(f"❌ Database error: {e}")
        return pd.DataFrame()

def train_from_processed_tables(system, symbol, timeframe, table_name):
    """تدريب مباشر من الجداول المعالجة"""
    try:
        logger.info(f"🤖 Training {symbol} {timeframe} from {table_name}...")
        
        conn = sqlite3.connect('./data/forex_ml.db')
        
        # تحويل الإطار الزمني
        tf_map = {
            'M15': 'PERIOD_M15',
            'M30': 'PERIOD_M30',
            'H1': 'PERIOD_H1'
        }
        period_tf = tf_map.get(timeframe, f'PERIOD_{timeframe}')
        
        # جلب البيانات
        query = f"""
        SELECT time, open, high, low, close, volume
        FROM {table_name}
        WHERE timeframe = '{period_tf}'
        ORDER BY time DESC
        LIMIT 10000
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(df) < 500:
            logger.warning(f"Not enough data: {len(df)} records")
            return False
        
        # تحضير البيانات
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df = df.sort_index()
        
        # تنظيف الرمز لاستخدامه في النظام
        clean_symbol = symbol.replace('m', '').replace('.ecn', '').upper()
        
        # تدريب النماذج
        return system.train_enhanced_models(clean_symbol, timeframe)
        
    except Exception as e:
        logger.error(f"❌ Training error: {e}")
        return False

def main():
    logger.info("\n" + "="*80)
    logger.info("🚀 Training with Available Data")
    logger.info("📊 Using reduced minimum requirements")
    logger.info("="*80)
    
    # إنشاء النظام
    logger.info("\n🔧 Initializing Enhanced Trading System...")
    system = EnhancedMLTradingSystem()
    
    # جلب الأزواج المتاحة بحد أدنى أقل
    pairs_df = get_all_available_pairs(min_candles=500)
    
    if pairs_df.empty:
        logger.error("\n❌ No pairs available even with reduced requirements!")
        
        # محاولة أخيرة - عرض كل البيانات المتاحة
        conn = sqlite3.connect('./data/forex_ml.db')
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name LIKE '%_processed'
        """)
        tables = cursor.fetchall()
        
        if tables:
            logger.info("\n📊 All available data:")
            for table_tuple in tables:
                table_name = table_tuple[0]
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                logger.info(f"   - {table_name}: {count:,} total records")
        
        conn.close()
        return
    
    logger.info(f"\n📊 Found {len(pairs_df)} pairs for training")
    
    # الأطر الزمنية
    timeframes = ['M15']  # نبدأ بـ M15 فقط
    
    total_models = 0
    failed_models = 0
    
    # تدريب كل زوج
    for idx, row in pairs_df.iterrows():
        symbol = row['symbol']
        count = row['count']
        table_name = row.get('table', f"{symbol}_processed")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"[{idx+1}/{len(pairs_df)}] Training {symbol} ({count:,} M15 candles)")
        
        for timeframe in timeframes:
            try:
                # محاولة التدريب من جدول price_data أولاً
                success = False
                
                # تنظيف الرمز
                clean_symbol = symbol.replace('m', '').replace('.ecn', '').upper()
                
                # محاولة التدريب العادي
                try:
                    success = system.train_enhanced_models(clean_symbol, timeframe)
                except:
                    # إذا فشل، جرب من الجدول المعالج مباشرة
                    logger.info("   🔄 Trying direct table training...")
                    success = train_from_processed_tables(system, symbol, timeframe, table_name)
                
                if success:
                    logger.info(f"   ✅ {timeframe} training completed")
                    total_models += 1
                else:
                    logger.warning(f"   ⚠️ {timeframe} training failed")
                    failed_models += 1
                    
            except Exception as e:
                logger.error(f"   ❌ Error: {e}")
                failed_models += 1
        
        # توقف بعد 5 أزواج للاختبار
        if idx >= 4:
            logger.info("\n📊 Stopping after 5 pairs for testing...")
            break
    
    # التقرير النهائي
    logger.info("\n" + "="*80)
    logger.info("🏁 Training Report")
    logger.info("="*80)
    logger.info(f"✅ Models trained: {total_models}")
    logger.info(f"❌ Failed: {failed_models}")
    
    if total_models > 0:
        logger.info(f"\n✅ Successfully trained {total_models} models!")
        logger.info("🚀 System is ready for trading with available models")
    else:
        logger.info("\n⚠️ No models were trained successfully")
        logger.info("💡 Try running: python3 direct_data_merger.py first")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n⚠️ Training interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()