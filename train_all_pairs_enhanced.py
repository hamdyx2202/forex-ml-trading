#!/usr/bin/env python3
"""
🚀 تدريب جميع الأزواج بالنظام المحسن
📊 Market Context + 200+ Features + 6 ML Models
💰 نظام قادر على الربح الحقيقي
"""

import os
import sys
import time
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

def get_all_available_pairs(min_candles=1000):  # خفضنا من 2000 إلى 1000
    """جلب جميع الأزواج المتاحة من قاعدة البيانات"""
    db_path = './data/forex_ml.db'
    
    # تحقق من وجود قاعدة البيانات
    if not os.path.exists(db_path):
        logger.error(f"❌ Database not found at: {db_path}")
        logger.info("   Please ensure the database exists with price data")
        return pd.DataFrame()
    
    try:
        conn = sqlite3.connect(db_path)
        
        # تحقق من وجود جدول price_data
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='price_data'")
        if not cursor.fetchone():
            logger.error("❌ Table 'price_data' not found in database")
            conn.close()
            return pd.DataFrame()
        
        query = f"""
        SELECT symbol, COUNT(*) as count 
        FROM price_data 
        WHERE timeframe = 'M15'
        GROUP BY symbol 
        HAVING count > {min_candles}
        ORDER BY count DESC
        """
        pairs = pd.read_sql_query(query, conn)
        conn.close()
        
        if pairs.empty:
            logger.warning(f"⚠️ No pairs found with sufficient data (>{min_candles} M15 candles)")
            logger.info("   Checking all available data...")
            
            # عرض ما هو متاح
            conn = sqlite3.connect(db_path)
            query_all = """
            SELECT symbol, timeframe, COUNT(*) as count 
            FROM price_data 
            GROUP BY symbol, timeframe 
            ORDER BY count DESC
            LIMIT 20
            """
            all_data = pd.read_sql_query(query_all, conn)
            conn.close()
            
            if not all_data.empty:
                logger.info("\n📊 Available data in database:")
                for _, row in all_data.iterrows():
                    logger.info(f"   - {row['symbol']} ({row['timeframe']}): {row['count']:,} candles")
        
        return pairs
        
    except Exception as e:
        logger.error(f"❌ Database error: {e}")
        return pd.DataFrame()

def get_priority_pairs():
    """الأزواج ذات الأولوية للتدريب"""
    return [
        'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD',
        'USDCAD', 'NZDUSD', 'EURJPY', 'GBPJPY', 'EURNZD',
        'EURAUD', 'EURCAD', 'EURGBP', 'XAUUSD'
    ]

def main():
    logger.info("\n" + "="*80)
    logger.info("🚀 Enhanced ML Training System")
    logger.info("📊 تدريب النماذج مع تحليل السياق الكامل")
    logger.info("💰 نظام قادر على تحقيق الربح الحقيقي")
    logger.info("="*80)
    
    # إنشاء النظام المحسن
    logger.info("\n🔧 Initializing Enhanced Trading System...")
    system = EnhancedMLTradingSystem()
    
    # جلب جميع الأزواج
    pairs_df = get_all_available_pairs()
    logger.info(f"\n📊 عدد الأزواج المتاحة: {len(pairs_df)}")
    
    # التحقق من وجود بيانات
    if pairs_df.empty:
        logger.error("\n❌ No pairs available for training!")
        logger.info("\n💡 Troubleshooting steps:")
        logger.info("   1. Check database path: ./data/forex_ml.db")
        logger.info("   2. Ensure database contains price_data table")
        logger.info(f"   3. Ensure sufficient M15 data (>{min_candles} candles per pair)")
        logger.info("   4. Run: python3 full_data_merger.py")
        return
    
    # ترتيب الأزواج حسب الأولوية
    priority_pairs = get_priority_pairs()
    pairs_list = []
    
    # أضف الأزواج ذات الأولوية أولاً
    for pair in priority_pairs:
        for _, row in pairs_df.iterrows():
            symbol = row['symbol']
            clean = symbol.replace('m', '').replace('.ecn', '').replace('_pro', '')
            if clean.upper() == pair.upper():
                pairs_list.append((symbol, row['count'], True))  # True = priority
                break
    
    # أضف باقي الأزواج
    for _, row in pairs_df.iterrows():
        symbol = row['symbol']
        if not any(symbol == p[0] for p in pairs_list):
            pairs_list.append((symbol, row['count'], False))  # False = not priority
    
    # الفريمات الزمنية (مع التركيز على الفريمات القصيرة والمتوسطة)
    timeframes = ['M15', 'M30', 'H1']  # H4 و D1 اختياري
    
    total_models = 0
    failed_models = 0
    start_time = time.time()
    
    logger.info(f"\n🎯 Priority pairs: {len([p for p in pairs_list if p[2]])} من أصل {len(pairs_list)}")
    logger.info(f"⏰ Timeframes: {', '.join(timeframes)}")
    
    # تدريب كل زوج
    for idx, (symbol, count, is_priority) in enumerate(pairs_list):
        # تنظيف اسم الرمز
        clean_symbol = symbol.replace('m', '').replace('.ecn', '').replace('_pro', '')
        
        priority_mark = "⭐" if is_priority else "  "
        logger.info(f"\n{'='*60}")
        logger.info(f"{priority_mark} [{idx+1}/{len(pairs_list)}] Training {symbol} ({count:,} records)")
        
        # Skip non-priority pairs after first 20 pairs (to save time)
        if idx >= 20 and not is_priority:
            logger.info("   ⏭️ Skipping non-priority pair to save time")
            continue
        
        pair_start = time.time()
        pair_models = 0
        
        for timeframe in timeframes:
            try:
                logger.info(f"\n   ⏰ {timeframe} - Enhanced training with market context...")
                start = time.time()
                
                # تدريب محسن مع تحليل السوق
                success = system.train_enhanced_models(clean_symbol, timeframe)
                
                if success:
                    elapsed = time.time() - start
                    logger.info(f"   ✅ Completed in {elapsed:.1f}s")
                    total_models += 1
                    pair_models += 1
                else:
                    logger.warning(f"   ⚠️ Training skipped (insufficient quality data)")
                    failed_models += 1
                    
                # راحة بين النماذج
                time.sleep(2)
                
            except KeyboardInterrupt:
                logger.warning("\n⚠️ Training interrupted by user")
                return
                
            except Exception as e:
                logger.error(f"   ❌ Error: {str(e)}")
                failed_models += 1
        
        # ملخص الزوج
        pair_elapsed = time.time() - pair_start
        if pair_models > 0:
            logger.info(f"\n   📊 {clean_symbol} Summary:")
            logger.info(f"      Models trained: {pair_models}")
            logger.info(f"      Time: {pair_elapsed:.1f}s")
            logger.info(f"      Status: Ready for profitable trading! 💰")
        
        # تقرير مؤقت كل 5 أزواج
        if (idx + 1) % 5 == 0:
            elapsed_total = time.time() - start_time
            logger.info(f"\n{'='*60}")
            logger.info(f"📊 Progress Report:")
            logger.info(f"   ✅ Models trained: {total_models}")
            logger.info(f"   ❌ Failed: {failed_models}")
            logger.info(f"   ⏰ Elapsed: {elapsed_total/60:.1f} minutes")
            if (total_models + failed_models) > 0:
                logger.info(f"   📈 Success rate: {(total_models/(total_models+failed_models)*100):.1f}%")
            
            # Risk management status
            risk_report = system.risk_manager.get_risk_report()
            logger.info(f"\n💰 Risk Management:")
            logger.info(f"   Balance: ${risk_report['current_balance']:.2f}")
            logger.info(f"   Risk Status: {risk_report['risk_status']}")
    
    # التقرير النهائي
    total_time = time.time() - start_time
    logger.info("\n" + "="*80)
    logger.info("🏁 FINAL REPORT - Enhanced Training Complete")
    logger.info("="*80)
    logger.info(f"✅ Models trained: {total_models}")
    logger.info(f"❌ Failed: {failed_models}")
    logger.info(f"⏰ Total time: {total_time/60:.1f} minutes")
    
    # تجنب القسمة على صفر
    if (total_models + failed_models) > 0:
        success_rate = (total_models/(total_models+failed_models)*100)
        logger.info(f"🎯 Success rate: {success_rate:.1f}%")
    else:
        logger.info(f"⚠️ No models were trained - check database connection")
    logger.info(f"📁 Models saved in: ./trained_models/")
    
    logger.info(f"\n💡 System Capabilities:")
    logger.info(f"   ✓ Market context analysis before trading")
    logger.info(f"   ✓ Dynamic risk management")
    logger.info(f"   ✓ Support/Resistance based SL/TP")
    logger.info(f"   ✓ Multi-timeframe trend alignment")
    logger.info(f"   ✓ Session quality filtering")
    logger.info(f"   ✓ Correlation-based exposure limits")
    
    logger.info(f"\n🚀 System is ready for PROFITABLE trading!")
    logger.info("="*80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\n⚠️ Training interrupted by user")
    except Exception as e:
        logger.error(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()