#!/usr/bin/env python3
"""
سكريبت للتعلم من البيانات التاريخية
يحلل السنوات الماضية ويتعلم من الصفقات الافتراضية
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Linux compatibility
try:
    import src.linux_compatibility
except:
    pass

from src.advanced_learner import AdvancedLearner
from loguru import logger
import json
from datetime import datetime

# Try to import MT5DataCollector
try:
    from src.data_collector import MT5DataCollector
except ImportError:
    logger.warning("MT5DataCollector not available - using mock data")
    MT5DataCollector = None


def main():
    """التعلم من التاريخ لجميع الأزواج"""
    logger.add("logs/historical_learning.log", rotation="1 day")
    
    print("=" * 60)
    print("🧠 نظام التعلم من التاريخ")
    print("=" * 60)
    
    # تحميل الإعدادات
    with open("config/config.json", 'r') as f:
        config = json.load(f)
    
    # إنشاء المتعلم المتقدم
    learner = AdvancedLearner()
    
    # الأزواج والإطارات الزمنية للتعلم
    pairs = config['trading']['pairs']
    timeframes = ['M15', 'H1', 'H4']  # إطارات زمنية متعددة للتعلم الشامل
    
    print(f"\n📊 سيتم تحليل {len(pairs)} أزواج عملات على {len(timeframes)} إطارات زمنية")
    print(f"📅 فترة التحليل: آخر 365 يوم")
    
    # التأكد من وجود البيانات
    print("\n1️⃣ فحص البيانات المتاحة...")
    collector = MT5DataCollector()
    
    for symbol in pairs:
        print(f"\n🔍 تحليل {symbol}:")
        
        for timeframe in timeframes:
            try:
                print(f"  ⏰ {timeframe}:", end=" ")
                
                # فحص البيانات
                df = collector.get_latest_data(symbol, timeframe, limit=100)
                if df.empty:
                    print("❌ لا توجد بيانات - يجب جمعها أولاً")
                    continue
                
                print(f"✅ {len(df)} شمعة متاحة")
                
                # بدء التعلم
                print(f"    🧪 بدء التحليل والتعلم...")
                learner.analyze_historical_opportunities(symbol, timeframe, lookback_days=365)
                
                print("    ✅ تم التحليل بنجاح")
                
            except Exception as e:
                print(f"❌ خطأ: {str(e)}")
                logger.error(f"Error analyzing {symbol} {timeframe}: {str(e)}")
    
    # عرض تقرير التعلم
    print("\n" + "=" * 60)
    print("📈 تقرير التعلم النهائي")
    print("=" * 60)
    
    report = learner.get_learning_report()
    
    if report['general_stats']['total_trades'] > 0:
        print(f"\n📊 الإحصائيات العامة:")
        print(f"  • إجمالي التجارب الافتراضية: {report['general_stats']['total_trades']:,}")
        print(f"  • التجارب الناجحة: {report['general_stats']['successful_trades']:,}")
        print(f"  • نسبة النجاح: {report['success_rate']:.1%}")
        print(f"  • متوسط النقاط: {report['general_stats']['avg_pips']:.1f} pips")
        print(f"  • أفضل صفقة: {report['general_stats']['best_trade']:.1f} pips")
        print(f"  • أسوأ صفقة: {report['general_stats']['worst_trade']:.1f} pips")
        
        if report['best_patterns']:
            print(f"\n🎯 أفضل أنماط الشموع:")
            for pattern in report['best_patterns'][:5]:
                win_rate = pattern['wins'] / pattern['count'] if pattern['count'] > 0 else 0
                print(f"  • {pattern['candle_pattern']}: {win_rate:.1%} نجاح ({pattern['count']} صفقة)")
        
        if report['best_hours']:
            print(f"\n⏰ أفضل ساعات التداول:")
            for hour_data in report['best_hours'][:5]:
                win_rate = hour_data['wins'] / hour_data['count'] if hour_data['count'] > 0 else 0
                print(f"  • الساعة {hour_data['hour']:02d}:00: {win_rate:.1%} نجاح ({hour_data['count']} صفقة)")
    else:
        print("\n⚠️ لم يتم العثور على تجارب افتراضية بعد")
        print("تأكد من توفر البيانات التاريخية الكافية")
    
    # البحث عن فرص حالية
    print("\n" + "=" * 60)
    print("🔍 البحث عن فرص التداول الحالية")
    print("=" * 60)
    
    all_opportunities = []
    
    for symbol in pairs[:3]:  # أول 3 أزواج فقط للسرعة
        print(f"\n{symbol}:")
        for timeframe in ['H1', 'H4']:  # الإطارات الأكبر للإشارات الأقوى
            opportunities = learner.find_high_quality_opportunities(symbol, timeframe)
            
            if opportunities:
                for opp in opportunities[:2]:  # أفضل فرصتين فقط
                    print(f"\n  💎 فرصة عالية الجودة:")
                    print(f"    • الإطار الزمني: {timeframe}")
                    print(f"    • الاتجاه: {opp['direction']}")
                    print(f"    • السعر الحالي: {opp['price']:.5f}")
                    print(f"    • الثقة: {opp['confidence']:.1%}")
                    print(f"    • النقاط: {opp['score']}/10")
                    print(f"    • Stop Loss: {opp['suggested_sl']:.5f}")
                    print(f"    • Take Profit: {opp['suggested_tp']:.5f}")
                    print(f"    • الأسباب:")
                    for reason in opp['reasons']:
                        print(f"      - {reason}")
                    
                    all_opportunities.append(opp)
            else:
                print(f"  • {timeframe}: لا توجد فرص قوية حالياً")
    
    # حفظ الفرص في ملف
    if all_opportunities:
        with open('data/current_opportunities.json', 'w') as f:
            json.dump(all_opportunities, f, indent=2, default=str)
        print(f"\n✅ تم حفظ {len(all_opportunities)} فرصة في data/current_opportunities.json")
    
    print("\n" + "=" * 60)
    print("✅ اكتمل التعلم من التاريخ!")
    print("=" * 60)
    
    print("\n💡 الخطوات التالية:")
    print("1. راجع الفرص المحفوظة في data/current_opportunities.json")
    print("2. شغل البوت للتداول: python main.py trade")
    print("3. راقب الأداء عبر: streamlit run dashboard.py")


if __name__ == "__main__":
    main()