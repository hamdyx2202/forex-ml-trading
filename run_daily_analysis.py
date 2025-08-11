#!/usr/bin/env python3
"""
سكريبت للتحليل اليومي والبحث عن أفضل الفرص
يُشغل يومياً للعثور على صفقات عالية الجودة
"""

from src.advanced_learner import AdvancedLearner
from src.data_collector import MT5DataCollector
from src.monitor import TradingMonitor
from datetime import datetime
import json
from loguru import logger
import asyncio


async def main():
    """التحليل اليومي للفرص"""
    logger.add("logs/daily_analysis.log", rotation="1 day", retention="30 days")
    
    print("\n" + "="*60)
    print(f"📅 التحليل اليومي - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*60)
    
    # تحميل الإعدادات
    with open("config/config.json", 'r') as f:
        config = json.load(f)
    
    # إنشاء المكونات
    learner = AdvancedLearner()
    collector = MT5DataCollector()
    monitor = TradingMonitor()
    
    # تحديث البيانات أولاً
    print("\n1️⃣ تحديث البيانات...")
    try:
        if collector.connect_mt5():
            collector.update_all_pairs()
            collector.disconnect_mt5()
            print("✅ تم تحديث البيانات بنجاح")
        else:
            print("❌ فشل الاتصال بـ MT5")
            return
    except Exception as e:
        print(f"❌ خطأ في تحديث البيانات: {str(e)}")
    
    # البحث عن الفرص
    print("\n2️⃣ البحث عن فرص التداول عالية الجودة...")
    
    all_opportunities = []
    opportunities_by_quality = {
        'excellent': [],  # نقاط 9-10
        'very_good': [],  # نقاط 7-8
        'good': []        # نقاط 5-6
    }
    
    pairs = config['trading']['pairs']
    timeframes = ['M15', 'H1', 'H4']
    
    for symbol in pairs:
        print(f"\n🔍 تحليل {symbol}:")
        
        for timeframe in timeframes:
            try:
                opportunities = learner.find_high_quality_opportunities(symbol, timeframe)
                
                if opportunities:
                    for opp in opportunities:
                        all_opportunities.append(opp)
                        
                        # تصنيف حسب الجودة
                        if opp['score'] >= 9:
                            opportunities_by_quality['excellent'].append(opp)
                        elif opp['score'] >= 7:
                            opportunities_by_quality['very_good'].append(opp)
                        else:
                            opportunities_by_quality['good'].append(opp)
                        
                        print(f"  ✅ {timeframe}: وجدت فرصة (نقاط: {opp['score']}/10)")
                else:
                    print(f"  • {timeframe}: لا توجد فرص قوية")
                    
            except Exception as e:
                print(f"  ❌ خطأ في {timeframe}: {str(e)}")
    
    # عرض ملخص الفرص
    print("\n" + "="*60)
    print("📊 ملخص الفرص المكتشفة")
    print("="*60)
    
    print(f"\n🏆 فرص ممتازة (9-10 نقاط): {len(opportunities_by_quality['excellent'])}")
    for opp in opportunities_by_quality['excellent'][:3]:
        print(f"\n  💎 {opp['symbol']} - {opp['timeframe']}")
        print(f"     • الاتجاه: {opp['direction']}")
        print(f"     • الثقة: {opp['confidence']:.1%}")
        print(f"     • السعر: {opp['price']:.5f}")
        print(f"     • الأسباب: {', '.join(opp['reasons'][:2])}")
    
    print(f"\n🥈 فرص جيدة جداً (7-8 نقاط): {len(opportunities_by_quality['very_good'])}")
    for opp in opportunities_by_quality['very_good'][:3]:
        print(f"\n  ⭐ {opp['symbol']} - {opp['timeframe']}")
        print(f"     • الاتجاه: {opp['direction']}")
        print(f"     • الثقة: {opp['confidence']:.1%}")
    
    print(f"\n🥉 فرص جيدة (5-6 نقاط): {len(opportunities_by_quality['good'])}")
    
    # حفظ الفرص
    if all_opportunities:
        # حفظ في ملف JSON
        with open('data/daily_opportunities.json', 'w') as f:
            json.dump({
                'date': datetime.now().isoformat(),
                'total_opportunities': len(all_opportunities),
                'excellent': len(opportunities_by_quality['excellent']),
                'very_good': len(opportunities_by_quality['very_good']),
                'good': len(opportunities_by_quality['good']),
                'opportunities': all_opportunities[:10]  # أفضل 10 فقط
            }, f, indent=2, default=str)
        
        print(f"\n✅ تم حفظ {len(all_opportunities)} فرصة في data/daily_opportunities.json")
    
    # إرسال تقرير Telegram
    if monitor.telegram_enabled and opportunities_by_quality['excellent']:
        message = f"🤖 *تقرير الفرص اليومية*\n"
        message += f"التاريخ: {datetime.now().strftime('%Y-%m-%d')}\n\n"
        message += f"🏆 *فرص ممتازة: {len(opportunities_by_quality['excellent'])}*\n"
        
        for opp in opportunities_by_quality['excellent'][:3]:
            message += f"\n{opp['symbol']} ({opp['timeframe']})\n"
            message += f"• {opp['direction']} - {opp['confidence']:.0%} ثقة\n"
            message += f"• السعر: {opp['price']:.5f}\n"
            message += f"• SL: {opp['suggested_sl']:.5f}\n"
            message += f"• TP: {opp['suggested_tp']:.5f}\n"
        
        await monitor.send_telegram_message(message)
        print("\n📱 تم إرسال التقرير عبر Telegram")
    
    # تقرير الأداء التاريخي
    print("\n" + "="*60)
    print("📈 تقرير الأداء التاريخي")
    print("="*60)
    
    learning_report = learner.get_learning_report()
    
    if learning_report['general_stats']['total_trades'] > 0:
        print(f"\n• إجمالي التجارب التاريخية: {learning_report['general_stats']['total_trades']:,}")
        print(f"• نسبة النجاح: {learning_report['success_rate']:.1%}")
        print(f"• متوسط الربح: {learning_report['general_stats']['avg_pips']:.1f} نقطة")
        
        # نصائح بناءً على التحليل
        print("\n💡 نصائح اليوم:")
        
        if learning_report['best_hours']:
            best_hour = learning_report['best_hours'][0]
            print(f"• أفضل وقت للتداول: الساعة {best_hour['hour']:02d}:00")
        
        if learning_report['best_patterns']:
            best_pattern = learning_report['best_patterns'][0]
            print(f"• ابحث عن نمط: {best_pattern['candle_pattern']}")
        
        if opportunities_by_quality['excellent']:
            print(f"• هناك {len(opportunities_by_quality['excellent'])} فرصة ممتازة اليوم!")
        else:
            print("• لا توجد فرص قوية اليوم، الأفضل الانتظار")
    
    print("\n" + "="*60)
    print("✅ اكتمل التحليل اليومي!")
    print("="*60)
    
    # توصيات نهائية
    print("\n📌 التوصيات:")
    if opportunities_by_quality['excellent']:
        print("✅ يوجد فرص ممتازة - يُنصح بالتداول اليوم")
        print("⚠️ لا تدخل أكثر من 2-3 صفقات")
        print("⚠️ التزم بإدارة المخاطر (1% لكل صفقة)")
    elif opportunities_by_quality['very_good']:
        print("⚡ يوجد فرص جيدة - يمكن التداول بحذر")
        print("⚠️ اختر أفضل فرصة واحدة فقط")
    else:
        print("🚫 لا توجد فرص قوية - الأفضل عدم التداول اليوم")


if __name__ == "__main__":
    asyncio.run(main())