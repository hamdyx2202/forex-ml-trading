#!/usr/bin/env python3
"""
سكريبت التحسين التلقائي - يُشغل يومياً
يحلل الأداء ويحسن النظام تلقائياً
"""

from src.continuous_learner import ContinuousLearner
from src.advanced_learner import AdvancedLearner
from datetime import datetime
import json
from loguru import logger
import schedule
import time


def daily_improvement():
    """دورة التحسين اليومية"""
    logger.add("logs/auto_improvement.log", rotation="1 day", retention="30 days")
    
    print("\n" + "="*60)
    print(f"🔧 التحسين التلقائي - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*60)
    
    # إنشاء المتعلم المستمر
    continuous_learner = ContinuousLearner()
    advanced_learner = AdvancedLearner()
    
    # 1. تشغيل دورة التحسين المستمر
    print("\n1️⃣ تحليل الأداء وتحديد نقاط التحسين...")
    continuous_learner.continuous_improvement_cycle()
    
    # 2. الحصول على رؤى التعلم
    print("\n2️⃣ استخراج رؤى التعلم...")
    insights = continuous_learner.get_learning_insights()
    
    if insights['general_stats'].get('total_trades_analyzed', 0) > 0:
        print(f"\n📊 إحصائيات التعلم:")
        print(f"  • صفقات محللة: {insights['general_stats']['total_trades_analyzed']}")
        print(f"  • صفقات ناجحة: {insights['general_stats']['successful_trades']}")
        print(f"  • متوسط النقاط: {insights['general_stats']['avg_pips']:.1f}")
        
        if insights['best_patterns']:
            print(f"\n✅ أفضل الأنماط:")
            for pattern in insights['best_patterns'][:3]:
                print(f"  • {pattern['pattern_key']}: {pattern['success_rate']:.1%} نجاح")
        
        if insights['worst_patterns']:
            print(f"\n❌ أسوأ الأنماط (سيتم تجنبها):")
            for pattern in insights['worst_patterns'][:3]:
                print(f"  • {pattern['pattern_key']}: {pattern['success_rate']:.1%} نجاح")
        
        if insights['improvements']:
            print(f"\n💡 اقتراحات التحسين:")
            for improvement in insights['improvements'][-5:]:
                print(f"  • {improvement}")
    
    # 3. إعادة تحليل البيانات التاريخية مع المعرفة الجديدة
    print("\n3️⃣ إعادة تحليل البيانات بالمعرفة الجديدة...")
    
    with open("config/config.json", 'r') as f:
        config = json.load(f)
    
    # تحليل سريع لآخر 30 يوم فقط
    for symbol in config['trading']['pairs'][:2]:  # أول زوجين فقط
        for timeframe in ['H1']:  # إطار واحد فقط للسرعة
            try:
                advanced_learner.analyze_historical_opportunities(symbol, timeframe, lookback_days=30)
                print(f"  ✅ {symbol} {timeframe} - تم التحليل")
            except Exception as e:
                print(f"  ❌ {symbol} {timeframe} - خطأ: {str(e)}")
    
    # 4. تحديث معايير الجودة
    print("\n4️⃣ تحديث معايير جودة الفرص...")
    update_quality_criteria(insights)
    
    # 5. تقرير التحسين
    print("\n" + "="*60)
    print("📈 ملخص التحسينات المطبقة")
    print("="*60)
    
    improvements_applied = []
    
    # الأنماط المحظورة
    try:
        with open('data/blacklisted_patterns.json', 'r') as f:
            blacklisted = json.load(f)
            if blacklisted:
                improvements_applied.append(f"حظر {len(blacklisted)} نمط فاشل")
    except:
        pass
    
    # معايير الجودة المحدثة
    try:
        with open('data/quality_criteria.json', 'r') as f:
            criteria = json.load(f)
            if criteria.get('min_confidence', 0) > 0.7:
                improvements_applied.append(f"رفع الحد الأدنى للثقة إلى {criteria['min_confidence']:.0%}")
    except:
        pass
    
    if improvements_applied:
        print("\n✅ التحسينات المطبقة:")
        for imp in improvements_applied:
            print(f"  • {imp}")
    else:
        print("\n💭 لا توجد تحسينات مطلوبة حالياً - الأداء جيد!")
    
    print("\n✅ اكتمل التحسين التلقائي!")
    print("🚀 النظام أصبح أذكى وأكثر دقة!")


def update_quality_criteria(insights: dict):
    """تحديث معايير جودة الفرص بناءً على التعلم"""
    criteria = {
        'min_confidence': 0.7,
        'min_score': 6,
        'blacklisted_patterns': [],
        'preferred_hours': [],
        'updated_at': datetime.now().isoformat()
    }
    
    # رفع معايير الثقة إذا كان الأداء ضعيف
    if insights['general_stats'].get('successful_trades', 0) > 0:
        success_rate = insights['general_stats']['successful_trades'] / insights['general_stats']['total_trades_analyzed']
        if success_rate < 0.5:
            criteria['min_confidence'] = 0.75
            criteria['min_score'] = 7
        elif success_rate < 0.6:
            criteria['min_confidence'] = 0.72
    
    # إضافة الأنماط السيئة للقائمة السوداء
    if insights['worst_patterns']:
        for pattern in insights['worst_patterns']:
            if pattern['success_rate'] < 0.4:
                criteria['blacklisted_patterns'].append(pattern['pattern_key'])
    
    # حفظ المعايير المحدثة
    with open('data/quality_criteria.json', 'w') as f:
        json.dump(criteria, f, indent=2)


def continuous_improvement_loop():
    """حلقة التحسين المستمر"""
    print("🔄 بدء حلقة التحسين المستمر...")
    print("سيتم التحسين يومياً في الساعة 02:00")
    
    # جدولة المهام
    schedule.every().day.at("02:00").do(daily_improvement)
    
    # تشغيل مرة أولى
    daily_improvement()
    
    # الحلقة المستمرة
    while True:
        schedule.run_pending()
        time.sleep(3600)  # فحص كل ساعة


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        # تشغيل مرة واحدة فقط
        daily_improvement()
    else:
        # تشغيل مستمر
        continuous_improvement_loop()