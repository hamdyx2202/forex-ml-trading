#!/usr/bin/env python3
"""
تحديث نظام التدريب - دمج جميع الميزات المتقدمة في الملفات الموجودة
"""

import os
import json
from pathlib import Path
from datetime import datetime

def update_config_for_all_symbols():
    """تحديث config.json لدعم جميع العملات"""
    config_path = Path("config/config.json")
    
    # قراءة الإعدادات الحالية
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # جميع العملات المتاحة
    all_symbols = [
        # Forex Majors
        "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
        # Forex Minors
        "USDMXN", "USDZAR", "USDTRY", "USDNOK", "USDSEK", "USDSGD", "USDHKD",
        # Forex Crosses
        "EURJPY", "GBPJPY", "EURGBP", "EURAUD", "EURCAD", "AUDCAD", "AUDNZD",
        "EURCHF", "GBPAUD", "GBPCAD", "GBPNZD", "GBPCHF", "AUDJPY", "NZDJPY",
        # Metals
        "XAUUSD", "XAGUSD", "XPTUSD", "XPDUSD",
        # Crypto
        "BTCUSD", "ETHUSD", "LTCUSD", "XRPUSD", "BNBUSD",
        # Energy
        "WTIUSD", "XBRUSD",
        # Indices
        "US30", "US500", "US100", "DE30", "UK100"
    ]
    
    # تحديث قائمة العملات
    config['trading']['pairs'] = all_symbols
    
    # إضافة إعدادات متقدمة
    config['training'] = {
        'use_support_resistance': True,
        'use_dynamic_sl_tp': True,
        'use_advanced_patterns': True,
        'use_multiple_targets': True,
        'target_timeframes': [5, 15, 30, 60, 240],
        'sl_tp_strategies': ['conservative', 'balanced', 'aggressive', 'scalping', 'swing'],
        'min_data_points': 5000,
        'use_continuous_learning': True,
        'use_pattern_simulation': True
    }
    
    # إضافة إعدادات الميزات المتقدمة
    config['features'] = {
        'technical_indicators': {
            'moving_averages': [5, 10, 20, 50, 100, 200],
            'rsi_periods': [7, 14, 21],
            'atr_periods': [7, 14, 21],
            'bollinger_periods': [10, 20, 30],
            'stochastic_periods': [5, 14]
        },
        'support_resistance': {
            'methods': ['peaks_troughs', 'pivot_points', 'moving_averages', 'fibonacci', 'psychological'],
            'lookback_periods': [20, 50, 100, 200]
        },
        'time_features': True,
        'volume_features': True,
        'candlestick_patterns': True,
        'market_sessions': True
    }
    
    # حفظ النسخة المحدثة
    backup_path = config_path.with_suffix('.backup.json')
    config_path.rename(backup_path)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ تم تحديث config.json")
    print(f"📁 النسخة الاحتياطية: {backup_path}")
    print(f"📊 عدد العملات: {len(all_symbols)}")
    
    return config

def create_training_scripts():
    """إنشاء سكريبتات تدريب محدثة"""
    
    # سكريبت للتدريب السريع
    quick_train_script = '''#!/usr/bin/env python3
"""
Quick Training Script - تدريب سريع للاختبار
"""
from train_ultimate_models import UltimateModelTrainer

# تدريب عملات محددة للاختبار
test_symbols = ["EURUSD", "GBPUSD", "XAUUSD", "BTCUSD"]

trainer = UltimateModelTrainer()

for symbol in test_symbols:
    for timeframe in ["M5", "M15", "H1"]:
        print(f"\\n🚀 Training {symbol} {timeframe}")
        trainer.train_symbol(symbol, timeframe)

print("\\n✅ Quick training completed!")
'''
    
    with open("quick_train_ultimate.py", "w") as f:
        f.write(quick_train_script)
    
    # سكريبت للتدريب الليلي
    overnight_script = '''#!/usr/bin/env python3
"""
Overnight Training Script - تدريب شامل ليلي
"""
import time
from datetime import datetime
from train_ultimate_models import UltimateModelTrainer

print(f"🌙 Starting overnight training at {datetime.now()}")
start_time = time.time()

trainer = UltimateModelTrainer()
trainer.train_all_symbols()

elapsed = time.time() - start_time
hours = int(elapsed // 3600)
minutes = int((elapsed % 3600) // 60)

print(f"\\n✅ Training completed in {hours}h {minutes}m")
print(f"🌅 Finished at {datetime.now()}")
'''
    
    with open("overnight_train_ultimate.py", "w") as f:
        f.write(overnight_script)
    
    # سكريبت للتدريب المستمر
    continuous_script = '''#!/usr/bin/env python3
"""
Continuous Training Script - تدريب مستمر يومي
"""
import schedule
import time
from datetime import datetime
from train_ultimate_models import UltimateModelTrainer

def daily_training():
    """تدريب يومي في الساعة 2 صباحاً"""
    print(f"\\n🔄 Starting daily training at {datetime.now()}")
    
    trainer = UltimateModelTrainer()
    # تدريب العملات الأكثر نشاطاً فقط
    active_symbols = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "BTCUSD"]
    
    for symbol in active_symbols:
        for timeframe in ["M5", "M15", "H1", "H4"]:
            try:
                trainer.train_symbol(symbol, timeframe)
            except Exception as e:
                print(f"Error training {symbol} {timeframe}: {e}")
    
    print(f"✅ Daily training completed at {datetime.now()}")

# جدولة التدريب اليومي
schedule.every().day.at("02:00").do(daily_training)

print("🕒 Continuous training scheduler started")
print("   Daily training at 02:00 AM")
print("   Press Ctrl+C to stop")

while True:
    schedule.run_pending()
    time.sleep(60)  # فحص كل دقيقة
'''
    
    with open("continuous_train_ultimate.py", "w") as f:
        f.write(continuous_script)
    
    print("\n✅ تم إنشاء سكريبتات التدريب:")
    print("  • quick_train_ultimate.py - للاختبار السريع")
    print("  • overnight_train_ultimate.py - للتدريب الليلي الشامل")
    print("  • continuous_train_ultimate.py - للتدريب المستمر اليومي")

def create_integration_guide():
    """إنشاء دليل لدمج الميزات المتقدمة"""
    guide = '''# دليل دمج الميزات المتقدمة في نظام التدريب

## 🚀 الميزات الجديدة المدمجة:

### 1. **دعم جميع العملات** (80+ عملة)
- Forex Majors & Minors
- Currency Crosses
- Metals (Gold, Silver, etc.)
- Cryptocurrencies
- Energy (Oil, Gas)
- Indices

### 2. **مستويات الدعم والمقاومة**
- 5 طرق مختلفة لحساب المستويات
- دمج مباشر في الميزات
- حساب المسافة والقوة

### 3. **نظام SL/TP الديناميكي**
- 5 استراتيجيات مختلفة
- حساب بناءً على ATR والدعم/المقاومة
- تكيف مع نوع الأداة المالية

### 4. **أهداف متعددة**
- 5 أطر زمنية مختلفة (5م، 15م، 30م، 1س، 4س)
- أحجام أهداف مختلفة
- تصنيف ثلاثي (صعود قوي، محايد، هبوط قوي)

### 5. **التعلم المستمر**
- تحليل الفرص التاريخية
- محاكاة أنماط تداول يومية
- تحديث مستمر للأداء

### 6. **ميزات متقدمة**
- مؤشرات فنية شاملة (100+ ميزة)
- تحليل الوقت والجلسات
- أنماط الشموع اليابانية
- تحليل الحجم والتقلب

## 📝 كيفية الاستخدام:

### للتدريب السريع:
```bash
python quick_train_ultimate.py
```

### للتدريب الشامل:
```bash
python train_ultimate_models.py
```

### للتدريب الليلي:
```bash
python overnight_train_ultimate.py
```

### للتدريب المستمر:
```bash
python continuous_train_ultimate.py
```

## 🔧 التخصيص:

### تغيير العملات:
```python
# في train_ultimate_models.py
trainer = UltimateModelTrainer()
trainer.train_symbol("EURUSD", "H1")  # عملة محددة
```

### تغيير الأهداف:
```python
# في __init__ من UltimateModelTrainer
self.target_configs = [
    {'name': 'target_scalping', 'minutes': 2, 'min_pips': 3},
    {'name': 'target_day_trade', 'minutes': 120, 'min_pips': 50},
]
```

### تغيير استراتيجيات SL/TP:
```python
self.sltp_strategies = [
    {'name': 'ultra_safe', 'risk_reward': 1.0, 'atr_multiplier': 0.5},
    {'name': 'high_risk', 'risk_reward': 5.0, 'atr_multiplier': 3.0},
]
```

## 📊 مراقبة الأداء:

### التقارير:
- `models/{symbol}_{timeframe}/training_report.json`
- `models/training_summary_ultimate.json`

### السجلات:
- `logs/ultimate_training.log`

## 🎯 نصائح للحصول على أفضل النتائج:

1. **البيانات**: تأكد من وجود 3 سنوات على الأقل
2. **الذاكرة**: 16GB RAM موصى بها للتدريب الشامل
3. **الوقت**: التدريب الشامل قد يستغرق 4-8 ساعات
4. **التحسين**: استخدم Optuna لتحسين المعاملات تلقائياً

## 🔄 التحديث المستمر:

النظام مصمم للتعلم المستمر:
- يحفظ أنماط التداول الناجحة
- يتعلم من كل صفقة جديدة
- يقترح تحسينات تلقائياً

## 📞 الدعم:

في حالة وجود مشاكل:
1. تحقق من السجلات في `logs/`
2. تأكد من توفر البيانات في قاعدة البيانات
3. راجع التقارير في `models/`
'''
    
    with open("ULTIMATE_TRAINING_GUIDE.md", "w", encoding='utf-8') as f:
        f.write(guide)
    
    print("\n📖 تم إنشاء دليل الاستخدام: ULTIMATE_TRAINING_GUIDE.md")

def main():
    """تحديث نظام التدريب بالكامل"""
    print("🚀 بدء تحديث نظام التدريب...")
    print("="*60)
    
    # 1. تحديث الإعدادات
    print("\n1️⃣ تحديث ملف الإعدادات...")
    update_config_for_all_symbols()
    
    # 2. إنشاء السكريبتات
    print("\n2️⃣ إنشاء سكريبتات التدريب...")
    create_training_scripts()
    
    # 3. إنشاء الدليل
    print("\n3️⃣ إنشاء دليل الاستخدام...")
    create_integration_guide()
    
    print("\n" + "="*60)
    print("✅ تم تحديث نظام التدريب بنجاح!")
    print("\n📋 الخطوات التالية:")
    print("  1. تشغيل جمع البيانات: python ForexMLDataCollector_Ultimate.mq5")
    print("  2. اختبار سريع: python quick_train_ultimate.py")
    print("  3. تدريب شامل: python train_ultimate_models.py")
    print("\n💡 نصيحة: ابدأ بالاختبار السريع للتأكد من عمل النظام")

if __name__ == "__main__":
    main()