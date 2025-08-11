#!/usr/bin/env python3
"""
فحص صحة النظام والتأكد من أنه يعمل ويتعلم
"""

import os
import sys
import sqlite3
from datetime import datetime
import json
import requests

print("=" * 70)
print("🏥 فحص صحة نظام Forex ML Trading")
print("=" * 70)

# 1. فحص الملفات الأساسية
print("\n1️⃣ فحص الملفات الأساسية:")
essential_files = [
    "config/config.json",
    "src/data_collector.py",
    "src/feature_engineer.py",
    "src/model_trainer.py",
    "src/predictor.py",
    "src/trader.py",
    "src/risk_manager.py",
    "src/advanced_learner.py",
    "src/continuous_learner.py",
    "src/mt5_bridge_server_linux.py",
    "ForexMLBot.mq5"
]

all_files_exist = True
for file in essential_files:
    if os.path.exists(file):
        print(f"  ✅ {file}")
    else:
        print(f"  ❌ {file} - مفقود!")
        all_files_exist = False

# 2. فحص قاعدة البيانات
print("\n2️⃣ فحص قاعدة البيانات:")
db_path = "data/forex_ml.db"
if os.path.exists(db_path):
    print(f"  ✅ قاعدة البيانات موجودة")
    
    # فحص الجداول
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # فحص جدول البيانات التاريخية
        cursor.execute("SELECT COUNT(*) FROM price_data")
        count = cursor.fetchone()[0]
        print(f"  📊 البيانات التاريخية: {count} سجل")
        
        # فحص جدول التعلم المستمر
        try:
            cursor.execute("SELECT COUNT(*) FROM continuous_learning")
            learning_count = cursor.fetchone()[0]
            print(f"  🧠 سجلات التعلم: {learning_count} سجل")
        except:
            print("  ⚠️  جدول التعلم المستمر غير موجود")
        
        # فحص جدول أنماط التداول
        try:
            cursor.execute("SELECT COUNT(*) FROM learned_patterns")
            patterns_count = cursor.fetchone()[0]
            print(f"  📈 الأنماط المتعلمة: {patterns_count} نمط")
        except:
            print("  ⚠️  جدول الأنماط غير موجود")
        
        conn.close()
    except Exception as e:
        print(f"  ❌ خطأ في قاعدة البيانات: {e}")
else:
    print(f"  ❌ قاعدة البيانات غير موجودة!")

# 3. فحص النماذج المدربة
print("\n3️⃣ فحص النماذج المدربة:")
models_dir = "models"
if os.path.exists(models_dir):
    models = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    if models:
        print(f"  ✅ عدد النماذج: {len(models)}")
        for model in models[:5]:  # أول 5 نماذج فقط
            print(f"     📦 {model}")
    else:
        print("  ⚠️  لا توجد نماذج مدربة")
else:
    print("  ❌ مجلد النماذج غير موجود!")

# 4. فحص السجلات
print("\n4️⃣ فحص السجلات:")
logs_dir = "logs"
if os.path.exists(logs_dir):
    logs = os.listdir(logs_dir)
    if logs:
        print(f"  ✅ عدد ملفات السجل: {len(logs)}")
        
        # فحص آخر سجل
        for log in ["advanced_learning.log", "continuous_learning.log", "mt5_bridge_linux.log"]:
            log_path = os.path.join(logs_dir, log)
            if os.path.exists(log_path):
                size = os.path.getsize(log_path) / 1024  # KB
                mtime = datetime.fromtimestamp(os.path.getmtime(log_path))
                print(f"     📄 {log} - {size:.1f} KB - آخر تحديث: {mtime.strftime('%Y-%m-%d %H:%M')}")
    else:
        print("  ⚠️  لا توجد سجلات")
else:
    print("  ❌ مجلد السجلات غير موجود!")

# 5. فحص الخادم (إذا كان يعمل محلياً)
print("\n5️⃣ فحص خادم API:")
try:
    response = requests.get("http://localhost:5000/health", timeout=2)
    if response.status_code == 200:
        data = response.json()
        print(f"  ✅ الخادم يعمل - {data.get('mode', 'Unknown')}")
    else:
        print(f"  ⚠️  الخادم يستجيب بكود: {response.status_code}")
except:
    print("  ❌ الخادم لا يعمل محلياً")

# 6. فحص إعدادات التعلم
print("\n6️⃣ فحص إعدادات التعلم:")
try:
    with open("config/config.json", 'r') as f:
        config = json.load(f)
    
    print("  📚 إعدادات التعلم:")
    print(f"     • حد المخاطرة: {config['risk']['max_risk_per_trade']*100}%")
    print(f"     • حد الخسارة اليومي: {config['risk']['max_daily_loss']*100}%")
    print(f"     • عدد الأزواج: {len(config['trading']['pairs'])}")
    print(f"     • الإطارات الزمنية: {', '.join(config['trading']['timeframes'])}")
except Exception as e:
    print(f"  ❌ خطأ في قراءة الإعدادات: {e}")

# 7. التوصيات
print("\n7️⃣ التوصيات:")
if all_files_exist:
    print("  ✅ جميع الملفات الأساسية موجودة")
else:
    print("  ❌ بعض الملفات مفقودة - تحتاج لإصلاح")

if os.path.exists(db_path):
    print("  ✅ قاعدة البيانات جاهزة")
    print("  💡 نصيحة: شغّل `python main.py collect` لجمع المزيد من البيانات")
else:
    print("  ❌ قاعدة البيانات غير موجودة")
    print("  💡 نصيحة: شغّل `python main.py test` لإنشائها")

if os.path.exists(models_dir) and len(os.listdir(models_dir)) > 0:
    print("  ✅ النماذج جاهزة للتداول")
else:
    print("  ⚠️  تحتاج لتدريب النماذج")
    print("  💡 نصيحة: شغّل `python train_models.py`")

print("\n📝 خطوات التشغيل الموصى بها:")
print("1. python main.py collect      # جمع البيانات")
print("2. python learn_from_history.py # التعلم من التاريخ")
print("3. python train_models.py      # تدريب النماذج")
print("4. python start_bridge_server.py # تشغيل الخادم")
print("5. تشغيل EA على MT5")

print("\n=" * 70)
print("✅ اكتمل فحص النظام!")
print("=" * 70)