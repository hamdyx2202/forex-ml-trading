#!/usr/bin/env python3
"""
Start Training - سكريبت البداية السريعة
"""

import os
import sys
from pathlib import Path

def main():
    print("🚀 نظام Forex ML Trading - البداية السريعة")
    print("="*60)
    
    # 1. التحقق من قاعدة البيانات
    print("\n1️⃣ فحص قاعدة البيانات...")
    if not Path("data/forex_data.db").exists():
        print("❌ قاعدة البيانات غير موجودة!")
        print("⏳ جاري إنشاء قاعدة البيانات...")
        os.system("python setup_database.py")
    else:
        print("✅ قاعدة البيانات موجودة")
    
    # 2. التحقق من البيانات
    print("\n2️⃣ فحص البيانات...")
    import sqlite3
    try:
        conn = sqlite3.connect("data/forex_data.db")
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM price_data")
        count = cursor.fetchone()[0]
        conn.close()
        
        if count == 0:
            print("❌ لا توجد بيانات في قاعدة البيانات!")
            print("\n📊 خطوات جمع البيانات:")
            print("  1. افتح MT5 على Windows")
            print("  2. أضف URL في WebRequest:")
            print("     Tools -> Options -> Expert Advisors")
            print("     Allow WebRequest for: http://YOUR_IP:5000")
            print("  3. شغّل إكسبيرت جمع البيانات:")
            print("     - ForexMLDataCollector_Ultimate.mq5 (جميع العملات)")
            print("     - ForexMLDataCollector_Advanced.mq5 (عملات محددة)")
            print("  4. انتظر حتى اكتمال جمع البيانات")
            print("\n⚠️ تأكد من تشغيل السيرفر أولاً:")
            print("  python main_linux.py server")
            return
        else:
            print(f"✅ يوجد {count:,} سجل في قاعدة البيانات")
            
    except Exception as e:
        print(f"❌ خطأ في قراءة البيانات: {e}")
        return
    
    # 3. اختيار نوع التدريب
    print("\n3️⃣ اختر نوع التدريب:")
    print("  1. تدريب سريع (عملة واحدة للاختبار)")
    print("  2. تدريب متوسط (نموذج مبسط)")
    print("  3. تدريب متقدم كامل (هدف 95%+ دقة)")
    print("  4. إلغاء")
    
    choice = input("\n👈 اختيارك (1-4): ").strip()
    
    if choice == "1":
        print("\n🚀 بدء التدريب السريع...")
        print("  عملة واحدة: EURUSD H1")
        os.system("python train_advanced_complete.py --quick")
        
    elif choice == "2":
        print("\n🚀 بدء التدريب المتوسط...")
        os.system("python train_models_simple.py")
        
    elif choice == "3":
        print("\n🚀 بدء التدريب المتقدم الكامل...")
        print("⚠️ تحذير: قد يستغرق عدة ساعات!")
        confirm = input("هل تريد المتابعة؟ (y/n): ").strip().lower()
        if confirm == 'y':
            os.system("python train_advanced_complete.py")
        else:
            print("تم الإلغاء")
            
    else:
        print("تم الإلغاء")
    
    print("\n✅ انتهى!")

if __name__ == "__main__":
    main()