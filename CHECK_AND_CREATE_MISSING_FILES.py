#!/usr/bin/env python3
"""
سكريبت للتحقق من الملفات المفقودة وإنشائها
"""

import os
import sys

# قائمة الملفات المطلوبة
REQUIRED_FILES = {
    "src/mt5_bridge_server.py": "خادم API للربط مع MT5",
    "start_bridge_server.py": "سكريبت تشغيل الخادم", 
    "ForexMLBot.mq5": "Expert Advisor لـ MT5",
    "docs/INSTALLATION_EA.md": "دليل تثبيت EA",
    "REAL_TRADING_GUIDE.md": "دليل التداول الحقيقي",
    "test_bridge_server.py": "سكريبت اختبار الخادم",
    "scripts/mt5-bridge.service": "خدمة systemd",
    "src/linux_adapter.py": "محول Linux للتوافق",
    "LINUX_VPS_UPDATE.md": "دليل تحديث Linux"
}

def check_files():
    """التحقق من وجود الملفات"""
    print("🔍 التحقق من الملفات المطلوبة...")
    print("=" * 50)
    
    missing_files = []
    existing_files = []
    
    for file_path, description in REQUIRED_FILES.items():
        if os.path.exists(file_path):
            existing_files.append(file_path)
            print(f"✅ موجود: {file_path}")
        else:
            missing_files.append(file_path)
            print(f"❌ مفقود: {file_path} - {description}")
    
    print("\n" + "=" * 50)
    print(f"📊 النتيجة: {len(existing_files)}/{len(REQUIRED_FILES)} ملفات موجودة")
    
    if missing_files:
        print(f"\n⚠️  الملفات المفقودة ({len(missing_files)}):")
        for f in missing_files:
            print(f"   - {f}")
        
        print("\n💡 الحل:")
        print("1. قم بإنشاء هذه الملفات من المحتوى المتوفر")
        print("2. أو قم بتنزيلها من الـ repository إذا كانت موجودة هناك")
        print("3. تأكد من أنك في المجلد الصحيح (forex-ml-trading)")
        
        print("\n📝 أوامر Git المطلوبة:")
        print("```bash")
        print("# للتحقق من الملفات غير المتتبعة")
        print("git status --porcelain")
        print("\n# لإضافة الملفات المفقودة")
        for f in missing_files:
            print(f"git add {f}")
        print("\n# لعمل commit")
        print('git commit -m "Add missing files for real trading"')
        print("\n# للرفع")
        print("git push origin main")
        print("```")
    else:
        print("\n✅ جميع الملفات موجودة!")
        print("\n📝 الخطوة التالية:")
        print("```bash")
        print("git add .")
        print('git commit -m "Add all real trading components"')
        print("git push origin main")
        print("```")
    
    # التحقق من requirements.txt
    print("\n🔍 التحقق من Flask في requirements.txt...")
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r") as f:
            content = f.read()
            if "flask" in content.lower():
                print("✅ Flask موجود في requirements.txt")
            else:
                print("⚠️  Flask غير موجود في requirements.txt")
                print("   أضف هذه الأسطر:")
                print("   # API Server")
                print("   flask==3.0.0")
                print("   flask-cors==4.0.0")
    
    # التحقق من البنية المكررة
    print("\n🔍 التحقق من البنية المكررة...")
    if os.path.exists("forex-ml-trading/forex-ml-trading"):
        print("⚠️  تحذير: يوجد مجلد forex-ml-trading مكرر!")
        print("   هذا قد يسبب مشاكل في Git")
        print("   تأكد من أنك تعمل في المجلد الصحيح")

if __name__ == "__main__":
    print("🚀 فحص مشروع Forex ML Trading")
    print(f"📁 المجلد الحالي: {os.getcwd()}")
    
    # التحقق من أننا في المجلد الصحيح
    if not os.path.exists("src") or not os.path.exists("config"):
        print("\n❌ خطأ: يبدو أنك لست في مجلد المشروع الصحيح!")
        print("   انتقل إلى: C:\\Users\\ACA-w10\\Desktop\\learn\\forex-ml-trading")
        sys.exit(1)
    
    check_files()