#!/usr/bin/env python3
"""
📊 فحص تقدم التدريب
🔍 يعرض حالة التدريب والنماذج المكتملة
"""

import os
import time
import sqlite3
from datetime import datetime
import glob

def check_progress():
    """فحص تقدم التدريب"""
    print("="*60)
    print("📊 Training Progress Check")
    print(f"🕐 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # 1. فحص النماذج المدربة
    models_dir = './trained_models'
    if os.path.exists(models_dir):
        all_models = glob.glob(f"{models_dir}/*.pkl")
        model_files = [f for f in all_models if 'scaler' not in f]
        
        print(f"\n🤖 Trained Models: {len(model_files)}")
        
        # تجميع حسب الزوج
        pairs = {}
        for model in model_files:
            filename = os.path.basename(model)
            parts = filename.split('_')
            if len(parts) >= 2:
                pair = parts[0]
                if pair not in pairs:
                    pairs[pair] = []
                pairs[pair].append(filename)
        
        print(f"📈 Pairs with models: {len(pairs)}")
        
        # عرض آخر 5 أزواج
        print("\n🏆 Latest trained pairs:")
        sorted_models = sorted(model_files, key=lambda x: os.path.getmtime(x), reverse=True)
        shown_pairs = set()
        count = 0
        
        for model in sorted_models:
            filename = os.path.basename(model)
            pair = filename.split('_')[0]
            if pair not in shown_pairs:
                mod_time = datetime.fromtimestamp(os.path.getmtime(model))
                time_ago = datetime.now() - mod_time
                mins = int(time_ago.total_seconds() / 60)
                
                if mins < 60:
                    time_str = f"{mins} minutes ago"
                else:
                    time_str = f"{mins//60} hours ago"
                
                print(f"   ✅ {pair} - {time_str}")
                shown_pairs.add(pair)
                count += 1
                if count >= 5:
                    break
    else:
        print("❌ No models directory found")
    
    # 2. فحص السجلات
    log_file = 'enhanced_ml_server.log'
    if os.path.exists(log_file):
        print(f"\n📝 Log Analysis:")
        
        # قراءة آخر 100 سطر
        with open(log_file, 'r') as f:
            lines = f.readlines()[-100:]
        
        # عد الأحداث
        training_count = sum(1 for line in lines if 'Training enhanced models' in line)
        success_count = sum(1 for line in lines if 'Successfully trained' in line or 'models saved' in line)
        error_count = sum(1 for line in lines if 'Error' in line or 'Failed' in line)
        
        print(f"   📊 Training attempts: {training_count}")
        print(f"   ✅ Successful: {success_count}")
        print(f"   ❌ Errors: {error_count}")
        
        # آخر نشاط
        if lines:
            last_line = lines[-1].strip()
            if len(last_line) > 80:
                last_line = last_line[:80] + "..."
            print(f"\n📍 Last activity: {last_line}")
    
    # 3. تقدير الوقت المتبقي
    if 'model_files' in locals() and len(model_files) > 0:
        # افتراض 60 نموذج إجمالي (20 زوج × 3 أطر)
        total_expected = 60
        completed = len(model_files)
        remaining = total_expected - completed
        
        if completed > 0:
            # حساب متوسط الوقت لكل نموذج
            oldest_model = min(model_files, key=lambda x: os.path.getmtime(x))
            time_elapsed = time.time() - os.path.getmtime(oldest_model)
            avg_time_per_model = time_elapsed / completed
            estimated_remaining = (remaining * avg_time_per_model) / 60
            
            print(f"\n⏰ Time Estimation:")
            print(f"   Progress: {completed}/{total_expected} ({completed/total_expected*100:.1f}%)")
            print(f"   Estimated remaining: {estimated_remaining:.0f} minutes")
    
    print("\n" + "="*60)

def watch_progress(interval=30):
    """مراقبة مستمرة للتقدم"""
    print("👁️ Watching training progress...")
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            os.system('clear' if os.name == 'posix' else 'cls')
            check_progress()
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\n✋ Monitoring stopped")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--watch':
        # مراقبة مستمرة
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        watch_progress(interval)
    else:
        # فحص مرة واحدة
        check_progress()
        print("\n💡 Tip: Use --watch for continuous monitoring")
        print("   Example: python3 check_training_progress.py --watch 10")