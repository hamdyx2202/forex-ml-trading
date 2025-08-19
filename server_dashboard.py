#!/usr/bin/env python3
"""
📊 لوحة تحكم السيرفر المحسن
🎯 عرض مباشر للأداء وأسباب القرارات
"""

import os
import json
import sqlite3
from datetime import datetime, timedelta
import pandas as pd

def get_server_status():
    """فحص حالة السيرفر"""
    # فحص العملية
    result = os.popen("ps aux | grep 'enhanced_ml_server' | grep -v grep").read()
    if 'gunicorn' in result or 'python' in result:
        return "🟢 Running"
    return "🔴 Stopped"

def analyze_recent_activity():
    """تحليل النشاط الأخير"""
    log_file = 'enhanced_ml_server.log'
    
    if not os.path.exists(log_file):
        return None
    
    # قراءة آخر 500 سطر
    with open(log_file, 'r') as f:
        lines = f.readlines()[-500:]
    
    # تحليل
    stats = {
        'last_request': None,
        'total_requests': 0,
        'predictions': {'BUY': 0, 'SELL': 0, 'HOLD': 0},
        'avg_confidence': 0,
        'hold_reasons': [],
        'models_loaded': 0,
        'auto_trains': 0
    }
    
    confidences = []
    
    for i, line in enumerate(lines):
        # آخر طلب
        if "Request:" in line:
            stats['total_requests'] += 1
            stats['last_request'] = line.strip()
        
        # التنبؤات
        if '"action":' in line:
            if '"action": 0' in line:
                stats['predictions']['BUY'] += 1
            elif '"action": 1' in line:
                stats['predictions']['SELL'] += 1
            elif '"action": 2' in line:
                stats['predictions']['HOLD'] += 1
                
                # البحث عن السبب
                for j in range(max(0, i-5), i):
                    if "confidence" in lines[j]:
                        import re
                        match = re.search(r'confidence["\s:]+(\d+\.?\d*)', lines[j])
                        if match:
                            conf = float(match.group(1))
                            if conf < 0.65:
                                stats['hold_reasons'].append(f"Low confidence: {conf:.2%}")
                    if "Score=" in lines[j]:
                        match = re.search(r'Score=(-?\d+)', lines[j])
                        if match:
                            score = int(match.group(1))
                            if abs(score) < 20:
                                stats['hold_reasons'].append(f"Weak market: {score}")
        
        # الثقة
        if "confidence" in line:
            import re
            match = re.search(r'confidence["\s:]+(\d+\.?\d*)', line)
            if match:
                confidences.append(float(match.group(1)))
        
        # النماذج
        if "ML Models:" in line and "loaded" in line:
            match = re.search(r'(\d+) loaded', line)
            if match:
                stats['models_loaded'] = int(match.group(1))
        
        # التدريب التلقائي
        if "Auto-training" in line:
            stats['auto_trains'] += 1
    
    if confidences:
        stats['avg_confidence'] = sum(confidences) / len(confidences)
    
    return stats

def main():
    """عرض لوحة التحكم"""
    os.system('clear' if os.name == 'posix' else 'cls')
    
    print("="*60)
    print("📊 Enhanced ML Server Dashboard")
    print(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # حالة السيرفر
    status = get_server_status()
    print(f"\n🖥️ Server Status: {status}")
    
    # تحليل النشاط
    stats = analyze_recent_activity()
    
    if stats:
        print(f"\n📈 Recent Activity (last 500 log lines):")
        print(f"   Total requests: {stats['total_requests']}")
        
        # التنبؤات
        total_preds = sum(stats['predictions'].values())
        if total_preds > 0:
            print(f"\n🎯 Predictions Distribution:")
            print(f"   ✅ BUY:  {stats['predictions']['BUY']} ({stats['predictions']['BUY']/total_preds*100:.1f}%)")
            print(f"   ❌ SELL: {stats['predictions']['SELL']} ({stats['predictions']['SELL']/total_preds*100:.1f}%)")
            print(f"   ⏸️  HOLD: {stats['predictions']['HOLD']} ({stats['predictions']['HOLD']/total_preds*100:.1f}%)")
            
            # نسبة الصفقات المفتوحة
            trade_rate = (stats['predictions']['BUY'] + stats['predictions']['SELL']) / total_preds * 100
            print(f"\n   📊 Trade Rate: {trade_rate:.1f}%")
            
            if trade_rate < 30:
                print("   ⚠️ Low trade rate - check market conditions")
        
        # متوسط الثقة
        if stats['avg_confidence'] > 0:
            print(f"\n🎯 Average Confidence: {stats['avg_confidence']:.1%}")
            if stats['avg_confidence'] < 0.65:
                print("   ⚠️ Low average confidence")
        
        # أسباب HOLD
        if stats['hold_reasons']:
            print(f"\n❓ HOLD Reasons:")
            from collections import Counter
            reason_counts = Counter(stats['hold_reasons'])
            for reason, count in reason_counts.most_common(3):
                print(f"   - {reason}: {count}x")
        
        # النماذج
        print(f"\n🤖 Models:")
        print(f"   Loaded: {stats['models_loaded']}")
        print(f"   Auto-trains: {stats['auto_trains']}")
        
        # آخر طلب
        if stats['last_request']:
            print(f"\n📍 Last Request:")
            print(f"   {stats['last_request']}")
    
    # فحص النماذج المحفوظة
    if os.path.exists('./trained_models'):
        models = [f for f in os.listdir('./trained_models') if f.endswith('.pkl') and 'scaler' not in f]
        enhanced = [m for m in models if '_enhanced.pkl' in m]
        auto = [m for m in models if '_auto.pkl' in m]
        
        print(f"\n💾 Saved Models:")
        print(f"   Enhanced (strong): {len(enhanced)}")
        print(f"   Auto (weak): {len(auto)}")
        print(f"   Total: {len(models)}")
    
    # نصائح
    print(f"\n💡 Commands:")
    print("   ./monitor_server.sh           - Live log monitoring")
    print("   python3 analyze_server_logs.py - Detailed analysis")
    print("   tail -f enhanced_ml_server.log - Raw logs")
    print("   tail -f server.log            - Gunicorn logs")

if __name__ == "__main__":
    main()