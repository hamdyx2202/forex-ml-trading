#!/usr/bin/env python3
"""
📊 تحليل سجلات السيرفر المحسن
🔍 يعرض أسباب رفض الصفقات والأداء
"""

import re
import sys
from datetime import datetime
from collections import defaultdict, Counter

def analyze_logs(log_file='enhanced_ml_server.log'):
    """تحليل سجلات السيرفر"""
    
    print("="*60)
    print("📊 Enhanced ML Server Log Analysis")
    print(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"❌ Log file not found: {log_file}")
        return
    
    # إحصائيات
    requests = 0
    predictions = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
    hold_reasons = []
    market_scores = []
    confidences = []
    symbols = Counter()
    news_events = 0
    high_volatility = 0
    auto_trains = 0
    models_saved = 0
    
    # تحليل السطور
    for i, line in enumerate(lines):
        # طلبات
        if "Request:" in line:
            requests += 1
            # استخراج الرمز
            match = re.search(r"Request: (\w+)", line)
            if match:
                symbols[match.group(1)] += 1
        
        # التنبؤات
        if "action" in line and ("BUY" in line or "SELL" in line or "HOLD" in line):
            if "BUY" in line and "action: 0" in line:
                predictions['BUY'] += 1
            elif "SELL" in line and "action: 1" in line:
                predictions['SELL'] += 1
            elif "HOLD" in line:
                predictions['HOLD'] += 1
                
                # البحث عن سبب HOLD
                for j in range(max(0, i-10), min(len(lines), i+5)):
                    if "confidence" in lines[j].lower():
                        match = re.search(r"confidence[:\s]+(\d+\.?\d*)", lines[j])
                        if match:
                            conf = float(match.group(1))
                            if conf < 0.65:
                                hold_reasons.append(f"Low confidence: {conf:.2f}")
                    if "market.*score" in lines[j].lower():
                        match = re.search(r"Score=(-?\d+)", lines[j])
                        if match:
                            score = int(match.group(1))
                            if abs(score) < 20:
                                hold_reasons.append(f"Weak market score: {score}")
        
        # Market scores
        match = re.search(r"Market Analysis:.*Score=(-?\d+)", line)
        if match:
            market_scores.append(int(match.group(1)))
        
        # Confidence
        match = re.search(r"confidence[:\s]+(\d+\.?\d*)", line)
        if match:
            confidences.append(float(match.group(1)))
        
        # أحداث خاصة
        if "News time" in line:
            news_events += 1
        if "Very high volatility" in line:
            high_volatility += 1
        if "Auto-training" in line:
            auto_trains += 1
        if "models saved" in line:
            models_saved += 1
    
    # عرض النتائج
    print(f"\n📈 Total Requests: {requests}")
    
    if requests > 0:
        print(f"\n🎯 Predictions:")
        total_preds = sum(predictions.values())
        if total_preds > 0:
            print(f"   ✅ BUY:  {predictions['BUY']} ({predictions['BUY']/total_preds*100:.1f}%)")
            print(f"   ❌ SELL: {predictions['SELL']} ({predictions['SELL']/total_preds*100:.1f}%)")
            print(f"   ⏸️  HOLD: {predictions['HOLD']} ({predictions['HOLD']/total_preds*100:.1f}%)")
    
    # أسباب HOLD
    if hold_reasons:
        print(f"\n❓ Top HOLD Reasons:")
        reason_counts = Counter(hold_reasons)
        for reason, count in reason_counts.most_common(5):
            print(f"   - {reason}: {count} times")
    
    # Market scores
    if market_scores:
        avg_score = sum(market_scores) / len(market_scores)
        print(f"\n📊 Market Scores:")
        print(f"   Average: {avg_score:.1f}")
        print(f"   Range: {min(market_scores)} to {max(market_scores)}")
        weak_scores = sum(1 for s in market_scores if abs(s) < 20)
        print(f"   Weak scores (<20): {weak_scores} ({weak_scores/len(market_scores)*100:.1f}%)")
    
    # Confidence levels
    if confidences:
        avg_conf = sum(confidences) / len(confidences)
        print(f"\n🎯 Confidence Levels:")
        print(f"   Average: {avg_conf:.2%}")
        print(f"   Range: {min(confidences):.2%} to {max(confidences):.2%}")
        low_conf = sum(1 for c in confidences if c < 0.65)
        print(f"   Low confidence (<65%): {low_conf} ({low_conf/len(confidences)*100:.1f}%)")
    
    # رموز الأزواج
    if symbols:
        print(f"\n💹 Top Symbols:")
        for symbol, count in symbols.most_common(5):
            print(f"   {symbol}: {count} requests")
    
    # أحداث خاصة
    print(f"\n📰 Special Events:")
    print(f"   News periods: {news_events}")
    print(f"   High volatility: {high_volatility}")
    print(f"   Auto-training: {auto_trains}")
    print(f"   Models saved: {models_saved}")
    
    # آخر 5 تنبؤات
    print(f"\n📍 Last 5 Predictions:")
    pred_count = 0
    for line in reversed(lines):
        if "action" in line and any(x in line for x in ["BUY", "SELL", "HOLD"]):
            # تنظيف السطر
            clean_line = line.strip()
            if len(clean_line) > 100:
                clean_line = clean_line[:100] + "..."
            print(f"   {clean_line}")
            pred_count += 1
            if pred_count >= 5:
                break

def watch_live():
    """مراقبة حية للسجلات"""
    import time
    import os
    
    print("👁️ Live monitoring - updates every 30 seconds")
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            os.system('clear' if os.name == 'posix' else 'cls')
            analyze_logs()
            time.sleep(30)
    except KeyboardInterrupt:
        print("\n\n✋ Monitoring stopped")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--live':
        watch_live()
    else:
        analyze_logs()
        print("\n💡 Use --live for continuous monitoring")