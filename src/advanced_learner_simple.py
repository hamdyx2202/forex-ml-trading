#!/usr/bin/env python3
"""
Advanced Pattern Learner - Simple Version without TA-Lib
نظام التعلم المتقدم - نسخة بسيطة بدون مكتبات خارجية
"""

import sqlite3
import json
import os
from datetime import datetime, timedelta
import time
from pathlib import Path

class SimpleAdvancedLearner:
    """متعلم متقدم بسيط للأنماط"""
    
    def __init__(self):
        self.db_path = "data/forex_ml.db"
        self.patterns_file = "models/discovered_patterns.json"
        self.patterns = self.load_patterns()
        
    def load_patterns(self):
        """تحميل الأنماط المكتشفة"""
        if os.path.exists(self.patterns_file):
            with open(self.patterns_file, 'r') as f:
                return json.load(f)
        return {"patterns": [], "last_update": None}
    
    def save_patterns(self):
        """حفظ الأنماط المكتشفة"""
        os.makedirs("models", exist_ok=True)
        with open(self.patterns_file, 'w') as f:
            json.dump(self.patterns, f, indent=2)
    
    def calculate_sma(self, data, period):
        """حساب المتوسط المتحرك البسيط"""
        if len(data) < period:
            return None
        return sum(data[-period:]) / period
    
    def find_support_resistance(self, highs, lows):
        """إيجاد مستويات الدعم والمقاومة"""
        if len(highs) < 20 or len(lows) < 20:
            return None, None
            
        # بسيط: أعلى قمة وأدنى قاع في آخر 20 شمعة
        resistance = max(highs[-20:])
        support = min(lows[-20:])
        
        return support, resistance
    
    def detect_trend(self, closes):
        """كشف الاتجاه"""
        if len(closes) < 50:
            return "UNKNOWN"
            
        sma_10 = self.calculate_sma(closes, 10)
        sma_30 = self.calculate_sma(closes, 30)
        sma_50 = self.calculate_sma(closes, 50)
        
        if not all([sma_10, sma_30, sma_50]):
            return "UNKNOWN"
            
        if sma_10 > sma_30 > sma_50:
            return "STRONG_UP"
        elif sma_10 < sma_30 < sma_50:
            return "STRONG_DOWN"
        elif sma_10 > sma_30:
            return "UP"
        elif sma_10 < sma_30:
            return "DOWN"
        else:
            return "SIDEWAYS"
    
    def find_patterns(self, symbol, timeframe):
        """البحث عن أنماط في البيانات"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # جلب آخر 100 شمعة
        cursor.execute("""
            SELECT time, open, high, low, close, volume
            FROM price_data
            WHERE symbol = ? AND timeframe = ?
            ORDER BY time DESC
            LIMIT 100
        """, (symbol, timeframe))
        
        data = cursor.fetchall()
        conn.close()
        
        if len(data) < 50:
            return []
            
        # عكس الترتيب ليكون من الأقدم للأحدث
        data = data[::-1]
        
        closes = [row[4] for row in data]
        highs = [row[2] for row in data]
        lows = [row[3] for row in data]
        
        patterns_found = []
        
        # 1. كشف الاتجاه
        trend = self.detect_trend(closes)
        if trend != "UNKNOWN":
            patterns_found.append({
                "type": "TREND",
                "name": f"Trend_{trend}",
                "symbol": symbol,
                "timeframe": timeframe,
                "confidence": 0.7,
                "description": f"السوق في اتجاه {trend}"
            })
        
        # 2. الدعم والمقاومة
        support, resistance = self.find_support_resistance(highs, lows)
        if support and resistance:
            current_price = closes[-1]
            
            # قرب من الدعم
            if abs(current_price - support) / current_price < 0.002:
                patterns_found.append({
                    "type": "SUPPORT_BOUNCE",
                    "name": "Near_Support",
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "confidence": 0.65,
                    "description": f"السعر قريب من الدعم عند {support:.5f}"
                })
            
            # قرب من المقاومة
            if abs(resistance - current_price) / current_price < 0.002:
                patterns_found.append({
                    "type": "RESISTANCE_TEST",
                    "name": "Near_Resistance",
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "confidence": 0.65,
                    "description": f"السعر قريب من المقاومة عند {resistance:.5f}"
                })
        
        # 3. نمط الارتداد
        if len(closes) >= 5:
            recent_change = (closes[-1] - closes[-5]) / closes[-5]
            
            if abs(recent_change) > 0.003:  # حركة قوية
                if recent_change > 0 and trend == "DOWN":
                    patterns_found.append({
                        "type": "REVERSAL",
                        "name": "Bullish_Reversal",
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "confidence": 0.6,
                        "description": "احتمال انعكاس صاعد"
                    })
                elif recent_change < 0 and trend == "UP":
                    patterns_found.append({
                        "type": "REVERSAL",
                        "name": "Bearish_Reversal",
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "confidence": 0.6,
                        "description": "احتمال انعكاس هابط"
                    })
        
        return patterns_found
    
    def analyze_all_pairs(self):
        """تحليل جميع الأزواج"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # الحصول على جميع الأزواج والأطر الزمنية
        cursor.execute("""
            SELECT DISTINCT symbol, timeframe
            FROM price_data
            GROUP BY symbol, timeframe
            HAVING COUNT(*) > 50
        """)
        
        pairs = cursor.fetchall()
        conn.close()
        
        all_patterns = []
        
        for symbol, timeframe in pairs:
            print(f"🔍 Analyzing {symbol} {timeframe}...")
            patterns = self.find_patterns(symbol, timeframe)
            all_patterns.extend(patterns)
        
        return all_patterns
    
    def run_continuous_learning(self):
        """التعلم المستمر"""
        print("="*60)
        print("🧠 Simple Advanced Pattern Learner Started")
        print("="*60)
        
        while True:
            try:
                print(f"\n🕐 Starting analysis at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # تحليل جميع الأزواج
                patterns = self.analyze_all_pairs()
                
                # حفظ الأنماط
                self.patterns = {
                    "patterns": patterns,
                    "last_update": datetime.now().isoformat(),
                    "total_patterns": len(patterns)
                }
                self.save_patterns()
                
                print(f"\n✅ Found {len(patterns)} patterns")
                
                # عرض أفضل الأنماط
                if patterns:
                    print("\n📊 Top Patterns:")
                    sorted_patterns = sorted(patterns, key=lambda x: x['confidence'], reverse=True)
                    for i, pattern in enumerate(sorted_patterns[:5]):
                        print(f"  {i+1}. {pattern['symbol']} {pattern['timeframe']}: {pattern['name']} ({pattern['confidence']:.1%})")
                        print(f"     → {pattern['description']}")
                
                # الانتظار 5 دقائق
                print("\n💤 Waiting 5 minutes for next analysis...")
                time.sleep(300)
                
            except KeyboardInterrupt:
                print("\n👋 Shutting down Advanced Learner...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                time.sleep(60)

def main():
    learner = SimpleAdvancedLearner()
    learner.run_continuous_learning()

if __name__ == "__main__":
    main()