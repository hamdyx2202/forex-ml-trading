#!/usr/bin/env python3
"""
Continuous Learning System - Simple Version
نظام التعلم المستمر - نسخة بسيطة
"""

import sqlite3
import json
import os
from datetime import datetime, timedelta
import time
from pathlib import Path

class SimpleContinuousLearner:
    """نظام التعلم المستمر البسيط"""
    
    def __init__(self):
        self.db_path = "data/forex_ml.db"
        self.learning_file = "models/continuous_learning.json"
        self.performance = self.load_performance()
        self.ensure_tables()
        
    def ensure_tables(self):
        """التأكد من وجود جداول التعلم"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # جدول التداولات
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                signal_time INTEGER NOT NULL,
                signal_type TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                profit_loss REAL,
                status TEXT DEFAULT 'OPEN',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # جدول الأداء
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                total_trades INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                losing_trades INTEGER DEFAULT 0,
                win_rate REAL DEFAULT 0,
                avg_profit REAL DEFAULT 0,
                avg_loss REAL DEFAULT 0,
                last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
        
    def load_performance(self):
        """تحميل بيانات الأداء"""
        if os.path.exists(self.learning_file):
            with open(self.learning_file, 'r') as f:
                return json.load(f)
        return {
            "total_signals": 0,
            "successful_signals": 0,
            "failed_signals": 0,
            "learning_history": []
        }
    
    def save_performance(self):
        """حفظ بيانات الأداء"""
        os.makedirs("models", exist_ok=True)
        with open(self.learning_file, 'w') as f:
            json.dump(self.performance, f, indent=2)
    
    def analyze_recent_trades(self):
        """تحليل التداولات الأخيرة"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # التداولات المغلقة في آخر 24 ساعة
        yesterday = datetime.now() - timedelta(days=1)
        cursor.execute("""
            SELECT symbol, timeframe, signal_type, profit_loss, status
            FROM trades
            WHERE status = 'CLOSED' AND created_at > ?
            ORDER BY created_at DESC
        """, (yesterday.isoformat(),))
        
        trades = cursor.fetchall()
        conn.close()
        
        if not trades:
            return None
            
        # حساب الإحصائيات
        total = len(trades)
        winners = sum(1 for t in trades if t[3] and t[3] > 0)
        losers = sum(1 for t in trades if t[3] and t[3] < 0)
        
        avg_profit = sum(t[3] for t in trades if t[3] and t[3] > 0) / max(winners, 1)
        avg_loss = sum(t[3] for t in trades if t[3] and t[3] < 0) / max(losers, 1)
        
        return {
            "total_trades": total,
            "winners": winners,
            "losers": losers,
            "win_rate": winners / total if total > 0 else 0,
            "avg_profit": avg_profit,
            "avg_loss": avg_loss
        }
    
    def update_model_parameters(self):
        """تحديث معاملات النموذج بناءً على الأداء"""
        # قراءة النموذج الحالي
        model_path = "models/simple_model.json"
        if not os.path.exists(model_path):
            return
            
        with open(model_path, 'r') as f:
            model = json.load(f)
        
        # تحليل الأداء الأخير
        stats = self.analyze_recent_trades()
        if not stats:
            return
            
        # تحديث المعاملات بناءً على الأداء
        if stats['win_rate'] < 0.4:  # أداء ضعيف
            print("⚠️ Low win rate detected. Adjusting parameters...")
            
            # زيادة فترة المتوسطات للحصول على إشارات أكثر موثوقية
            model['fast_period'] = min(model['fast_period'] + 2, 20)
            model['slow_period'] = min(model['slow_period'] + 5, 50)
            
        elif stats['win_rate'] > 0.7:  # أداء ممتاز
            print("🎯 High win rate! Optimizing for more signals...")
            
            # تقليل الفترات قليلاً للحصول على إشارات أكثر
            model['fast_period'] = max(model['fast_period'] - 1, 5)
            model['slow_period'] = max(model['slow_period'] - 2, 20)
        
        # تحديث النموذج
        model['last_update'] = datetime.now().isoformat()
        model['performance'] = {
            "win_rate": stats['win_rate'],
            "total_trades": stats['total_trades']
        }
        
        with open(model_path, 'w') as f:
            json.dump(model, f, indent=2)
            
        print(f"✅ Model updated: Fast={model['fast_period']}, Slow={model['slow_period']}")
    
    def learn_from_patterns(self):
        """التعلم من الأنماط المكتشفة"""
        patterns_file = "models/discovered_patterns.json"
        if not os.path.exists(patterns_file):
            return
            
        with open(patterns_file, 'r') as f:
            patterns_data = json.load(f)
            
        if not patterns_data.get('patterns'):
            return
            
        # تحليل الأنماط الناجحة
        high_confidence_patterns = [
            p for p in patterns_data['patterns'] 
            if p['confidence'] > 0.65
        ]
        
        if high_confidence_patterns:
            print(f"\n📊 Learning from {len(high_confidence_patterns)} high-confidence patterns")
            
            # حفظ الدروس المستفادة
            lesson = {
                "timestamp": datetime.now().isoformat(),
                "patterns_analyzed": len(patterns_data['patterns']),
                "high_confidence": len(high_confidence_patterns),
                "top_pattern": high_confidence_patterns[0]['name'] if high_confidence_patterns else None
            }
            
            self.performance['learning_history'].append(lesson)
            self.save_performance()
    
    def run_continuous_learning(self):
        """تشغيل التعلم المستمر"""
        print("="*60)
        print("🔄 Simple Continuous Learning System Started")
        print("="*60)
        
        iteration = 0
        while True:
            try:
                iteration += 1
                print(f"\n🔄 Learning Iteration #{iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # 1. تحليل التداولات الأخيرة
                print("\n1️⃣ Analyzing recent trades...")
                stats = self.analyze_recent_trades()
                if stats:
                    print(f"   • Total trades: {stats['total_trades']}")
                    print(f"   • Win rate: {stats['win_rate']:.1%}")
                    print(f"   • Winners: {stats['winners']}, Losers: {stats['losers']}")
                else:
                    print("   • No recent trades to analyze")
                
                # 2. تحديث معاملات النموذج
                print("\n2️⃣ Updating model parameters...")
                self.update_model_parameters()
                
                # 3. التعلم من الأنماط
                print("\n3️⃣ Learning from discovered patterns...")
                self.learn_from_patterns()
                
                # 4. حفظ التقدم
                self.performance['total_signals'] = iteration
                self.performance['last_update'] = datetime.now().isoformat()
                self.save_performance()
                
                print("\n✅ Learning cycle completed")
                
                # الانتظار 10 دقائق
                print("\n💤 Waiting 10 minutes for next learning cycle...")
                time.sleep(600)
                
            except KeyboardInterrupt:
                print("\n👋 Shutting down Continuous Learner...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                time.sleep(60)

def main():
    learner = SimpleContinuousLearner()
    learner.run_continuous_learning()

if __name__ == "__main__":
    main()