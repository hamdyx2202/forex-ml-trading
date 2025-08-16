#!/usr/bin/env python3
"""
Full Simple Training - التدريب الشامل البسيط
يدرب جميع العملات المتاحة بالنموذج البسيط
"""

import time
from datetime import datetime
from train_models_simple import SimpleModelTrainer
import sqlite3
import pandas as pd
from pathlib import Path

class FullSimpleTrainer:
    def __init__(self):
        self.trainer = SimpleModelTrainer()
        self.results = []
        
    def get_all_available_pairs(self):
        """الحصول على جميع الأزواج المتاحة"""
        try:
            conn = sqlite3.connect("data/forex_ml.db")
            query = """
                SELECT symbol, timeframe, COUNT(*) as count
                FROM price_data
                GROUP BY symbol, timeframe
                HAVING count >= 1000
                ORDER BY symbol, timeframe
            """
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            return [(row['symbol'], row['timeframe'], row['count']) 
                   for _, row in df.iterrows()]
            
        except Exception as e:
            print(f"❌ خطأ في قراءة البيانات: {e}")
            return []
    
    def train_all(self):
        """تدريب جميع الأزواج"""
        print("🚀 التدريب الشامل البسيط")
        print("="*80)
        
        # الحصول على جميع الأزواج
        pairs = self.get_all_available_pairs()
        
        if not pairs:
            print("❌ لا توجد بيانات متاحة!")
            return
        
        print(f"📊 عدد الأزواج المتاحة: {len(pairs)}")
        print(f"⏱️  الوقت المقدر: {len(pairs) * 0.5:.0f}-{len(pairs) * 1.5:.0f} دقيقة")
        
        # بدء التدريب
        start_time = time.time()
        successful = 0
        failed = 0
        
        for idx, (symbol, timeframe, count) in enumerate(pairs, 1):
            print(f"\n{'='*60}")
            print(f"📈 [{idx}/{len(pairs)}] {symbol} {timeframe} ({count:,} سجل)")
            
            try:
                symbol_start = time.time()
                scores = self.trainer.train_symbol(symbol, timeframe)
                symbol_time = time.time() - symbol_start
                
                if scores:
                    successful += 1
                    self.results.append({
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'accuracy': scores['test_accuracy'],
                        'precision': scores['precision'],
                        'recall': scores['recall'],
                        'f1': scores['f1'],
                        'time': symbol_time
                    })
                    print(f"✅ نجح في {symbol_time:.1f} ثانية")
                else:
                    failed += 1
                    
            except Exception as e:
                failed += 1
                print(f"❌ خطأ: {e}")
            
            # تقدير الوقت المتبقي
            elapsed = time.time() - start_time
            avg_time = elapsed / idx
            remaining = avg_time * (len(pairs) - idx)
            
            print(f"⏱️  الوقت المتبقي المقدر: {remaining/60:.1f} دقيقة")
        
        # حفظ النتائج
        self.save_results()
        
        # طباعة الملخص
        self.print_summary(successful, failed, time.time() - start_time)
    
    def save_results(self):
        """حفظ نتائج التدريب"""
        if not self.results:
            return
        
        # حفظ كـ CSV
        df = pd.DataFrame(self.results)
        df = df.sort_values('accuracy', ascending=False)
        
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = output_dir / f"simple_training_results_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        
        print(f"\n💾 تم حفظ النتائج في: {csv_path}")
    
    def print_summary(self, successful, failed, total_time):
        """طباعة الملخص النهائي"""
        print("\n" + "="*80)
        print("📊 الملخص النهائي - التدريب البسيط")
        print("="*80)
        
        print(f"\n📈 الإحصائيات:")
        print(f"   • نجح: {successful}")
        print(f"   • فشل: {failed}")
        print(f"   • الإجمالي: {successful + failed}")
        print(f"   • معدل النجاح: {successful/(successful+failed)*100:.1f}%")
        
        if self.results:
            df = pd.DataFrame(self.results)
            
            print(f"\n🏆 أفضل 10 نماذج:")
            top_10 = df.nlargest(10, 'accuracy')
            for idx, row in top_10.iterrows():
                print(f"   {row['symbol']:<15} {row['timeframe']:<5} - "
                      f"الدقة: {row['accuracy']:.4f}, F1: {row['f1']:.4f}")
            
            print(f"\n📊 متوسط الأداء:")
            print(f"   • متوسط الدقة: {df['accuracy'].mean():.4f}")
            print(f"   • أعلى دقة: {df['accuracy'].max():.4f}")
            print(f"   • أقل دقة: {df['accuracy'].min():.4f}")
        
        print(f"\n⏱️  الوقت الإجمالي: {total_time/60:.1f} دقيقة")
        print(f"   متوسط الوقت لكل نموذج: {total_time/max(successful+failed,1):.1f} ثانية")

def main():
    trainer = FullSimpleTrainer()
    trainer.train_all()

if __name__ == "__main__":
    main()