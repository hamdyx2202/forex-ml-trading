#!/usr/bin/env python3
"""
Full Advanced Training - التدريب الشامل المتقدم المحسن
نظام كامل بجميع الميزات مع معالجة المقاطعة وحفظ التقدم
"""

import time
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
from train_advanced_complete import AdvancedCompleteTrainer
import sqlite3
import pandas as pd
from pathlib import Path
import json
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
import multiprocessing
import signal
import sys
import pickle
import os

class FullAdvancedTrainer:
    def __init__(self):
        self.trainer = AdvancedCompleteTrainer()
        self.results = []
        self.failed_pairs = []
        self.progress_file = Path("results/advanced/training_progress.pkl")
        self.completed_pairs = set()
        self._interrupted = False
        
        # تحميل التقدم السابق إن وجد
        self.load_progress()
        
    def load_progress(self):
        """تحميل التقدم المحفوظ من جلسة سابقة"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'rb') as f:
                    progress_data = pickle.load(f)
                    self.results = progress_data.get('results', [])
                    self.failed_pairs = progress_data.get('failed_pairs', [])
                    self.completed_pairs = progress_data.get('completed_pairs', set())
                    
                print(f"✅ تم تحميل التقدم السابق:")
                print(f"   • نماذج مكتملة: {len(self.completed_pairs)}")
                print(f"   • نتائج محفوظة: {len(self.results)}")
                print(f"   • أزواج فاشلة: {len(self.failed_pairs)}")
                
                # السؤال عن الاستمرار
                choice = input("\nهل تريد الاستمرار من حيث توقفت؟ (y/n): ").strip().lower()
                if choice != 'y':
                    self.reset_progress()
                    
            except Exception as e:
                print(f"⚠️ تعذر تحميل التقدم السابق: {e}")
                self.reset_progress()
    
    def reset_progress(self):
        """إعادة تعيين التقدم"""
        self.results = []
        self.failed_pairs = []
        self.completed_pairs = set()
        if self.progress_file.exists():
            self.progress_file.unlink()
        print("🔄 تم إعادة تعيين التقدم")
    
    def save_progress(self):
        """حفظ التقدم الحالي"""
        try:
            self.progress_file.parent.mkdir(parents=True, exist_ok=True)
            
            progress_data = {
                'results': self.results,
                'failed_pairs': self.failed_pairs,
                'completed_pairs': self.completed_pairs,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.progress_file, 'wb') as f:
                pickle.dump(progress_data, f)
                
            print(f"\n💾 تم حفظ التقدم ({len(self.completed_pairs)} نموذج مكتمل)")
            
        except Exception as e:
            print(f"⚠️ تعذر حفظ التقدم: {e}")
        
    def get_all_available_pairs(self):
        """الحصول على جميع الأزواج المتاحة مع بيانات كافية"""
        try:
            conn = sqlite3.connect("data/forex_ml.db")
            query = """
                SELECT symbol, timeframe, COUNT(*) as count
                FROM price_data
                GROUP BY symbol, timeframe
                HAVING count >= 10000
                ORDER BY count DESC
            """
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            # تصنيف العملات
            categories = {
                'majors': [],
                'crosses': [],
                'metals': [],
                'crypto': [],
                'energy': [],
                'indices': []
            }
            
            for _, row in df.iterrows():
                symbol = row['symbol']
                
                # تخطي الأزواج المكتملة
                pair_key = f"{symbol}_{row['timeframe']}"
                if pair_key in self.completed_pairs:
                    continue
                
                # تصنيف العملات
                if any(major in symbol for major in ['EUR', 'GBP', 'USD', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']):
                    if 'USD' in symbol:
                        categories['majors'].append((row['symbol'], row['timeframe'], row['count']))
                    else:
                        categories['crosses'].append((row['symbol'], row['timeframe'], row['count']))
                elif any(metal in symbol for metal in ['XAU', 'XAG', 'GOLD', 'SILVER']):
                    categories['metals'].append((row['symbol'], row['timeframe'], row['count']))
                elif any(crypto in symbol for crypto in ['BTC', 'ETH', 'CRYPTO']):
                    categories['crypto'].append((row['symbol'], row['timeframe'], row['count']))
                elif any(energy in symbol for energy in ['OIL', 'WTI', 'BRENT', 'NATGAS']):
                    categories['energy'].append((row['symbol'], row['timeframe'], row['count']))
                elif any(index in symbol for index in ['US30', 'NAS100', 'SP500', 'DAX', 'FTSE']):
                    categories['indices'].append((row['symbol'], row['timeframe'], row['count']))
                else:
                    categories['majors'].append((row['symbol'], row['timeframe'], row['count']))
            
            return categories
            
        except Exception as e:
            print(f"❌ خطأ في قراءة البيانات: {e}")
            return {}
    
    def signal_handler(self, signum, frame):
        """معالج إشارة المقاطعة"""
        print("\n\n⚠️ تم استلام إشارة المقاطعة...")
        self._interrupted = True
        
        # حفظ التقدم
        self.save_progress()
        self.save_results()
        
        print("\n✅ تم حفظ التقدم. يمكنك استئناف التدريب لاحقاً.")
        print("🔄 لاستئناف التدريب، قم بتشغيل البرنامج مرة أخرى.")
        
        # إنهاء البرنامج
        sys.exit(0)
    
    def train_single_pair(self, symbol, timeframe, count):
        """تدريب زوج واحد مع timeout"""
        try:
            print(f"\n🔄 تدريب {symbol} {timeframe} ({count:,} سجل)")
            start_time = time.time()
            
            # تعيين timeout للتدريب (30 دقيقة)
            timeout = 1800
            
            # تدريب متقدم
            results = self.trainer.train_symbol_advanced(symbol, timeframe)
            
            if results and results.get('best_accuracy', 0) > 0:
                training_time = time.time() - start_time
                
                # حفظ النتائج
                result_data = {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'records': count,
                    'best_accuracy': results['best_accuracy'],
                    'best_strategy': results['best_strategy'],
                    'ensemble_accuracy': results.get('ensemble_accuracy', 0),
                    'models': results.get('model_results', {}),
                    'confidence_threshold': results.get('confidence_threshold', 0.7),
                    'expected_win_rate': results.get('expected_win_rate', 0),
                    'training_time': training_time,
                    'timestamp': datetime.now().isoformat()
                }
                
                # طباعة النتائج
                print(f"✅ {symbol} {timeframe}:")
                print(f"   • أفضل دقة: {results['best_accuracy']:.4f}")
                print(f"   • أفضل استراتيجية: {results['best_strategy']}")
                print(f"   • معدل الفوز المتوقع: {results.get('expected_win_rate', 0):.2%}")
                print(f"   • الوقت: {training_time/60:.1f} دقيقة")
                
                return result_data
            else:
                print(f"❌ فشل تدريب {symbol} {timeframe}")
                return None
                
        except Exception as e:
            print(f"❌ خطأ في {symbol} {timeframe}: {str(e)}")
            return None
    
    def train_all_advanced(self, parallel=True, max_workers=None):
        """تدريب جميع الأزواج بالنظام المتقدم مع معالجة المقاطعة"""
        
        # تسجيل معالج المقاطعة
        signal.signal(signal.SIGINT, self.signal_handler)
        
        print("🚀 التدريب الشامل المتقدم - نسبة النجاح المستهدفة 95%+")
        print("="*80)
        
        if self.completed_pairs:
            print(f"\n📊 استئناف التدريب ({len(self.completed_pairs)} نموذج مكتمل)")
        
        # الحصول على جميع الأزواج
        categories = self.get_all_available_pairs()
        
        if not categories:
            print("❌ لا توجد بيانات متاحة أو تم إكمال جميع الأزواج!")
            return
        
        # حساب الإجمالي
        total_pairs = sum(len(pairs) for pairs in categories.values())
        
        print(f"\n📊 إحصائيات البيانات:")
        for category, pairs in categories.items():
            if pairs:
                print(f"   • {category}: {len(pairs)} زوج")
        
        print(f"\n📈 إجمالي الأزواج المتبقية: {total_pairs}")
        print(f"⏱️  الوقت المقدر: {total_pairs * 5:.0f}-{total_pairs * 15:.0f} دقيقة")
        
        # السؤال عن التأكيد
        print("\n⚠️  تحذير: هذا التدريب سيستغرق وقتاً طويلاً!")
        print("💡 يمكنك إيقاف التدريب في أي وقت بالضغط على Ctrl+C وسيتم حفظ التقدم")
        confirm = input("هل تريد المتابعة؟ (y/n): ").strip().lower()
        
        if confirm != 'y':
            print("❌ تم إلغاء التدريب")
            return
        
        # بدء التدريب
        start_time = time.time()
        successful = len([r for r in self.results if r])
        failed = len(self.failed_pairs)
        
        # جمع جميع الأزواج
        all_pairs = []
        for category, pairs in categories.items():
            for pair in pairs:
                all_pairs.append((*pair, category))
        
        if parallel:
            # تدريب متوازي محسن
            if max_workers is None:
                max_workers = min(multiprocessing.cpu_count() - 1, 4)
            
            print(f"\n🚀 بدء التدريب المتوازي ({max_workers} عمليات)")
            
            # استخدام معالجة أفضل للأخطاء
            try:
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    # إرسال المهام
                    future_to_pair = {}
                    
                    for symbol, timeframe, count, category in all_pairs:
                        if not self._interrupted:
                            future = executor.submit(self.train_single_pair, symbol, timeframe, count)
                            future_to_pair[future] = (symbol, timeframe, category)
                    
                    # معالجة النتائج
                    for future in as_completed(future_to_pair):
                        if self._interrupted:
                            break
                            
                        symbol, timeframe, category = future_to_pair[future]
                        pair_key = f"{symbol}_{timeframe}"
                        
                        try:
                            # تعيين timeout لانتظار النتيجة
                            result = future.result(timeout=1800)  # 30 دقيقة
                            
                            if result:
                                successful += 1
                                result['category'] = category
                                self.results.append(result)
                                self.completed_pairs.add(pair_key)
                            else:
                                failed += 1
                                self.failed_pairs.append((symbol, timeframe))
                                
                        except TimeoutError:
                            failed += 1
                            self.failed_pairs.append((symbol, timeframe))
                            print(f"⏱️ انتهت مهلة {symbol} {timeframe}")
                            
                        except Exception as e:
                            failed += 1
                            self.failed_pairs.append((symbol, timeframe))
                            print(f"❌ خطأ في {symbol} {timeframe}: {e}")
                        
                        # تحديث التقدم
                        total_processed = successful + failed
                        progress = total_processed / (total_pairs + len(self.completed_pairs)) * 100
                        elapsed = time.time() - start_time
                        avg_time = elapsed / max(total_processed - len(self.completed_pairs), 1)
                        remaining = avg_time * (total_pairs - (total_processed - len(self.completed_pairs)))
                        
                        print(f"\n📊 التقدم: {progress:.1f}% ({total_processed}/{total_pairs + len(self.completed_pairs)})")
                        print(f"⏱️  الوقت المتبقي المقدر: {remaining/60:.1f} دقيقة")
                        
                        # حفظ التقدم كل 10 نماذج
                        if total_processed % 10 == 0:
                            self.save_progress()
                            
            except KeyboardInterrupt:
                print("\n⚠️ تم إيقاف التدريب...")
                self.save_progress()
                self.save_results()
                print("✅ تم حفظ التقدم")
                return
        
        else:
            # تدريب تسلسلي محسن
            print("\n🚀 بدء التدريب التسلسلي")
            
            try:
                for idx, (symbol, timeframe, count, category) in enumerate(all_pairs, 1):
                    if self._interrupted:
                        break
                        
                    print(f"\n{'='*60}")
                    print(f"📈 [{idx}/{total_pairs}] {category}: {symbol} {timeframe}")
                    
                    pair_key = f"{symbol}_{timeframe}"
                    result = self.train_single_pair(symbol, timeframe, count)
                    
                    if result:
                        successful += 1
                        result['category'] = category
                        self.results.append(result)
                        self.completed_pairs.add(pair_key)
                    else:
                        failed += 1
                        self.failed_pairs.append((symbol, timeframe))
                    
                    # تقدير الوقت المتبقي
                    elapsed = time.time() - start_time
                    avg_time = elapsed / idx
                    remaining = avg_time * (total_pairs - idx)
                    
                    print(f"⏱️  الوقت المتبقي المقدر: {remaining/60:.1f} دقيقة")
                    
                    # حفظ التقدم كل 5 نماذج
                    if idx % 5 == 0:
                        self.save_progress()
                        
            except KeyboardInterrupt:
                print("\n⚠️ تم إيقاف التدريب...")
                self.save_progress()
                self.save_results()
                print("✅ تم حفظ التقدم")
                return
        
        # حفظ النتائج النهائية
        self.save_results()
        
        # طباعة الملخص
        self.print_final_summary(successful, failed, time.time() - start_time)
        
        # حذف ملف التقدم بعد الانتهاء
        if self.progress_file.exists():
            self.progress_file.unlink()
    
    def save_results(self):
        """حفظ نتائج التدريب المتقدم"""
        if not self.results:
            return
        
        output_dir = Path("results/advanced")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. حفظ النتائج التفصيلية كـ JSON
        json_path = output_dir / f"advanced_training_results_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        # 2. حفظ ملخص النتائج كـ CSV
        summary_data = []
        for result in self.results:
            if result:  # التأكد من أن النتيجة ليست None
                summary_data.append({
                    'symbol': result['symbol'],
                    'timeframe': result['timeframe'],
                    'category': result.get('category', 'unknown'),
                    'records': result['records'],
                    'best_accuracy': result['best_accuracy'],
                    'best_strategy': result['best_strategy'],
                    'ensemble_accuracy': result.get('ensemble_accuracy', 0),
                    'expected_win_rate': result.get('expected_win_rate', 0),
                    'training_time_minutes': result['training_time'] / 60
                })
        
        if summary_data:
            df_summary = pd.DataFrame(summary_data)
            df_summary = df_summary.sort_values('best_accuracy', ascending=False)
            csv_path = output_dir / f"advanced_training_summary_{timestamp}.csv"
            df_summary.to_csv(csv_path, index=False)
            
            # 3. حفظ أفضل النماذج
            best_models = df_summary[df_summary['best_accuracy'] >= 0.85]
            if not best_models.empty:
                best_path = output_dir / f"best_models_85plus_{timestamp}.csv"
                best_models.to_csv(best_path, index=False)
            
            print(f"\n💾 تم حفظ النتائج:")
            print(f"   • النتائج التفصيلية: {json_path}")
            print(f"   • الملخص: {csv_path}")
            if not best_models.empty:
                print(f"   • أفضل النماذج (85%+): {best_path}")
    
    def print_final_summary(self, successful, failed, total_time):
        """طباعة الملخص النهائي المتقدم"""
        print("\n" + "="*80)
        print("🏆 الملخص النهائي - التدريب المتقدم الشامل")
        print("="*80)
        
        print(f"\n📊 الإحصائيات العامة:")
        print(f"   • نجح: {successful}")
        print(f"   • فشل: {failed}")
        print(f"   • الإجمالي: {successful + failed}")
        if successful + failed > 0:
            print(f"   • معدل النجاح: {successful/(successful+failed)*100:.1f}%")
        
        if self.results:
            # تصفية النتائج الصالحة فقط
            valid_results = [r for r in self.results if r is not None]
            
            if valid_results:
                df = pd.DataFrame(valid_results)
                
                # إحصائيات حسب الفئة
                print(f"\n📈 الأداء حسب الفئة:")
                for category in df['category'].unique():
                    cat_data = df[df['category'] == category]
                    print(f"\n   {category.upper()}:")
                    print(f"   • عدد النماذج: {len(cat_data)}")
                    print(f"   • متوسط الدقة: {cat_data['best_accuracy'].mean():.4f}")
                    print(f"   • أعلى دقة: {cat_data['best_accuracy'].max():.4f}")
                
                # أفضل النماذج
                print(f"\n🏆 أفضل 20 نموذج:")
                top_20 = df.nlargest(20, 'best_accuracy')
                for idx, row in top_20.iterrows():
                    win_rate = row.get('expected_win_rate', 0)
                    print(f"   {row['symbol']:<15} {row['timeframe']:<5} - "
                          f"الدقة: {row['best_accuracy']:.4f}, "
                          f"معدل الفوز: {win_rate:.2%}, "
                          f"الاستراتيجية: {row['best_strategy']}")
                
                # النماذج التي حققت الهدف
                target_models = df[df['best_accuracy'] >= 0.95]
                if not target_models.empty:
                    print(f"\n🎯 النماذج التي حققت الهدف (95%+): {len(target_models)}")
                    for idx, row in target_models.iterrows():
                        print(f"   • {row['symbol']} {row['timeframe']}: {row['best_accuracy']:.4f}")
                
                # الإحصائيات المتقدمة
                print(f"\n📊 الإحصائيات المتقدمة:")
                print(f"   • متوسط الدقة العام: {df['best_accuracy'].mean():.4f}")
                print(f"   • الانحراف المعياري: {df['best_accuracy'].std():.4f}")
                print(f"   • النماذج فوق 90%: {len(df[df['best_accuracy'] >= 0.90])}")
                print(f"   • النماذج فوق 85%: {len(df[df['best_accuracy'] >= 0.85])}")
                print(f"   • النماذج فوق 80%: {len(df[df['best_accuracy'] >= 0.80])}")
                
                # توزيع الاستراتيجيات
                print(f"\n🎯 توزيع الاستراتيجيات الناجحة:")
                strategy_counts = df['best_strategy'].value_counts()
                for strategy, count in strategy_counts.items():
                    avg_acc = df[df['best_strategy'] == strategy]['best_accuracy'].mean()
                    print(f"   • {strategy}: {count} نموذج (متوسط الدقة: {avg_acc:.4f})")
        
        # الأزواج الفاشلة
        if self.failed_pairs:
            print(f"\n❌ الأزواج الفاشلة ({len(self.failed_pairs)}):")
            for symbol, timeframe in self.failed_pairs[:10]:
                print(f"   • {symbol} {timeframe}")
            if len(self.failed_pairs) > 10:
                print(f"   • ... و {len(self.failed_pairs) - 10} زوج آخر")
        
        print(f"\n⏱️  الوقت الإجمالي: {total_time/3600:.1f} ساعة")
        if successful + failed > 0:
            print(f"   متوسط الوقت لكل نموذج: {total_time/max(successful+failed,1)/60:.1f} دقيقة")
        
        print("\n✅ اكتمل التدريب الشامل المتقدم!")
        print("\n💡 الخطوات التالية:")
        print("   1. مراجعة النتائج في مجلد results/advanced")
        print("   2. اختبار أفضل النماذج على بيانات حقيقية")
        print("   3. نشر النماذج الناجحة للتداول الفعلي")

def main():
    # خيارات التشغيل
    print("🚀 التدريب الشامل المتقدم - النسخة المحسنة")
    print("="*60)
    print("\n✨ الميزات الجديدة:")
    print("   • معالجة المقاطعة (Ctrl+C) مع حفظ التقدم")
    print("   • استئناف التدريب من حيث توقف")
    print("   • معالجة أفضل للأخطاء")
    print("   • حفظ تلقائي للتقدم")
    
    print("\n\nخيارات التشغيل:")
    print("1. تدريب متوازي (أسرع، يستخدم عدة معالجات)")
    print("2. تدريب تسلسلي (أبطأ، معالج واحد)")
    
    choice = input("\nاختيارك (1 أو 2): ").strip()
    
    parallel = choice == '1'
    
    if parallel:
        max_workers = input("عدد العمليات المتوازية (اتركه فارغاً للقيمة الافتراضية): ").strip()
        max_workers = int(max_workers) if max_workers else None
    else:
        max_workers = None
    
    # بدء التدريب
    trainer = FullAdvancedTrainer()
    
    try:
        trainer.train_all_advanced(parallel=parallel, max_workers=max_workers)
    except Exception as e:
        print(f"\n❌ خطأ غير متوقع: {e}")
        trainer.save_progress()
        trainer.save_results()
        print("✅ تم حفظ التقدم")

if __name__ == "__main__":
    main()