#!/usr/bin/env python3
"""
اختبار بسيط للتدريب
"""

print("🔍 اختبار التدريب البسيط...")

try:
    from train_models_simple import SimpleModelTrainer
    
    trainer = SimpleModelTrainer()
    
    # اختبار على EURUSD M5
    print("\n📊 اختبار تدريب EURUSD M5...")
    scores = trainer.train_symbol("EURUSD", "M5")
    
    if scores:
        print("✅ نجح التدريب البسيط!")
        print(f"   • دقة التدريب: {scores['train_accuracy']:.4f}")
        print(f"   • دقة الاختبار: {scores['test_accuracy']:.4f}")
        print(f"   • Precision: {scores['precision']:.4f}")
        print(f"   • Recall: {scores['recall']:.4f}")
        print(f"   • F1-Score: {scores['f1']:.4f}")
        
        # حفظ النموذج
        model_path = trainer.save_model("EURUSD", "M5", trainer.model, trainer.scaler, scores)
        print(f"\n💾 تم حفظ النموذج في: {model_path}")
        
        print("\n✅ النظام يعمل بشكل صحيح!")
        print("🚀 يمكنك الآن تشغيل التدريب الكامل: python train_full_advanced.py")
        
    else:
        print("❌ فشل التدريب")
        
except Exception as e:
    print(f"❌ خطأ: {e}")
    import traceback
    traceback.print_exc()