#!/usr/bin/env python3
"""
Quick Server Fix for Feature Passing
إصلاح سريع لتمرير الميزات
"""

import os
import sys

print("🔧 Quick fix for feature passing...")
print("="*60)

# قراءة الخادم الحالي
server_file = "src/mt5_bridge_server_advanced.py"

if os.path.exists(server_file):
    with open(server_file, 'r') as f:
        content = f.read()
    
    # نسخة احتياطية
    with open(server_file + '.backup2', 'w') as f:
        f.write(content)
    
    print("✅ Backup created")
    
    # البحث والاستبدال
    # 1. تغيير predict_with_confidence ليستخدم df_features
    old_predict = """                    # التنبؤ
                    result = self.predictor.predict_with_confidence(
                        symbol=symbol,
                        timeframe=model_timeframe,
                        current_data=None,
                        historical_data=bars_data
                    )"""
    
    new_predict = """                    # التنبؤ باستخدام الميزات المُعدة
                    # تحضير الميزات للتنبؤ
                    feature_cols = [col for col in df_features.columns 
                                   if col not in ['target', 'target_binary', 'target_3class', 
                                                 'future_return', 'time', 'open', 'high', 
                                                 'low', 'close', 'volume', 'spread', 'datetime']]
                    
                    # استخدام آخر صف
                    X = df_features[feature_cols].iloc[-1:].values
                    logger.info(f"🎯 Prediction features shape: {X.shape}")
                    
                    # التنبؤ المباشر
                    if model_key in self.predictor.models:
                        try:
                            model = self.predictor.models[model_key]
                            scaler = self.predictor.scalers[model_key]
                            
                            # التطبيع والتنبؤ
                            X_scaled = scaler.transform(X)
                            y_pred = model.predict(X_scaled)[0]
                            y_proba = model.predict_proba(X_scaled)[0]
                            
                            # النتيجة
                            result = {
                                'signal': 'BUY' if y_pred == 1 else 'SELL',
                                'action': 'BUY' if y_pred == 1 else 'SELL',
                                'confidence': float(max(y_proba)),
                                'probability_up': float(y_proba[1] if len(y_proba) > 1 else 0.5),
                                'probability_down': float(y_proba[0] if len(y_proba) > 0 else 0.5),
                                'model_accuracy': self.predictor.metrics.get(model_key, {}).get('accuracy', 0.75)
                            }
                        except Exception as e:
                            logger.error(f"Prediction error: {str(e)}")
                            result = {
                                'signal': 'ERROR',
                                'confidence': 0,
                                'message': str(e)
                            }
                    else:
                        result = {
                            'signal': 'NO_MODEL',
                            'confidence': 0,
                            'message': f'No model for {model_key}'
                        }"""
    
    # تطبيق التغيير
    if old_predict in content:
        content = content.replace(old_predict, new_predict)
        print("✅ Updated prediction code")
    else:
        print("⚠️ Could not find exact match, trying alternative...")
        
        # محاولة بديلة - إضافة الكود بعد إنشاء df_features
        lines = content.split('\n')
        new_lines = []
        
        for i, line in enumerate(lines):
            new_lines.append(line)
            
            # بعد إنشاء df_features وقبل التنبؤ
            if "df_features = self.feature_engineer.create_features(df.copy())" in line:
                # إضافة سطر لحفظ الميزات
                indent = len(line) - len(line.lstrip())
                new_lines.append(' ' * indent + "# حفظ الميزات للتنبؤ")
                new_lines.append(' ' * indent + "prepared_features = df_features.copy()")
            
            # استبدال استدعاء predict_with_confidence
            if "result = self.predictor.predict_with_confidence(" in line:
                # استبدال بكود التنبؤ المباشر
                indent = len(line) - len(line.lstrip())
                new_lines[-1] = ' ' * indent + "# التنبؤ المباشر باستخدام الميزات المُعدة"
                
                # إضافة كود التنبؤ
                prediction_code = """
                    # تحضير الميزات للتنبؤ
                    feature_cols = [col for col in prepared_features.columns 
                                   if col not in ['target', 'target_binary', 'target_3class', 
                                                 'future_return', 'time', 'open', 'high', 
                                                 'low', 'close', 'volume', 'spread', 'datetime']]
                    
                    # استخدام آخر صف
                    X = prepared_features[feature_cols].iloc[-1:].values
                    logger.info(f"🎯 Prediction features shape: {X.shape}")
                    
                    # التنبؤ المباشر
                    if model_key in self.predictor.models:
                        try:
                            model = self.predictor.models[model_key]
                            scaler = self.predictor.scalers[model_key]
                            
                            # التطبيع والتنبؤ
                            X_scaled = scaler.transform(X)
                            y_pred = model.predict(X_scaled)[0]
                            y_proba = model.predict_proba(X_scaled)[0]
                            
                            # النتيجة
                            result = {
                                'signal': 'BUY' if y_pred == 1 else 'SELL',
                                'action': 'BUY' if y_pred == 1 else 'SELL',
                                'confidence': float(max(y_proba)),
                                'probability_up': float(y_proba[1] if len(y_proba) > 1 else 0.5),
                                'probability_down': float(y_proba[0] if len(y_proba) > 0 else 0.5),
                                'model_accuracy': self.predictor.metrics.get(model_key, {}).get('accuracy', 0.75)
                            }
                        except Exception as e:
                            logger.error(f"Prediction error: {str(e)}")
                            result = {
                                'signal': 'ERROR',
                                'confidence': 0,
                                'message': str(e)
                            }
                    else:
                        result = {
                            'signal': 'NO_MODEL',
                            'confidence': 0,
                            'message': f'No model for {model_key}'
                        }"""
                
                for pred_line in prediction_code.strip().split('\n'):
                    new_lines.append(' ' * indent + pred_line)
                
                # تخطي الأسطر التالية من predict_with_confidence
                skip_count = 5
                for j in range(i+1, min(i+1+skip_count, len(lines))):
                    if j < len(lines) and ')' in lines[j]:
                        i = j
                        break
        
        content = '\n'.join(new_lines)
    
    # حفظ النسخة المحدثة
    with open(server_file, 'w') as f:
        f.write(content)
    
    print("✅ Server updated!")
    
else:
    print("❌ Server file not found!")

print("\n" + "="*60)
print("✅ Quick fix applied!")
print("\n🚀 الآن على VPS:")
print("1. انسخ الملف المحدث:")
print("   scp src/mt5_bridge_server_advanced.py root@69.62.121.53:/home/forex-ml-trading/src/")
print("\n2. أعد تشغيل الخادم:")
print("   ssh root@69.62.121.53")
print("   cd /home/forex-ml-trading")
print("   pkill -f 'python.*mt5_bridge_server'")
print("   source venv_pro/bin/activate")
print("   python src/mt5_bridge_server_advanced.py &")