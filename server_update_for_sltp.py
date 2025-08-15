#!/usr/bin/env python3
"""
تحديث السيرفر لإرسال SL/TP مع التنبؤات
Server Update to Send SL/TP with Predictions
"""

import os
import sys
import json
from pathlib import Path

class ServerSLTPUpdater:
    """محدث السيرفر لدعم SL/TP"""
    
    def __init__(self):
        self.server_file = "src/mt5_bridge_server_advanced.py"
        self.backup_file = f"{self.server_file}.backup"
    
    def update_server(self):
        """تحديث ملف السيرفر لإضافة SL/TP"""
        
        print("🔄 تحديث السيرفر لدعم SL/TP...")
        
        # قراءة الملف الحالي
        with open(self.server_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # حفظ نسخة احتياطية
        with open(self.backup_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # إضافة التحديثات
        updates = []
        
        # 1. تحديث دالة process_prediction لإضافة SL/TP
        prediction_update = '''
    def process_prediction(self, data):
        """معالجة طلب التنبؤ وإرجاع الإشارة مع SL/TP"""
        try:
            # ... (الكود الحالي) ...
            
            # بعد الحصول على التنبؤ
            signal = self.determine_signal(probabilities[0])
            confidence = float(np.max(probabilities[0]))
            
            # حساب SL/TP بناءً على التنبؤ والسوق
            symbol = data.get('symbol', 'UNKNOWN')
            sl_tp_info = self.calculate_ml_based_sltp(
                symbol=symbol,
                signal=signal,
                confidence=confidence,
                df_features=df_features,
                current_price=float(df['close'].iloc[-1])
            )
            
            # إضافة SL/TP للاستجابة
            response = {
                'signal': signal,
                'action': signal,
                'confidence': confidence,
                'probability_up': float(probabilities[0][0]),
                'probability_down': float(probabilities[0][1]),
                'timestamp': datetime.now().isoformat(),
                # معلومات SL/TP الجديدة
                'sl_tp': {
                    'stop_loss': sl_tp_info['sl'],
                    'take_profit': sl_tp_info['tp'],
                    'sl_pips': sl_tp_info['sl_pips'],
                    'tp_pips': sl_tp_info['tp_pips'],
                    'risk_reward': sl_tp_info['risk_reward'],
                    'method': sl_tp_info['method'],
                    'confidence_adjusted': sl_tp_info['confidence_adjusted']
                }
            }
            
            return response
'''
        
        # 2. إضافة دالة حساب SL/TP بناءً على ML
        ml_sltp_function = '''
    def calculate_ml_based_sltp(self, symbol, signal, confidence, df_features, current_price):
        """
        حساب SL/TP بناءً على:
        1. ثقة النموذج
        2. تقلب السوق (volatility)
        3. مستويات الدعم والمقاومة
        4. نوع الأداة المالية
        """
        try:
            # استيراد الأدوات المطلوبة
            from dynamic_sl_tp_system import DynamicSLTPSystem
            from support_resistance import calculate_support_resistance
            
            # حساب التقلب
            volatility = df_features['ATR'].iloc[-1] if 'ATR' in df_features else 0.001
            
            # حساب مستويات S/R
            sr_levels = calculate_support_resistance(df_features, symbol)
            
            # تحديد نوع الأداة
            instrument_type = self._get_instrument_type(symbol)
            
            # حساب SL/TP الأساسي
            base_sl_distance = volatility * 1.5  # 1.5 ATR
            base_tp_distance = volatility * 3.0  # 3 ATR
            
            # تعديل بناءً على الثقة
            confidence_factor = 0.5 + (confidence * 0.5)  # 0.5 to 1.0
            
            # تعديل بناءً على نوع الأداة
            if instrument_type == 'metals':
                base_sl_distance *= 2
                base_tp_distance *= 2.5
            elif instrument_type == 'crypto':
                base_sl_distance *= 3
                base_tp_distance *= 4
            elif instrument_type == 'indices':
                base_sl_distance *= 1.5
                base_tp_distance *= 2
            
            # حساب القيم النهائية
            if signal == 'BUY':
                # البحث عن أقرب دعم للـ SL
                nearest_support = self._find_nearest_level(
                    sr_levels['support'], current_price, 'below'
                )
                sl = nearest_support if nearest_support else current_price - base_sl_distance
                
                # البحث عن أقرب مقاومة للـ TP
                nearest_resistance = self._find_nearest_level(
                    sr_levels['resistance'], current_price, 'above'
                )
                tp = nearest_resistance if nearest_resistance else current_price + base_tp_distance
                
            else:  # SELL
                # البحث عن أقرب مقاومة للـ SL
                nearest_resistance = self._find_nearest_level(
                    sr_levels['resistance'], current_price, 'above'
                )
                sl = nearest_resistance if nearest_resistance else current_price + base_sl_distance
                
                # البحث عن أقرب دعم للـ TP
                nearest_support = self._find_nearest_level(
                    sr_levels['support'], current_price, 'below'
                )
                tp = nearest_support if nearest_support else current_price - base_tp_distance
            
            # حساب النقاط
            pip_value = self._get_pip_value(symbol)
            sl_pips = abs(current_price - sl) / pip_value
            tp_pips = abs(tp - current_price) / pip_value
            
            # تعديل نهائي بناءً على الثقة
            if confidence < 0.7:
                # تقليل المخاطرة في حالة الثقة المنخفضة
                sl_pips *= 0.8
                tp_pips *= 0.8
            
            return {
                'sl': round(sl, 5),
                'tp': round(tp, 5),
                'sl_pips': round(sl_pips, 1),
                'tp_pips': round(tp_pips, 1),
                'risk_reward': round(tp_pips / sl_pips, 2) if sl_pips > 0 else 2.0,
                'method': 'ml_based_with_sr',
                'confidence_adjusted': True,
                'volatility_factor': round(volatility / current_price * 100, 2)
            }
            
        except Exception as e:
            logger.error(f"Error calculating ML-based SL/TP: {str(e)}")
            # قيم افتراضية
            return {
                'sl': 0,
                'tp': 0,
                'sl_pips': 30,
                'tp_pips': 60,
                'risk_reward': 2.0,
                'method': 'default',
                'confidence_adjusted': False
            }
    
    def _find_nearest_level(self, levels, current_price, direction):
        """إيجاد أقرب مستوى دعم أو مقاومة"""
        if not levels:
            return None
        
        if direction == 'below':
            valid_levels = [l for l in levels if l['price'] < current_price]
            if valid_levels:
                return max(valid_levels, key=lambda x: x['price'])['price']
        else:  # above
            valid_levels = [l for l in levels if l['price'] > current_price]
            if valid_levels:
                return min(valid_levels, key=lambda x: x['price'])['price']
        
        return None
    
    def _get_instrument_type(self, symbol):
        """تحديد نوع الأداة المالية"""
        symbol_upper = symbol.upper()
        
        if any(metal in symbol_upper for metal in ['XAU', 'GOLD', 'XAG', 'SILVER']):
            return 'metals'
        elif any(crypto in symbol_upper for crypto in ['BTC', 'ETH', 'XRP']):
            return 'crypto'
        elif any(idx in symbol_upper for idx in ['US30', 'NAS', 'DAX', 'SP500']):
            return 'indices'
        elif any(oil in symbol_upper for oil in ['OIL', 'WTI', 'BRENT']):
            return 'energy'
        else:
            return 'forex'
    
    def _get_pip_value(self, symbol):
        """الحصول على قيمة النقطة"""
        if 'JPY' in symbol.upper():
            return 0.01
        elif any(x in symbol.upper() for x in ['XAU', 'GOLD']):
            return 0.01
        elif any(x in symbol.upper() for x in ['BTC', 'US30', 'NAS']):
            return 1.0
        else:
            return 0.0001
'''
        
        # 3. تحديث التدريب لحفظ معلومات SL/TP
        training_update = '''
    def enhance_training_with_sltp(self, df, symbol):
        """تحسين بيانات التدريب بإضافة أهداف SL/TP الفعلية"""
        
        # حساب الأهداف المثلى بناءً على البيانات التاريخية
        for i in range(len(df) - 100):  # ترك 100 شمعة للتحليل المستقبلي
            current_price = df['close'].iloc[i]
            
            # تحليل الحركة المستقبلية
            future_prices = df['close'].iloc[i+1:i+101]
            future_highs = df['high'].iloc[i+1:i+101]
            future_lows = df['low'].iloc[i+1:i+101]
            
            # حساب أفضل SL/TP ممكن
            if df['signal'].iloc[i] == 'BUY':
                # أقصى ربح ممكن
                max_profit = future_highs.max() - current_price
                # أقصى خسارة قبل الربح
                max_loss = current_price - future_lows[:future_highs.argmax()].min()
                
                df.loc[i, 'optimal_tp'] = current_price + max_profit * 0.8  # 80% من أقصى ربح
                df.loc[i, 'optimal_sl'] = current_price - max_loss * 1.2   # 120% من أقصى خسارة
                
            elif df['signal'].iloc[i] == 'SELL':
                # أقصى ربح ممكن
                max_profit = current_price - future_lows.min()
                # أقصى خسارة قبل الربح
                max_loss = future_highs[:future_lows.argmin()].max() - current_price
                
                df.loc[i, 'optimal_tp'] = current_price - max_profit * 0.8
                df.loc[i, 'optimal_sl'] = current_price + max_loss * 1.2
        
        return df
'''
        
        print("✅ تم إنشاء ملف التحديث")
        print("\n📝 التحديثات المطلوبة:")
        print("1. تحديث دالة process_prediction لإرجاع SL/TP")
        print("2. إضافة دالة calculate_ml_based_sltp")
        print("3. تدريب النماذج على التنبؤ بـ SL/TP الأمثل")
        
        # كتابة ملف مساعد للتحديث
        update_script = '''#!/usr/bin/env python3
"""
سكريبت تحديث السيرفر لدعم SL/TP
"""

import sys
import os

# أضف المسار
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# التحديثات المطلوبة:
# 1. في mt5_bridge_server_advanced.py:
#    - تحديث process_prediction() لإرجاع sl_tp في الاستجابة
#    - إضافة calculate_ml_based_sltp() للحساب الذكي
#    - تحديث نماذج التدريب لتشمل optimal_sl و optimal_tp

# 2. في advanced_learner_unified.py:
#    - إضافة حساب الأهداف المثلى من البيانات التاريخية
#    - تدريب نموذج إضافي للتنبؤ بـ SL/TP

# 3. في الإكسبيرت:
#    - تحديث ParseResponse() لاستخراج SL/TP
#    - استخدام SL/TP من السيرفر إذا كان متاحاً

print("يرجى تطبيق هذه التحديثات يدوياً على ملفات السيرفر")
'''
        
        with open('apply_sltp_updates.py', 'w', encoding='utf-8') as f:
            f.write(update_script)
        
        return True


if __name__ == "__main__":
    updater = ServerSLTPUpdater()
    updater.update_server()