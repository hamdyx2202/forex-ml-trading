#!/usr/bin/env python3
"""
تحديث نظام التدريب المتقدم ليشمل Stop Loss و Take Profit ديناميكي
"""

import os

# قراءة الملف الحالي
with open("train_advanced_complete.py", "r", encoding="utf-8") as f:
    content = f.read()

# إضافة دالة حساب SL/TP الديناميكي
dynamic_sl_tp_code = '''
    def calculate_dynamic_sl_tp(self, df, position_type, entry_idx, strategy_params):
        """حساب Stop Loss و Take Profit ديناميكي بناءً على ظروف السوق"""
        
        # الحصول على ATR الحالي
        atr_period = 14
        if f'atr_{atr_period}' in df.columns:
            current_atr = df.iloc[entry_idx][f'atr_{atr_period}']
        else:
            # حساب ATR يدوياً
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            current_atr = true_range.rolling(atr_period).mean().iloc[entry_idx]
        
        current_price = df['close'].iloc[entry_idx]
        
        # حساب pip value
        if 'JPY' in str(df.index.name):
            pip_value = 0.01
        else:
            pip_value = 0.0001
        
        # Stop Loss ديناميكي
        sl_atr_multiplier = strategy_params.get('stop_loss_atr', 2.0)
        stop_loss_distance = current_atr * sl_atr_multiplier
        
        # Take Profit متعدد المستويات
        tp_ratio = strategy_params.get('take_profit_ratio', 2.0)
        
        if position_type == 'long':
            stop_loss = current_price - stop_loss_distance
            take_profit_1 = current_price + (stop_loss_distance * 1.0)  # TP1: 1:1
            take_profit_2 = current_price + (stop_loss_distance * tp_ratio)  # TP2: 1:2+
            take_profit_3 = current_price + (stop_loss_distance * tp_ratio * 1.5)  # TP3: 1:3+
        else:  # short
            stop_loss = current_price + stop_loss_distance
            take_profit_1 = current_price - (stop_loss_distance * 1.0)
            take_profit_2 = current_price - (stop_loss_distance * tp_ratio)
            take_profit_3 = current_price - (stop_loss_distance * tp_ratio * 1.5)
        
        # تعديل بناءً على Support/Resistance
        support_resistance = self.find_support_resistance_levels(df, entry_idx)
        
        if position_type == 'long':
            # تعديل SL ليكون تحت أقرب دعم
            nearest_support = min([s for s in support_resistance['support'] if s < current_price], default=stop_loss)
            stop_loss = min(stop_loss, nearest_support - (5 * pip_value))
            
            # تعديل TP ليكون قبل أقرب مقاومة
            nearest_resistance = min([r for r in support_resistance['resistance'] if r > current_price], default=take_profit_2)
            take_profit_2 = min(take_profit_2, nearest_resistance - (5 * pip_value))
        else:
            # العكس للبيع
            nearest_resistance = max([r for r in support_resistance['resistance'] if r > current_price], default=stop_loss)
            stop_loss = max(stop_loss, nearest_resistance + (5 * pip_value))
            
            nearest_support = max([s for s in support_resistance['support'] if s < current_price], default=take_profit_2)
            take_profit_2 = max(take_profit_2, nearest_support + (5 * pip_value))
        
        return {
            'stop_loss': stop_loss,
            'take_profit_1': take_profit_1,
            'take_profit_2': take_profit_2,
            'take_profit_3': take_profit_3,
            'risk_reward_ratio': abs(take_profit_2 - current_price) / abs(stop_loss - current_price)
        }
    
    def find_support_resistance_levels(self, df, current_idx, lookback=100):
        """إيجاد مستويات الدعم والمقاومة"""
        start_idx = max(0, current_idx - lookback)
        price_data = df['close'].iloc[start_idx:current_idx]
        
        # إيجاد القمم والقيعان
        highs = []
        lows = []
        
        for i in range(2, len(price_data) - 2):
            # قمة محلية
            if (price_data.iloc[i] > price_data.iloc[i-1] and 
                price_data.iloc[i] > price_data.iloc[i-2] and
                price_data.iloc[i] > price_data.iloc[i+1] and 
                price_data.iloc[i] > price_data.iloc[i+2]):
                highs.append(price_data.iloc[i])
            
            # قاع محلي
            if (price_data.iloc[i] < price_data.iloc[i-1] and 
                price_data.iloc[i] < price_data.iloc[i-2] and
                price_data.iloc[i] < price_data.iloc[i+1] and 
                price_data.iloc[i] < price_data.iloc[i+2]):
                lows.append(price_data.iloc[i])
        
        # تجميع المستويات القريبة
        def cluster_levels(levels, threshold=0.001):
            if not levels:
                return []
            
            levels = sorted(levels)
            clusters = [[levels[0]]]
            
            for level in levels[1:]:
                if abs(level - clusters[-1][-1]) / clusters[-1][-1] < threshold:
                    clusters[-1].append(level)
                else:
                    clusters.append([level])
            
            # متوسط كل مجموعة
            return [sum(cluster) / len(cluster) for cluster in clusters]
        
        resistance_levels = cluster_levels(highs)
        support_levels = cluster_levels(lows)
        
        return {
            'resistance': resistance_levels,
            'support': support_levels
        }
    
    def create_advanced_targets_with_sl_tp(self, df, strategy):
        """إنشاء أهداف متقدمة مع معلومات SL/TP"""
        lookahead = strategy['lookahead']
        min_pips = strategy['min_pips']
        
        # حساب pip value
        if 'JPY' in str(df.index.name):
            pip_value = 0.01
        else:
            pip_value = 0.0001
        
        targets = []
        confidences = []
        sl_tp_info = []
        
        for i in range(len(df) - lookahead):
            future_prices = df['close'].iloc[i+1:i+lookahead+1].values
            current_price = df['close'].iloc[i]
            
            # حساب أقصى حركة
            max_up = (future_prices.max() - current_price) / pip_value
            max_down = (current_price - future_prices.min()) / pip_value
            
            # تحديد نوع الصفقة
            if max_up >= min_pips * 2:  # Long
                targets.append(2)
                confidences.append(min(max_up / (min_pips * 3), 1.0))
                
                # حساب SL/TP للشراء
                sl_tp = self.calculate_dynamic_sl_tp(df, 'long', i, strategy)
                sl_tp_info.append(sl_tp)
                
            elif max_down >= min_pips * 2:  # Short
                targets.append(0)
                confidences.append(min(max_down / (min_pips * 3), 1.0))
                
                # حساب SL/TP للبيع
                sl_tp = self.calculate_dynamic_sl_tp(df, 'short', i, strategy)
                sl_tp_info.append(sl_tp)
                
            else:  # No Trade
                targets.append(1)
                confidences.append(0.5)
                sl_tp_info.append(None)
        
        # ملء القيم الأخيرة
        targets.extend([1] * lookahead)
        confidences.extend([0] * lookahead)
        sl_tp_info.extend([None] * lookahead)
        
        return np.array(targets), np.array(confidences), sl_tp_info
'''

# البحث عن مكان إدراج الكود
insert_position = content.find("def create_advanced_targets(self, df, strategy):")

if insert_position != -1:
    # إدراج الكود الجديد قبل الدالة الموجودة
    content = content[:insert_position] + dynamic_sl_tp_code + "\n    " + content[insert_position:]
    
    # حفظ النسخة المحدثة
    backup_path = "train_advanced_complete_backup.py"
    with open(backup_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    print("✅ تم تحديث نظام التدريب المتقدم!")
    print(f"📁 النسخة المحدثة محفوظة في: {backup_path}")
    print("\n🎯 الميزات الجديدة:")
    print("   • Stop Loss ديناميكي بناءً على ATR")
    print("   • Take Profit متعدد المستويات (3 مستويات)")
    print("   • تعديل SL/TP بناءً على الدعم والمقاومة")
    print("   • حساب Risk/Reward Ratio")
    print("   • دعم العملات المختلفة (JPY vs others)")
else:
    print("❌ لم يتم العثور على الموضع المناسب للتحديث!")

# إنشاء مثال للاستخدام
example_code = '''#!/usr/bin/env python3
"""
مثال على استخدام نظام SL/TP الديناميكي
"""

from train_advanced_complete import AdvancedCompleteTrainer

# إنشاء المدرب
trainer = AdvancedCompleteTrainer()

# مثال على استراتيجية مع SL/TP ديناميكي
strategy = {
    'name': 'dynamic_risk_management',
    'lookahead': 20,
    'min_pips': 30,
    'confidence_threshold': 0.75,
    'stop_loss_atr': 2.0,      # SL = 2 × ATR
    'take_profit_ratio': 2.5    # TP = 2.5 × SL distance
}

# التدريب سيستخدم الآن SL/TP الديناميكي تلقائياً
print("🎯 استراتيجية إدارة المخاطر الديناميكية:")
print(f"   • Stop Loss: {strategy['stop_loss_atr']} × ATR")
print(f"   • Take Profit: {strategy['take_profit_ratio']} × SL distance")
print("   • يتم التعديل بناءً على مستويات الدعم/المقاومة")
'''

with open("example_dynamic_sl_tp.py", "w", encoding="utf-8") as f:
    f.write(example_code)

print("\n📝 تم إنشاء مثال الاستخدام: example_dynamic_sl_tp.py")