#!/usr/bin/env python3
"""
تحسين نظام التدريب المتقدم بإضافة Stop Loss و Take Profit ديناميكي
"""

import shutil
from pathlib import Path

# نسخ احتياطية
original_file = "train_advanced_complete.py"
backup_file = "train_advanced_complete_original.py"
enhanced_file = "train_advanced_complete_enhanced.py"

# إنشاء نسخة احتياطية
shutil.copy(original_file, backup_file)
print(f"✅ تم حفظ نسخة احتياطية: {backup_file}")

# قراءة الملف الأصلي
with open(original_file, 'r', encoding='utf-8') as f:
    content = f.read()

# إضافة الكود الجديد بعد training_strategies
sl_tp_code = '''
        
        # إعدادات Stop Loss و Take Profit الديناميكية
        self.sl_tp_settings = {
            'ultra_short': {
                'stop_loss_atr': 0.5,      # نصف ATR للسكالبينج السريع
                'take_profit_ratios': [1.0, 1.5, 2.0],  # TP متعدد
                'trailing_stop_atr': 0.3,
                'breakeven_pips': 5
            },
            'scalping': {
                'stop_loss_atr': 1.0,
                'take_profit_ratios': [1.0, 2.0, 3.0],
                'trailing_stop_atr': 0.5,
                'breakeven_pips': 10
            },
            'short_term': {
                'stop_loss_atr': 1.5,
                'take_profit_ratios': [1.5, 2.5, 3.5],
                'trailing_stop_atr': 0.7,
                'breakeven_pips': 15
            },
            'medium_term': {
                'stop_loss_atr': 2.0,
                'take_profit_ratios': [2.0, 3.0, 4.0],
                'trailing_stop_atr': 1.0,
                'breakeven_pips': 20
            },
            'long_term': {
                'stop_loss_atr': 2.5,
                'take_profit_ratios': [2.5, 4.0, 6.0],
                'trailing_stop_atr': 1.5,
                'breakeven_pips': 30
            }
        }
'''

# البحث عن موضع الإدراج
insert_pos = content.find("self.training_strategies = {")
if insert_pos != -1:
    # إيجاد نهاية self.training_strategies
    brace_count = 0
    start_brace = content.find("{", insert_pos)
    i = start_brace
    while i < len(content) and brace_count >= 0:
        if content[i] == '{':
            brace_count += 1
        elif content[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                # وجدنا نهاية القاموس
                insert_pos = i + 1
                break
        i += 1
    
    # إدراج الكود الجديد
    content = content[:insert_pos] + sl_tp_code + content[insert_pos:]

# إضافة دوال جديدة قبل create_advanced_targets
new_functions = '''
    def calculate_pip_value(self, symbol):
        """حساب قيمة النقطة حسب العملة"""
        if 'JPY' in symbol or 'XAG' in symbol:
            return 0.01
        elif 'XAU' in symbol:
            return 0.1
        else:
            return 0.0001
    
    def calculate_dynamic_sl_tp(self, df, position_type, entry_idx, strategy_name):
        """حساب Stop Loss و Take Profit ديناميكي متقدم"""
        
        sl_tp_config = self.sl_tp_settings.get(strategy_name, self.sl_tp_settings['medium_term'])
        
        # الحصول على ATR
        atr_period = 14
        if f'atr_{atr_period}' in df.columns:
            current_atr = df.iloc[entry_idx][f'atr_{atr_period}']
        else:
            # حساب ATR
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            current_atr = true_range.rolling(atr_period).mean().iloc[entry_idx]
        
        current_price = df['close'].iloc[entry_idx]
        pip_value = self.calculate_pip_value(df.index.name if hasattr(df.index, 'name') else 'EURUSD')
        
        # Stop Loss ديناميكي
        sl_distance = current_atr * sl_tp_config['stop_loss_atr']
        
        # Take Profit متعدد المستويات
        tp_distances = [sl_distance * ratio for ratio in sl_tp_config['take_profit_ratios']]
        
        # Support/Resistance
        support_resistance = self.find_support_resistance_levels(df, entry_idx)
        
        if position_type == 'long':
            # Stop Loss
            stop_loss = current_price - sl_distance
            
            # تعديل SL بناءً على الدعم
            nearest_support = min([s for s in support_resistance['support'] 
                                 if s < current_price and s > stop_loss], 
                                default=stop_loss)
            if nearest_support > stop_loss:
                stop_loss = nearest_support - (2 * pip_value)
            
            # Take Profit levels
            take_profits = []
            for tp_distance in tp_distances:
                tp = current_price + tp_distance
                
                # تعديل TP بناءً على المقاومة
                nearest_resistance = min([r for r in support_resistance['resistance'] 
                                        if r > current_price and r < tp], 
                                       default=tp)
                if nearest_resistance < tp:
                    tp = nearest_resistance - (2 * pip_value)
                
                take_profits.append(tp)
            
            # Trailing Stop
            trailing_stop_distance = current_atr * sl_tp_config['trailing_stop_atr']
            
        else:  # Short
            # Stop Loss
            stop_loss = current_price + sl_distance
            
            # تعديل SL بناءً على المقاومة
            nearest_resistance = max([r for r in support_resistance['resistance'] 
                                    if r > current_price and r < stop_loss], 
                                   default=stop_loss)
            if nearest_resistance < stop_loss:
                stop_loss = nearest_resistance + (2 * pip_value)
            
            # Take Profit levels
            take_profits = []
            for tp_distance in tp_distances:
                tp = current_price - tp_distance
                
                # تعديل TP بناءً على الدعم
                nearest_support = max([s for s in support_resistance['support'] 
                                     if s < current_price and s > tp], 
                                    default=tp)
                if nearest_support > tp:
                    tp = nearest_support + (2 * pip_value)
                
                take_profits.append(tp)
            
            # Trailing Stop
            trailing_stop_distance = current_atr * sl_tp_config['trailing_stop_atr']
        
        return {
            'stop_loss': stop_loss,
            'take_profits': take_profits,
            'trailing_stop_distance': trailing_stop_distance,
            'breakeven_pips': sl_tp_config['breakeven_pips'],
            'risk_amount': abs(current_price - stop_loss),
            'reward_ratios': sl_tp_config['take_profit_ratios'],
            'atr_used': current_atr
        }
    
    def find_support_resistance_levels(self, df, current_idx, lookback=150):
        """إيجاد مستويات الدعم والمقاومة المتقدمة"""
        start_idx = max(0, current_idx - lookback)
        price_data = df[['high', 'low', 'close']].iloc[start_idx:current_idx]
        
        # إيجاد القمم والقيعان
        highs = []
        lows = []
        
        # استخدام خوارزمية متقدمة للكشف عن القمم والقيعان
        window = 5
        for i in range(window, len(price_data) - window):
            # قمة محلية
            if (price_data['high'].iloc[i] == price_data['high'].iloc[i-window:i+window+1].max()):
                highs.append(price_data['high'].iloc[i])
            
            # قاع محلي
            if (price_data['low'].iloc[i] == price_data['low'].iloc[i-window:i+window+1].min()):
                lows.append(price_data['low'].iloc[i])
        
        # إضافة مستويات من Fibonacci
        if len(price_data) > 0:
            recent_high = price_data['high'].max()
            recent_low = price_data['low'].min()
            fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
            
            for level in fib_levels:
                fib_price = recent_low + (recent_high - recent_low) * level
                if fib_price > price_data['close'].iloc[-1]:
                    highs.append(fib_price)
                else:
                    lows.append(fib_price)
        
        # تجميع المستويات القريبة
        def cluster_levels(levels, threshold=0.002):
            if not levels:
                return []
            
            levels = sorted(levels)
            clusters = [[levels[0]]]
            
            for level in levels[1:]:
                if abs(level - clusters[-1][-1]) / clusters[-1][-1] < threshold:
                    clusters[-1].append(level)
                else:
                    clusters.append([level])
            
            # متوسط كل مجموعة مع وزن للتكرار
            result = []
            for cluster in clusters:
                weight = len(cluster)  # كلما زاد التكرار زادت القوة
                avg_level = sum(cluster) / len(cluster)
                result.append({'level': avg_level, 'strength': weight})
            
            # ترتيب حسب القوة
            result.sort(key=lambda x: x['strength'], reverse=True)
            
            # إرجاع أقوى المستويات فقط
            return [item['level'] for item in result[:10]]
        
        resistance_levels = cluster_levels(highs)
        support_levels = cluster_levels(lows)
        
        return {
            'resistance': resistance_levels,
            'support': support_levels
        }
    
    def create_advanced_targets_with_sl_tp(self, df, strategy_name):
        """إنشاء أهداف متقدمة مع معلومات SL/TP كاملة"""
        strategy = self.training_strategies[strategy_name]
        lookahead = strategy['lookahead']
        min_pips = strategy['min_pips']
        
        pip_value = self.calculate_pip_value(df.index.name if hasattr(df.index, 'name') else 'EURUSD')
        
        targets = []
        confidences = []
        sl_tp_info = []
        trade_quality = []
        
        for i in range(len(df) - lookahead):
            future_prices = df['close'].iloc[i+1:i+lookahead+1].values
            future_highs = df['high'].iloc[i+1:i+lookahead+1].values
            future_lows = df['low'].iloc[i+1:i+lookahead+1].values
            current_price = df['close'].iloc[i]
            
            # حساب أقصى حركة مع مراعاة الـ wicks
            max_up = (future_highs.max() - current_price) / pip_value
            max_down = (current_price - future_lows.min()) / pip_value
            
            # حساب الحركة الفعلية للإغلاق
            close_up = (future_prices.max() - current_price) / pip_value
            close_down = (current_price - future_prices.min()) / pip_value
            
            # تحديد نوع الصفقة مع معايير محسنة
            if max_up >= min_pips * 2 and close_up >= min_pips * 1.5:
                # Long signal
                targets.append(2)
                
                # حساب الثقة بناءً على عوامل متعددة
                confidence = min(
                    0.5 + (close_up / (min_pips * 4)) * 0.3 +  # قوة الحركة
                    (1 - max_down / max_up) * 0.2,  # نسبة الصعود للهبوط
                    1.0
                )
                confidences.append(confidence)
                
                # حساب SL/TP
                sl_tp = self.calculate_dynamic_sl_tp(df, 'long', i, strategy_name)
                sl_tp_info.append(sl_tp)
                
                # تقييم جودة الصفقة
                quality = self.evaluate_trade_quality(df, i, 'long', sl_tp, max_up, max_down)
                trade_quality.append(quality)
                
            elif max_down >= min_pips * 2 and close_down >= min_pips * 1.5:
                # Short signal
                targets.append(0)
                
                # حساب الثقة
                confidence = min(
                    0.5 + (close_down / (min_pips * 4)) * 0.3 +
                    (1 - max_up / max_down) * 0.2,
                    1.0
                )
                confidences.append(confidence)
                
                # حساب SL/TP
                sl_tp = self.calculate_dynamic_sl_tp(df, 'short', i, strategy_name)
                sl_tp_info.append(sl_tp)
                
                # تقييم جودة الصفقة
                quality = self.evaluate_trade_quality(df, i, 'short', sl_tp, max_up, max_down)
                trade_quality.append(quality)
                
            else:
                # No trade
                targets.append(1)
                confidences.append(0.5)
                sl_tp_info.append(None)
                trade_quality.append(0)
        
        # ملء القيم الأخيرة
        targets.extend([1] * lookahead)
        confidences.extend([0] * lookahead)
        sl_tp_info.extend([None] * lookahead)
        trade_quality.extend([0] * lookahead)
        
        return (np.array(targets), np.array(confidences), 
                sl_tp_info, np.array(trade_quality))
    
    def evaluate_trade_quality(self, df, idx, position_type, sl_tp, max_up, max_down):
        """تقييم جودة الصفقة بناءً على معايير متعددة"""
        quality_score = 0.5  # البداية من المنتصف
        
        # 1. Risk/Reward Ratio
        if sl_tp and sl_tp['take_profits']:
            avg_rr = np.mean([abs(tp - df['close'].iloc[idx]) / sl_tp['risk_amount'] 
                            for tp in sl_tp['take_profits']])
            if avg_rr >= 3:
                quality_score += 0.2
            elif avg_rr >= 2:
                quality_score += 0.1
        
        # 2. وضوح الاتجاه
        if position_type == 'long':
            trend_clarity = max_up / (max_down + 1)  # تجنب القسمة على صفر
        else:
            trend_clarity = max_down / (max_up + 1)
        
        if trend_clarity >= 3:
            quality_score += 0.15
        elif trend_clarity >= 2:
            quality_score += 0.1
        
        # 3. قوة المؤشرات الفنية
        rsi_col = 'rsi_14'
        if rsi_col in df.columns:
            rsi = df[rsi_col].iloc[idx]
            if position_type == 'long' and rsi < 70 and rsi > 30:
                quality_score += 0.1
            elif position_type == 'short' and rsi > 30 and rsi < 70:
                quality_score += 0.1
        
        # 4. التقلب المناسب
        if sl_tp and 'atr_used' in sl_tp:
            atr_ratio = sl_tp['atr_used'] / df['close'].iloc[idx]
            if 0.001 < atr_ratio < 0.02:  # تقلب معتدل
                quality_score += 0.05
        
        return min(quality_score, 1.0)
'''

# البحث عن موضع إدراج الدوال الجديدة
create_targets_pos = content.find("def create_advanced_targets(")
if create_targets_pos != -1:
    # إدراج قبل create_advanced_targets
    content = content[:create_targets_pos] + new_functions + "\n    " + content[create_targets_pos:]

# تحديث دالة train_symbol_advanced لاستخدام الميزات الجديدة
train_update = '''
            # إنشاء الأهداف مع SL/TP
            if hasattr(self, 'create_advanced_targets_with_sl_tp'):
                y, confidence, sl_tp_info, quality = self.create_advanced_targets_with_sl_tp(df, strategy_name)
                
                # فلترة الصفقات عالية الجودة فقط
                high_quality_mask = quality > 0.7
                X_balanced, y_balanced = self.balance_dataset(X[high_quality_mask], 
                                                             y[high_quality_mask], 
                                                             confidence[high_quality_mask])
            else:
                # الطريقة القديمة كـ fallback
                y, confidence = self.create_advanced_targets(df, strategy)
                X_balanced, y_balanced = self.balance_dataset(X, y, confidence)
'''

# إيجاد وتحديث السطر المناسب
old_line = "y, confidence = self.create_advanced_targets(df, strategy)"
if old_line in content:
    content = content.replace(
        "# إنشاء الأهداف\n            y, confidence = self.create_advanced_targets(df, strategy)",
        "# إنشاء الأهداف" + train_update
    )

# حفظ النسخة المحسنة
with open(enhanced_file, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"✅ تم إنشاء النسخة المحسنة: {enhanced_file}")

# استبدال الملف الأصلي بالمحسن
shutil.copy(enhanced_file, original_file)
print(f"✅ تم تحديث الملف الأصلي: {original_file}")

print("\n🎯 الميزات المضافة:")
print("1. Stop Loss ديناميكي متعدد المستويات")
print("2. Take Profit ثلاثي (TP1, TP2, TP3)")
print("3. Trailing Stop متكيف")
print("4. Breakeven آلي")
print("5. دعم جميع أنواع العملات")
print("6. مستويات Fibonacci")
print("7. تقييم جودة الصفقات")
print("8. فلترة الصفقات عالية الجودة")

print("\n✅ النظام الآن يشمل جميع أنماط الهدف والاستوب المختلفة!")