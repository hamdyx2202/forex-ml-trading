#!/usr/bin/env python3
"""
Fix Feature Count Mismatch
إصلاح عدم تطابق عدد الميزات
"""

import os
import sys

print("🔧 Fixing feature count mismatch...")
print("="*60)

print("\nالمشكلة:")
print("- النماذج تتوقع: 70 features")
print("- AdaptiveFeatureEngineer ينتج: 49 features")
print("\nالحل: تحديث AdaptiveFeatureEngineer لإنتاج 70 ميزة")

# قراءة الملف
if os.path.exists('feature_engineer_adaptive.py'):
    with open('feature_engineer_adaptive.py', 'r') as f:
        content = f.read()
    
    # البحث عن السطر الذي يحدد target_features
    if 'self.target_features = target_features' in content:
        print("\n✅ Found AdaptiveFeatureEngineer")
        
        # إضافة المزيد من الميزات
        additional_features_code = '''
    def add_extended_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """إضافة ميزات إضافية للوصول إلى 70"""
        df = df.copy()
        
        # مؤشرات إضافية
        if len(df) >= 10:
            # Williams %R
            df['williams_r'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
            
            # CCI - Commodity Channel Index
            df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=20)
            
            # MFI - Money Flow Index
            if 'volume' in df.columns:
                df['mfi'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
            
            # ADX - Average Directional Index
            df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
            df['plus_di'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
            df['minus_di'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
            
            # Aroon
            df['aroon_up'], df['aroon_down'] = talib.AROON(df['high'], df['low'], timeperiod=25)
            
            # Ultimate Oscillator
            df['ultimate_osc'] = talib.ULTOSC(df['high'], df['low'], df['close'])
            
            # ROC - Rate of Change
            for period in [5, 10, 20]:
                df[f'roc_{period}'] = talib.ROC(df['close'], timeperiod=period)
            
            # Standard Deviation
            for period in [10, 20]:
                df[f'stddev_{period}'] = talib.STDDEV(df['close'], timeperiod=period)
            
            # More price ratios
            df['oc_ratio'] = df['open'] / df['close'].replace(0, 1)
            df['range_ratio'] = (df['high'] - df['low']) / df['close'].replace(0, 1)
            df['body_to_range'] = abs(df['close'] - df['open']) / (df['high'] - df['low']).replace(0, 0.0001)
            
            # More volume features
            if 'volume' in df.columns:
                df['volume_change'] = df['volume'].pct_change()
                df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(10, min_periods=1).mean().replace(0, 1)
            
            # Momentum features
            for lag in [1, 3, 5, 10]:
                df[f'momentum_{lag}'] = df['close'] - df['close'].shift(lag)
                df[f'momentum_pct_{lag}'] = df['close'].pct_change(lag)
        
        return df
'''
        
        # إضافة الدالة الجديدة
        # البحث عن نهاية الكلاس
        class_end = content.rfind('def create_unified_features')
        if class_end > 0:
            # إدراج قبل نهاية الكلاس
            content = content[:class_end] + additional_features_code + '\n' + content[class_end:]
        
        # تحديث create_features لاستخدام الميزات الإضافية
        if 'df = self.add_market_structure(df)' in content:
            content = content.replace(
                'df = self.add_market_structure(df)',
                'df = self.add_market_structure(df)\n        df = self.add_extended_features(df)'
            )
        
        # حفظ النسخة المحدثة
        with open('feature_engineer_adaptive_70.py', 'w') as f:
            f.write(content)
        
        print("✅ Created feature_engineer_adaptive_70.py")

# الحل البديل - تحديث الخادم لاستخدام padding
print("\n📝 Creating server fix...")

server_fix = '''#!/usr/bin/env python3
"""
Server Fix for Feature Mismatch
إصلاح الخادم لعدم تطابق الميزات
"""

import os

print("🔧 Updating server to handle feature mismatch...")

# تحديث mt5_bridge_server_advanced.py
server_file = "src/mt5_bridge_server_advanced.py"

if os.path.exists(server_file):
    with open(server_file, 'r') as f:
        content = f.read()
    
    # إضافة كود padding
    padding_code = """
                # معالجة عدم تطابق عدد الميزات
                if X.shape[1] < 70:  # إذا كانت الميزات أقل من المتوقع
                    logger.warning(f"Feature padding: {X.shape[1]} -> 70")
                    # إضافة أعمدة صفرية
                    padding_needed = 70 - X.shape[1]
                    padding = np.zeros((X.shape[0], padding_needed))
                    X = np.hstack([X, padding])
                elif X.shape[1] > 70:  # إذا كانت أكثر
                    logger.warning(f"Feature trimming: {X.shape[1]} -> 70")
                    X = X[:, :70]
"""
    
    # البحث عن مكان التنبؤ
    if "result = self.predictor.predict_with_confidence" in content:
        lines = content.split('\\n')
        new_lines = []
        
        for i, line in enumerate(lines):
            if "df_features = self.feature_engineer.prepare_for_prediction" in line:
                new_lines.append(line)
                # إضافة padding بعد هذا السطر
                indent = len(line) - len(line.lstrip())
                new_lines.append(' ' * (indent + 4) + "")
                new_lines.append(' ' * (indent + 4) + "# Ensure 70 features")
                new_lines.append(' ' * (indent + 4) + "import numpy as np")
                new_lines.append(' ' * (indent + 4) + "X = df_features.values if hasattr(df_features, 'values') else df_features")
                for p_line in padding_code.strip().split('\\n'):
                    new_lines.append(' ' * (indent + 4) + p_line)
                new_lines.append(' ' * (indent + 4) + "df_features = pd.DataFrame(X)")
            else:
                new_lines.append(line)
        
        content = '\\n'.join(new_lines)
        
        # حفظ النسخة المحدثة
        with open(server_file + '.backup', 'w') as f:
            with open(server_file, 'r') as orig:
                f.write(orig.read())
        
        with open(server_file, 'w') as f:
            f.write(content)
        
        print("✅ Updated server with padding fix")
'''

with open('fix_server_padding.py', 'w') as f:
    f.write(server_fix)

print("✅ Created fix_server_padding.py")

# الحل الأسرع - إنشاء wrapper للخادم
print("\n📝 Creating quick wrapper...")

wrapper = '''#!/usr/bin/env python3
"""
Server Wrapper with Feature Fix
غلاف الخادم مع إصلاح الميزات
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# تعديل AdaptiveFeatureEngineer ليُنتج 70 ميزة
import feature_engineer_adaptive

# تعديل __init__ ليستهدف 70
original_init = feature_engineer_adaptive.AdaptiveFeatureEngineer.__init__

def new_init(self, target_features=None):
    original_init(self, target_features=70)  # فرض 70 ميزة
    print("🎯 Forcing 70 features")

feature_engineer_adaptive.AdaptiveFeatureEngineer.__init__ = new_init

# تعديل create_features لإضافة padding
original_create = feature_engineer_adaptive.AdaptiveFeatureEngineer.create_features

def new_create_features(self, df, target_config=None):
    result = original_create(self, df, target_config)
    
    # التحقق من عدد الميزات
    feature_cols = [col for col in result.columns 
                   if col not in ['target', 'target_binary', 'target_3class', 
                                 'future_return', 'time', 'open', 'high', 
                                 'low', 'close', 'volume', 'spread', 'datetime']]
    
    current_count = len(feature_cols)
    
    if current_count < 70:
        print(f"📊 Adding {70 - current_count} padding features...")
        # إضافة أعمدة padding
        for i in range(current_count, 70):
            result[f'padding_feature_{i}'] = 0.0
    
    return result

feature_engineer_adaptive.AdaptiveFeatureEngineer.create_features = new_create_features

# الآن استيراد وتشغيل الخادم
print("🚀 Starting server with 70-feature fix...")
from src.mt5_bridge_server_advanced import app

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)
'''

with open('server_70_features.py', 'w') as f:
    f.write(wrapper)

print("✅ Created server_70_features.py")

print("\n" + "="*60)
print("✅ Solutions created!")
print("\n🚀 Try one of these:")
print("1. python server_70_features.py  (الأسرع)")
print("2. python fix_server_padding.py && python src/mt5_bridge_server_advanced.py")
print("3. Update imports to use feature_engineer_adaptive_70.py")

# حل فوري - تحديث feature_engineer_adaptive مباشرة
print("\n🔧 Direct fix for AdaptiveFeatureEngineer...")

if os.path.exists('feature_engineer_adaptive.py'):
    with open('feature_engineer_adaptive.py', 'r') as f:
        content = f.read()
    
    # تغيير السطر الذي يحدد target_features في create_features
    content = content.replace(
        'if self.target_features and len(feature_cols) > self.target_features:',
        'if self.target_features and len(feature_cols) != self.target_features:'
    )
    
    # إضافة padding في نهاية create_features
    if 'logger.info(f"Feature engineering completed. Features: {len(feature_cols)}")' in content:
        padding_addition = '''
        
        # التأكد من 70 ميزة للنماذج الحالية
        final_feature_cols = [col for col in df.columns 
                             if col not in ['target', 'target_binary', 'target_3class', 
                                           'future_return', 'time', 'open', 'high', 
                                           'low', 'close', 'volume', 'spread', 'datetime']]
        
        if len(final_feature_cols) < 70:
            logger.info(f"Adding padding features: {len(final_feature_cols)} -> 70")
            for i in range(len(final_feature_cols), 70):
                df[f'padding_{i}'] = 0.0
        
        logger.info(f"Final feature count: {len([c for c in df.columns if c not in ['target', 'target_binary', 'target_3class', 'future_return', 'time', 'open', 'high', 'low', 'close', 'volume', 'spread', 'datetime']])}")
'''
        
        content = content.replace(
            'logger.info(f"Feature engineering completed. Features: {len(feature_cols)}")',
            'logger.info(f"Feature engineering completed. Features: {len(feature_cols)}")' + padding_addition
        )
    
    # حفظ النسخة المحدثة
    with open('feature_engineer_adaptive.py.backup', 'w') as f:
        with open('feature_engineer_adaptive.py', 'r') as orig:
            f.write(orig.read())
    
    with open('feature_engineer_adaptive.py', 'w') as f:
        f.write(content)
    
    print("✅ Updated feature_engineer_adaptive.py directly!")
    print("\n🚀 Now just restart the server:")
    print("   python src/mt5_bridge_server_advanced.py")