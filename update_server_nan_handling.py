#!/usr/bin/env python3
"""
Update server to handle NaN values intelligently
تحديث الخادم للتعامل مع NaN بذكاء
"""

import shutil

# نسخ احتياطية
shutil.copy('src/mt5_bridge_server_advanced.py', 'src/mt5_bridge_server_advanced_backup.py')
shutil.copy('feature_engineer_fixed_v3.py', 'feature_engineer_fixed_v4.py')

# تحديث feature_engineer_fixed_v4.py
with open('feature_engineer_fixed_v4.py', 'r', encoding='utf-8') as f:
    content = f.read()

# إضافة دالة جديدة للتعامل مع NaN
nan_handling_code = '''
    def handle_nan_values(self, df: pd.DataFrame, min_valid_ratio: float = 0.7) -> pd.DataFrame:
        """التعامل مع قيم NaN بطريقة ذكية"""
        df = df.copy()
        
        # 1. إزالة الأعمدة التي بها الكثير من NaN
        nan_ratio = df.isna().sum() / len(df)
        cols_to_drop = nan_ratio[nan_ratio > (1 - min_valid_ratio)].index.tolist()
        
        if cols_to_drop:
            logger.info(f"Dropping {len(cols_to_drop)} columns with >30% NaN")
            df = df.drop(columns=cols_to_drop)
        
        # 2. ملء NaN في المؤشرات بطرق مناسبة
        for col in df.columns:
            if df[col].isna().any():
                if 'SMA' in col or 'EMA' in col:
                    # للمتوسطات: استخدم forward fill ثم backward fill
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                elif 'RSI' in col or 'STOCH' in col:
                    # للمذبذبات: استخدم القيمة المحايدة 50
                    df[col] = df[col].fillna(50)
                elif 'volume' in col.lower():
                    # للحجم: استخدم المتوسط
                    df[col] = df[col].fillna(df[col].mean())
                else:
                    # للباقي: استخدم الاستيفاء الخطي
                    df[col] = df[col].interpolate(method='linear', limit_direction='both')
        
        return df
'''

# إضافة الكود قبل دالة create_features
insert_pos = content.find('def create_features(')
content = content[:insert_pos] + nan_handling_code + '\n' + content[insert_pos:]

# تحديث دالة create_features
old_dropna = "df = df.dropna()"
new_dropna = """# معالجة NaN بدلاً من حذف كل الصفوف
        df = self.handle_nan_values(df)
        
        # حذف الصفوف المتبقية مع NaN في الأعمدة الحرجة فقط
        critical_cols = ['open', 'high', 'low', 'close', 'target'] if 'target' in df.columns else ['open', 'high', 'low', 'close']
        df = df.dropna(subset=[col for col in critical_cols if col in df.columns])"""

content = content.replace(old_dropna, new_dropna)

# حفظ النسخة المحدثة
with open('feature_engineer_fixed_v4.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Created feature_engineer_fixed_v4.py with smart NaN handling")

# تحديث الخادم ليستخدم النسخة الجديدة
with open('src/mt5_bridge_server_advanced.py', 'r', encoding='utf-8') as f:
    server_content = f.read()

server_content = server_content.replace(
    'from feature_engineer_fixed_v3 import FeatureEngineer',
    'from feature_engineer_fixed_v4 import FeatureEngineer'
)

# إضافة معالجة أفضل للبيانات القليلة
additional_check = '''
                # التحقق من كفاية البيانات للمؤشرات
                min_bars_needed = 200  # للـ SMA_200
                if len(bars_data) < min_bars_needed:
                    logger.warning(f"Bars received: {len(bars_data)}, recommended: {min_bars_needed}")
                    # نستمر لكن مع تحذير
'''

# إضافة التحقق
check_pos = server_content.find("if len(bars_data) < 50:")
if check_pos > 0:
    server_content = server_content[:check_pos] + additional_check + '\n                ' + server_content[check_pos:]

with open('src/mt5_bridge_server_advanced.py', 'w', encoding='utf-8') as f:
    f.write(server_content)

print("✅ Updated server with better data handling")
print("\n🚀 Now:")
print("1. Update EA: BarsToSend = 200")
print("2. Restart server: python src/mt5_bridge_server_advanced.py")