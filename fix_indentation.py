#!/usr/bin/env python3
"""
Fix indentation error in feature_engineer_fixed_v4.py
"""

# استخدم النسخة v3 بدلاً من v4 المعطوبة
import shutil

print("🔧 Fixing indentation error...")

# نسخ v3 إلى v4 مع التعديلات
shutil.copy('feature_engineer_fixed_v3.py', 'feature_engineer_fixed_v4.py')

# قراءة المحتوى
with open('feature_engineer_fixed_v4.py', 'r', encoding='utf-8') as f:
    content = f.read()

# إضافة معالجة NaN المحسنة
nan_handling = '''
    def smart_fillna(self, df: pd.DataFrame) -> pd.DataFrame:
        """ملء قيم NaN بطريقة ذكية حسب نوع المؤشر"""
        df = df.copy()
        
        for col in df.columns:
            if df[col].isna().any():
                # تخطي الأعمدة الأساسية
                if col in ['open', 'high', 'low', 'close', 'volume', 'time']:
                    continue
                    
                # للمتوسطات المتحركة
                if 'SMA' in col or 'EMA' in col:
                    # استخدم أقرب قيمة سابقة
                    df[col] = df[col].fillna(method='ffill')
                    # إذا بقيت NaN في البداية، استخدم السعر
                    if df[col].isna().any():
                        df[col] = df[col].fillna(df['close'])
                
                # للمذبذبات
                elif 'RSI' in col or 'STOCH' in col or 'CCI' in col:
                    df[col] = df[col].fillna(50)  # قيمة محايدة
                
                # لمؤشرات الحجم
                elif 'volume' in col.lower() or 'OBV' in col:
                    df[col] = df[col].fillna(df[col].mean())
                
                # للـ MACD وإشاراته
                elif 'MACD' in col:
                    df[col] = df[col].fillna(0)
                
                # للباقي
                else:
                    # محاولة الاستيفاء أولاً
                    df[col] = df[col].interpolate(method='linear', limit_direction='both')
                    # إذا فشل، استخدم المتوسط
                    if df[col].isna().any():
                        df[col] = df[col].fillna(df[col].mean())
                        
        return df
'''

# إضافة الدالة قبل create_features
insert_pos = content.find('    def create_features(')
if insert_pos > 0:
    content = content[:insert_pos] + nan_handling + '\n' + content[insert_pos:]

# تحديث create_features لاستخدام المعالجة الجديدة
old_dropna = """        # Remove NaN values
        initial_rows = len(df)
        df = df.dropna()
        logger.info(f"Removed {initial_rows - len(df)} rows with NaN values")"""

new_dropna = """        # معالجة NaN بذكاء بدلاً من حذف كل شيء
        initial_rows = len(df)
        
        # ملء NaN بطريقة ذكية
        df = self.smart_fillna(df)
        
        # حذف الصفوف فقط إذا كانت البيانات الأساسية ناقصة
        critical_cols = ['open', 'high', 'low', 'close']
        if 'target' in df.columns:
            critical_cols.append('target')
            
        df = df.dropna(subset=critical_cols)
        
        if len(df) < initial_rows:
            logger.info(f"Removed {initial_rows - len(df)} rows with critical NaN values")
        else:
            logger.info("All rows preserved after smart NaN handling")"""

content = content.replace(old_dropna, new_dropna)

# حفظ الملف المصحح
with open('feature_engineer_fixed_v4.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Fixed feature_engineer_fixed_v4.py")

# تحديث الخادم أيضاً
with open('src/mt5_bridge_server_advanced.py', 'r', encoding='utf-8') as f:
    server_content = f.read()

# إضافة تحقق أفضل من البيانات
check_addition = '''
                # تسجيل معلومات عن البيانات المستلمة
                logger.info(f"Received {len(bars_data)} bars for {symbol} {timeframe}")
                logger.info(f"Date range: {pd.to_datetime(bars_data[0]['time'], unit='s')} to {pd.to_datetime(bars_data[-1]['time'], unit='s')}")
'''

# البحث عن المكان المناسب للإضافة
pos = server_content.find("# تحويل البيانات إلى DataFrame")
if pos > 0:
    server_content = server_content[:pos] + check_addition + '\n                ' + server_content[pos:]

# حفظ الخادم المحدث
with open('src/mt5_bridge_server_advanced.py', 'w', encoding='utf-8') as f:
    f.write(server_content)

print("✅ Updated server with better logging")
print("\n🚀 Now run:")
print("python src/mt5_bridge_server_advanced.py")