#!/usr/bin/env python3
"""
تحديث ملف التدريب المتقدم لإزالة suffix من الأسماء
"""

def update_advanced_training():
    # قراءة الملف
    with open("train_advanced_complete.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    # البحث عن السطر الذي يحتوي على symbol.replace('m', '')
    # وتحديثه ليزيل suffix بشكل صحيح
    content = content.replace(
        "symbol = row['symbol'].replace('m', '')  # إزالة suffix",
        "symbol = row['symbol']  # استخدام الاسم كما هو"
    )
    
    # أيضاً في train_full_advanced.py
    content = content.replace(
        "pip_value = self.calculate_pip_value(df.index.name if hasattr(df.index, 'name') else 'EURUSD')",
        "pip_value = self.calculate_pip_value(symbol if symbol else 'EURUSD')"
    )
    
    # حفظ الملف
    with open("train_advanced_complete.py", "w", encoding="utf-8") as f:
        f.write(content)
    
    print("✅ تم تحديث train_advanced_complete.py")
    
    # نفس الشيء لـ train_full_advanced.py
    try:
        with open("train_full_advanced.py", "r", encoding="utf-8") as f:
            content = f.read()
        
        content = content.replace(
            "symbol = row['symbol'].replace('m', '')  # إزالة suffix",
            "symbol = row['symbol']  # استخدام الاسم كما هو"
        )
        
        with open("train_full_advanced.py", "w", encoding="utf-8") as f:
            f.write(content)
        
        print("✅ تم تحديث train_full_advanced.py")
    except:
        pass

if __name__ == "__main__":
    update_advanced_training()
