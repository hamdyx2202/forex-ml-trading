#!/usr/bin/env python3
"""
إصلاح مشاكل الصيغة في ملفات السيرفر
"""

import re

def fix_server_file():
    """إصلاح ملف unified_trading_learning_system_server.py"""
    
    # قراءة الملف
    with open('unified_trading_learning_system_server.py', 'r') as f:
        content = f.read()
    
    # إصلاحات محددة
    fixes = [
        # إصلاح التعليقات في القواميس
        (r"'M15': # mt5\.TIMEFRAME_M15,", "'M15': None,  # mt5.TIMEFRAME_M15"),
        (r"'M30': # mt5\.TIMEFRAME_M30,", "'M30': None,  # mt5.TIMEFRAME_M30"),
        (r"'H1': # mt5\.TIMEFRAME_H1,", "'H1': None,  # mt5.TIMEFRAME_H1"),
        (r"'H4': # mt5\.TIMEFRAME_H4,", "'H4': None,  # mt5.TIMEFRAME_H4"),
        (r"'D1': # mt5\.TIMEFRAME_D1,", "'D1': None,  # mt5.TIMEFRAME_D1"),
        
        # إصلاح استدعاءات mt5
        (r"# mt5\.copy_rates_from_pos", "# rates = mt5.copy_rates_from_pos"),
        (r"# mt5\.shutdown\(\)", "pass  # mt5.shutdown()"),
        
        # إصلاح أي تعليقات أخرى تسبب مشاكل
        (r"^\s*# mt5\.", "        pass  # mt5."),
    ]
    
    for pattern, replacement in fixes:
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    # حفظ الملف المصلح
    with open('unified_trading_learning_system_server.py', 'w') as f:
        f.write(content)
    
    print("✅ تم إصلاح ملف unified_trading_learning_system_server.py")

if __name__ == "__main__":
    fix_server_file()