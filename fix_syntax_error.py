#!/usr/bin/env python3
"""
إصلاح أخطاء الصيغة في unified_trading_learning_system_server.py
"""

import re

def fix_syntax_errors():
    """إصلاح جميع أخطاء الصيغة"""
    
    # قراءة الملف
    with open('unified_trading_learning_system_server.py', 'r') as f:
        content = f.read()
    
    # إصلاحات محددة
    fixes = [
        # إصلاح if False statements
        (r'if False  # MT5 not on server:', 'if False:  # MT5 not on server'),
        (r'if False:', 'if False:'),
        
        # إصلاح أي أخطاء أخرى في الصيغة
        (r'(\s+)if\s+False\s+#([^:]+)$', r'\1if False:  #\2'),
        (r'(\s+)if\s+False\s*$', r'\1if False:'),
        
        # إصلاح comment-only lines that should be pass
        (r'^(\s*)# mt5\.\w+\(\)$', r'\1pass  # mt5 function'),
        (r'^(\s*)# mt5\.$', r'\1pass  # mt5'),
    ]
    
    for pattern, replacement in fixes:
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    # حفظ الملف المصلح
    with open('unified_trading_learning_system_server.py', 'w') as f:
        f.write(content)
    
    print("✅ تم إصلاح أخطاء الصيغة")

if __name__ == "__main__":
    fix_syntax_errors()