#!/usr/bin/env python3
"""
Add debug logging to server
إضافة سجلات تصحيح للخادم
"""

import os

server_file = "src/mt5_bridge_server_advanced.py"
if os.path.exists(server_file):
    with open(server_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add debug logging after model_key construction
    # Find where model_key is constructed
    lines = content.split('\n')
    new_lines = []
    
    for i, line in enumerate(lines):
        new_lines.append(line)
        
        # Add debug log after model_key assignment
        if 'model_key = f"{symbol}_{model_timeframe}"' in line:
            indent = len(line) - len(line.lstrip())
            debug_line = ' ' * indent + 'logger.info(f"🔍 Model key: {model_key} (symbol={symbol}, timeframe={model_timeframe})")'
            new_lines.append(debug_line)
        
        # Also log when searching for model
        if 'if model_key not in self.predictor.models:' in line:
            indent = len(line) - len(line.lstrip())
            debug_line = ' ' * indent + 'logger.info(f"🔍 Searching for model: {model_key}")'
            new_lines.insert(-1, debug_line)
    
    content = '\n'.join(new_lines)
    
    with open(server_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Added debug logging to server")
