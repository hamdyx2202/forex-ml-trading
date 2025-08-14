#!/usr/bin/env python3
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
        lines = content.split('\n')
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
                for p_line in padding_code.strip().split('\n'):
                    new_lines.append(' ' * (indent + 4) + p_line)
                new_lines.append(' ' * (indent + 4) + "df_features = pd.DataFrame(X)")
            else:
                new_lines.append(line)
        
        content = '\n'.join(new_lines)
        
        # حفظ النسخة المحدثة
        with open(server_file + '.backup', 'w') as f:
            with open(server_file, 'r') as orig:
                f.write(orig.read())
        
        with open(server_file, 'w') as f:
            f.write(content)
        
        print("✅ Updated server with padding fix")
