#!/usr/bin/env python3
"""
Final fix for model loading - uses absolute paths
إصلاح نهائي لتحميل النماذج
"""

import os
import shutil

print("🔧 Final fix for model loading...")

# Fix 1: Update advanced_predictor_95.py
predictor_file = "src/advanced_predictor_95.py"
if os.path.exists(predictor_file):
    with open(predictor_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace relative path with absolute path
    old_line = '        model_dir = Path("models/advanced")'
    new_line = '        model_dir = Path(os.path.abspath("models/advanced"))'
    
    # Also add import if not exists
    if 'import os' not in content:
        content = 'import os\n' + content
    
    content = content.replace(old_line, new_line)
    
    with open(predictor_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Fixed advanced_predictor_95.py")
else:
    print("❌ advanced_predictor_95.py not found!")

# Fix 2: Update the import in server if needed
server_file = "src/mt5_bridge_server_advanced.py"
if os.path.exists(server_file):
    print("✅ Server file exists")
    
    # Make sure advanced_predictor imports os
    with open(server_file, 'r', encoding='utf-8') as f:
        server_content = f.read()
    
    # Check if model loading is happening correctly
    if 'self.predictor = AdvancedPredictor()' in server_content:
        print("✅ Server correctly initializes AdvancedPredictor")
else:
    print("❌ Server file not found!")

# Fix 3: Create a test script to verify models are loadable
test_script = '''#!/usr/bin/env python3
"""
Test model loading
"""
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print(f"Current directory: {os.getcwd()}")
print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

# Check models directory
models_dir = os.path.abspath("models/advanced")
print(f"\\nLooking for models in: {models_dir}")
print(f"Directory exists: {os.path.exists(models_dir)}")

if os.path.exists(models_dir):
    pkl_files = list(Path(models_dir).glob("*.pkl"))
    print(f"\\nFound {len(pkl_files)} model files:")
    for f in pkl_files[:5]:  # Show first 5
        print(f"  • {f.name}")
        
    # Try loading the predictor
    print("\\n🔍 Testing AdvancedPredictor...")
    try:
        from src.advanced_predictor_95 import AdvancedPredictor
        predictor = AdvancedPredictor()
        print(f"✅ Loaded {len(predictor.models)} models")
        print("\\nAvailable models:")
        for key in list(predictor.models.keys())[:5]:  # Show first 5
            print(f"  • {key}")
    except Exception as e:
        print(f"❌ Error loading predictor: {e}")
else:
    print("\\n❌ Models directory not found!")
    print("\\nSearching for .pkl files in current directory tree...")
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith("_ensemble_") and file.endswith(".pkl"):
                print(f"Found: {os.path.join(root, file)}")
                break
'''

with open('test_model_loading.py', 'w') as f:
    f.write(test_script)

print("✅ Created test_model_loading.py")

# Fix 4: Alternative - create symlink if models are elsewhere
print("\n🔍 Checking for models in different locations...")
possible_paths = [
    "/home/forex-ml-trading/models/advanced",
    "../models/advanced",
    "../../models/advanced",
    "/root/models/advanced"
]

for path in possible_paths:
    if os.path.exists(path) and os.path.isdir(path):
        pkl_count = len([f for f in os.listdir(path) if f.endswith('.pkl')])
        if pkl_count > 0:
            print(f"✅ Found {pkl_count} models at: {path}")
            
            # Create symlink if not exists
            if not os.path.exists("models/advanced") and path != "models/advanced":
                os.makedirs("models", exist_ok=True)
                os.symlink(os.path.abspath(path), "models/advanced")
                print(f"✅ Created symlink: models/advanced -> {path}")
            break

print("\n✅ Done! Now:")
print("1. python test_model_loading.py")
print("2. Restart server: python src/mt5_bridge_server_advanced.py")