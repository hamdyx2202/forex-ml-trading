#!/bin/bash
echo "🚀 Quick Installation for Server"
echo "================================"

# Check if we can use pip3
if command -v pip3 &> /dev/null; then
    echo "✅ pip3 found, installing packages..."
    pip3 install pandas numpy scikit-learn flask --user
elif command -v python3 -m pip &> /dev/null; then
    echo "✅ Using python3 -m pip..."
    python3 -m pip install pandas numpy scikit-learn flask --user
else
    echo "❌ pip not found. Installing pip..."
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python3 get-pip.py --user
    ~/.local/bin/pip install pandas numpy scikit-learn flask --user
fi

echo "✅ Installation complete!"
echo "Now run: python3 run_complete_system.py"