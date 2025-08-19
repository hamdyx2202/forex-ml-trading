#!/bin/bash
# 🔍 فحص أسباب رفض الصفقات (HOLD)

echo "🔍 Analyzing HOLD/rejection reasons..."
echo "================================"

# البحث عن HOLD مع السياق
echo -e "\n📊 Recent HOLD decisions with context:"
grep -B5 -A2 "HOLD" enhanced_ml_server.log | tail -50

echo -e "\n❓ Common rejection patterns:"
echo -e "\n1️⃣ Low Confidence (<65%):"
grep -E "confidence.*0\.[0-5]" enhanced_ml_server.log | tail -5

echo -e "\n2️⃣ Weak Market Score (<20):"
grep -E "Score=(-)?[0-1]?[0-9][^0-9]" enhanced_ml_server.log | tail -5

echo -e "\n3️⃣ News Time:"
grep "News time" enhanced_ml_server.log | tail -5

echo -e "\n4️⃣ High Volatility:"
grep "high volatility" enhanced_ml_server.log | tail -5

echo -e "\n5️⃣ No Models:"
grep -E "No models|models not found" enhanced_ml_server.log | tail -5

echo -e "\n📈 Summary:"
echo "Total HOLDs: $(grep -c '"action": 2' enhanced_ml_server.log)"
echo "Total BUYs:  $(grep -c '"action": 0' enhanced_ml_server.log)"
echo "Total SELLs: $(grep -c '"action": 1' enhanced_ml_server.log)"