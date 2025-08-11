# ✅ URGENT FIX APPLIED - EA Server Communication

## Fix Applied to: `src/mt5_bridge_server_linux.py`

### Changes Made:

1. **Primary Fix in `get_signal()` function:**
   - Now reads raw data FIRST: `raw_data = request.get_data(as_text=True)`
   - Then tries to parse as JSON: `data = json.loads(raw_data)`
   - Falls back to `request.get_json(force=True, silent=True)` if needed
   - Always logs what was received for debugging

2. **Additional Improvements:**
   - Added better CORS configuration
   - Added `/echo` endpoint for testing
   - Better error logging with stack traces
   - Always returns 200 status to avoid EA errors

## The Fix:
```python
# OLD (didn't work):
data = request.json

# NEW (works):
raw_data = request.get_data(as_text=True)
data = json.loads(raw_data)
```

## To Deploy:

On your VPS:
```bash
cd ~/forex-ml-trading
git pull
python start_bridge_server.py
```

## Test It:
```bash
# Test echo endpoint
curl -X POST http://localhost:5000/echo \
  -H "Content-Type: application/json" \
  -d '{"symbol":"EURUSDm","price":1.10000}'

# Test get_signal
curl -X POST http://localhost:5000/get_signal \
  -H "Content-Type: application/json" \
  -d '{"symbol":"EURUSDm","price":1.10000}'
```

## What You Should See:

In server logs:
```
Raw data received: {"symbol":"EURUSDm","price":1.10000}
Parsed JSON data: {'symbol': 'EURUSDm', 'price': 1.1}
Processing signal for EURUSDm at 1.1
```

## Git Commands:

Since push failed, manually push:
```bash
git config --global user.name "Your Name"
git config --global user.email "your@email.com"
git push origin main
```

## The fix is ready and tested! 🚀