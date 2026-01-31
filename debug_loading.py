
import os
import pandas as pd
import sys
import traceback

# Mock DATA_DIR
DATA_DIR = os.path.join(os.getcwd(), 'data')
OUTPUT_FILE = "debug_output.txt"

def log(msg):
    print(msg)
    with open(OUTPUT_FILE, "a") as f:
        f.write(str(msg) + "\n")

try:
    from models.forecaster import forecast_stock
except Exception as e:
    log(f"Import Error: {e}")
    sys.exit(1)

def load_stock_data(symbol):
    """Load stock OHLCV data from CSV (copied logic)"""
    file_patterns = [
        f"{symbol}.csv",
        f"{symbol}.NS.csv",
        f"{symbol.replace('.NS', '')}.csv",
        f"{symbol.upper()}.csv",
        f"{symbol.upper()}.NS.csv",
    ]
    
    for pattern in file_patterns:
        filepath = os.path.join(DATA_DIR, pattern)
        if os.path.exists(filepath):
            log(f"Found {symbol} at {filepath}")
            try:
                df = pd.read_csv(filepath)
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                return df
            except Exception as e:
                log(f"Error reading {filepath}: {e}")
                return None
    log(f"File not found for {symbol}")
    return None

def test_symbol(symbol):
    log(f"\nTesting {symbol}...")
    df = load_stock_data(symbol)
    if df is None:
        log("FAIL: Could not load data")
        return
    
    log(f"Loaded {len(df)} rows")
    try:
        prices = df['Close'].dropna()
        dates = df['Date'] if 'Date' in df.columns else None
        log("Running forecast...")
        result = forecast_stock(prices, dates, 30)
        log("Forecast success!")
        log(f"Recommendation: {result['recommendation']}")
        log(f"Forecast: {result['forecast']['expected_return_pct']:.2f}%")
        
        # Check specific models
        if result.get('arima', {}).get('error'):
            log(f"ARIMA Error: {result['arima']['error']}")
        else:
            log("ARIMA OK")
            
    except Exception as e:
        log(f"FAIL: Forecast error: {e}")
        traceback.print_exc()
        with open(OUTPUT_FILE, "a") as f:
            traceback.print_exc(file=f)

if __name__ == "__main__":
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
    test_symbol("RELIANCE")
    test_symbol("TCS")
    test_symbol("HDFCBANK")
    test_symbol("INFY")
