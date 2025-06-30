import pandas as pd
import yfinance as yf
import pandas_ta as ta
import traceback
#indicators
print("--- Starting Minimal Debug Script ---")

try:
    ticker = "GOOGL"
    print(f"--- Downloading data for {ticker} ---")
    # Using auto_adjust=True is modern and correct
    df = yf.download(ticker, start="2022-01-01", end="2023-01-01", auto_adjust=True)

    if df.empty:
        print("Failed to download data.")
    else:
        print("Data downloaded successfully. Columns:", df.columns)

        print("\n--- Applying RSI ---")
        df.ta.rsi(append=True)
        print("RSI applied successfully.")

        print("\n--- Applying MACD ---")
        df.ta.macd(append=True)
        print("MACD applied successfully.")

        print("\n--- Applying BBANDS ---")
        df.ta.bbands(append=True)
        print("BBANDS applied successfully.")

        print("\n--- Applying ATR ---")
        df.ta.atr(append=True)
        print("ATR applied successfully.")

        print("\n\n--- ALL INDICATORS APPLIED SUCCESSFULLY! ---")
        print("Final columns:", df.columns)

except Exception as e:
    print(f"\n--- AN ERROR OCCURRED ---")
    print(f"Error Type: {type(e)}")
    print(f"Error Message: {e}")
    print("\n--- FULL TRACEBACK ---")
    traceback.print_exc()

print("\n--- Script Finished ---")