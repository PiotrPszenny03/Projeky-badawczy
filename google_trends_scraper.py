"""
Google Trends & Yahoo Finance Data Scraper (Weekly Data)
Okres badawczy: 2021-01-01 do 2025-12-31
Interwał: Tygodniowy (1wk)

Skrypt pobiera notowania giełdowe (yfinance) oraz dane o popularności haseł (pytrends).
Wyniki są zapisywane do odpowiednich folderów w formacie CSV z uwzględnieniem polskich znaków (utf-8-sig).
Zaimplementowano exponential backoff dla zapytań do Google Trends oraz logowanie zdarzeń (FAIR principles).
"""

import os
import csv
import time
import random
import pandas as pd
import yfinance as yf
from datetime import datetime
from pytrends.request import TrendReq
from requests.exceptions import RequestException

# ----------------- KONFIGURACJA -----------------
START_DATE_STR = '2021-01-01'
END_DATE_STR = '2025-12-31'
GT_TIMEFRAME = f"{START_DATE_STR} {END_DATE_STR}"

DIR_TRENDS = "01_google_trends"
DIR_FINANCE = "02_yahoo_finance"
DIR_LOGS = "03_logs"
LOG_FILE = os.path.join(DIR_LOGS, "download_log.csv")

# Struktura badawcza
RESEARCH_PAIRS = [
    {
        'pair_id': 1,
        'tickers': ['SPY', 'IWM'],
        'gt_region': 'US',
        'gt_keywords': ['stock market', 'buy stocks', 'S&P 500', 'Russell 2000']
    },
    {
        'pair_id': 2,
        'tickers': ['AAPL', 'PLTR'],
        'gt_region': 'US',
        'gt_keywords': ['tech stocks', 'buy shares', 'apple stock', 'palantir stock']
    },
    {
        'pair_id': 3,
        'tickers': ['WIG20.WA', 'SWIG80.WA'],
        'gt_region': 'PL',
        'gt_keywords': ['giełda', 'XTB', 'wig20', 'swig80']
    },
    {
        'pair_id': 4,
        'tickers': ['BTC-USD', 'DOGE-USD'],
        'gt_region': '',
        'gt_keywords': ['Binance', 'bitcoin price', 'bitcoin', 'dogecoin']
    },
    {
        'pair_id': 5,
        'tickers': ['XLF', 'ARKK'],
        'gt_region': 'US',
        'gt_keywords': ['growth stocks', 'dividend stocks', 'cathie wood', 'bank stocks']
    }
]
# ------------------------------------------------

def setup_directories():
    """Tworzy niezbędne foldery wyjściowe."""
    os.makedirs(DIR_TRENDS, exist_ok=True)
    os.makedirs(DIR_FINANCE, exist_ok=True)
    os.makedirs(DIR_LOGS, exist_ok=True)

def log_download(asset_type, identifier, region, status, error_msg=""):
    """Zapisuje zdarzenie pobierania do pliku logu."""
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'type', 'identifier', 'region', 'timeframe', 'status', 'error_msg'])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            asset_type, identifier, region, GT_TIMEFRAME, status, error_msg
        ])

def fetch_yfinance_data(ticker):
    """Pobiera historyczne notowania giełdowe z Yahoo Finance z interwałem tygodniowym."""
    print(f"  -> Pobieranie [YFinance]: {ticker}")
    try:
        data = yf.download(ticker, start=START_DATE_STR, end=END_DATE_STR, interval='1wk', progress=False)
        
        if isinstance(data.columns, pd.MultiIndex):
            # Uproszczenie MultiIndex columns od nowszych wersji yfinance
            if ticker in data.columns.get_level_values(1):
                data = data.xs(ticker, level=1, axis=1)
            else:
                data.columns = [col[0] for col in data.columns]
                
        if data.empty:
            print(f"     [!] Brak danych finansowych dla {ticker}.")
            log_download("YFinance", ticker, "Global", "Brak danych")
            return None
        
        # Zachowanie oczekiwanych kolumn
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        available_columns = [col for col in expected_columns if col in data.columns]
        data = data[available_columns]
        
        safe_ticker = ticker.replace('.', '_')
        filepath = os.path.join(DIR_FINANCE, f"{safe_ticker}_weekly.csv")
        data.to_csv(filepath, encoding='utf-8-sig')
        log_download("YFinance", ticker, "Global", "Sukces")
        return data
    except Exception as e:
        print(f"     [!] Błąd pobierania {ticker}: {e}")
        log_download("YFinance", ticker, "Global", "Błąd", str(e))
        return None

def fetch_google_trends_with_backoff(pytrends, keyword, region, max_retries=6):
    """Pobiera dane z Google Trends używając Exponential Backoff."""
    for attempt in range(max_retries):
        try:
            print(f"  -> Pobieranie [PyTrends]: '{keyword}' | Próba {attempt+1}")
            pytrends.build_payload(kw_list=[keyword], timeframe=GT_TIMEFRAME, geo=region)
            df = pytrends.interest_over_time()
            
            # Losowy sleep aby unikać blokad
            sleep_time = random.uniform(5, 15)
            time.sleep(sleep_time)
            
            if df.empty:
                print(f"     [!] Pusta odpowiedź dla '{keyword}'.")
                return df
                
            if 'isPartial' in df.columns:
                df = df.drop(columns=['isPartial'])
            return df
            
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "400" in error_str or isinstance(e, RequestException):
                wait_time = (2 ** attempt) * 15 + random.uniform(5, 15)
                print(f"     [!] API Rate limit / Timeout. Oczekiwanie {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                print(f"     [!] Krytyczny błąd pobierania '{keyword}': {e}")
                log_download("PyTrends", keyword, region, "Błąd", str(e))
                return None
                
    print(f"     [!] Przekroczono limit prób dla '{keyword}'.")
    log_download("PyTrends", keyword, region, "Błąd", "Przekroczono limit prób")
    return None

def process_research_pairs():
    """Główna funkcja orkiestrująca pobieranie danych finansowych i z Google Trends."""
    setup_directories()
    
    # Inicjalizacja klienta pytrends
    pytrends = TrendReq(hl='en-US', tz=360, timeout=(10,25))
    
    for pair in RESEARCH_PAIRS:
        print(f"\n=== PRZETWARZANIE PARY {pair['pair_id']} ===")
        
        # 1. Pobieranie danych finansowych
        print("--- Dane finansowe (yfinance) ---")
        for ticker in pair['tickers']:
            fetch_yfinance_data(ticker)
            
        # 2. Pobieranie danych Google Trends
        print("\n--- Dane Google Trends (pytrends) ---")
        region = pair['gt_region']
        region_label = region if region else 'Global'
        
        for keyword in pair['gt_keywords']:
            safe_keyword = keyword.replace(' ', '_')
            filepath = os.path.join(DIR_TRENDS, f"{safe_keyword}_{region_label}_weekly.csv")
            
            # Pomiń jeśli plik już istnieje (prosty cache)
            if os.path.exists(filepath):
                print(f"  -> Cache znaleziony dla: '{keyword}'")
                continue
                
            df = fetch_google_trends_with_backoff(pytrends, keyword, region)
            
            if df is not None and not df.empty:
                df.to_csv(filepath, encoding='utf-8-sig')
                log_download("PyTrends", keyword, region_label, "Sukces")
            elif df is not None and df.empty:
                log_download("PyTrends", keyword, region_label, "Puste dane")

def main():
    try:
        process_research_pairs()
        print("\n[ZAKOŃCZONO] Proces pobierania przebiegł pomyślnie. Sprawdź logi w folderze 03_logs.")
    except Exception as e:
        print(f"\n[BŁĄD KRYTYCZNY] Wystąpił nieoczekiwany błąd: {e}")

if __name__ == "__main__":
    main()
