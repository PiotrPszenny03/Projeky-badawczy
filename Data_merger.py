import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import os

# Konfiguracja wszystkich par, przypisanie plików giełdowych i plików z Google Trends
PAIRS = {
    "1_Indeksy_US": {
        "assets": {"SPY": "SPY_weekly.csv", "IWM": "IWM_weekly.csv"},
        "finance_date_col": "Date",
        "finance_close_col": "Close",
        "trends": ["S&P_500_US_weekly.csv", "Russell_2000_US_weekly.csv", "stock_market_US_weekly.csv", "buy_stocks_US_weekly.csv"]
    },
    "2_Spolki_US": {
        "assets": {"AAPL": "AAPL_weekly.csv", "PLTR": "PLTR_weekly.csv"},
        "finance_date_col": "Date",
        "finance_close_col": "Close",
        "trends": ["apple_stock_US_weekly.csv", "palantir_stock_US_weekly.csv", "tech_stocks_US_weekly.csv", "buy_shares_US_weekly.csv"]
    },
    "3_Indeksy_PL": {
        "assets": {"WIG20": "wig20_w.csv", "sWIG80": "swig80_w.csv"},
        "finance_date_col": "Data", # Zwróć uwagę, polskie pliki mają "Data" i "Zamkniecie"
        "finance_close_col": "Zamkniecie",
        "trends": ["wig20_PL_weekly.csv", "giełda_PL_weekly.csv", "XTB_PL_weekly.csv", "swig80_PL_weekly.csv"]
    },
    "4_Krypto": {
        "assets": {"BTC": "BTC-USD_weekly.csv", "DOGE": "DOGE-USD_weekly.csv"},
        "finance_date_col": "Date",
        "finance_close_col": "Close",
        "trends": ["Binance_Global_weekly.csv", "bitcoin_Global_weekly.csv", "dogecoin_Global_weekly.csv", "bitcoin_price_Global_weekly.csv"]
    },
    "5_Sektory_US": {
        "assets": {"XLF": "XLF_weekly.csv", "ARKK": "ARKK_weekly.csv"},
        "finance_date_col": "Date",
        "finance_close_col": "Close",
        "trends": ["bank_stocks_US_weekly.csv", "dividend_stocks_US_weekly.csv", "growth_stocks_US_weekly.csv", "cathie_wood_US_weekly.csv"]
    }
}

def run_adf(series, name):
    """Pomocnicza funkcja do przeprowadzania testu stacjonarności ADF"""
    series_clean = series.dropna() # ADF nie znosi NaN
    if len(series_clean) > 10:
        result = adfuller(series_clean)
        p_value = result[1]
        is_stationary = "STACJONARNE" if p_value < 0.05 else "NIESTACJONARNE (!)"
        print(f"   [ADF] {name}: p-value = {p_value:.4f} -> {is_stationary}")
    else:
        print(f"   [ADF] {name}: Zbyt mało danych do testu.")

def process_all_pairs():
    for pair_name, config in PAIRS.items():
        print(f"\n{'='*50}\nPrzetwarzanie pary: {pair_name}\n{'='*50}")

        merged_df = pd.DataFrame()

        # 1. PRZETWARZANIE DANYCH FINANSOWYCH
        for asset_name, file_name in config["assets"].items():
            try:
                finance_path = os.path.join("02_yahoo_finance", file_name)
                df = pd.read_csv(finance_path)
                # Konwersja daty i index
                df[config["finance_date_col"]] = pd.to_datetime(df[config["finance_date_col"]])
                df.set_index(config["finance_date_col"], inplace=True)

                # Resampling do końca tygodnia (Niedziela)
                df = df.resample('W-SUN').last()

                # Wyciągamy tylko kolumnę z ceną zamknięcia, zmieniamy nazwę
                df_close = df[[config["finance_close_col"]]].rename(columns={config["finance_close_col"]: f"{asset_name}_Close"})

                # Łączymy do głównego DataFrame dla pary
                if merged_df.empty:
                    merged_df = df_close
                else:
                    merged_df = merged_df.join(df_close, how='outer')
            except FileNotFoundError:
                print(f"   [BŁĄD] Nie znaleziono pliku {finance_path}")

        # 2. PRZETWARZANIE DANYCH GOOGLE TRENDS
        for gt_file in config["trends"]:
            try:
                gt_path = os.path.join("01_google_trends", gt_file)
                # Zazwyczaj pytrends wypluwa datę jako 'date' i wartość jako nazwa hasła, założymy wczytywanie wszystkiego
                df_gt = pd.read_csv(gt_path)
                # Zakładamy że pierwsza kolumna to data, druga to wartość
                date_col_gt = df_gt.columns[0]
                val_col_gt = df_gt.columns[1]

                df_gt[date_col_gt] = pd.to_datetime(df_gt[date_col_gt])
                df_gt.set_index(date_col_gt, inplace=True)

                # Resampling do niedzieli, by na 100% zgrało się z finansowymi (choć GT zazwyczaj jest już niedzielne)
                df_gt = df_gt.resample('W-SUN').last()

                # Zmiana nazwy kolumny na nazwę pliku/hasła (bez .csv)
                col_name = f"GT_{gt_file.replace('.csv', '')}"
                df_gt = df_gt[[val_col_gt]].rename(columns={val_col_gt: col_name})

                merged_df = merged_df.join(df_gt, how='outer')
            except FileNotFoundError:
                print(f"   [BŁĄD] Nie znaleziono pliku {gt_path}")

        if merged_df.empty:
            continue

        # 3. CZYSZCZENIE BRAKÓW (Zgodnie z instrukcją Kierownika)
        merged_df.interpolate(method='linear', inplace=True)
        merged_df.ffill(inplace=True)
        merged_df.bfill(inplace=True)

        # 4. OBLICZANIE LOG ZWROTÓW I ZMIENNOŚCI DLA AKTYWÓW
        for asset_name in config["assets"].keys():
            close_col = f"{asset_name}_Close"
            if close_col in merged_df.columns:
                # Log Returns
                merged_df[f"{asset_name}_Log_Return"] = np.log(merged_df[close_col] / merged_df[close_col].shift(1))
                # Volatility 4W
                merged_df[f"{asset_name}_Vol_4W"] = merged_df[f"{asset_name}_Log_Return"].rolling(window=4).std()

        # Usuwamy pierwsze rzędy, które obcięło przez .shift i .rolling
        merged_df.dropna(inplace=True)

        # 5. TESTY ADF
        print("--- Wyniki Testu ADF ---")
        for col in merged_df.columns:
            # Pomijamy surowe ceny przy testowaniu, interesują nas zwroty, zmienność i GT
            if "Close" not in col:
                run_adf(merged_df[col], col)

        # 6. ZAPIS DO CSV
        os.makedirs("03_prepared_data", exist_ok=True)
        output_name = os.path.join("03_prepared_data", f"{pair_name}_ARIMA_Ready.csv")
        merged_df.to_csv(output_name)
        print(f"\n[SUKCES] Wygenerowano plik: {output_name}")

if __name__ == "__main__":
    process_all_pairs()