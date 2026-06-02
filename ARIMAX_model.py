# ==============================================================================
# Modelowanie Ekonometryczne (ARIMAX) - Wpływ Google Trends na błąd predykcji
# ==============================================================================
# Instrukcja dla Google Colab:
# Odkomentuj poniższą linię, aby zainstalować wymagane biblioteki przed startem:
# !pip install pmdarima pandas numpy scikit-learn

import os
import glob
import math
import warnings
import numpy as np
import pandas as pd
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

# Katalog z przygotowanymi plikami CSV (W Colab zmień np. na '/content/')
DATA_DIR = "03_prepared_data"

# Klasyfikacja aktywów (Instytucjonalne vs Detaliczne)
# Założenie: Pierwsze aktywo w badanej parze to Instytucjonalne, drugie to Detaliczne
ASSET_TYPES = {
    'SPY': 'Instytucjonalne', 'IWM': 'Detaliczne',
    'AAPL': 'Instytucjonalne', 'PLTR': 'Detaliczne',
    'WIG20': 'Instytucjonalne', 'sWIG80': 'Detaliczne',
    'BTC': 'Instytucjonalne', 'DOGE': 'Detaliczne',
    'XLF': 'Instytucjonalne', 'ARKK': 'Detaliczne'
}

def get_asset_type(asset_name):
    """Pomocnicza funkcja do pobierania typu aktywa na podstawie słownika."""
    return ASSET_TYPES.get(asset_name, "Nieznany")

def calculate_rmse(y_true, y_pred):
    """Zwraca pierwiastek z błędu średniokwadratowego (RMSE)."""
    return math.sqrt(mean_squared_error(y_true, y_pred))

def evaluate_models():
    """
    Główna funkcja wykonująca pętlę po parach aktywów, trenująca modele benchmarkowe
    oraz eksperymentalne ARIMAX dla poszczególnych zmiennych Google Trends.
    """
    all_results = []
    
    # Znajdź wszystkie pliki z danymi
    csv_files = glob.glob(os.path.join(DATA_DIR, "*_ARIMA_Ready.csv"))
    
    if not csv_files:
        print(f"Nie znaleziono plików CSV w katalogu {DATA_DIR}!")
        return pd.DataFrame()
        
    for filepath in csv_files:
        filename = os.path.basename(filepath)
        group_name = filename.replace("_ARIMA_Ready.csv", "")
        print(f"\n{'='*60}\nRozpoczynam modelowanie dla grupy: {group_name}\n{'='*60}")
        
        # Wczytanie danych
        df = pd.read_csv(filepath)
        
        # Ustawienie indeksu czasowego (zawsze pierwsza kolumna z datą)
        date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
        df.index.name = 'Date'
        df.sort_index(inplace=True)
        
        # Znajdowanie docelowych kolumn
        log_return_cols = [c for c in df.columns if c.endswith("_Log_Return")]
        gt_cols = [c for c in df.columns if c.startswith("GT_")]
        
        # -------------------------------------------------------------
        # 1. LAGGING (OPÓŹNIENIE) DANYCH GOOGLE TRENDS
        # -------------------------------------------------------------
        # Aby zapobiec "data leakage" (wyciekowi danych z przyszłości), 
        # opóźniamy dane GT o 1 okres, żeby na ich podstawie prognozować
        for gt_col in gt_cols:
            df[gt_col] = df[gt_col].shift(1)
            
        # 2. Usuwamy rzędy z wartościami NaN powstałymi po opóźnieniu
        df.dropna(inplace=True)
        
        # -------------------------------------------------------------
        # 3. TRAIN / TEST SPLIT (STATYCZNY)
        # -------------------------------------------------------------
        # Zbiór Treningowy (Train) = do końca roku 2024
        # Zbiór Testowy (Test) = od początku roku 2025
        train_df = df.loc[:'2024-12-31']
        test_df = df.loc['2025-01-01':]
        
        if len(train_df) == 0 or len(test_df) == 0:
            print(f"Brak wystarczającej ilości danych po podziale dla {group_name}. Pomijam.")
            continue
            
        for asset_col in log_return_cols:
            asset_name = asset_col.replace("_Log_Return", "")
            asset_type = get_asset_type(asset_name)
            
            y_train = train_df[asset_col]
            y_test = test_df[asset_col]
            
            print(f"\n--- Aktywo: {asset_name} ({asset_type}) ---")
            
            # =========================================================
            # MODEL BAZOWY (BENCHMARK) - CZYSTE ARIMA
            # =========================================================
            try:
                # Szukamy optymalnych parametrów na zbiorze Train
                base_model = auto_arima(
                    y_train,
                    X=None,
                    stationary=True,
                    seasonal=False,
                    suppress_warnings=True,
                    error_action="ignore"
                )
                
                # Prognoza dla zbioru Testowego (OOS)
                forecast_base = base_model.predict(n_periods=len(y_test), X=None)
                rmse_base = calculate_rmse(y_test, forecast_base)
                
            except Exception as e:
                print(f"[!] Błąd w modelu bazowym dla {asset_name}: {e}")
                continue
                
            # =========================================================
            # MODELE EKSPERYMENTALNE (ARIMAX) DLA KAŻDEGO HASŁA GT
            # =========================================================
            for gt_col in gt_cols:
                gt_keyword = gt_col.replace("GT_", "").replace("_weekly", "")
                
                X_train = train_df[[gt_col]]
                X_test = test_df[[gt_col]]
                
                try:
                    gt_model = auto_arima(
                        y_train,
                        X=X_train,
                        stationary=True,
                        seasonal=False,
                        suppress_warnings=True,
                        error_action="ignore"
                    )
                    
                    forecast_gt = gt_model.predict(n_periods=len(y_test), X=X_test)
                    rmse_gt = calculate_rmse(y_test, forecast_gt)
                    
                    # Różnica predykcyjna (Improvement Ratio)
                    improvement = ((rmse_base - rmse_gt) / rmse_base) * 100
                    
                    # Tekstowe podsumowanie wyników w konsoli
                    if improvement > 0:
                        wynik_str = f"zmniejszyło błąd predykcji o {improvement:.2f}%"
                    else:
                        wynik_str = f"zwiększyło błąd predykcji o {abs(improvement):.2f}%"
                        
                    print(f"Dla {asset_name} ({asset_type}) dodanie hasła '{gt_keyword}' {wynik_str}")
                    
                    # Dodanie wyników do głównej puli
                    all_results.append({
                        "Group": group_name,
                        "Asset": asset_name,
                        "Asset_Type": asset_type,
                        "GT_Keyword": gt_keyword,
                        "RMSE_Base": rmse_base,
                        "RMSE_GT": rmse_gt,
                        "Improvement_%": improvement
                    })
                    
                except Exception as e:
                    print(f"[!] Błąd modelu ARIMAX dla {asset_name} + {gt_keyword}: {e}")
                    
    # =========================================================
    # ZAPIS WYNIKÓW I PODSUMOWANIE
    # =========================================================
    results_df = pd.DataFrame(all_results)
    
    if not results_df.empty:
        results_path = "ARIMAX_Results.csv"
        results_df.to_csv(results_path, index=False)
        print("\n" + "="*60)
        print("PODSUMOWANIE KOŃCOWE")
        print("="*60)
        print(f"[SUKCES] Zapisano zestawienie wyników do pliku: '{results_path}'")
        
        print("\n[TOP 5] Hasła przynoszące największą poprawę predykcji:")
        top5 = results_df.sort_values(by="Improvement_%", ascending=False).head(5)
        for _, row in top5.iterrows():
            print(f"- {row['Asset']} ({row['Asset_Type']}) | Hasło: {row['GT_Keyword']} | Poprawa: {row['Improvement_%']:.2f}%")
            
    return results_df

if __name__ == "__main__":
    results = evaluate_models()
