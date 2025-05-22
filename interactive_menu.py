"""
interactive_menu.py
Script a menu-driven CLI to explore insurance calculation insights and predict premiums
"""
import pandas as pd
import numpy as np
import joblib
import sys
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error

# Caricamento risorse (dataset e modello salvato)
try:
    # Carica il dataset preprocessato e il modello salvato
    data = pd.read_csv('train_fe.csv')
    pipeline = joblib.load('baseline_lgbm_pipeline.pkl')
except Exception:
    # Se il caricamento fallisce, imposta data e pipeline a None
    data = None
    pipeline = None

# Definizione bins e mappe per categorizzare alcune feature
AGE_BINS = [0, 25, 35, 45, 55, 65, 100]  # Fasce di età
CREDIT_BINS = [300, 580, 670, 740, 800, 850]  # Fasce di punteggio di credito
VEHICLE_BINS = [0, 1, 3, 5, 10, 20]  # Fasce di età del veicolo
SMOKE_MAP = {'Never': 0, 'Former': 1, 'Current': 2}  # Mappatura dello stato di fumatore
EXERCISE_MAP = {'None': 0, 'Rare': 1, 'Regular': 2, 'Daily': 3}  # Mappatura della frequenza di esercizio

# Utility per calcolo RMSLE (Root Mean Squared Logarithmic Error)
def rmsle(y_true, y_pred):
    # Assicura che le predizioni siano >= 0
    y_pred = np.maximum(0, y_pred)
    # Calcola RMSLE
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

# 1. Riepilogo dataset
def show_dataset_summary():
    """
    Mostra un riepilogo statistico del dataset.
    """
    if data is None:
        print("Dataset non disponibile. Esegui prima i moduli di preprocessing e validazione.")
        return
    print("\n=== Dataset Summary ===")
    # Mostra statistiche descrittive per tutte le colonne
    print(data.describe(include='all').transpose())

# 2. Definizioni bins
def show_feature_bins():
    """
    Mostra le definizioni delle fasce (bins) utilizzate per categorizzare le feature.
    """
    print("\n=== Feature Bins Definitions ===")
    print(f"Age bins: {AGE_BINS}")
    print(f"Credit Score bins: {CREDIT_BINS}")
    print(f"Vehicle Age bins: {VEHICLE_BINS}")

# 3. Logica HealthRisk Index
def show_healthrisk_logic():
    """
    Mostra la logica utilizzata per calcolare l'indice di rischio sanitario (HealthRisk Index).
    """
    print("\n=== Health Risk Index Logic ===")
    print("Smoking mapping:", SMOKE_MAP)
    print("Exercise mapping:", EXERCISE_MAP)
    print("HealthRisk_index = smoking_map - exercise_map")

# 4. Performance modello (CV cross-validation usando pipeline già addestrata su log1p)
def show_model_performance():
    """
    Mostra le performance del modello utilizzando la cross-validation (K-Fold).
    """
    print("\n=== Model Performance ===")
    if pipeline is None or data is None:
        print("Modello o dati non disponibili. Esegui prima preprocessing e validazione.")
        return
    # Prepara X (feature) e y (target)
    X = data.drop(columns=['id', 'Premium Amount'], errors='ignore')
    y = data['Premium Amount']
    # Configura la cross-validation K-Fold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), start=1):
        # Suddivide i dati in training e validation set
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]
        # pipeline.predict restituisce log1p(premium)
        log_preds = pipeline.predict(X_val)
        # Converte le predizioni alla scala originale
        preds = np.expm1(log_preds)
        # Calcola RMSLE per il fold corrente
        score = rmsle(y_val, preds)
        scores.append(score)
        print(f"Fold {fold} RMSLE: {score:.4f}")
    # Calcola media e deviazione standard dei punteggi
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    print(f"Mean RMSLE: {mean_score:.4f} (+/- {std_score:.4f})")

# 5. Predizione premio per nuovo cliente
def predict_premium():
    """
    Permette di inserire i dati di un nuovo cliente e predire il premio assicurativo.
    """
    if pipeline is None or data is None:
        print("Modello o dati non disponibili. Esegui prima preprocessing e validazione.")
        return
    print("\nInserisci i dati del nuovo cliente:")
    # Definizione delle feature numeriche e categoriche
    base_numeric = ['Age', 'Annual Income', 'Number of Dependents',
                    'Health Score', 'Previous Claims', 'Vehicle Age',
                    'Credit Score', 'Insurance Duration']
    base_categorical = ['Gender', 'Marital Status', 'Education Level', 'Occupation',
                        'Location', 'Policy Type', 'Policy Start Date',
                        'Customer Feedback', 'Smoking Status', 'Exercise Frequency', 'Property Type']
    record = {}
    # Richiede input per ogni feature
    for feat in base_numeric + base_categorical:
        val = input(f"{feat}: ")
        if feat in base_numeric:
            try:
                record[feat] = float(val) if val.strip() else np.nan
            except ValueError:
                record[feat] = np.nan
        else:
            record[feat] = val.strip() if val.strip() else np.nan
    # Crea un DataFrame con i dati inseriti
    df = pd.DataFrame([record])
    # Feature engineering inline
    df['Policy Start Date'] = pd.to_datetime(df['Policy Start Date'], errors='coerce')
    df['Policy_Start_Year'] = df['Policy Start Date'].dt.year
    df['Policy_Start_Month'] = df['Policy Start Date'].dt.month
    df['Policy_Start_DayOfWeek'] = df['Policy Start Date'].dt.dayofweek
    df['Policy_Age_Days'] = (pd.Timestamp.now() - df['Policy Start Date']).dt.days
    for feat in base_numeric + base_categorical:
        df[f"{feat}_missing_flag"] = df[feat].isna().astype(int)
    df['Age_bin'] = pd.cut(df['Age'], bins=AGE_BINS, labels=False)
    df['CreditScore_bin'] = pd.cut(df['Credit Score'], bins=CREDIT_BINS, labels=False)
    df['VehicleAge_bin'] = pd.cut(df['Vehicle Age'], bins=VEHICLE_BINS, labels=False)
    smoke = SMOKE_MAP.get(df['Smoking Status'].iloc[0], 0)
    ex = EXERCISE_MAP.get(df['Exercise Frequency'].iloc[0], 0)
    df['HealthRisk_index'] = smoke - ex
    df['Feedback_length'] = df['Customer Feedback'].fillna('').str.len()
    # Predizione del premio assicurativo
    log_pred = pipeline.predict(df[pipeline.named_steps['preprocessor'].feature_names_in_])
    pred = np.expm1(log_pred)
    print(f"Premio stimato: {pred[0]:.2f}\n")

# 6. Visualizzazione di tutti i grafici
def show_all_graphs():
    """
    Mostra tutti i grafici salvati nella directory corrente.
    """
    print("\n=== Visualizzazione Grafici ===")
    # Cerca i file immagine con prefisso 'hist_' o il file 'correlation_heatmap.png'
    images = glob.glob('hist_*.png') + ['correlation_heatmap.png']
    if not images:
        print("Nessun grafico trovato. Esegui il modulo EDA.")
        return
    for img in images:
        print(f"Mostro: {img}")
        try:
            # Carica e mostra l'immagine
            im = plt.imread(img)
            plt.figure(figsize=(8, 6))
            plt.imshow(im)
            plt.axis('off')
            plt.show()
        except Exception as e:
            print(f"Errore nel mostrare {img}: {e}")

# Menù interattivo
def main_menu():
    """
    Mostra il menu interattivo per navigare tra le funzionalità del programma.
    """
    menu = {
        '1': ('Mostra riepilogo dataset', show_dataset_summary),
        '2': ('Mostra definizioni bins', show_feature_bins),
        '3': ('Mostra logica HealthRisk', show_healthrisk_logic),
        '4': ('Mostra performance modello', show_model_performance),
        '5': ('Predici premio per nuovo cliente', predict_premium),
        '6': ('Mostra tutti i grafici', show_all_graphs),
        '0': ('Esci', lambda: sys.exit(0))
    }
    while True:
        print("\n=== Menu Interattivo Assicurazioni ===")
        for key, (desc, _) in menu.items():
            print(f"{key}. {desc}")
        choice = input("Seleziona opzione: ")
        if choice in menu:
            # Esegue la funzione associata alla scelta
            menu[choice][1]()
        else:
            print("Scelta non valida: inserisci un valore tra 0 e 6.")

if __name__ == '__main__':
    main_menu()
