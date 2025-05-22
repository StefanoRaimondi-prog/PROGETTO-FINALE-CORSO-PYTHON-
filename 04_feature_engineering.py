import pandas as pd
import numpy as np

# Modulo 1.4: Feature Engineering Addizionale

def main():
    # 1. Caricamento dati con temporal features e missing flags
    # Carichiamo i dataset di training e test che contengono già le feature temporali e i flag per i valori mancanti.
    train = pd.read_csv('train_with_missing_flags.csv')
    test  = pd.read_csv('test_with_missing_flags.csv')

    # 2. Binning di alcune variabili numeriche
    # Creiamo intervalli (bin) per categorizzare alcune variabili numeriche in gruppi.
    # Questo aiuta a ridurre la granularità dei dati e a catturare pattern più generali.
    bins_age = [0, 25, 35, 45, 55, 65, 100]  # Intervalli per l'età
    bins_credit = [300, 580, 670, 740, 800, 850]  # Intervalli per il punteggio di credito
    bins_vehicle = [0, 1, 3, 5, 10, 20]  # Intervalli per l'età del veicolo

    # Applichiamo il binning ai dataset di training e test per le variabili selezionate.
    train['Age_bin'] = pd.cut(train['Age'], bins=bins_age, labels=False)
    test['Age_bin']  = pd.cut(test['Age'],  bins=bins_age, labels=False)

    train['CreditScore_bin'] = pd.cut(train['Credit Score'], bins=bins_credit, labels=False)
    test['CreditScore_bin']  = pd.cut(test['Credit Score'],  bins=bins_credit, labels=False)

    train['VehicleAge_bin'] = pd.cut(train['Vehicle Age'], bins=bins_vehicle, labels=False)
    test['VehicleAge_bin']  = pd.cut(test['Vehicle Age'],  bins=bins_vehicle, labels=False)

    # 3. Composite Health Risk: Smoking & Exercise
    # Creiamo un indice di rischio per la salute basato sullo stato di fumatore e sulla frequenza di esercizio fisico.
    # Mappiamo i valori testuali di "Smoking Status" e "Exercise Frequency" su valori numerici.
    smoking_map = {'Never': 0, 'Former': 1, 'Current': 2}  # Mappatura per lo stato di fumatore
    exercise_map = {'None': 0, 'Rare': 1, 'Regular': 2, 'Daily': 3}  # Mappatura per la frequenza di esercizio

    # Applichiamo la mappatura ai dataset di training e test, sostituendo i valori mancanti con 0.
    train_smoke = train['Smoking Status'].map(smoking_map).fillna(0).astype(int)
    test_smoke  = test['Smoking Status'].map(smoking_map).fillna(0).astype(int)

    train_ex = train['Exercise Frequency'].map(exercise_map).fillna(0).astype(int)
    test_ex  = test['Exercise Frequency'].map(exercise_map).fillna(0).astype(int)

    # Calcoliamo l'indice di rischio per la salute come differenza tra lo stato di fumatore e la frequenza di esercizio.
    train['HealthRisk_index'] = train_smoke - train_ex
    test['HealthRisk_index']  = test_smoke  - test_ex

    # 4. Customer Feedback: text length
    # Creiamo una nuova feature che misura la lunghezza del feedback del cliente (numero di caratteri).
    # Se il feedback è mancante (NaN), lo sostituiamo con una stringa vuota prima di calcolarne la lunghezza.
    train['Feedback_length'] = train['Customer Feedback'].fillna('').str.len()
    test['Feedback_length']  = test['Customer Feedback'].fillna('').str.len()

    # 5. Salvataggio dei dataset con feature engineered
    # Salviamo i dataset di training e test con le nuove feature aggiunte in file CSV.
    train.to_csv('train_fe.csv', index=False)
    test.to_csv('test_fe.csv',  index=False)

if __name__ == '__main__':
    main()
