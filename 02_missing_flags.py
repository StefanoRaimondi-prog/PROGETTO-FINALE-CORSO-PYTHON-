import pandas as pd

# 1. Caricamento dei dati già arricchiti dal passo precedente
# Si caricano i dataset di train e test, che sono stati precedentemente arricchiti con informazioni temporali.
train = pd.read_csv('train_with_temporal.csv')
test  = pd.read_csv('test_with_temporal.csv')

# 2. Riepilogo dei missing sul train
# Si crea un riepilogo dei valori mancanti (NaN) nel dataset di train.
# Per ogni colonna, si calcola il numero di valori mancanti (missing_count) e la percentuale di valori mancanti (missing_pct).
# Il risultato viene ordinato in ordine decrescente di percentuale di valori mancanti.
missing_summary = (
    train.isna()  # Verifica quali valori sono NaN
         .sum()  # Conta il numero di NaN per ogni colonna
         .rename('missing_count')  # Rinomina la serie risultante
         .to_frame()  # Converte la serie in un DataFrame
         .assign(missing_pct = lambda df: df['missing_count'] / len(train))  # Aggiunge una colonna con la percentuale di NaN
         .sort_values('missing_pct', ascending=False)  # Ordina il DataFrame in base alla percentuale di NaN
)
print("=== Missing Values Summary (Train) ===")
print(missing_summary)

# 3. Creazione dei flag di missing per ogni colonna che presenta NaN
# Per ogni colonna del dataset di train, si verifica se contiene valori mancanti (NaN).
# Se una colonna contiene NaN, si crea una nuova colonna con il suffisso '_missing_flag'.
# Questa nuova colonna contiene 1 se il valore originale è NaN, altrimenti 0.
# Lo stesso processo viene applicato al dataset di test.
for col in train.columns:
    if train[col].isna().any():  # Controlla se ci sono NaN nella colonna
        train[f'{col}_missing_flag'] = train[col].isna().astype(int)  # Crea il flag per il train
        test[f'{col}_missing_flag']  = test[col].isna().astype(int)  # Crea il flag per il test

# 4. Salvataggio dei nuovi CSV
# Si salvano i dataset di train e test arricchiti con i flag di missing in nuovi file CSV.
train.to_csv('train_with_missing_flags.csv', index=False)  # Salva il train
test.to_csv('test_with_missing_flags.csv',  index=False)  # Salva il test
