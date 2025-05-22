import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Modulo 1.3: Statistiche descrittive e EDA approfondita

def main():
    # 1. Caricamento dei dati
    # Carichiamo i dataset di training e test dai file CSV
    train = pd.read_csv('train_with_missing_flags.csv')
    test  = pd.read_csv('test_with_missing_flags.csv')

    # 2. Statistiche descrittive numeriche
    # Selezioniamo le colonne numeriche e calcoliamo le statistiche descrittive
    numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    numeric_stats = train[numeric_cols].describe().T
    print("=== Statistiche Descrittive (Numeric) ===")
    print(numeric_stats)
    # Salviamo le statistiche descrittive in un file CSV
    numeric_stats.to_csv('numeric_stats.csv')

    # 3. Distribuzione variabili numeriche (istogrammi)
    # Creiamo istogrammi per visualizzare la distribuzione di ogni variabile numerica
    for col in numeric_cols:
        plt.figure(figsize=(6,4))
        train[col].hist(bins=30)  # Istogramma con 30 bin
        plt.title(f'Distribution of {col}')  # Titolo del grafico
        plt.xlabel(col)  # Etichetta asse X
        plt.ylabel('Frequency')  # Etichetta asse Y
        plt.tight_layout()  # Migliora il layout
        plt.savefig(f'hist_{col}.png')  # Salviamo il grafico come immagine
        plt.close()  # Chiudiamo la figura per evitare sovrapposizioni

    # 4. Analisi variabili categoriche
    # Selezioniamo le colonne categoriche e analizziamo le frequenze delle categorie
    cat_cols = train.select_dtypes(include=['object']).columns.tolist()
    print("=== Frequenze Top 10 (Categorie) ===")
    freq = {}
    for col in cat_cols:
        # Calcoliamo le 10 categorie più frequenti per ogni colonna
        top = train[col].value_counts().nlargest(10)
        freq[col] = top
        print(f"\n-- {col} --")
        print(top)

    # Salvataggio frequenze in file CSV
    for col, series in freq.items():
        series.to_frame(name='count').to_csv(f'{col}_freq.csv')

    # 5. Matrice di correlazione
    # Calcoliamo la matrice di correlazione per le variabili numeriche
    corr = train[numeric_cols].corr()
    # Salviamo la matrice di correlazione in un file CSV
    corr.to_csv('correlation_matrix.csv')
    # Creiamo una heatmap per visualizzare la matrice di correlazione
    plt.figure(figsize=(10,8))
    plt.matshow(corr, fignum=1)  # Visualizzazione della matrice come immagine
    plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=90)  # Etichette asse X
    plt.yticks(range(len(numeric_cols)), numeric_cols)  # Etichette asse Y
    plt.title('Correlation Matrix')  # Titolo del grafico
    plt.colorbar()  # Barra dei colori per la heatmap
    plt.tight_layout()  # Migliora il layout
    plt.savefig('correlation_heatmap.png')  # Salviamo la heatmap come immagine
    plt.close()  # Chiudiamo la figura

if __name__ == '__main__':
    main()
