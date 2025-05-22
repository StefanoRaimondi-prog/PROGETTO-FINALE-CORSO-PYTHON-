import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import mean_squared_log_error
import lightgbm as lgb
import joblib

# Funzione per calcolare la metrica RMSLE (Root Mean Squared Logarithmic Error)
def rmsle(y_true, y_pred):
    # Assicuriamoci che le predizioni non siano negative
    y_pred = np.maximum(0, y_pred)
    # Calcoliamo la radice quadrata dell'errore quadratico logaritmico medio
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

def main():
    # 1. Caricamento del dataset con feature engineering
    # Leggiamo il dataset preprocessato (train_fe.csv) che contiene le feature già ingegnerizzate
    data = pd.read_csv('train_fe.csv')
    # Separiamo le feature (X) dal target (y)
    X = data.drop(columns=['Premium Amount', 'id'])  # Rimuoviamo la colonna target e l'id
    y = data['Premium Amount']  # La colonna target è il premio assicurativo

    # 2. Identificazione colonne numeriche e categoriche
    # Identifichiamo le colonne numeriche
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    # Identifichiamo le colonne categoriche
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # 3. Preprocessing pipelines
    # Pipeline per il preprocessing delle colonne numeriche
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),  # Sostituiamo i valori mancanti con la mediana
        ('scaler', StandardScaler())  # Standardizziamo i valori numerici
    ])
    # Pipeline per il preprocessing delle colonne categoriche
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # Sostituiamo i valori mancanti con "missing"
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))  # Codifichiamo le categorie in valori ordinali
    ])
    # Combiniamo entrambe le pipeline in un unico preprocessore
    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, numeric_cols),  # Applichiamo la pipeline numerica alle colonne numeriche
        ('cat', categorical_pipeline, categorical_cols)  # Applichiamo la pipeline categorica alle colonne categoriche
    ])

    # 4. Setup LightGBM Regressor
    # Configuriamo il modello LightGBM con iperparametri di base
    model = lgb.LGBMRegressor(
        n_estimators=1000,  # Numero massimo di alberi
        learning_rate=0.05,  # Tasso di apprendimento
        random_state=42  # Fissiamo il seed per la riproducibilità
    )

    # 5. K-Fold Validation
    # Configuriamo la validazione incrociata con 5 fold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)  # Shuffle per mescolare i dati
    rmsle_scores = []  # Lista per salvare gli score RMSLE di ogni fold

    # Ciclo sui fold della validazione incrociata
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
        # Dividiamo i dati in training e validation per il fold corrente
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Creiamo una pipeline completa che include il preprocessore e il modello
        clf = Pipeline([
            ('preprocessor', preprocessor),  # Preprocessing dei dati
            ('regressor', model)  # Modello LightGBM
        ])
        # Addestriamo il modello sui dati di training (applichiamo log1p al target per gestire la scala logaritmica)
        clf.fit(X_train, np.log1p(y_train))

        # Prediciamo sui dati di validation e applichiamo expm1 per riportare i valori alla scala originale
        y_pred = np.expm1(clf.predict(X_val))
        # Calcoliamo l'RMSLE per il fold corrente
        score = rmsle(y_val, y_pred)
        print(f"Fold {fold} RMSLE: {score:.4f}")  # Stampiamo il risultato del fold
        rmsle_scores.append(score)  # Salviamo il risultato

    # 6. Metriche aggregate
    # Calcoliamo la media e la deviazione standard degli score RMSLE
    mean_score = np.mean(rmsle_scores)
    std_score = np.std(rmsle_scores)
    print(f"\nMean RMSLE: {mean_score:.4f} (+/- {std_score:.4f})")  # Stampiamo i risultati aggregati

    # 7. Fit finale e salvataggio del modello completo
    # Addestriamo il modello finale su tutto il dataset
    clf_full = Pipeline([
        ('preprocessor', preprocessor),  # Preprocessing dei dati
        ('regressor', model)  # Modello LightGBM
    ])
    clf_full.fit(X, np.log1p(y))  # Addestriamo il modello sull'intero dataset
    # Salviamo il modello addestrato in un file .pkl
    joblib.dump(clf_full, 'baseline_lgbm_pipeline.pkl')

if __name__ == '__main__':
    main()
