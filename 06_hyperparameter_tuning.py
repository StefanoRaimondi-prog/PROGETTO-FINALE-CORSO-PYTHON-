import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import mean_squared_log_error
import lightgbm as lgb
import joblib

# Modulo 2.2: Ottimizzazione Iperparametri semplificata con Optuna

# Funzione per calcolare la metrica RMSLE (Root Mean Squared Logarithmic Error)
# Questa metrica è utile per problemi di regressione dove si vuole penalizzare
# maggiormente errori relativi piuttosto che assoluti.
def rmsle(y_true, y_pred):
    # Impedisce valori negativi nei predittori
    y_pred = np.maximum(0, y_pred)
    # Calcola la radice quadrata dell'errore quadratico logaritmico medio
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


def main():
    # Caricamento del dataset preprocessato
    # Il file 'train_fe.csv' contiene i dati di addestramento con feature engineering già applicato
    data = pd.read_csv('train_fe.csv')
    # Separazione delle feature (X) e del target (y)
    X = data.drop(columns=['Premium Amount', 'id'])  # Rimuove la colonna target e l'id
    y = data['Premium Amount']  # Colonna target

    # Identificazione delle colonne numeriche e categoriche
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()  # Colonne numeriche
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()  # Colonne categoriche

    # Creazione del preprocessore per il pipeline di trasformazione
    # Le colonne numeriche vengono imputate con la mediana e scalate
    # Le colonne categoriche vengono imputate con un valore costante ('missing') e codificate ordinalmente
    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_cols),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))]), categorical_cols)
    ])
    # Applica il preprocessore ai dati
    X_proc = preprocessor.fit_transform(X)

    # Funzione obiettivo per Optuna
    # Questa funzione definisce il problema di ottimizzazione, ovvero minimizzare l'RMSLE
    def objective(trial):
        # Definizione degli iperparametri da ottimizzare
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300, step=100),  # Numero di alberi
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),  # Tasso di apprendimento
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),  # Numero massimo di foglie per albero
            'max_depth': trial.suggest_int('max_depth', 3, 8),  # Profondità massima dell'albero
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 100),  # Minimo numero di dati in una foglia
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),  # Regolarizzazione L1
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),  # Regolarizzazione L2
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),  # Percentuale di feature usate
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),  # Percentuale di dati usati per il bagging
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 5)  # Frequenza del bagging
        }
        # Creazione di un K-Fold cross-validator con 3 split
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        scores = []  # Lista per memorizzare gli score RMSLE
        # Ciclo sui fold del K-Fold
        for train_idx, val_idx in kf.split(X_proc):
            # Divisione dei dati in train e validation
            X_train, X_val = X_proc[train_idx], X_proc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            # Creazione del modello LightGBM con i parametri suggeriti
            model = lgb.LGBMRegressor(**params, random_state=42)
            # Addestramento del modello sui dati di training (log1p per gestire target skewed)
            model.fit(X_train, np.log1p(y_train))
            # Predizione sui dati di validation (expm1 per riportare i valori alla scala originale)
            preds = np.expm1(model.predict(X_val))
            # Calcolo dello score RMSLE e aggiunta alla lista
            scores.append(rmsle(y_val, preds))
        # Restituisce la media degli score RMSLE sui fold
        return np.mean(scores)

    # Creazione dello studio Optuna per minimizzare l'RMSLE
    study = optuna.create_study(direction='minimize')
    # Avvio dell'ottimizzazione con un numero limitato di tentativi (n_trials=5)
    study.optimize(objective, n_trials=5)

    # Stampa dei risultati migliori
    print('Best trial:')
    trial = study.best_trial  # Miglior trial trovato
    print(f'  RMSLE: {trial.value:.4f}')  # Miglior valore di RMSLE
    print('  Params:')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')  # Parametri del miglior modello

    # Salvataggio dei parametri migliori in un file per utilizzi futuri
    joblib.dump(trial.params, 'best_lgbm_params.pkl')

# Punto di ingresso principale
if __name__ == '__main__':
    main()
