import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import lightgbm as lgb

# Modulo 2.3: Fit finale e creazione submission con Ordinal Encoding

def main():
    # 1. Caricamento dati con feature engineering
    # Carichiamo i dataset di train e test che contengono le feature già ingegnerizzate
    train = pd.read_csv('train_fe.csv')
    test = pd.read_csv('test_fe.csv')

    # 2. Separazione X e y
    # Separiamo le feature (X) dal target (y) nel dataset di training
    # Rimuoviamo anche la colonna 'id' che non è utile per il modello
    X_train = train.drop(columns=['Premium Amount', 'id'])
    y_train = train['Premium Amount']
    # Nel dataset di test rimuoviamo solo la colonna 'id'
    X_test = test.drop(columns=['id'])

    # 3. Caricamento dei migliori parametri da Optuna
    # Carichiamo i parametri ottimizzati per il modello LightGBM salvati in precedenza
    best_params = joblib.load('best_lgbm_params.pkl')

    # 4. Definizione del preprocessore
    # Identifichiamo le colonne numeriche e categoriche nel dataset di training
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    # Creiamo una pipeline per il preprocessing delle colonne numeriche
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),  # Sostituiamo i valori mancanti con la mediana
        ('scaler', StandardScaler())  # Standardizziamo i valori numerici
    ])
    # Creiamo una pipeline per il preprocessing delle colonne categoriche
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # Sostituiamo i valori mancanti con 'missing'
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))  # Codifichiamo le categorie in valori ordinali
    ])
    # Combiniamo le due pipeline in un unico preprocessore
    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, numeric_cols),  # Applichiamo la pipeline numerica alle colonne numeriche
        ('cat', categorical_pipeline, categorical_cols)  # Applichiamo la pipeline categorica alle colonne categoriche
    ])

    # 5. Costruzione ed esecuzione pipeline finale
    # Creiamo il modello LightGBM con i parametri ottimizzati
    model = lgb.LGBMRegressor(**best_params, random_state=42)
    # Creiamo una pipeline che combina il preprocessore e il modello
    clf = Pipeline([
        ('preprocessor', preprocessor),  # Preprocessing dei dati
        ('regressor', model)  # Modello di regressione
    ])

    # Addestriamo la pipeline sul dataset di training
    # Applichiamo il logaritmo naturale (log1p) al target per gestire meglio la distribuzione
    clf.fit(X_train, np.log1p(y_train))

    # 6. Predizioni sul test
    # Generiamo le predizioni sul dataset di test
    preds = np.expm1(clf.predict(X_test))  # Applichiamo l'esponenziale inverso per riportare i valori alla scala originale
    preds = np.maximum(0, preds)  # Impostiamo un limite inferiore di 0 per evitare valori negativi

    # 7. Allineamento con sample_submission tramite merge su id
    # Carichiamo il file di esempio per la submission
    submission = pd.read_csv('sample_submission.csv')
    # Creiamo un DataFrame con le predizioni e gli id del dataset di test
    preds_df = pd.DataFrame({
        'id': test['id'],
        'Premium Amount': preds
    })
    # Facciamo un merge con il file di esempio per mantenere l'ordine degli id e gestire eventuali id extra
    submission = submission[['id']].merge(preds_df, on='id', how='left')
    # Riempimento dei valori mancanti con la media delle predizioni (se qualche id non è stato trovato)
    submission['Premium Amount'].fillna(preds.mean(), inplace=True)

    # 8. Salvataggio del file submission
    # Salviamo il file finale in formato CSV
    submission.to_csv('submission.csv', index=False)
    print("File 'submission.csv' creato con successo.")
    
if __name__ == '__main__':
    main()
