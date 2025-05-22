# PROGETTO-FINALE-CORSO-PYTHON-
PROGETTO-FINALE- Progetto Finale Assicurazioni Questo progetto affronta la competizione Kaggle Playground Series S4E12, con l'obiettivo di predire il Premium Amount di polizze assicurative sui dati di treno/test forniti, raggiungendo un RMSLE ≤ 1.05.

Struttura e logica del progetto Il flusso generale è suddiviso in 7 moduli Python, che vanno eseguiti in sequenza dall'estrazione delle feature fino alla creazione della submission:

01_feature_temporali.py

Carica train.csv e test.csv, parse della data Policy Start Date.

Le estrae sono caratterizzate da temporali:

Policy_Start_Year, Policy_Start_Month, Policy_Start_DayOfWeek Policy_Age_Days (distanza dalla data di esecuzione) Salva *_with_temporal.csv.

02_bandiere_mancanti.py

Carica i CSV con le funzionalità temporali. Per ogni colonna originale (numeriche e categorie), crea un flag booleano _missing_flag che indica valori nulli. Salva *_with_missing_flags.csv. 03_eda.py

Esegui un'analisi esplorativa del dataset (train_with_missing_flags.csv):

Distribuzioni statistiche (istogrammi, trama del violino) delle variabili numeriche. Frequenze e conteggi delle categorie. Matrice di correlazione e grafico a dispersione tra punteggio sanitario, punteggio di credito, reddito annuo e importo del premio. Generi report e grafici in output (cartella report/).

04_feature_engineering.py

Carica *_with_missing_flags.csv.

Binning numerico:

Age_bin, CreditScore_bin, VehicleAge_bin con tagli predefiniti. Caratteristica composita HealthRisk_index = mappatura di abitudine al fumo (0–2) meno mappatura di frequenza di esercizio (0–3).

Feedback_length dalla lunghezza testo di Customer Feedback.

Salva train_fe.csv e test_fe.csv.

05_validazione_e_linea_di_base.py

Carica train_fe.csv.

Seleziona X/y, log1p trasforma Importo Premium per stabilità numerica.

Costruisci pipeline:

Imputer (mediana) + StandardScaler per variabili numeriche. Imputer (costante) + OrdinalEncoder per categorie. Aggiungi un LightGBMRegressor di base su 5 volte CV, misura RMSLE per piega e media.

Fit finale su tutti i dati, salva pipeline in baseline_lgbm_pipeline.pkl.

06_hyperparameter_tuning.py

Carica train_fe.csv e preprocessa una volta.

Usa Optuna per la ricerca su 5-fold CV (trigger 5 trial rapidi):

Parametri di campionamento: n_estimators, learning_rate, num_leaves, max_depth, min_data_in_leaf, lambda_l1, lambda_l2, feature_fraction, bagging_fraction, bagging_freq. Restituisce il Best trial (RMSLE + parametri) e lo salva in best_lgbm_params.pkl.

07_generate_submission.py

Carica train_fe.csv, test_fe.csv, i migliori parametri da Optuna e sample_submission.csv. Ricostruisce la stessa pipeline di preprocessing + LGBMRegressor con best_lgbm_params.pkl. Fit su tutti i dati di train (log1p target), predice su test, esponenziale inversa e clip ≥0. Allinea le predizioni con gli ID via merge, salva submission.csv pronta per Kaggle. Strumenti e librerie utilizzate Python 3.10 Pandas / NumPy per gestione dati e feature engineering scikit-learn per pipeline, preprocessing (Imputer, StandardScaler, OrdinalEncoder), K-Fold CV, metriche (RMSLE) LightGBM per il modello base e ottimizzato Optuna per hyperparameter tuning Joblib per serializzazione modelli e parametri Matplotlib / Seaborn (nel modulo EDA) per grafici esplorativi CLI script (interactive_menu.py) per un menu interattivo di esplorazione e previsione Come eseguire il progetto Posizionarsi nella cartella del progetto con i file train.csv, test.csv, sample_submission.csv.

Eseguire in ordine i moduli:

python 01_feature_temporali.py python 02_missing_flags.py python 03_eda.py # opzionale, generi grafici python 04_feature_engineering.py python 05_validation_and_baseline.py python 06_hyperparameter_tuning.py python 07_generate_submission.py Caricare submission.csv su Kaggle per ottenere il punteggio RMSLE.

(Opzionale) Avviare python Interactive_menu.py per un'interfaccia CLI.
