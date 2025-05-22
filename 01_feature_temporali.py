import pandas as pd

# 1. Caricamento dei dati
# Leggiamo i file CSV `train.csv` e `test.csv` e li carichiamo in due DataFrame Pandas.
# La colonna 'Policy Start Date' viene interpretata come una colonna di tipo datetime grazie al parametro `parse_dates`.
train = pd.read_csv('train.csv', parse_dates=['Policy Start Date'])
test  = pd.read_csv('test.csv',  parse_dates=['Policy Start Date'])

# 2. Definizione della data di riferimento
# Creiamo un oggetto Timestamp che rappresenta la data di riferimento (ipotizziamo che "oggi" sia il 9 maggio 2025).
# Questa data verrà utilizzata per calcolare l'età della polizza in giorni.
reference_date = pd.Timestamp('2025-05-09')

# 3. Estrazione di informazioni temporali e calcolo dell'età della polizza
# Iteriamo sui due DataFrame (`train` e `test`) per aggiungere nuove colonne con informazioni temporali.
for df in (train, test):
    # Estraggo l'anno dalla colonna 'Policy Start Date' e lo salvo in una nuova colonna 'Policy_Start_Year'.
    df['Policy_Start_Year']      = df['Policy Start Date'].dt.year
    # Estraggo il mese dalla colonna 'Policy Start Date' e lo salvo in una nuova colonna 'Policy_Start_Month'.
    df['Policy_Start_Month']     = df['Policy Start Date'].dt.month
    # Estraggo il giorno della settimana (0 = lunedì, 6 = domenica) e lo salvo in 'Policy_Start_DayOfWeek'.
    df['Policy_Start_DayOfWeek'] = df['Policy Start Date'].dt.dayofweek
    # Calcolo l'età della polizza in giorni come differenza tra la data di riferimento e la data di inizio polizza.
    # Il risultato viene salvato in una nuova colonna 'Policy_Age_Days'.
    df['Policy_Age_Days']        = (reference_date - df['Policy Start Date']).dt.days

# 4. Salvataggio delle nuove versioni dei dataset
# Salviamo i DataFrame aggiornati con le nuove colonne in due nuovi file CSV.
# `index=False` evita di salvare l'indice del DataFrame come colonna nel file CSV.
train.to_csv('train_with_temporal.csv', index=False)
test.to_csv('test_with_temporal.csv',  index=False)
