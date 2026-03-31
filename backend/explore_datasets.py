import pandas as pd

for f in ['CEAS_08.csv', 'Nazario.csv', 'SpamAssasin.csv']:
    print(f"--- {f} ---")
    try:
        df = pd.read_csv(f, nrows=5)
        print("Columns:", df.columns.tolist())
        print("Label values sample (if column exists):")
        if 'label' in df.columns:
            full_df = pd.read_csv(f, usecols=['label'])
            print(full_df['label'].value_counts())
        else:
            print("No 'label' column found.")
        print("\n")
    except Exception as e:
        print(f"Error reading {f}: {e}")
