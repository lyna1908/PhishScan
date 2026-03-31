import pandas as pd

ceas = pd.read_csv('CEAS_08.csv')
nazario = pd.read_csv('Nazario.csv')
spam = pd.read_csv('SpamAssasin.csv')

print("CEAS columns:", ceas.columns.tolist())
print("Nazario columns:", nazario.columns.tolist())
print("SpamAssassin columns:", spam.columns.tolist())

# Check a sample of raw data
print("\nCEAS sample row:")
print(ceas.iloc[0].to_string())