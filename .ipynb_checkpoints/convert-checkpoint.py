import pandas as pd

pd.read_excel('data/raw/flowsheet.xlsx').to_pickle('data/processed/flowsheet.pickle')