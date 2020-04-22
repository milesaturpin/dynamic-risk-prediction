import pandas as pd

enc = pd.read_pickle('data/processed/encounters.pickle')
enc = enc[lambda df: df.sbo_poa].sort_index()

past_sbo = (
    enc
    .sort_index()
    .reset_index()
    .groupby('mrn')
    .age
    .expanding()
    .count()
    - 1)

past_sbo = past_sbo.rename('past_sbo_enc').reset_index()

past_sbo['id'] = (
    enc
    .sort_index()
    .reset_index()
    ['id'])

past_sbo = (
    past_sbo
    .drop('level_1',1)
    .set_index(['mrn','id']))

past_sbo.to_pickle('data/processed/past_sbo.pickle')