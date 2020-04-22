
# Load libraries
import pandas as pd

target_path = 'data/test/'

# Read xlsx
vitals1 = pd.read_excel("~/projects/Duke Surgery - SBO/July Data Pull 2018/gi_obst_vitals_hr.xlsx", sheet_name=0)
# Export csv
vitals1.to_csv(target_path + 'vitals1.csv')

# Read xlsx
vitals2 = pd.read_excel("~/projects/Duke Surgery - SBO/July Data Pull 2018/gi_obst_vitals_bp.xlsx")
# Export csv
vitals2.to_csv(target_path + 'vitals2.csv')

# Read xlsx
vitals3 = pd.read_excel("~/projects/Duke Surgery - SBO/July Data Pull 2018/gi_obst_vitals_rr.xlsx", sheet_name=0)
# Export csv
vitals3.to_csv(target_path + 'vitals3.csv')

vitals4 = pd.read_excel("~/projects/Duke Surgery - SBO/July Data Pull 2018/gi_obst_vitals_spo2.xlsx", sheet_name=0)
vitals4.to_csv(target_path + 'vitals4.csv')

# second sheet of hr excel file, doesn't have headers
vitals5 = pd.read_excel("~/projects/Duke Surgery - SBO/July Data Pull 2018/gi_obst_vitals_hr.xlsx",
    sheet_name=1,
    header=None,
    names=vitals1.columns)
vitals5.to_csv(target_path + 'vitals5.csv')

vitals6 = pd.read_excel("~/projects/Duke Surgery - SBO/July Data Pull 2018/gi_obst_vitals_rr.xlsx",
    sheet_name=1,
    header=None,
    names=vitals3.columns)
vitals6.to_csv(target_path + 'vitals6.csv')

vitals7 = pd.read_excel("~/projects/Duke Surgery - SBO/July Data Pull 2018/gi_obst_vitals_spo2.xlsx",
    sheet_name=1,
    header=None,
    names=vitals4.columns)
vitals7.to_csv(target_path + 'vitals7.csv')


# Labs
bmp = pd.read_excel("~/projects/Duke Surgery - SBO/July Data Pull 2018/gi_obst_labs_BMP(1).xlsx")
bmp.to_csv(target_path + 'bmp.csv')

cbc = pd.read_excel("~/projects/Duke Surgery - SBO/July Data Pull 2018/gi_obst_labs_CBC.xlsx")
cbc.to_csv(target_path + 'cbc.csv')

cmp = pd.read_excel("~/projects/Duke Surgery - SBO/July Data Pull 2018/gi_obst_labs_CMP.xlsx")
cmp.to_csv(target_path + 'cmp.csv')


# IO
io1 = pd.read_excel("~/projects/Duke Surgery - SBO/July Data Pull 2018/gi_obst_IN_OUT.xlsx", sheet_name=0)
io1.to_csv(target_path + 'io1.csv')

io2 = pd.read_excel("~/projects/Duke Surgery - SBO/July Data Pull 2018/gi_obst_IN_OUT.xlsx",
    sheet_name=1,
    header=None,
    names=io1.columns)
io2.to_csv(target_path + 'io2.csv')

io_unmeasured = pd.read_excel("~/projects/Duke Surgery - SBO/July Data Pull 2018/gi_obst_IN_OUT_unmeasured.xlsx")
io_unmeasured.to_csv(target_path + 'io_unmeasured.csv')

# Patient outcomes
outcomes = pd.read_excel("~/projects/Duke Surgery - SBO/SBO_Final_v1.xlsx", sheet_name=1)
outcomes.to_csv(target_path + 'outcomes.csv')

# Flowsheet
flow = pd.read_excel("~/projects/Duke Surgery - SBO/July Data Pull 2018/flowsheet.xlsx")
flow.to_pickle(target_path + 'flowsheet.pickle')


# Read xlsx
imaging = pd.read_excel("~/projects/Duke Surgery - SBO/July Data Pull 2018/gi_obst_imaging(2).xlsx")
# Export csv
imaging.to_csv(target_path + 'imaging.csv')




