import numpy as np
import pandas as pd
import os
import os.path as op

import itertools

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

if os.uname()[1] == 'antogeo-XPS':
    db_path = '/home/antogeo/data/PET/pet_suv_db/'
elif os.uname()[1] == 'comameth':
    db_path = '/home/coma_meth/Documents/PET/pet_suv_db/'
elif os.uname()[1] in ['mia.local', 'mia']:
    db_path = '/Users/fraimondo/data/pet_suv_db/'
meta_fname = op.join(db_path, 'extra', 'SUV_database10172017_2.xlsx')

df_Exp = pd.read_excel(meta_fname)
df = df_Exp.query('QC_PASS == True and ML_VALIDATION == False')
