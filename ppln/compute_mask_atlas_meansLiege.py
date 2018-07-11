import os
import os.path as op
import pandas as pd
from pypet.features import compute_regional_features

if os.uname()[1] == 'antogeo-XPS':
    db_path = '/home/antogeo/data/PET/pet_suv_db/Liege'
elif os.uname()[1] == 'comameth':
    db_path = '/home/coma_meth/Documents/PET/pet_suv_db/Liege'
elif os.uname()[1] in ['mia.local', 'mia']:
    db_path = '/Users/fraimondo/data/pet_suv_db/Liege'

results = 'group_results_SUV'

meta_fname = op.join(db_path, 'extra', 'SUV_database10172017.xlsx')
all_df = compute_regional_features(db_path, meta_fname)

dframe = pd.DataFrame(all_df)

dframe.to_csv(op.join(db_path, results, 'db_GM_masks_atlas.csv'))
