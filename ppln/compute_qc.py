import os
import os.path as op
import pandas as pd

if os.uname()[1] == 'antogeo-XPS':
    db_path = '/home/antogeo/data/PET/pet_suv_db/'
elif os.uname()[1] == 'comameth':
    db_path = '/home/coma_meth/Documents/PET/pet_suv_db/'
elif os.uname()[1] in ['mia.local', 'mia']:
    db_path = '/Users/fraimondo/data/pet_suv_db/'

groups = ['Liege']

for group in groups:
    all_df = pd.read_csv(op.join(db_path, group, 'group_results_SUV',
                         group + '_db_GM_masks_full_atlas.csv'))
    thresh = all_df[
        all_df['Final diagnosis (behav)'] == 'CTRL']['GMIndex'].mean() + \
        3 * all_df[
            all_df['Final diagnosis (behav)'] == 'CTRL']['GMIndex'].std()
    all_df[] 
