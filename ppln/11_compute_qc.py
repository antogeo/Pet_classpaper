import os
import os.path as op
import pandas as pd
from scipy.stats import zscore

if os.uname()[1] == 'antogeo-XPS':
    db_path = '/home/antogeo/dox/pet_suv_db/'
elif os.uname()[1] == 'comameth':
    db_path = '/home/coma_meth/dox/pet_suv_db/'
elif os.uname()[1] in ['mia.local', 'mia']:
    db_path = '/Users/fraimondo/data/pet_suv_db/'

groups = ['Liege']

for group in groups:
    all_df = pd.read_csv(op.join(db_path, group, 'group_results_SUV',
                         group + '_db_GM_AAL_nocereb.csv'))
    thresh = all_df[
        all_df['Final diagnosis (behav)'] == 'CTRL']['GMIndex'].mean() + \
        3 * all_df[
            all_df['Final diagnosis (behav)'] == 'CTRL']['GMIndex'].std()
    all_df['QC_PASS'] = all_df['GMIndex'] < thresh
    all_df.to_csv(op.join(db_path, group, 'group_results_SUV',
                          group + '_db_GM_AAL_nocereb.csv'))
