import os
import os.path as op
import pandas as pd
import numpy as np
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
                         group + '_db_GM_AAL_controls_inl.csv'))
    # thresh = all_df[
    #     all_df['Final diagnosis (behav)'] == 'CTRL']['GMIndex'].mean() + \
    #     3 * all_df[
    #         all_df['Final diagnosis (behav)'] == 'CTRL']['GMIndex'].std()
    # all_df['QC_PASS'] = all_df['GMIndex'] < thresh
    # all_df.to_csv(op.join(db_path, group, 'group_results_SUV',
    #                       group + '_db_GM_AAL_nocereb.csv'))
qc = dict()
qc['suv_max'] = []
qc['suv_mean'] = []
markers = [x for x in all_df.columns if 'aal' in x]
X = all_df[markers].values
for i, line in enumerate(X):
    print(i, max(line))
    qc['suv_max'].append(max(line))
    qc['suv_mean'].append(np.mean(line))
df_res = pd.DataFrame(qc)

z_suv_max = np.around(zscore(df_res['suv_max'], axis=0)) > 3
z_suv_mean = np.around(zscore(df_res['suv_mean'], axis=0)) > 3

# np.set_printoptions(suppress=True)

# z_pat_gm = np.around(zscore(all_df
#                       ['GMIndex'][0:212], axis=0), 3) > 4
z_pat_l = np.around(zscore(all_df
                    ['GMIndexLeft'][0:212], axis=0), 3) > 3
z_pat_r = np.around(zscore(all_df
                    ['GMIndexRight'][0:212], axis=0), 3) > 3
z_pat_prr = np.around(zscore(all_df
                      ['GMIndexPreRight'][0:212], axis=0), 3) > 4
z_pat_prl = np.around(zscore(all_df
                      ['GMIndexPreLeft'][0:212], axis=0), 3) > 4
z_pat_pol = np.around(zscore(all_df
                      ['GMIndexPostLeft'][0:212], axis=0), 3) > 4
z_pat_por = np.around(zscore(all_df
                      ['GMIndexPostRight'][0:212], axis=0), 3) > 4

# z_ctrl_gm = np.around(zscore(all_df
#                        ['GMIndex'][211:], axis=0), 3) < -2
# z_ctrl_l = np.around(zscore(all_df
#                       ['GMIndexLeft'][211:], axis=0), 3) < -2
# z_ctrl_r = np.around(zscore(all_df
#                       ['GMIndexRight'][211:], axis=0), 3) < -2
# z_ctrl_prr = np.around(zscore(all_df
#                         ['GMIndexPreRight'][211:], axis=0), 3) < -2
# z_ctrl_prl = np.around(zscore(all_df
#                         ['GMIndexPreLeft'][211:], axis=0), 3) < -2
# z_ctrl_pol = np.around(zscore(all_df
#                         ['GMIndexPostLeft'][211:], axis=0), 3) < -2
# z_ctrl_por = np.around(zscore(all_df
#                         ['GMIndexPostRight'][211:], axis=0), 3) < -2
