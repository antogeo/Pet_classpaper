import os
import os.path as op
import pypet
import pypet.viz
import pandas as pd
import numpy as np


if os.uname()[1] == 'antogeo-XPS':
    db_path = '/home/antogeo/data/PET/pet_suv_db/'
elif os.uname()[1] == 'comameth':
    db_path = '/home/coma_meth/dox/pet_suv_db/'
elif os.uname()[1] in ['mia.local', 'mia']:
    db_path = '/Users/fraimondo/data/pet_suv_db/'

groups = ['Liege']

for group in groups:
    all_df = pd.read_csv(op.join(db_path, group, 'group_results_SUV',
                         group + '_db_GM_AAL.csv'))
    all_df = all_df.query('ML_VALIDATION == False')
    if 'Final diagnosis (behav)' not in all_df:
        all_df['Final diagnosis (behav)'] = 'to test'

    vals = np.unique(all_df['Label'])

    fig_GM, axes_GM = pypet.viz.plot_values(
        all_df,
        values='GMIndex', target='Label',
        classes=vals, diam=3)

    ctrl_mean = all_df[
        all_df['Label'] == 'CTRL']['GMIndex'].mean()

    # Max of selected column and diagnosis
    ctrl_max = all_df[
        all_df['Label'] == 'CTRL']['GMIndex'].max()


    axes_GM[0].axhline(ctrl_mean, color='k', ls='--')

    # fig_outers, axes_out =
    # Plot Left Right
    fig_mean, axes = pypet.viz.plot_values(
        all_df,
        values=['GMIndexLeft', 'GMIndexRight'],
        target='Label',
        classes=vals, diam=3)

    # Plot the four sides
    fig_mean, axes = pypet.viz.plot_values(
        all_df,
        values=['GMIndexPreLeft', 'GMIndexPreRight',
                'GMIndexPostLeft', 'GMIndexPostRight'],
        target='Label',
        classes=vals, diam=3,
        n_cols=2)
