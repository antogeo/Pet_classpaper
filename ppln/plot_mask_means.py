import os
import os.path as op
import pypet
import pypet.viz
import pandas as pd


if os.uname()[1] == 'antogeo-XPS':
    db_path = '/home/antogeo/data/PET/pet_suv_db/Liege'
elif os.uname()[1] == 'comameth':
    db_path = '/home/coma_meth/Documents/PET/pet_suv_db/Liege'
elif os.uname()[1] in ['mia.local', 'mia']:
    db_path = '/Users/fraimondo/data/pet_suv_db/Liege'

all_df = pd.read_csv(op.join(db_path, 'results_SUV', 'db_GM_masks_atlas.csv'))

fig_GM, axes_GM = pypet.viz.plot_values(
    all_df,
    values='GMIndex', target='Final diagnosis (behav)',
    classes=['VS', 'MCS', 'MCS-', 'MCS+', 'CTRL'])

ctrl_mean = all_df[
    all_df['Final diagnosis (behav)'] == 'CTRL']['GMIndex'].mean()

# Max of selected column and diagnosis
ctrl_max = all_df[
    all_df['Final diagnosis (behav)'] == 'CTRL']['GMIndex'].max()

axes_GM[0].axhline(ctrl_mean, color='k', ls='--')

# fig_outers, axes_out =
# Plot Left Right
fig_mean, axes = pypet.viz.plot_values(
    all_df,
    values=['GMIndexLeft', 'GMIndexRight'],
    target='Final diagnosis (behav)',
    classes=['VS', 'MCS', 'MCS-', 'MCS+', 'CTRL'])

# Plot the four sides
fig_mean, axes = pypet.viz.plot_values(
    all_df,
    values=['GMIndexPreLeft', 'GMIndexPreLeft',
            'GMIndexPostLeft', 'GMIndexPostRight'],
    target='Final diagnosis (behav)',
    classes=['VS', 'MCS', 'MCS-', 'MCS+', 'CTRL'],
    n_cols=2)
