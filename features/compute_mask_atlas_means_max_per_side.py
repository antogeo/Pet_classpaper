import os
import os.path as op
import pandas as pd
from pypet.features import compute_regional_features

if os.uname()[1] == 'antogeo-XPS':
    db_path = '/home/antogeo/data/PET/pet_suv_db/'
elif os.uname()[1] == 'comameth':
    db_path = '/home/coma_meth/dox/pet_suv_db/'
elif os.uname()[1] in ['mia.local', 'mia']:
    db_path = '/Users/fraimondo/data/pet_suv_db/'

results = 'group_results_SUV'
folders = ['Paris', 'Liege']
all_df = []
for folder in folders:
    meta_fname = op.join(db_path, folder, 'extra', folder + '_meta.csv')
    s_path = op.join(db_path, folder)
    all_df = compute_regional_features(s_path, meta_fname, ftype='s8SUV')
    print('{} {}'.format(s_path, meta_fname))
    dframe = pd.DataFrame(all_df)
    dframe = dframe.reset_index(drop=True)
    dframe.to_csv(op.join(
        db_path, folder, results,
        folder + '_db_GM_masks_nAAL_roinames.csv'))

# Rename columns of mean values with Region names

atlas_reg = pd.read_csv(op.join(db_path, folders[1], 'extra', 'aal_rois.csv'))

# rename regions using csv file
for x in all_df.columns:
    if 'aal_atlas_' in x:
        new_name = atlas_reg['roi_name'][int(x.split("aal_atlas_", 1)[1])]
        all_df.rename(index=str, columns={x: 'aal' + new_name}, inplace=True)
# calculate and save max between regions

rois_data = {}
rois_data['Code'] = []
for num, sub in enumerate(all_df['Code']):
    rois_data['Code'].append(sub)
    for i in range(len(all_df.columns)-1):
        if "_L" in all_df.columns[i]:
            reg_1 = all_df.columns[i].split("_L", 1)[0]
            reg_2 = all_df.columns[i+1].split("_R", 1)[0]
            if reg_1 == reg_2:
                if 'max_' + reg_1 not in rois_data.keys():
                    rois_data['max_' + reg_1] = []
                rois_data['max_' + reg_1].append(max(
                    all_df[all_df['Code'] == sub][
                        all_df.columns[i]].values,
                    all_df[all_df['Code'] == sub][
                        all_df.columns[i+1]].values)[0])

df_max = pd.DataFrame.from_dict(rois_data, orient='columns')
df = all_df.join(df_max.set_index('Code'), on='Code')
df.to_csv(op.join(db_path, folder, results,
          folder + '_db_GM_masks_nAAL_roinames_maxside.csv'))
