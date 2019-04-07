import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd
import seaborn as sns

if os.uname()[1] == 'antogeo-XPS':
    db_path = '/home/antogeo/data/PET/pet_suv_db/'
elif os.uname()[1] == 'comameth':
    db_path = '/home/coma_meth/Documents/PET/pet_suv_db/'
elif os.uname()[1] in ['mia.local', 'mia']:
    db_path = '/Users/fraimondo/data/pet_suv_db/'

group = 'Liege'

df = pd.read_csv(op.join(db_path, group, 'group_results_SUV',
                 group + '_db_GM_masks_3_atlases_trim.csv'))
df = df.query('QC_PASS == True and ML_VALIDATION == False')

y = (df['Final diagnosis (behav)'] == 'VS').values.astype(np.int)


atlases = ['yeo', 'aal', 'cog']
results = dict()
results['atlas'] = []
results['importances'] = []
results['region'] = []
for atlas in atlases:
    markers = [x for x in df.columns if atlas in x]
    X = df[markers].values
    print(atlas)
    forest = ExtraTreesClassifier(
        n_estimators=1000, max_features=None,
        n_jobs=2, random_state=55)
    forest.fit(X, y)
    results['importances'].extend(forest.feature_importances_)
    results['region'].extend(range(len(forest.feature_importances_)))
    results['atlas'].extend([atlas]*len(forest.feature_importances_))
df_f = pd.DataFrame(results)

for atlas in atlases:
    plt.figure()
    sns.barplot(x='region', y='importances', data=df_f[df_f['atlas'] == atlas],
                color='red')
    plt.show()
