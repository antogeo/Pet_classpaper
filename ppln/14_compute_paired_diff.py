# plot scores
import pandas as pd
import numpy as np

if os.uname()[1] == 'antogeo-XPS':
    db_path = '/home/antogeo/data/PET/pet_suv_db/'
elif os.uname()[1] == 'comameth':
    db_path = '/home/coma_meth/dox/pet_suv_db/'
elif os.uname()[1] in ['mia.local', 'mia']:
    db_path = '/Users/fraimondo/data/pet_suv_db/'

group = 'Liege'

df = pd.read_csv(op.join(db_path, group, 'group_results_SUV',
                'perf_estim_1000iter_f10_AAL90_ctrlout.csv'),
                 index_col=['Iteration', 'Classifier'])[
                    ['AUC', 'Precision', 'Recall']]

df = df.reset_index()

classifiers = np.unique(df['Classifier'])
diffs = []
for cl1 in classifiers:
    for cl2 in classifiers:
        if cl1 == cl2:
            continue
        df1 = df.query('Classifier == "{}"'.format(cl1))
        df2 = df.query('Classifier == "{}"'.format(cl2))
        t_diff = (df1.set_index('Iteration')[['AUC', 'Precision', 'Recall']] -
                  df2.set_index('Iteration')[['AUC', 'Precision', 'Recall']])
        t_diff['Contrast'] = ['{}-{}'.format(cl1, cl2)] * len(t_diff)
        diffs.append(t_diff)
all_diffs = pd.concat(diffs)


all_diffs.to_csv(op.join(db_path, group, 'group_results_SUV',
                 'Clfs_contrast_f10_AAL90_ctrlout.csv'))


def _compute_p_vals(df, column, ci=.95):
    p_low = ((1.0 - ci) / 2.0) * 100
    p_up = ci * 100 + p_low

    # Get CI from Dist
    cols = np.unique(df[column])
    results = {column: [], 'AUC': [], 'Precision': [], 'Recall': []}
    for t_col in cols:
        t_df = df.query('{} == "{}"'.format(column, t_col))
        for t_var in ['AUC', 'Precision', 'Recall']:
            ci_up = np.percentile(t_df[t_var].values, p_up)
            ci_low = np.percentile(t_df[t_var].values, p_low)
            est = t_df[t_var].mean()
            se = (ci_up - ci_low) / (2 * 1.96)
            z = est / se
            p = np.exp(-0.717 * z - 0.416 * z * z)
            results[t_var].append(p)
        results[column].append(t_col)

    return pd.DataFrame(results)


p_vals = _compute_p_vals(all_diffs, 'Contrast')
df = pd.DataFrame(p_vals)
df.to_csv('p_values.csv')
