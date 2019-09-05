    import os
    import os.path as op
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from collections import OrderedDict
    from sklearn.model_selection import RepeatedKFold
    from sklearn.svm import SVC
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.dummy import DummyClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import RobustScaler
    from sklearn.metrics import (roc_auc_score, precision_score,
                                 recall_score)
    from sklearn.feature_selection import f_classif, SelectKBest

    if os.uname()[1] == 'antogeo-XPS':
        db_path = '/home/antogeo/dox/pet_suv_db/'
    elif os.uname()[1] == 'comameth':
        db_path = '/home/coma_meth/dox/pet_suv_db/'
    elif os.uname()[1] in ['mia.local', 'mia']:
        db_path = '/Users/fraimondo/data/pet_suv_db/'

    group = 'Liege'

    df = pd.read_csv(op.join(db_path, group, 'group_results_SUV',
                     group + '_db_GM_AAL.csv'))
    df = df.query('QC_PASS == True and ML_VALIDATION == False')

    weight_val = np.arange(1, 10, .2)
    weight_val = sorted(np.concatenate((1 / weight_val, weight_val)))

    classifiers = OrderedDict()

    markers = [x for x in df.columns if 'aal' in x]
    X = df[markers].values
    y = 1 - (df['Final diagnosis (behav)'] == 'VS').values.astype(np.int)

    results = dict()
    results['Iteration'] = []
    results['Weight Val'] = []
    results['Classifier'] = []
    results['AUC'] = []
    results['Precision'] = []
    results['Recall'] = []

    sss = RepeatedKFold(n_splits=5, n_repeats=50, random_state=42)

    for t_iter, (train, test) in enumerate(sss.split(X, y)):
        for val in weight_val:
            classifiers['SVC_fs10'] = Pipeline([
                    ('scaler', RobustScaler()),
                    ('select', SelectKBest(f_classif, 10)),
                    ('clf', SVC(kernel="linear", C=1,  probability=True,
                                class_weight={0: 1, 1: val}))
                ])
            classifiers['XRF'] = Pipeline([
                    ('scaler', RobustScaler()),
                    ('clf', ExtraTreesClassifier(
                            max_depth=5, n_estimators=2000, max_features='auto',
                            class_weight={0: 1, 1: val}))
                ])
            classifiers['Dummy'] = Pipeline([
                    ('clf', DummyClassifier(
                     strategy="most_frequent", random_state=42))
                ])
            print('Iteration {}'.format(t_iter))
            X_train, X_test = X[train], X[test]
            y_train, y_test = y[train], y[test]
            for clf_name, clf in classifiers.items():
                # Fit the model on the training set
                clf.fit(X_train, y_train)

                # Predict the test set
                y_pred_proba = clf.predict_proba(X_test)[:, 1]
                y_pred_class = clf.predict(X_test)

                auc_score = roc_auc_score(y_test, y_pred_proba)
                prec_score = precision_score(y_test, y_pred_class)
                rec_score = recall_score(y_test, y_pred_class)

                results['Weight Val'].append(val)
                results['Iteration'].append(t_iter)
                results['Classifier'].append(clf_name)
                results['AUC'].append(auc_score)
                results['Precision'].append(prec_score)
                results['Recall'].append(rec_score)

    df = pd.DataFrame(results)

    # classif = ['SVC_fs20', 'SVC_fs38']
    #
    # fig, ax = plt.subplots(3, 1)
    # paper_rc = {'lines.linewidth': .6, 'lines.markersize': 6}
    # sns.set_context("paper", rc=paper_rc)
    # df['Weight Val'] = df['Weight Val'].round(4)
    # ax[0] = sns.pointplot(x="Weight Val", y="Recall", hue="Classifier",
    #                       data=df, ax=ax, hue_order=classif)
    # ax[1] = sns.pointplot(x="Weight Val", y="Precision", hue="Classifier",
    #                       data=df, ax=ax, hue_order=classif)
    # ax[2] = sns.pointplot(x="Weight Val", y="AUC", hue="Classifier",
    #                       data=df, ax=ax, hue_order=classif)
    #
    # ax.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
    # ax.tick_params(axis='x', direction='out', length=3, width=1, grid_color='r',
    #                labelrotation=90, grid_alpha=0.5)

    df.to_csv(op.join(db_path, group, 'group_results_SUV',
                      'weights_eval_AAL_XRF_kfold.csv'))
