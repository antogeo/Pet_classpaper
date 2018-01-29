import numpy as np
import pandas pd
from sklearn.utils import resample
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, precision_score, recall_score)
from matplotlib import pyplot

# load dataset
if os.uname()[1] == 'antogeo-XPS':
    db_path = '/home/antogeo/data/PET/pet_suv_db/'
elif os.uname()[1] == 'comameth':
    db_path = '/home/coma_meth/Documents/PET/pet_suv_db/'
elif os.uname()[1] in ['mia.local', 'mia']:
    db_path = '/Users/fraimondo/data/pet_suv_db/'

meta_fname = op.join(db_path, 'extra', 'SUV_database10172017_2.xlsx')
df = compute_regional_features(db_path, meta_fname)
df_train = df.query('QC_PASS == True and ML_VALIDATION == False')
df_eval = df.query('QC_PASS == True and ML_VALIDATION == True')

classifiers = OrderedDict()

classifiers['SVC_fs_W40_10'] = Pipeline([
        ('scaler', RobustScaler()),
        ('select', SelectPercentile(f_classif, 10.)),
        ('clf', SVC(kernel="linear", C=1, probability=True,
                    class_weight={0: 1, 1: .25}))
    ])
classifiers['SVC_fs_W10_26'] = Pipeline([
        ('scaler', RobustScaler()),
        ('select', SelectPercentile(f_classif, 10.)),
        ('clf', SVC(kernel="linear", C=1,  probability=True,
                    class_weight={0: 1, 1: 2.6}))
    ])
classifiers['RF_w'] = Pipeline([
    ('scaler', RobustScaler()),
    ('clf', RandomForestClassifier(
        max_depth=5, n_estimators=2000, max_features='auto',
        class_weight={0: 1, 1: .7}))
])
classifiers['Dummy'] = Pipeline([
        ('clf', DummyClassifier(strategy="stratified", random_state=42))
    ])

markers = [x for x in df.columns if 'aal' in x]
X = df[markers].values
y = 1 - (df['Final diagnosis (behav)'] == 'VS').values.astype(np.int)

# configure bootstrap
n_iterations = 1000
n_size = int(len(data) * 0.50)
# run bootstrap

results = list()
results['Iteration'] = []
results['Classifier'] = []
results['AUC'] = []
results['Precision'] = []
results['Recall'] = []

for i in range(n_iterations):
	# prepare train and test sets
	train = resample(values, n_samples=n_size)
	test = numpy.array([x for x in values if x.tolist() not in train.tolist()])
	# fit model
	model = DecisionTreeClassifier()
	model.fit(train[:,:-1], train[:,-1])
	# evaluate model
	predictions = model.predict(test[:,:-1])
	score = accuracy_score(test[:,-1], predictions)
	print(score)
	results.append(score)
# plot scores
pyplot.hist(results)
pyplot.show()
# confidence intervals
alpha = 0.95
p = ((1.0-alpha)/2.0) * 100
lower = max(0.0, numpy.percentile(results, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upper = min(1.0, numpy.percentile(results, p))
print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))
