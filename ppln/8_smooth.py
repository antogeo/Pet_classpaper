import os
import os.path as op
import pypet.preprocessing as pre
from glob import glob

# load dataset
if os.uname()[1] == 'antogeo-XPS':
    db_path = '/home/antogeo/data/PET/pet_suv_db/'
elif os.uname()[1] == 'comameth':
    db_path = '/home/coma_meth/Documents/PET/pet_suv_db/'
elif os.uname()[1] in ['mia.local', 'mia']:
    db_path = '/Users/fraimondo/data/pet_suv_db/'

folders = ['Liege', 'Paris']

for f_name in folders:
    subjects = [op.basename(x) for x in glob(
        op.join(db_path, f_name, 'subjects', '*')) if op.isdir(x)]
    for subject in subjects:
        nii = glob(
            op.join(db_path, f_name, 'subjects', subject, '*_w*.nii'))
        for img in nii:
            print(img)
            pre.smooth_img(img)
