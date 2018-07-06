import os
import os.path as op
from glob import glob
import shutil

if os.uname()[1] == 'antogeo-XPS':
    db_path = '/home/antogeo/data/PET/pet_suv_db/Liege/subjects'
elif os.uname()[1] == 'comameth':
    db_path = '/home/coma_meth/Documents/PET/pet_suv_db/Liege/subjects'
elif os.uname()[1] in ['mia.local', 'mia']:
    db_path = '/Users/fraimondo/data/pet_suv_db/'

subjects = sorted([op.basename(x) for x in glob(op.join(db_path, '*'))])

for subject in subjects:
    s_path = op.join(db_path, subject)
    files = [op.basename(x) for x in glob(
        op.join(s_path, '*')) if not op.isdir(x)]

    print('--------------------------')
    print('{} ({} files)'.format(subject, len(files)))
    print('--------------------------')
    for fname in files:
        new_fname = op.join(s_path, 'frst_norm')
        if not op.exists(new_fname):
            print('creating folder: {} in {}'.format(op.basename(new_fname),
                  subject))
            os.makedirs(new_fname)
        if (((fname.endswith('nsSUV.nii') or fname.endswith('nsRAW.nii')) and
                fname.startswith('tmplt_HO')) or fname.endswith('_sn.mat')):
            print('{} -> {}'.format(fname, op.join(new_fname, fname)))
            shutil.move(op.join(s_path, fname), op.join(
                new_fname, fname))
