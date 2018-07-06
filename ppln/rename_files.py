import os
import os.path as op
from glob import glob

if os.uname()[1] == 'antogeo-XPS':
    db_path = '/home/antogeo/data/PET/pet_suv_db/Paris/subjects'
elif os.uname()[1] == 'comameth':
    db_path = '/home/coma_meth/Documents/PET/pet_suv_db/Paris/subjects'
elif os.uname()[1] in ['mia.local', 'mia']:
    db_path = '/Users/fraimondo/data/pet_suv_db/Paris/subjects'

subjects = sorted([op.basename(x) for x in glob(op.join(db_path, '*'))])

for subject in subjects:
    s_path = op.join(db_path, subject)
    files = [op.basename(x) for x in glob(
        op.join(s_path, '*')) if not op.isdir(x)]

    print('--------------------------')
    print('{} ({} files)'.format(subject, len(files)))
    print('--------------------------')
    for fname in files:
        if ((fname.endswith('nsSUV.nii') or fname.endswith('nsRAW.nii')) and
                fname.startswith('sec_')):
            new_fname = fname.replace("sec_", "")
            new_fname = new_fname.replace("ns", "w")
            print('{} ==> {}'.format(fname, new_fname))
            os.rename(op.join(db_path, subject, fname),
                      op.join(db_path, subject, new_fname))
