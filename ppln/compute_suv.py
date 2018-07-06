import os
import pypet.preprocessing as pre
from nipype.interfaces.dcm2nii import Dcm2nii

if os.uname()[1] == 'antogeo-XPS':
    db_path = '/home/antogeo/data/PET/pet_suv_db/TEST/subjects'
elif os.uname()[1] == 'comameth':
    db_path = '/home/coma_meth/Documents/PET/pet_suv_db/Paris/subjects'
elif os.uname()[1] in ['mia.local', 'mia']:
    db_path = '/Users/fraimondo/data/pet_suv_db/'
converter = Dcm2nii()


for subject in os.listdir(db_path):
    pre.create_suv(os.join(db_path, subject))
