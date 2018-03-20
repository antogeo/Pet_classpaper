import os
import os.path as op
import pydicom
import numpy as np
import pandas as pd
from glob import glob

if os.uname()[1] == 'antogeo-XPS':
    db_path = '/home/antogeo/data/PET/pet_suv_db/'
elif os.uname()[1] == 'comameth':
    db_path = '/home/coma_meth/Documents/PET/pet_suv_db/'
elif os.uname()[1] in ['mia.local', 'mia']:
    db_path = '/Users/fraimondo/data/pet_suv_db/'

subj_path = op.join(db_path, 'Paris/subjects')

subjects = sorted(
        [x.split('/')[-1] for x in glob(op.join(subj_path, '*'))])

# Read the nifti/img file
images = []
read_subjects = []
for subject in subjects:
    s_path = op.join(subj_path, subject)
    print(s_path)
    conts = sorted([x.split('/')[-1] for x in glob(op.join(s_path, '*'))])
    if 'PET_dicom' in conts:
        print('yeah')
        dicom_fd = sorted([x.split('/')[-1] for x in glob(
            op.join(s_path, 'PET_dicom', '*'))])
        if len(dicom_fd) < 88:
            print('No dicom files in {}' .format(subject))
            continue
        dataset = dicom.read_file(op.join(s_path, 'PET_dicom', dicom_fd[0]))
        weight = dataset.PatientWeight
        dose = dataset.RadiopharmaceuticalInformationSequence[
            0].RadionuclideTotalDose / 1000000
        weight/dose
        suv_factor = dataset[0x7053, 0x1000].value
        inj_time = dataset.SUV
        st_fold = sorted([x.split('/')[-1] for x in glob(
            op.join(s_path, 'DICOM', 'PET*'))])
        if len(files) == 0:
            files = glob(op.join(s_path, '*wSUV.img'))
        if len(files) == 0:
            # print('No {} files found for {}'.format(what, subject))
            logging.warning(
                'No {} files found for {}'.format(what, subject))
            continue
        this_image = nib.load(files[0]).get_data().reshape(-1)
        images.append(this_image)
        read_subjects.append(subject)
    else:
        raise ValueError('No dicom folder {}'.format())

logging.info(
    "Reading subjects' volume. Total number of subjects: {}"
    .format(len(read_subjects)))
