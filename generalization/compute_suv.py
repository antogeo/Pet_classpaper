import os.path as op
import dicom
import numpy as np
import pandas as pd
from glob import glob

subj_path = '/home/coma_meth/Documents/PET/pet_suv_db/Paris/subjects'
subjects = sorted(
        [x.split('/')[-1] for x in glob(op.join(subj_path, '*'))])

# Read the nifti/img file
images = []
read_subjects = []
for subject in subjects:
    s_path = op.join(subj_path, subject)
    if what == 'wSUV':
        files = glob(op.join(s_path, '*wSUV.nii'))
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
        raise ValueError('I do not know how to read {}'.format(what))

logging.info(
    "Reading subjects' volume. Total number of subjects: {}"
    .format(len(read_subjects)))
