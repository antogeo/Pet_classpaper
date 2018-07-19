import os
import os.path as op
import pypet
from nipype.interfaces.fsl.maths import MeanImage
from nipype.interfaces.fsl.utils import Merge
from nipype.interfaces.spm import Smooth
from glob import glob
import nipype.pipeline.engine as pe
import nibabel as nib

if os.uname()[1] == 'antogeo-XPS':
    db_path = '/home/antogeo/data/PET/pet_suv_db/Liege/'
elif os.uname()[1] == 'comameth':
    db_path = '/home/coma_meth/Documents/PET/pet_suv_db/Liege/'
elif os.uname()[1] in ['mia.local', 'mia']:
    db_path = '/Users/fraimondo/data/pet_suv_db/'

# Get metadata

meta_fname = op.join(db_path, 'extra', 'SUV_database10172017.xlsx')

subjects = sorted([op.basename(x) for x in glob(
                   op.join(db_path, 'subjects', '*'))])
metadata = pypet.io.read_metadata(subjects, meta_fname)
subj_list = metadata.loc[metadata['ML_VALIDATION'] == 0, 'Code']
img_list = []
i = 0
for subject in subj_list:
    s_path = op.join(db_path, 'subjects', subject)
    files = [op.basename(x) for x in glob(
             op.join(s_path, 'tmplt*SUV*')) if not op.isdir(x)]
    for file in files:
        print(file)
        # print(i)
        # i = i + 1
        img_list.append(op.join(s_path, file))


def mni_tmplt(db_path, img_list):
    merger = pe.Node(Merge(), name='merger')
    # merger = Merge()
    # merger.inputs.merged_file = os.path.join(db_path, 'extras', 'merged.nii')
    merger.inputs.in_files = img_list
    merger.inputs.dimension = 't'
    merger.inputs.output_type = 'NIFTI'
    # merger.run()
    mean = pe.Node(MeanImage(), name='mean')
    mean.inputs.output_type = 'NIFTI'
    sm = pe.Node(Smooth(), name='sm')
    sm.inputs.fwhm = 8
    # sm.inputs.output_type = 'NIFTI'
    mean.inputs.out_file = os.path.join(db_path, 'extra', 'mean.nii')

    ppln = pe.Workflow(name='ppln')
    ppln.connect([(merger, mean,  [('merged_file', 'in_file')]),
                 (mean, sm, [('out_file', 'in_files')]),
                  ])
    ppln.run()

    img = nib.load(os.path.join(db_path, 'extra', 'mean.nii'))
    scld_vox = (img.get_data() / img.get_data().max())
    new_img = nib.Nifti1Image(scld_vox, img.affine, img.header)
    nib.save(new_img, os.path.join(db_path, 'extra', 'st_sp_tmpl.nii'))
