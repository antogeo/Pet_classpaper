function mni_norm(ref_img, mov_img)

    spm_jobman('initcfg');

    matlabbatch{1}.spm.tools.oldnorm.estwrite.subj.source = {ref_img};
    matlabbatch{1}.spm.tools.oldnorm.estwrite.subj.wtsrc = '';
    matlabbatch{1}.spm.tools.oldnorm.estwrite.subj.resample = mov_img(:, 1);
    fprintf('%s \n', class(mov_img))
    fprintf('%s \n', class(ref_img))
    
    matlabbatch{1}.spm.tools.oldnorm.estwrite.eoptions.template = {...
            '/home/antogeo/progs/matlab/spm12/toolbox/OldNorm/PET.nii,1'
            '/home/antogeo/progs/matlab/spm12/toolbox/OldSeg/grey.nii,1'
            '/home/antogeo/progs/matlab/spm12/toolbox/OldSeg/white.nii,1'};
    matlabbatch{1}.spm.tools.oldnorm.estwrite.eoptions.weight = '';
    matlabbatch{1}.spm.tools.oldnorm.estwrite.eoptions.smosrc = 8;
    matlabbatch{1}.spm.tools.oldnorm.estwrite.eoptions.smoref = 0;
    matlabbatch{1}.spm.tools.oldnorm.estwrite.eoptions.regtype = 'mni';
    matlabbatch{1}.spm.tools.oldnorm.estwrite.eoptions.cutoff = 25;
    matlabbatch{1}.spm.tools.oldnorm.estwrite.eoptions.nits = 16;
    matlabbatch{1}.spm.tools.oldnorm.estwrite.eoptions.reg = 2;
    matlabbatch{1}.spm.tools.oldnorm.estwrite.roptions.preserve = 0;
    matlabbatch{1}.spm.tools.oldnorm.estwrite.roptions.bb = [-78 -112 -70
                                                                78 76 85];
    matlabbatch{1}.spm.tools.oldnorm.estwrite.roptions.vox = [2 2 2];
    matlabbatch{1}.spm.tools.oldnorm.estwrite.roptions.interp = 1;
    matlabbatch{1}.spm.tools.oldnorm.estwrite.roptions.wrap = [0 0 0];
    matlabbatch{1}.spm.tools.oldnorm.estwrite.roptions.prefix = 'tmplt_HO_';
    spm_jobman('serial', matlabbatch);