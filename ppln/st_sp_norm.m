function arg1 = st_sp_norm(ref_img, mov_img, tmplt)

  spm_jobman('initcfg');

  matlabbatch{1}.spm.tools.oldnorm.estwrite.subj.source = {ref_img};
  matlabbatch{1}.spm.tools.oldnorm.estwrite.subj.wtsrc = '';
  matlabbatch{1}.spm.tools.oldnorm.estwrite.subj.resample = mov_img(:, 1);
  fprintf('%s \n', class(mov_img))
  fprintf('%s \n', class(ref_img))

  matlabbatch{1}.spm.tools.oldnorm.estwrite.eoptions.template = {tmplt};
  matlabbatch{1}.spm.tools.oldnorm.estwrite.eoptions.weight = '';
  matlabbatch{1}.spm.tools.oldnorm.estwrite.eoptions.smosrc = 8;
  matlabbatch{1}.spm.tools.oldnorm.estwrite.eoptions.smoref = 0;
  matlabbatch{1}.spm.tools.oldnorm.estwrite.eoptions.regtype = 'mni';
  matlabbatch{1}.spm.tools.oldnorm.estwrite.eoptions.cutoff = 25;
  matlabbatch{1}.spm.tools.oldnorm.estwrite.eoptions.nits = 16;
  matlabbatch{1}.spm.tools.oldnorm.estwrite.eoptions.reg = 3;
  matlabbatch{1}.spm.tools.oldnorm.estwrite.roptions.preserve = 0;
  matlabbatch{1}.spm.tools.oldnorm.estwrite.roptions.bb = [-78 -112 -70      %%%  -78 -112 -70 Giati einai 70 se mena kai 50 ston Christoph??
                                                           78 76 85];
  matlabbatch{1}.spm.tools.oldnorm.estwrite.roptions.vox = [2 2 2];
  matlabbatch{1}.spm.tools.oldnorm.estwrite.roptions.interp = 1;
  matlabbatch{1}.spm.tools.oldnorm.estwrite.roptions.wrap = [0 0 0];
  % matlabbatch{2}.spm.spatial.smooth.data(1) = cfg_dep('Old Normalise: Estimate & Write: Normalised Images (Subj 1)', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('()',{1}, '.','files'));
  % matlabbatch{1}.spm.spatial.smooth.fwhm = [14 14 14];
  % matlabbatch{1}.spm.spatial.smooth.dtype = 0;
  % matlabbatch{1}.spm.spatial.smooth.im = 0;
  matlabbatch{1}.spm.tools.oldnorm.estwrite.roptions.prefix = 'sec_';

  spm_jobman('serial', matlabbatch);
