% Load spm_check_reg() for a list of subjects in the "subjects" folder.
% folders structure: subjects -> subject1, subject2, subject3 ... ->
% -> niftis including keyword (line 25)
% - Chose via the first GUI the folder containing the subjects ()
% - chose via the GUI the template you want to use
% - change subject by clicking enter in the command window
% antogeo 2018

% clear memory and command line (comment out if not needed)
clear; clc;
[~,pc_name]= system('hostname');
if regexp(string(pc_name), 'comameth')
    db_path = '/home/coma_meth/Documents/PET/pet_suv_db/Liege/subjects';
elseif regexp(pc_name, 'antogeo-XPS')
    db_path = '/home/antogeo/data/PET/pet_suv_db/Liege/subjects';
elseif regexp(pc_name, 'antogeo')
    db_path = 'home/antogeo/Documents/';
end

subj_name = dir(db_path);
for i =3: size(subj_name, 1)
    files = dir(fullfile(db_path, subj_name(i).name));
    % look for files containing keyword
    subjs_suv = files(contains({files.name}, "nsSUV.nii"));
    subjs_raw = files(contains({files.name}, "nsRAW.nii"));
    if ~isempty(subjs_raw) || ~isempty(subjs_suv)
        fprintf("Subject #%d \n", i-2);
        new_folder = fullfile(db_path, subj_name(i).name);
        oldFolder = cd(new_folder);
        mni_norm( ...
            fullfile(db_path, subj_name(i).name, subjs_suv.name), ...
            {fullfile(db_path, subj_name(i).name, subjs_suv.name); ...
            fullfile(db_path, subj_name(i).name, subjs_raw.name)})
        cd(oldFolder)
    end
end
