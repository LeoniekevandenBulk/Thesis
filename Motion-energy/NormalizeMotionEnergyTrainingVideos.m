% Converted Code from https://github.com/gallantlab/motion_energy_matlab
% for the use of a Master thesis in Artificial Intelligence by Leonieke van
% den Bulk.

% Z-scores and crops the Motion Energy features from the Training Videos

% Set Parameters
from_dir = 'MotionEnergyTrainingVideosUnnormalized/';
to_dir = 'MotionEnergyTrainingVideosNormalized/';

% Create folder to write features to or check if it already exists
if ~exist(to_dir, 'dir')
   mkdir(to_dir) 
end

% add relevant paths
addpath('gallantlab-motion_energy_matlab');
addpath('gallantlab-motion_energy_matlab/utils');

% Extract all file names from the folder containint the matrices
files = dir(strcat(from_dir,'*.mat'));

% Loop through all Motion Energy features per video
for i = 1:size(files,1)
    
    % Load file
    file_name = files(i).name;
    disp(file_name)
    fname = strcat(from_dir,file_name);
    file = load(fname);
    S_ds = file.S_fin;
   
    % Z-score and crop
    nrmparams = preprocNormalize_GetMetaParams(3);
    [S_fin, nrmparams] = preprocNormalize(S_ds, nrmparams);

    % Save result to folder
    file_name = strsplit(file_name,'Unnormalized_');
    save_path = strcat(to_dir,file_name{2});
    save(save_path, 'S_fin');

end

