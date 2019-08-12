% Code created for the use of a Master thesis in Artificial Intelligence by 
% Leonieke van den Bulk.
%
% Calculate mean and variance of the Motion Energy features of the Training
% Videos set

% Set Parameters
from_dir = 'MotionEnergyTrainingVideosUnnormalized/';
files = dir(strcat(from_dir,'*.mat'));
numArrays = size(files,1); %Amount of videos

% Create cell to save all matrices in
A = cell(numArrays,1);
for n = 1:numArrays
    file = load(strcat(from_dir, 'Unnormalized_ME_TrainingVideorun_', ...
        num2str(n.','%03d')));
	A{n} = file.S_fin;
end

% Calculate mean and variance across cells
m = mean(cell2mat(A));
s = std(cell2mat(A));

% Save to struct
meanAndStd = struct('mean',m,'std',s);
save_path = strcat(from_dir,'meanAndStd.mat');
save(save_path, 'meanAndStd');