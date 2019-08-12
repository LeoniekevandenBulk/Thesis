% Converted Code from https://github.com/gallantlab/motion_energy_matlab
% for the use of a Master thesis in Artificial Intelligence by Leonieke van
% den Bulk.

% This code calculates the Motion Energy features of the test videos
% without cropping them

%% Setup
% To run this demo, you will need a stack of movie frames stored as a 4D 
% (X x Y x Color x Time) array in the folder to be set below.
clear;
from_dir = '../Data/DoctorWho_TestVideos/';
to_dir = 'MotionEnergyTestVideos/';

% add relevant paths
addpath('gallantlab-motion_energy_matlab');
addpath('gallantlab-motion_energy_matlab/utils');

% Create folder to write features to or check if it already exists
if ~exist(to_dir, 'dir')
   mkdir(to_dir) 
end

% Extract all file names from the folder containing the matrices
files = dir(strcat(from_dir,'*.webm'));

%% Start loop through all clips
for i = 1:size(files,1)

    %% Load images
    % Set name of clip to be transformed
    file_name = files(i).name;
    disp(file_name)
    file_path = strcat(from_dir,file_name);
    
    % Try to read the test video
    try
        reader = VideoReader(file_path);
        number_of_frames = floor(reader.Duration*reader.FrameRate);
        resized_frames = zeros(96,96,3,number_of_frames); %Frames need to be 96x96
        ii = 1;
        while hasFrame(reader)
            frame = readFrame(reader);
            resized_frames(:,:,:,ii) = imresize(frame,[96 96]);
            ii = ii+1;
        end
        clear reader;
    % Display if a file could not be transformed to a matrix
    catch
        error = strcat('The file ',file_path,' could not be read');
        disp(error);
        return
    end
    
    % the variable resized_frames is an array that is (96 x 96 x 3 x Frames); (X x Y x Color x
    % Images).  The images are stored as 8-bit integer arrays (no decimal
    % places, with pixel values from 0-255). These should be converted to
    % floating point decimals from 0-1:
    S  = single(resized_frames)/255;

    %% Preprocessing
    % Conver to grayscale (luminance only)
    % The argument 1 here indicates a pre-specified set of parameters to feed
    % to the preprocColorSpace function to convert from RGB images to 
    % luminance values by converting from RGB to L*A*B colorspace and then
    % keeping only the luminance channel. (You could also use matlab's
    % rgb2gray.m function, but this is more principled.) Inspect cparams to see
    % what those parameters are.
    cparams = preprocColorSpace_GetMetaParams(1);
    [S_lum, cparams] = preprocColorSpace(S, cparams);

    %% Gabor wavelet processing
    % Process with Gabor wavelets
    % The numerical argument here specifies a set of parameters for the
    % preprocWavelets_grid function, that dictate the locations, spatial
    % frequencies, phases, and orientations of Gabors to use. 2 specifies Gabor
    % wavelets with three different temporal frequencies (0, 2, and 4 hz),
    % suitable for computing motion energy in movies.  
    gparams = preprocWavelets_grid_GetMetaParams(2);
    [S_gab, gparams] = preprocWavelets_grid(S_lum, gparams);

    %% Optional additions
    % Compute log of each channel to scale down very large values
    nlparams = preprocNonLinearOut_GetMetaParams(1);
    [S_nl, nlparams] = preprocNonLinearOut(S_gab, nlparams);

    % Downsample data to the sampling rate of your fMRI data (the TR)
    dsparams = preprocDownsample_GetMetaParams(7); % (1) for TR=1; use (2) for TR=2
    [S_ds, dsparams] = preprocDownsample(S_nl, dsparams);

    % Z-score each channel
    nrmparams = preprocNormalize_GetMetaParams(3);
    [S_fin, nrmparams] = preprocNormalize(S_ds, nrmparams);
    
    % Save result to folder
    save_path = strcat(to_dir,'ME_',strtok(file_name,'.'),'.mat');
    save(save_path, 'S_fin');
end