function [] = ComputeMotionEnergyPriorBatchJob(txt_number)
% Converted Code from https://github.com/gallantlab/motion_energy_matlab
% for the use of a Master thesis in Artificial Intelligence by Leonieke van
% den Bulk.

% This code calculates the Motion Energy features of the prior clips in
% mp4 format from a specified txt file given as input as a number, e.g. X 
% which belongs to a file called 'prior_namesX.txt'. This was made this way
% in order to parallelize the motion-energy computations. This code requires
% there to be a directory filled with txt files that contain the names of the
% collected prior clips, this can be set in the variable txt_path below.

    % Check if input is a string
    disp(txt_number)
    if(isa(txt_number, 'char') == 0)
        disp('Either none or a wrong type was given as input.')
        return
    end
	
    %% Setup
    from_dir = '../Prior/Clips/';
    to_dir = 'MotionEnergyPriorClips/';
	txt_path = strcat('../Prior/prior_names',txt_number,'.txt');
    disp(from_dir)
    disp(to_dir)
    
    % add relevant paths
    addpath('gallantlab-motion_energy_matlab');
    addpath('gallantlab-motion_energy_matlab/utils');

    % Create folder to write features to or check if it already exists
    if ~exist(to_dir, 'dir')
       mkdir(to_dir) 
    end

    % Extract all file names from a txt file containing the names of the files
    files = {};
    fid = fopen(txt_path);
    tline = fgetl(fid);
    ind = 1;
    while ischar(tline)
        files{ind} = tline;
        tline = fgetl(fid);
        ind = ind+1;
    end
    fclose(fid);

    %% Start loop through all clips
    for i = 1:size(files,2)
        
        disp(i)
        file_name = files{i};

        % Check if the clip is a subclip of an original full length clip
        if(size(strfind(file_name,'-'),1) > 0)

            % Set file to be loaded
            disp(file_name);
            file_path = strcat(from_dir,file_name);

            %% Load clips
            % Load the clips as mats files
             try
                reader = VideoReader(file_path);
                
                if(not(reader.width == 112 && reader.height == 112))
                    error = ['The file ', file_path, ' does not have the right dimensions'];
                    disp(error)
                    continue
                end
                
                resized_frames = zeros(96,96,3,17); %Frames need to be 96x96
                ii = 1;
                while hasFrame(reader)
                    frame = readFrame(reader);
                    resized_frames(:,:,:,ii) = imresize(frame,[96 96]);
                    ii = ii+1;
                end
                resized_frames(:,:,:,17) = []; % Delete last frame as it does not contain relevant information
                clear reader;
            % Display if a file could not be transformed to a matrix
             catch exception
                error = ['The file ', file_path, ' could not be read'];
                disp(error)
                disp(exception.message)
                continue
            end

            % the variable resized_frames is an array that is (96 x 96 x 3 x 16); (X x Y x Color x
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
        else
            continue
        end
    end
end