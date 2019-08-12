# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 12:55:25 2019

@author: Leonieke

Script to make reconstruction videos on basis of the top 100 prior clips per TR for features
"""

import os
import cv2
import pickle
import numpy as np

# Set files and directories to load and choose how many clips from the top to reconstruct with
top_clips_file = "ME_top_100_best_overall_clips.pkl"
clips_dir = "../Prior/Clips"
save_dir = "ME_Reconstructions_Top100_with_normalization"
top = 100
normalization = True
reconstruct_individual_TRs = False

# Check if save directory already exists, else make it
if not(os.path.isdir(os.path.join(os.getcwd(), save_dir))):
    os.makedirs(os.path.join(os.getcwd(), save_dir))

# Load matrix containing the top 100's per test video
top_clips = pickle.load(open(top_clips_file, 'rb'))#, encoding='latin1')

# Intialize name and path for final conatenated reconstruction video
concatenated_save_name = "Reconstruction_TestVideo_Top" + str(top) + "_Concatenated.mp4"
concatenated_save_path = os.path.join(os.getcwd(), save_dir, concatenated_save_name)
concatenated_out = cv2.VideoWriter(concatenated_save_path, cv2.VideoWriter_fourcc('m','p','4','v'), 22.86, (112,112))

# Loop over the top 100's for each TR within the test videos
for TR in range(len(top_clips)):

    print("TR:" + str(TR))
    
    # Initialize matrix to store all frames of the 100 clips
    clips = np.empty((top, 16, 112, 112, 3), np.dtype('float32'))

    # Initialize mean and std for final normalization
    means = []
    stds = []
    clip_corrs = []
    
    # Loop over tuples with correlations and clip names in the top 100
    for num,tup in enumerate(top_clips[str(TR)]):

        # Retrieve correlation and clip name and set its path
        clip_corr = tup[0]
        clip_corrs.append(clip_corr)
        clip_name = tup[1] + ".mp4"
        clip_path = os.path.join(os.getcwd(), clips_dir, clip_name)
        
        # Load video with cv2 and save clip as matrix
        capture = cv2.VideoCapture(clip_path)
        frames = np.empty((16, 112, 112, 3), np.dtype('float32'))
        fr = 0
        while (fr < 16):
            success, frame = capture.read()
            if(not(success)):
                print(clip_name)
                raise RuntimeError("Frame could not be read")
            frames[fr] = frame.astype(np.float32)
            fr += 1
        capture.release()

        # Update mean and std with this clip's mean and std
        mean = np.mean(frames)
        std = np.std(frames)
        means.append(mean)
        stds.append(std)

        # Normalize clip to have unit std if normalization = True and save
        if(normalization):
            normalized_frames = frames/std
            clips[num] = normalized_frames
            
        # Else just save the read frames as they are
        else:
            clips[num] = frames
        
        # Break if the amount of numbers in top have been reached
        if(num == top-1):
            break

    # Average all normalized clips
    avg_clip = np.mean(clips,axis=0)

    # Post-normalize to average mean and std of all clips if normalization = True
    if(normalization):
        # Transform mean and std to the average mean and std of the top clips
        avg_clip = np.mean(means) + ((avg_clip - np.mean(avg_clip)) * (np.mean(stds)/np.std(avg_clip)))
        
        # Clip values to the right range of 0-255
        avg_clip = np.clip(avg_clip,0,255)
        
    # Save video for this TR
    if(reconstruct_individual_TRs):
        # If reconstruct_individual_TRs is True, each separate TR is saved as video as well
        save_name = "Reconstruction_TestVideo_TR" + str(TR) + "_Top" + str(top) + ".mp4"
        save_path = os.path.join(os.getcwd(), save_dir, save_name)
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('m','p','4','v'), 22.86, (112,112))
        
        for frame_num in range(avg_clip.shape[0]):
            frame = avg_clip[frame_num]
            out.write(np.uint8(frame))
            concatenated_out.write(np.uint8(frame))
        
        out.release()
        
    else:
        # Else only save the concatenated video of all TRs
        for frame_num in range(avg_clip.shape[0]):
            frame = avg_clip[frame_num]
            concatenated_out.write(np.uint8(frame))

concatenated_out.release()