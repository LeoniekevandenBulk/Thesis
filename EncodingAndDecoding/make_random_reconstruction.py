# -*- coding: utf-8 -*-
"""
Created on Tue May 28 15:25:14 2019

@author: Leonieke

Script to make a reconstruction based on randomly drawn clips from the prior in 
order to do a Wilcoxon rank-sum test between this random reconstruction and the 
actual reconstructions
"""

import os
import cv2
import numpy as np

# Set files and directories
prior_names_file = "../Prior/prior_names.txt" # File with all prior files listed
clips_dir = "../Prior/Clips"
save_dir = "Random_Reconstruction_Top100"
top = 100
TRs = 919
normalization = True
reconstruct_individual_TRs = False

# Check if save directory already exists, else make it
if not(os.path.isdir(os.path.join(os.getcwd(), save_dir))):
    os.makedirs(os.path.join(os.getcwd(), save_dir))

# Put all prior names in a list and randomize it
prior_names = []
with open(prior_names_file, 'r') as txt:
    for name in txt:
        prior_names.append(name.strip())
np.random.shuffle(prior_names)

# Intialize name and path for final conatenated reconstruction video
concatenated_save_name = "Reconstruction_TestVideo_Top" + str(top) + "_Concatenated.mp4"
concatenated_save_path = os.path.join(os.getcwd(), save_dir, concatenated_save_name)
concatenated_out = cv2.VideoWriter(concatenated_save_path, cv2.VideoWriter_fourcc('m','p','4','v'), 22.86, (112,112))

# Loop over needed TRs and select a subset of top clips from the random list
index=0
for TR in range(TRs):
    
    print("TR:" + str(TR))
    
    # Initialize matrix to store all frames of the top clips
    clips = np.empty((top, 16, 112, 112, 3), np.dtype('float32'))

    # Initialize mean and std for final normalization
    means = []
    stds = []
    clip_corrs = []
    
    # Loop over top amounts of clips per TR
    nr_of_clips = 0
    while(nr_of_clips < top):
        
        # Set the clip
        clip_name = prior_names[index]
        index = index+1
        print(clip_name + "," + str(index))
        
        # Check if the clip_name is a valid one
        if(not "-" in clip_name):
            continue
        
        # Set the path for the clip
        clip_path = os.path.join(os.getcwd(), clips_dir, clip_name)

        # Load video with cv2
        capture = cv2.VideoCapture(clip_path)
        width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if(not int(width) is 112 or not int(height) is 112):
            continue
        
        # Save frames as matrix
        frames = np.empty((16, 112, 112, 3), np.dtype('float32'))
        fr = 0
        while (fr < 16):
            success, frame = capture.read()
            #frame = frame[:,:,(2,1,0)] # Swap from BGR to RGB encoding
            if(not(success)):
               print("Frame could not be read")
               continue
            frames[fr] = frame.astype(np.float32)
            fr += 1
        capture.release()
        
        # Update mean and std with this clip's mean and std
        mean = np.mean(frames)
        std = np.std(frames)
        if(std == 0):
            continue
        means.append(mean)
        stds.append(std)

        # Normalize clip to have unit std if normalization = True and save
        if(normalization):
            normalized_frames = frames/std
            clips[nr_of_clips] = normalized_frames
            
        # Else just save the read frames as they are
        else:
            clips[nr_of_clips] = frames
        
        nr_of_clips = nr_of_clips+1
    
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