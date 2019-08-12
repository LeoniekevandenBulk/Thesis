# -*- coding: utf-8 -*-
"""
Adapted code from https://github.com/jfzhang95/pytorch-video-recognition/blob/master/dataloaders/dataset.py
by Leonieke van den Bulk in order to load the kinetics dataset
"""

import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    """A Dataset for a folder of videos. Expects the directory structure to be
    directory->[train/val/test]->[class labels]->[videos]. Initializes with a list
    of all file names, along with an array of labels, with label being automatically
    inferred from the respective folder names.
        Args:
            dataset_dir (str): Directory of dataset. Defaults to CURRENT DIR/kinetics
            split (str): Determines which folder of the directory the dataset will read from. Defaults to 'train'.
            clip_len (int): Determines how many frames are there in each clip. Defaults to 16.
    """

    def __init__(self, dataset_dir=os.path.join(os.getcwd(),'kinetics'), split='train', clip_len=16):
        self.dataset_dir = dataset_dir
        self.clip_len = clip_len
        self.split = split

        if not os.path.exists(self.dataset_dir):
            raise RuntimeError("Dataset not found or corrupted." +
                               " You need to download it from official website.")

        # Obtain all the filenames of files inside all the class folders
        # Going through each class folder one at a time
        self.fnames, labels = [], []
        folder = os.path.join(self.dataset_dir, split)
        for label in sorted(os.listdir(folder)):
            for fname in os.listdir(os.path.join(folder, label)):
                self.fnames.append(os.path.join(folder, label, fname))
                labels.append(label)

        assert len(labels) == len(self.fnames)
        print("Number of {} videos: {:d}".format(split, len(self.fnames)))

        # Prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
        # Convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)


    def __len__(self):
        """Returns the amount of videos in dataset"""
        return len(self.fnames) 


    def __getitem__(self, index):
        """Loads clip_len frames from a video, preprocesses it and returns it as pytorch tensor"""
        # Load clip_len consequtive frames from a random point in the video and save the label
        frames = self.load_frames(self.fnames[index])
        labels = np.array(self.label_array[index])
        
        # Preprocessing
        if self.split == 'train':
            frames = self.randomflip(frames)
        frames = self.normalize(frames)
        
        # Return as pytorch tensor with the right order of dimensions
        frames = frames.transpose((3, 0, 1, 2))
        return torch.from_numpy(frames), torch.from_numpy(labels)


    def randomflip(self, frames):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

        if np.random.random() < 0.5:
            for i, frame in enumerate(frames):
                frame = cv2.flip(frames[i], flipCode=1)
                frames[i] = cv2.flip(frame, flipCode=1)

        return frames


    def normalize(self, frames):
        """Normalize frames by subtracting the mean and dividing by the std"""
        for i, frame in enumerate(frames):
            frame = frame - np.array([[[110.201, 100.64, 95.9966]]]) # Subtract mean
            frame = frame / np.array([[[58.1489, 56.4701, 55.3324]]]) # Divide by std
            frames[i] = frame

        return frames


    def load_frames(self, file_dir):
        """Load clip_len consequtive frames from a random point in the video and transform to a numpy array"""
        # Load video with cv2
        capture = cv2.VideoCapture(file_dir)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if(frame_count < self.clip_len):
            raise RuntimeError("Amount of frames in a video was smaller than the requested amount of frames.")
        
        # Save clip_len consequtive frames from a random point in the video
        frames = np.empty((self.clip_len, frame_height, frame_width, 3), np.dtype('float32'))
        start_frame = np.random.randint(frame_count-(self.clip_len+1))
        fr = 0
        while (fr < start_frame+self.clip_len):
            success, frame = capture.read()
            frame = frame[:,:,(2,1,0)] # Swap from BGR to RGB encoding
            if(not(success)):
               raise RuntimeError("Frame could not be read")
            if(fr >= start_frame):
                frames[fr-start_frame] = frame.astype(np.float32)
            fr += 1
        capture.release()
        
        return frames