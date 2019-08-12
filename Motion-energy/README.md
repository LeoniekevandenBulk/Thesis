The code in this folder was based on the code by Nishimoto et al. and can be used to determine the motion-energy features of the training set videos and the prior videos. The mean and standard deviation of the motion-energy features of the traning set are used for the prior videos as normalization step, so there is a fixed order to follow the matlab files: 
ComputeUnnormalizedMotionEnergyTrainingVideos.m,
CalculateMeanAndStdTrainingVideos.m,
NormalizeMotionEnergyTrainingVideos.m, 
ComputeMotionEnergyPriorVideos.m

Note that this code only seems to work in Matlab 2016b with the TMW_ALTERNATE_LM_THREAD environment variable set to 1.

Warning: Calculating motion-energy is computationally very costly, and it takes a very long time to calculate the features for a big set of prior videos.
