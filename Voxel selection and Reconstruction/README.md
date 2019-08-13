The code in this folder was used to go from the encoding models to actual reconstructions. 
The script "voxel_selection.py" selects the top 2000 voxels with their best hemodynamic delay from given encoding models. The script "combine_voxel_selections.py" combines two voxel selections into a combined voxel selection, keeping only the best or unique voxels.

With these voxel selections, we can determine the best prior videos for the reconstruction using the script "all_models_saving_best_priors_parallel.py". This makes use of the already computed motion-energy features on the prior set, but computes the R(2+1)D features on-the-fly. It is advised to run multiple instances of this script to speed up computation time. The output of all the instances can be transformed to one final set of prior videos via "combining_best_prior_dictionaries.py".

This final set is transformed in an actual reconstruction in "reconstructing_videos.py". 
As a baseline, a random reconstruction can also be created via "make_random_reconstruction.py".

