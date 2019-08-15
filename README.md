# Reconstructing naturalistic movies from fMRI brain responses: Comparing motion-energy features with convolutional neural network representations
GitHub repository for my Masters Thesis in Artificial Intelligence.

My research is based on the work by Nishimoto et al. [1], where the goal was to reconstruct naturalistic movies from fMRI brain responses. We repeated their experiment with a bigger dataset containing 24 hours of densely sampled fMRI data of a single subject watching the television series "Doctor Who". Originally, motion-energy was used to create features for the encoding model. However, motion-energy solely represents lower-order information. We compared the performance of motion-energy features to the performance of an encoding model that uses features from the last layer of a trained convolutional action recognition neural network, called the R(2+1)D network [2], in order to cover more higher-order information. These two types of features were also combined to create an encoding model selective to both lower- and higher-order information. 

The folders in this directory contain the code and reconstructions from my work. More explanation on the code can be found in the READMEs of the respective folders and the comments in the code itself. 


[1] Nishimoto S, Vu AT, Naselaris T, Benjamini Y, Yu B, Gallant JL. (2011). Reconstructing visual experiences from brain activity evoked by natural movies. Current Biology 21(19), 1641-1646. 

[2] Tran, D., Wang, H., Torresani, L., Ray, J., LeCun, Y., & Paluri, M. (2018). A closer look at spatiotemporal convolutions for action recognition. In Proceedings of the IEEE conference on Computer Vision and Pattern Recognition (pp. 6450-6459).