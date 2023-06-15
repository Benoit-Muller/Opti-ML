# Opti-ML
**Course project - Optimization for Machine Learning - EPFL - 2023**
## Team members
- Florence Osmont
- Haochen Su
- Benoît Müller

## Install
To run the code of this project, you need to install the libraries listed here:
- matplotlib
- numpy
- torch
- torchvision
- tqdm

The following two libraries can also be installed inside the notebooks:
-torch_cka (https://github.com/AntixK/PyTorch-Model-Compare/tree/main)
-anatome through the command : pip install -U git+https://github.com/moskomule/anatome (https://github.com/moskomule/anatome/tree/master)

## Folders
- model_weights contains the precomputed models
- graphics all the graphics obtained
- loss_and_acc the loss and accuracy obtained at each epochs during the training
- 
## Run
The runnable files are the `.ipynb` python notebooks files, and are adapted to be runned on Google Colab. The paths are set such that the content of this git is in a folder named Opti-ML in the root of Drive.
To get the results presented in the paper from the precomputed models, run the MesureSimilarity notebook. Inside, the models to compare can be chosen and it is possible to run the comparaison for two specific models or for a list of them. To obtain the results on the normalization, run the normalization notebook.

To recompute the models, run the SGD and the Adahessian notebooks.
