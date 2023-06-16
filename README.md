# Opti-ML
**Course project - Optimization for Machine Learning - EPFL - 2023**
## Team members
- Florence Osmont
- Haochen Su
- Benoît Müller

## Install
Language: **Python**

To run the code of this project, you need to install the libraries listed here:
- matplotlib
- numpy
- torch
- torchvision
- tqdm

The following libraries can also be installed inside the notebooks:
- torch_cka (https://github.com/AntixK/PyTorch-Model-Compare/tree/main)
- anatome through the command : `pip install -U git+https://github.com/moskomule/anatome` (https://github.com/moskomule/anatome/tree/master)
- adahessian (https://github.com/amirgholami/adahessian/tree/master)

## Folders
- [`model_weights`](model_weights) contains the computed models
- [`graphics`](graphics) all the graphics obtained
- [`loss_and_acc`](loss_and_acc) the loss and accuracy obtained at each epochs during the training

## Run
The runnable files are the `.ipynb` python notebooks files, and are adapted to be runned on Google Colab. The paths are set such that the content of this git is in a folder named `Opti-ML` in the root of Drive.


To get the training results presented in the paper, run the 'train_and_test.ipynb' notebook. All of the parameters can be modified to train. One can choose to use the pretrained version of ResNet18 or initialize the weight with a seed. Changing the optimizer_name to 'SGD' 'adahessian' or 'adam' to train with different optimizers. The end of the notebook has some visualized result of our best model. The path to best model is included in the notebook.


To get the results presented in the paper from the precomputed models, run the `MeasureSimilarity.ipynb` notebook. Inside, the models to compare can be chosen and it is possible to run the comparaison for two specific models or for a list of them. To obtain the results on the normalization, run the `normalization.ipynb` notebook.

Functions used in 'train_and_test.ipynb' notebook and `MeasureSimilarity.ipynb` notebook are defined in optimizer.py
