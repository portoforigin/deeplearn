# General Notes and Jupyter Notebooks for learning deeply

This repository is a dumpiong ground of notes, side projects, and jupyter notebooks used to explore deep learning.

### AFAIK these are the main deep learning libraries:
- Tensorflow
  - Keras: Abstracted interface built on Tensorflow
- PyTorch
  - FastAI: Abstracted interface built on PyTorch
- MXNET (no clue...)

## Things to explore:
- mlflow: https://mlflow.org/docs/latest/quickstart.html

## Jupyter Notebooks (organized by dataset):
- PETS Dataset: (https://www.robots.ox.ac.uk/~vgg/data/pets/)
  - [PETS_Playground.ipynb](https://colab.research.google.com/github/portoforigin/deeplearn/blob/main/fastai/PETS_Playground.ipynb)
    - Just playing around w/ PETS dataset
  - [PETS_Learning_Rate.ipynb](https://colab.research.google.com/github/portoforigin/deeplearn/blob/main/fastai/PETS_Learning_Rate.ipynb)
    - Expirement w/ tuning Learning Rates  
  - [PETS_SimilarCats_v1.ipynb](https://colab.research.google.com/github/portoforigin/deeplearn/blob/main/fastai/PETS_SimilarCats_v1.ipynb)
    - Challenge problem identifying Birman vs Ragdoll cats
    - First attempt
  - [PETS_SimilarCats_CAM.ipynb](https://colab.research.google.com/github/portoforigin/deeplearn/blob/main/fastai/PETS_SimilarCats_CAM.ipynb)
    - Refined Birman vs Ragdoll cat classifier
    - Started looking at Class Activation Maps (CAM) to understand what is going on...still don't understand...
  - [PETS_Breed_Classifier.ipynb](https://colab.research.google.com/github/portoforigin/deeplearn/blob/main/fastai/PETS_Breed_Classifier.ipynb)
    - Extend classifier to breed classification, work in progress

Where I'm running:
- Google Colab
  - There are limits and I've been locked out for a day due to resource limits w/ the free version
- Local GPU
  - Nvidia GTX1070ti w/ 8GB Memory

## Links:
- FastAI:
  - Course: https://course.fast.ai/videos/?lesson=1
  - Jupyter Notebooks: https://github.com/fastai/fastbook
  - Code: https://github.com/fastai/fastai
- Class Activation Map:
  - FastAI Course Ch18: https://github.com/fastai/fastbook/blob/master/18_CAM.ipynb
  - CAM Applied to AstroPhysics: https://jwuphysics.github.io/blog/galaxies/astrophysics/deep%20learning/visualization/2020/08/27/image-attribution-for-galaxies.html

## Papers:
- TBD
