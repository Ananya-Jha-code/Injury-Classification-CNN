# Injury-Classification-CNN
A CNN model to predict the type of injury using PyTorch

**CNN** (Convolutional Neural Network) is a class of deep neural network, most commonly applied to analyze visual imagery. CNN is a Deep Learning algorithm which can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other.

**Pytorch** is an open source machine learning library based on the Torch library, used for applications such as computer vision and natural language processing.

## Contents 
1. Libraries used
2. Uploading datasets for use
3. Transformations
4. Loading datasets
5. Building and training network
6. Function for validation pass
7. Training classifier
8. Test accuracy

### 1. Libraries used
  ''import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import cv2
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models''
