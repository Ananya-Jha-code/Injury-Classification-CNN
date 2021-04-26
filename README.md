# Injury-Classification-CNN
A CNN model to predict the type of injury using PyTorch

**CNN** (Convolutional Neural Network) is a class of deep neural network, most commonly applied to analyze visual imagery. CNN is a Deep Learning algorithm which can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other.

**Pytorch** is an open source machine learning library based on the Torch library, used for applications such as computer vision and natural language processing.

I am using GPU in Google Colab.

    import torch
    train_on_gpu = torch.cuda.is_available()
    if not train_on_gpu:
       print('CUDA is not available.  Training on CPU ...')
    else:
       print('CUDA is available!  Training on GPU ...')
    -CUDA is available!  Training on GPU ...
    
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
    import numpy as np
    import pandas as pd
numpy and pandas libraries are used to work with arrays

    import matplotlib.pyplot as plt
matplotlib is used for plotting

    import seaborn as sb
Seaborn library is used to ease the challenging task of data visualization

    import cv2
CV-Python is a library of Python bindings designed to solve computer vision problems.

    import torch
    from torch import nn
    from torch import optim
    import torch.nn.functional as F
    from torchvision import datasets, transforms, models
import torch library to imply pytorch in the model.
nn is a neural network module in torch.
optim is a gradient descent optimizer. 
torch.nn.functional is a module used to for additional functions.
datasets, transforms, models are other such modules in torchvision to write a cnn model.

### 2. Uploading datasets to use
As I am using Google colab, which is not hosted on my local computer, the files are not accessible. 
To access the files I uploaded the files containing the datasets on my Google Drive. 
The Google Drive can be accessed by Google colab by the process of mounting.

    from google.colab import drive
    drive.mount("/content/drive")
    - Mounted at /content/drive
    
    !ls "/content/drive/MyDrive/injury"
    -test  train  valid
    
    data_dir = '/content/drive/MyDrive/injury'
    train_dir = '/content/drive/MyDrive/injury/train'
    valid_dir = '/content/drive/MyDrive/injury/valid'
    test_dir = '/content/drive/MyDrive/injury/test'
The folder injury is successfully uploaded on MyDrive. Injury contains the files test, train, valid. 
It is a great procedure to split the dataset in three parts, training, testing and validation so that the model can train on one, test on the other and have cross validation in the third. It improves performance.
I assigned each folder to a directory so that it can be accessible in the later parts of the model.

## 3.Transformations
    training_transforms = transforms.Compose([transforms.RandomRotation(30),
                                              transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485,0.456,0.406],
                                                                   [0.229,0.224,0.225])])
The dataset may not contain all the pictures of the same size, colour, angle of rotation, etc. For the model to work efficiently, we need to standardize the dataset by running a series of transformations. 
transforms.RandomRotation(30) randomly rotates the image by an angle 30 degrees. 
RandomResizedCrop(224) randomly crops the image and resizes it. RandomHorizontalFlip flips the image horizontally. 
Normalize exists in tensor module that is why I used .ToTensor. 
All the values used here are hyperparameters and can be tuned to get the desired output.
Similar process is used for testing_transforms and validation_trandforms.

## 4.Loading Datasets
    training_dataset = datasets.ImageFolder(train_dir, transform=training_transforms)
I used the ImgFolder existing in the datasets module to pass directory. The parameters used are the directory containing the dataset and the transformations to be performed on the same.

    train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=64, shuffle=True)
The uploaded dataset is then loaded in the train_loader using torch.utils.data.DataLoader. The parameters used are the dataset, batch size, and shuffle enable. 
Batch size is a hyperparameter. Any value can be assigned based on the memory available. If there is an error regarding the memory unavailability reduce the batch size, preferably in the power of 2. Eg. 64 to 32.
Shuffle is enabled as the model be biased. To have a fair prediction, we shuffle the pictures.
Similar process is followed for testing and validation.

### 5.Building and Training network
    from torchvision import models
    model = models.vgg16(pretrained=True)
    model
For building the network, I have used a pretrained model called vgg16.

    VGG(
       (features): Sequential(
        (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU(inplace=True)
        (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (3): ReLU(inplace=True)
        (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        .
        .
        .
The list that is displayed is the architecture of the pretrained model vgg16. It acts like a feature selector. 
In this model, ReLu is used as the activation function.

     (6): Linear(in_features=4096, out_features=1000, bias=True)
The classifier in this model uses 1000 different linear features but we have only 3 classes, so we have to make a custom classifier.

    for parameter in model.parameters():
        parameter.requires_grad = False
While making a classifier, we need not backpropogate on a pretrained model. So we freeze gradient descent by 
parameter.requires_grad = False. This will save time as well as memory.

    from collections import OrderedDict
    from torch import nn
    classifier = nn.Sequential(OrderedDict([('fc1',nn.Linear(25088,5000)),
                                            ('relu',nn.ReLU()),
                                            ('drop',nn.Dropout(p=0.5)),#to avoid overfitting
                                            ('fc2',nn.Linear(5000,102)),
                                            ('output',nn.LogSoftmax(dim=1))]))
To build the classifier, we use sequential function from nn module. 
('fc1',nn.Linear(25088,5000)) is the input layer. The input is taken linearly and has features 25088, the same as mention in the vgg16 model. 
5000 is the number of neurons of the next layer.
('relu',nn.ReLU()) is used for the linear output. Here ReLu is used as the activation function.
('drop',nn.Dropout(p=0.5)) is the dropout function. We use it to avoid overfitting. It is set to 0.5 which is a hyperparameter and can be changed. 
0.5 represents that 50% can be dropped out if the model is trainig more on one node.
('fc2',nn.Linear(5000,102)) is a hidden layer with 5000 neurons.
('output',nn.LogSoftmax(dim=1)) is the output layer. If the model was a binary classifier we could use sigmoid function which would give probabilities 0 or 1. 
But as we have 3 classes, we use LogSoftmax function which gives probabilities in the form of log but within 0 and 1.

### 6.Function for validation pass
     def validation (model, validateloader, criterion):
    val_loss = 0
    accuracy = 0
    for images, labels in iter(validateloader):
    images, labels = images.to('cuda'), labels.to('cuda')
    output = model.forward(images)
    val_loss += criterion(output, labels).item()

    probabilities = torch.exp(output)
    equality = (labels.data == probabilities.max(dim=1)[1])
    accuracy += equality.type(torch.FloatTensor).mean()
    return val_loss, accuracy
We define a validation function to see how the network is improving.
The function takes model, validate loader and criterion as the parameters. The val_loss and accuracy has been to set to 0 and will be updated as and when the code runs.
Images and labels are converted into cuda. The probabilities are obtained using the exp function, which is used for exponential purposes as the LogSoftmax function returns probabilities in the form of log.

    from torch import optim
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
To minimize the loss, we use a loss function NLLoss(). This function works well with LogSoftmax.
Optimizer is a gradient descent optimize. Adam increases the chance of finding a local minimum.
It uses two parameters, the model classifier parameters and the learning rate. lr is again a hyperparameter.

### 7.Training classifier
We define a train_classifier to train the classifier we built earlier. 
Epoch is a hyperparameter and can be tuned as per requirement.

    for e in range(epochs):
        
            model.train()
    
            running_loss = 0
    
            for images, labels in iter(train_loader):
        
                steps += 1
        
                images, labels = images.to('cuda'), labels.to('cuda')
        
                optimizer.zero_grad()

                output = model.forward(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
        
                running_loss += loss.item()
We run a for loop for the specified epoch number of times. running_loss and steps will be updated as and when the code runs.
model.train indicates that the model is set to training mode.
optimizer.zero_grad() is written so that after every loop the gradient is set to zero and not added up.
In the output the model.forward indicates a forward pass.
optimizer.step() is used so that the gradient descent step updates the weights and biases.
              
               if steps % print_every == 0:
                
                    model.eval()
                
                    # Turn off gradients for validation, saves memory and computations
                    with torch.no_grad():
                      validation_loss, accuracy = validation(model, validate_loader, criterion)
            
                    print("Epoch: {}/{}.. ".format(e+1, epochs),
                          "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                          "Validation Loss: {:.3f}.. ".format(validation_loss/len(validate_loader)),
                          "Validation Accuracy: {:.3f}".format(accuracy/len(validate_loader)))
            
                    running_loss = 0
                    model.train()
                    
    train_classifier()
We use an if statement that if the steps divided by print_every does not give any remainder, then the model can be set to evaluation mode.
The training loss, validation loss and the validation accuracy is printed. Ideally the training loss should decrease and the validation accuracy should increase consistently.
If not, then tune the hyperparameters.

### 8.Test accuracy
We define a test_accuracy function which takes two parameters, the model which is used and test_loader. The output states the test accuracy.
