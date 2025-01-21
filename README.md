**AI & Machine Learning (KAN-CINTO4003U) - Copenhagen Business School | Fall 2025**

# Mandatory assignment 1

This repository contains the first mandatory assignment (MA1) for AIML25. 

***

### Group members
| Student name | Student ID |
| --- | --- |
| #NAME# | #ID# |
| #NAME# | #ID# |
| #NAME# | #ID# |

***

## Part 1: Multi-Layer Perceptrons

### 1.1. Introduction



### 1.2. Assignment
In this assignment, your task will be to construct an MLP for the well-known MNIST dataset. Build a 2-layer MLP for MNIST digit classfication in the `nn.ipynb` notebook (assigntments/nn.ipynb).

Feel free to play around with the model architecture and see how the training time/performance changes, but to begin, try the following:

* Input Layer: Image (784 dimensions) ->  
* Fully connected layer (500 hidden units) -> 
* Nonlinearity (ReLU) ->  
* Fully connected (10 hidden units - digits 0-9) -> 

*Some hints*:
- Even as we add additional layers, we still only require a single optimizer to learn the parameters. 
- To get the best performance, you may want to play with the learning rate and the number of training epochs.

***

## Part 2: Convolutional Neural Networks

### 2.1. Introduction
Feedforward neural networks, like Multi-Layer Perceptrons (MLPs), may still struggle with high-dimensional data like images because they don't account for the spatial structure in such data. Convolutional Neural Networks (CNNs) address this by using convolutional layers that focus on small regions of the input, detecting patterns such as edges and textures. This allows CNNs to learn hierarchical features, with early layers capturing simple patterns and deeper layers recognizing more complex ones.

Unlike MLPs, CNNs use convolutional operations instead of fully connected layers. This reduces the number of parameters and makes the model more efficient, while still capturing important spatial relationships in the data. Pooling layers are also used to downsample feature maps, reducing their size and emphasizing the most important features.

<img src="./media/CNN.png" width="500"/>

#### Why Convolutions Matter
Convolutions are what enables CNNs to generalize well in complex data such as images. A convolution is a mathematical operation that applies a small filter to a region of the input, producing a *feature map*. By sharing the same filter across the entire input, the model learns spatially invariant features. This means that patterns like edges or textures can be detected anywhere in the image.

CNNs often combine several important building blocks, including:

- Convolutional Layers: Extract local patterns and generate feature maps.
- Activation Functions: Introduce nonlinearity, often using ReLU to ensure the model can approximate complex functions.
- Pooling Layers: Downsample feature maps to reduce computational complexity and emphasize the most important features.
- Fully Connected Layers: Connect high-level features learned by convolutional layers to output predictions.

### 2.2. Assignment
In this assignment, your task is to build a CNN for multi-class classification on the famous CIFAR-10 dataset. You will practice building networks with convolutional layers and pooling to capture abstract patterns in order to classify different objects.

Build a *6-layer* (or deeper) CNN for object classification in the `cnn.ipynb` notebook (assignments/nn.ipynb).

Feel free to play around with the model architecture - add/delete layers, adjust layer parameters etc. - and see how the training time/performance changes, but start with the following:

* 2D Convolutional layer
* Non-linearity (ReLU)
* Max pooling layer
* Flatten the output from the last convolutional layer (this part is already given in the code)
* Fully Connected Layer with 128 hidden units
* Non-linearity (ReLU)
* Fully Connected Layer (output layer for 10 classes)

*Some hints*:
- You need to define the number of filters (also known as channels) in each convolutional layer. Try starting with a small number (e.g., 16 or 32) and increase it in deeper layers.
- The kernel size determines how much of the image each filter sees at a time. Smaller kernels (like 3x3) are commonly used and work well.
- Pick a small number for stride and padding. Padding helps preserve the spatial size of the input, and stride determines how much the kernel moves.
- Pooling layers reduce the spatial dimensions of the feature maps and help reduce computation and overfitting. A common choice is to use 2x2 max pooling with a stride of 2, which halves the spatial dimensions.


___
___



### Submission

Please see sections VII - XI in the `AIML25-dev-setup.pdf` document on Canvas for instructions on 

- how to submit your assignment.
- setting up your development environment.
- how to work with the assignment repository.
- setting up virtual environments.
- using git and GitHub.