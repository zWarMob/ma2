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

Traditional machine learning methods, like logistic regression, are often sufficient for datasets that are linearly separable. However, more complex problems, such as image recognition and natural language understanding/processing (NLU/P), sometimes demand a more intricate approach. Neural networks, which incorporate additional layers, excel at learning non-linear relationships. These extra layers, known as *hidden* layers, process the input into one or more intermediate forms before generating the final prediction.

Logistic regression achieves this transformation using a single fully-connected layer. We can think of this as a Single-Layer Perceptron. This layer performs a linear transformation (a matrix multiplication combined with a bias). In contrast, a neural network with multiple connected layers is typically referred to as a Multi-Layer Perceptron (MLP). For instance, in the simple MLP shown below, a 4-dimensional input is mapped to a 5-dimensional hidden representation, which is subsequently transformed into a single output used for prediction. This is a "simple" architecture: A so-called feedforward neural network. 

<img src="./media/MLP.png" width="500"/>


#### Nonlinearities revisited

Nonlinearities are usually applied between the layers of a neural network. As discussed in class 2, there are several reasons for this. A key reason is that without any nonlinearity, a sequence of linear transformations (fully connected layers) reduces to a single linear transformation, limiting the model's expressiveness to that of a single layer. Including nonlinearities between layers prevents this reduction, enabling neural networks to approximate far more complex functions. This is what makes neural networks so powerful.

Numerous nonlinear activation functions are frequently employed in neural networks, but one of the most commonly used is the [rectified linear unit (ReLU)](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)):

```math
\begin{align}
x = \max(0,x)
\end{align}
```

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