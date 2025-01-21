**AI & Machine Learning (KAN-CINTO4003U) - Copenhagen Business School | Fall 2025**

***

### Group members
| Student name | Student ID |
| --- | --- |
| #NAME# | #ID# |
| #NAME# | #ID# |
| #NAME# | #ID# |

***

<br>

# Mandatory assignment 1

This repository contains the first mandatory assignment (MA1) for AIML25. It consists of two parts: Neural Networks and Convolutional Neural Networks.

You are given a lot of boilerplate code to help you get started. You will need to fill in the missing parts to complete the assignment. Everything necessary to complete the assignments are given in the notebooks under `/materials`. Here you will find a step-by-step notebook for both parts of the assignment.

| Assignment | Assignment notebook | Guide notebook | Description |
| --- | --- | --- | --- |
| Part 1 | assignments/nn.ipynb | materials/nns_pytorch.ipynb | Multi-Layer Perceptrons |
| Part 2 | assignments/cnn.ipynb | materials/cnns_pytorch.ipynb | Convolutional Neural Networks |

Please see due dates and other relevant information in the assignment description on Canvas.


## Part 1: Multi-Layer Perceptrons

In this assignment, your task will be to construct an MLP for the well-known MNIST dataset. Build a n-layer MLP (artificial neural network) for MNIST digit classfication in the `nn.ipynb` notebook (`assigntments/nn.ipynb`). See how good of a model you can build!

Feel free to play around with the model architecture and see how the training time/performance changes, but to begin, you can try the following:

* Input Layer: Image (784 dimensions) ->  
* Fully connected layer (500 hidden units) -> 
* Nonlinearity (ReLU) ->  
* Fully connected (10 hidden units - digits 0-9) -> 

*Some hints*:
- Everything needed to complete the assignment is in the notebook. 
- To get the best performance, you may want to play with the learning rate and the number of training epochs.

***

## Part 2: Convolutional Neural Networks

In this assignment, your task is to build a CNN for multi-class classification on the famous CIFAR-10 dataset. You will practice building networks with convolutional and pooling layers, as well as fully connected layers.

Build a *n-layer*  CNN for object classification in the `cnn.ipynb` notebook (assignments/nn.ipynb).

Feel free to play around with the model architecture - add/delete layers, adjust layer parameters etc. - and see how the training time/performance changes. Try to get the best performance you can!

You can start with the following:

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
<br>
<br>

___



### Submission

Please see sections VII - XI in the `AIML25-dev-setup.pdf` document on Canvas for instructions on 

- how to submit your assignment.
- setting up your development environment.
- how to work with the assignment repository.
- setting up virtual environments.
- using git and GitHub.