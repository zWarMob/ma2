import numpy as np
import matplotlib.pyplot as plt
import torch

def imshow(image : "torch.Tensor", label : str, classes : list) -> None:

    image = image.numpy().transpose((1, 2, 0))  # Convert from CxHxW to HxWxC
    mean = np.array([0.4914, 0.4822, 0.4465])   # CIFAR-10 normalization mean
    std = np.array([0.2023, 0.1994, 0.2010])    # CIFAR-10 normalization std
    image = std * image + mean                  # Unnormalize
    image = np.clip(image, 0, 1)                # Clip values to [0, 1]
    plt.title(f"Label: {classes[label]}")
    plt.imshow(image)
    plt.show()


# Function to unnormalize and prepare the image for display
def unnormalize(image : "torch.Tensor") -> "np.ndarray":
    mean = np.array([0.4914, 0.4822, 0.4465])   # CIFAR-10 normalization mean
    std = np.array([0.2023, 0.1994, 0.2010])    # CIFAR-10 normalization std
    image = image.numpy().transpose((1, 2, 0))  # Convert to HxWxC
    image = std * image + mean                  # Unnormalize
    image = np.clip(image, 0, 1)                # Clip values to [0, 1]
    return image

def plot_probabilities(image : "torch.Tensor", label : str, probabilities : "torch.Tensor", classes : list, n : int = 10) -> None:

    # Get the top n predictions
    top_prob, top_classes = torch.topk(probabilities, n)

    # Plot the image and the top-5 probabilities
    fig, ax = plt.subplots(figsize=(8, 4))

    # Plot the image on the left
    ax_image = plt.subplot(1, 2, 1)
    ax_image.imshow(unnormalize(image))
    ax_image.axis('off')
    ax_image.set_title(f"True Label: {classes[label]}")

    # Plot the probabilities on the right
    ax_probs = plt.subplot(1, 2, 2)
    class_names = [classes[idx] for idx in top_classes]
    probabilities = top_prob.cpu().numpy()  # Move to CPU for compatibility
    ax_probs.barh(class_names, probabilities, color='blue')  # Horizontal bar chart
    ax_probs.set_xlim(0, 1)
    ax_probs.set_xlabel('Probability')
    ax_probs.set_title('Top 10 Prediction probabilities')

    plt.tight_layout()
    plt.show()