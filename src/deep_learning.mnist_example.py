import tqdm
import mnist
import random
import matplotlib.pyplot as plt
from typing import List, Optional

from deep_learning import (
    Tanh,
    Loss,
    Layer,
    Linear,
    Dropout,
    Momentum,
    Optimizer,
    Sequential,
    SoftmaxCrossEntropy
)
from neural_networks import argmax
from basics.tensor import tensor_sum, Tensor


def one_hot_encode(i: int, num_labels: int = 10) -> List[float]:
    return [1.0 if j == i else 0.0 for j in range(num_labels)]


def loop(model: Layer,
         images: List[Tensor],
         labels: List[Tensor],
         loss: Loss,
         optimizer: Optional[Optimizer] = None) -> None:
        
        correct = 0         # Track number of correct predictions.
        total_loss = 0.0    # Track total loss.
    
        with tqdm.trange(len(images)) as t:
            for i in t:
                predicted = model.forward(images[i])             # Predict.
                if argmax(predicted) == argmax(labels[i]):       # Check for
                    correct += 1                                 # correctness.
                total_loss += loss.loss(predicted, labels[i])    # Compute loss.
    
                # If we're training, backpropagate gradient and update weights.
                if optimizer is not None:
                    gradient = loss.gradient(predicted, labels[i])
                    model.backward(gradient)
                    optimizer.step(model)
    
                # And update our metrics in the progress bar.
                avg_loss = total_loss / (i + 1)
                acc = correct / (i + 1)
                t.set_description(f"mnist loss: {avg_loss:.3f} acc: {acc:.3f}")


def main():
    random.seed(0)

    mnist.temporary_dir = lambda: '/tmp'
    
    # Each of these functions first downloads the data and returns a numpy array.
    # We call .tolist() because our "tensors" are just lists.
    train_images = mnist.train_images().tolist()
    train_labels = mnist.train_labels().tolist()

    fig, ax = plt.subplots(10, 10)
    
    for i in range(10):
        for j in range(10):
            # Plot each image in black and white and hide the axes.
            ax[i][j].imshow(train_images[10 * i + j], cmap='Greys')
            ax[i][j].xaxis.set_visible(False)
            ax[i][j].yaxis.set_visible(False)

    # plt.show()

    # Load the MNIST test data
    
    test_images = mnist.test_images().tolist()
    test_labels = mnist.test_labels().tolist()

    train_labels = [one_hot_encode(label) for label in train_labels]
    test_labels = [one_hot_encode(label) for label in test_labels]

    # Recenter the images
    
    # Compute the average pixel value
    avg = tensor_sum(train_images) / 60000 / 28 / 28
    
    # Recenter, rescale, and flatten
    train_images = [[(pixel - avg) / 256 for row in image for pixel in row]
                    for image in train_images]
    test_images = [[(pixel - avg) / 256 for row in image for pixel in row]
                   for image in test_images]


    # Name them so we can turn train on and off
    dropout1 = Dropout(0.1)
    dropout2 = Dropout(0.1)
    
    model = Sequential([
        Linear(784, 30),  # Hidden layer 1: size 30
        dropout1,
        Tanh(),
        Linear(30, 10),   # Hidden layer 2: size 10
        dropout2,
        Tanh(),
        Linear(10, 10)    # Output layer: size 10
    ])

    # Training the deep model for MNIST
    
    optimizer = Momentum(learning_rate=0.01, momentum=0.99)
    loss = SoftmaxCrossEntropy()
    
    # Enable dropout and train (takes > 20 minutes on my laptop!)
    dropout1.train = dropout2.train = True
    loop(model, train_images, train_labels, loss, optimizer)
    
    # Disable dropout and evaluate
    dropout1.train = dropout2.train = False
    loop(model, test_images, test_labels, loss)


if __name__ == "__main__":
    main()