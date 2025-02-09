# Denoising Autoencoder & Random Forest on MNIST

## Overview

This code implements a **Denoising Autoencoder** trained on the **MNIST dataset**, using an embedding size of **64**. The learned embeddings are then used to train a **Random Forest classifier**, achieving an **accuracy of over 90%** on the test set.

## Features

- **Denoising Autoencoder**: Trained to generate compressed embeddings of MNIST images.
- **Embedding Extraction**: Extracts embeddings of size **(1000, 64)** for training and **(10000, 64)** for testing.
- **Random Forest Classification**: Trains a `RandomForestClassifier(random_state=0)` on the extracted embeddings.
- **Accuracy**: Achieves **>90% accuracy** on the test set.

## Dataset

- **MNIST** (handwritten digits, 28x28 grayscale images, resized to 32x32)
- **Train embeddings**: Shape **(1000, 64)**
- **Test embeddings**: Shape **(10000, 64)**

## Model & Training

- **Autoencoder**: Uses convolutional layers with batch normalization and LeakyReLU activation.
- **Denoising Mechanism**: Adds Gaussian noise during training for robust feature learning.
- **Embedding Size**: **64-dimensional latent representation**.
- **Optimizer**: Adam (`lr=1e-3`).
- **Training Duration**: 7 epochs.

## Evaluation & Results

- The **Random Forest classifier** trained on the embeddings achieves an accuracy **above 90%**.
- Embeddings are stored in `embeddings.pt` for reproducibility.

## Visualization

- The notebook includes **image reconstruction** comparisons (original vs. reconstructed).
- **Latent space interpolation** is implemented to explore the embedding space.

## References

- **[Denoising Autoencoders Paper](https://www.cs.toronto.edu/~hinton/absps/NIPS2006_0935.pdf)**
- **[Scikit-learn Random Forest Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)**
