# A Test Statistic Estimation-Based Approach for Establishing Self-Interpretable CNN-Based Binary Classifiers

This repository contains the codebase for the paper **"A Test Statistic Estimation-Based Approach for Establishing Self-Interpretable CNN-Based Binary Classifiers"** (Sengupta and Anastasio, IEEE TMI 2024).

## Abstract

**Interpretability** is highly desired for deep neural network-based classifiers, especially when addressing high-stakes decisions in medical imaging. Commonly used post-hoc interpretability methods can produce plausible but different interpretations of a given model, leading to ambiguity about which one to choose. To address this problem, we investigate a novel decision-theory-inspired approach to establish a self-interpretable model, given a pre-trained deep binary black-box medical image classifier.

This approach involves utilizing a self-interpretable encoder-decoder model in conjunction with a single-layer fully connected network with unity weights. The model is trained to estimate the test statistic of the given trained black-box deep binary classifier to maintain similar accuracy. The decoder output image, referred to as an equivalency map, represents a transformed version of the to-be-classified image that, when processed by the fixed fully connected layer, produces the same test statistic value as the original classifier. The equivalency map provides a visualization of the transformed image features that directly contribute to the test statistic value and permits quantification of their relative contributions. Unlike traditional post-hoc interpretability methods, the proposed method is self-interpretable and quantitative. Detailed quantitative and qualitative analyses have been performed with three different medical image binary classification tasks.

## Overview

This repository contains the following components:

1. **Train the CNN Classifier** (`classification_relu.py`): Handles a classification task using a neural network with ReLU activations.
2. **Train the Interpretable Classifier** (`estimation_relu_skip.py`): Handles the estimation task using a neural network with ReLU activations and skip connections.
3. **Test the Performance and Visualization**: Jupyter Notebook (`test-relu-skip.ipynb`): Provides a test bed for evaluating and visualizing.


Important Libraries
keras==2.2.4
tensorflow==1.15
numpy
matplotlib
jupyter
Contact:
If you have any questions or feedback, feel free to reach out:

Email: souryas2@illinois.edu
