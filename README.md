# Self-interpretable-classifier
Codebase for the paper, A Test Statistic Estimation-Based Approach for Establishing Self-Interpretable CNN-Based Binary Classifiers (Sengupta and Anastasio, IEEE TMI 2024)

Abstract : Interpretability is highly desired for deep neural network-based classifiers, especially when addressing high-stake decisions in medical imaging. Commonly used post-hoc interpretability methods have the limitation that they can produce plausible but different interpretations of a given model, leading to ambiguity about which one to choose. To address this problem, a novel decision-theory-inspired approach is investigated to establish a self-interpretable model, given a pre-trained deep binary black-box medical image classifier. This approach involves utilizing a self-interpretable encoder-decoder model in conjunction with a single-layer fully connected network with unity weights. The model is trained to estimate the test statistic of the given trained black-box deep binary classifier to maintain a similar accuracy. The decoder output image, referred to as an equivalency map, is an image that represents a transformed version of the to-be-classified image that, when processed by the fixed fully connected layer, produces the same test statistic value as the original classifier. The equivalency map provides a visualization of the transformed image features that directly contribute to the test statistic value and, moreover, permits quantification of their relative contributions. Unlike the traditional post-hoc interpretability methods, the proposed method is self-interpretable, quantitative. Detailed quantitative and qualitative analyses have been performed with three different medical image binary classification tasks.

Overview
This repository contains the following components:

1. Train the CNN classifier (classification_relu.py): Handles a classification task using a neural network with ReLU activations.
2. Train the interpretable Classifier: Handles the estimation task using a neural network with ReLU activations and skip connections.
3. Test the performance and visualization: Jupyter Notebook (test-relu-skip.ipynb): Provides a test bed for evaluating and visualizing.

Important libraries :keras==2.2.4 tensorflow==1.15 numpy matplotlib jupyter

Contact:
If you have any questions or feedback, feel free to reach out:

Email: souryas2@illinois.edu
