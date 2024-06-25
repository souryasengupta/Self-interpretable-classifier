# A Test Statistic Estimation-Based Approach for Establishing Self-Interpretable CNN-Based Binary Classifiers

This repository contains the codebase for the paper **"A Test Statistic Estimation-Based Approach for Establishing Self-Interpretable CNN-Based Binary Classifiers"** (Sengupta and Anastasio, IEEE TMI 2024).

## Abstract

**Interpretability** is highly desired for deep neural network-based classifiers, especially when addressing high-stakes decisions in medical imaging. Commonly used post-hoc interpretability methods can produce plausible but different interpretations of a given model, leading to ambiguity about which one to choose. To address this problem, we investigate a novel decision-theory-inspired approach to establish a self-interpretable model, given a pre-trained deep binary black-box medical image classifier.

This approach involves utilizing a self-interpretable encoder-decoder model in conjunction with a single-layer fully connected network with unity weights. The model is trained to estimate the test statistic of the given trained black-box deep binary classifier to maintain similar accuracy. The decoder output image, referred to as an equivalency map, represents a transformed version of the to-be-classified image that, when processed by the fixed fully connected layer, produces the same test statistic value as the original classifier. The equivalency map provides a visualization of the transformed image features that directly contribute to the test statistic value and permits quantification of their relative contributions. Unlike traditional post-hoc interpretability methods, the proposed method is self-interpretable and quantitative. Detailed quantitative and qualitative analyses have been performed with three different medical image binary classification tasks.

## Overview

This repository contains the following components:

1. **Train the Black-box CNN Classifier** (`classification_blackbox.py`)
2. **Train the Self-interpretable Classifier** (`self-interpretable.py`)
3. **Test the Performance and Visualization**: Jupyter Notebook (`test.ipynb`): Provides a test bed for evaluating and visualizing.


### Running the Python Scripts

1. **Black-box Classifier Training**:
   - Execute the following command to run the black-box classifier training script:

    ```sh
    python classification_blackbox.py
    ```

2. **Self-interpretable Classifier Trainings**:
   - Execute the following command to run the self-interpretable encoder-decoder network:

    ```sh
    python estimation_relu.py --total 4 --randomrestart 1 --data_path /path/to/your/data_directory --best_blackbox_ckpt /path/to/your/best_checkpoint_file.hdf5 --best_interpretable_ckpt /path/to/save/checkpoint_files

    ```
    
### Running the Jupyter Notebook

1. **<span style="font-family: Arial, sans-serif;">Start Jupyter Notebook</span>**:

    ```sh
    jupyter notebook
    ```

2. **<span style="font-family: Arial, sans-serif;">Open `test.ipynb`</span>**:
   - Navigate to the `test.ipynb` file in the Jupyter Notebook interface and open it.
   - Run the cells sequentially to see the neural network training and evaluation process.


## Important Libraries

- `keras==2.2.4`
- `tensorflow==1.15`
- `numpy`
- `matplotlib`
- `jupyter`

## Contact

If you have any questions or feedback, feel free to reach out:

- **Email**: souryas2@illinois.edu

## Citation

If you use this code in your research, please cite the following paper:

```bibtex
@article{sengupta2024test,
  title={A Test Statistic Estimation-based Approach for Establishing Self-interpretable CNN-based Binary Classifiers},
  author={Sengupta, Sourya and Anastasio, Mark A},
  journal={IEEE transactions on medical imaging},
  year={2024},
  publisher={IEEE}
}
@article{sengupta2023revisiting,
  title={Revisiting model self-interpretability in a decision-theoretic way for binary medical image classification},
  author={Sengupta, Sourya and Anastasio, Mark A},
  journal={arXiv preprint arXiv:2303.06876},
  year={2023}
}


