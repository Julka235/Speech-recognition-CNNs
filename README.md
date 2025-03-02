# Speech Command Recognition Classifier

## Project Overview

This project focuses on the classification of spoken command audio files using deep learning techniques. The primary goal is to explore different convolutional neural network architectures for speech recognition and assess their performance. The project was developed as part of the recruitment process for the **Artificial Intelligence Society GOLEM** at Warsaw University of Technology (WUT).

## Contributors

- Filip Kopyt
- Julia Czosnek

## Dataset

We used the TensorFlow Speech Recognition Challenge dataset from Kaggle: [Dataset](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data)

### Dataset Preprocessing

- Label Assignment: The training dataset lacked direct labels, so we utilized validation and test lists to assign file paths to appropriate categories.
- NPZ Conversion: To streamline data loading and improve efficiency, all processed files were saved in .npz format.

## Neural Network Architectures

Three different neural networks were implemented and evaluated:

- **2D Convolutional Neural Network (CNN)** – A custom-built model utilizing 2D convolutions on spectrogram representations.

- **1D Convolutional Neural Network (CNN)** – A custom-built model that processes raw audio waveforms using 1D convolutions.

- **Pre-trained Model** – A ResNet18 model to compare our CNNs accuracies to a pretrained model.

## Audio Processing Pipeline

To ensure high-quality input data, several preprocessing steps were applied:

1. **Resampling**. Standardized all audio signals to 8 kHz to maintain consistency and reduce computational complexity.

2. **Voice Activity Detection (VAD)** Removed silent segments to focus on speech content and improve model efficiency.

3. **Padding for Equal Length** Audio samples were padded with zeros to ensure uniform input shapes for batch processing.

4. **Mel Spectrogram Conversion** Converted raw waveforms into log spectrograms using Short-Time Fourier Transform (STFT) and logarithmic scaling.

5. **Feature Normalization** Applied mean and standard deviation normalization to stabilize training and prevent large values from dominating.

## Model Training & Evaluation

- **Training Variables**: We experimented with different hyperparameters, including learning rate, batch size, and activation functions.

- **Performance Metrics**:

  - **Confusion Matrix**: To analyze misclassified words.

  - **Train vs. Validation Loss Plot**: To detect potential overfitting and optimize training.

- **Results Visualization**: A detailed presentation (included in the repository) contains plots and diagrams visualizing audio processing steps and model performance.

## Tools & Libraries

- Python
- Librosa (audio processing)
- PyTorch (deep learning framework)
- gdown (for downloading files from Google Drive)
- Google Colab (for training and experimentation)
- Recommended: A high-performance GPU for faster training.

## Repository Structure
```
📂 project-root
 ├── Golem_1DCNN.ipynb    # Jupyter notebook with 1D CNN
 ├── Golem_2DCNN.pt       # Python file with 2D CNN
 ├── presentation.pdf     # Visual documentation of the project
 └── README.md            # Project documentation
 ```

## Future Improvements

- Experimenting with Recurrent Neural Networks (RNNs) and Transformer models.
- Implementing data augmentation techniques for better generalization.
- Enhancing real-time speech recognition capabilities.

