## Project Report: Improving Time Series Classification Accuracy Using Self-Supervised Learning

### 1. Objective
The primary objective of this project was to improve the accuracy of time-series classification for gesture recognition by leveraging self-supervised learning. Self-supervised learning enables the model to learn meaningful representations from unlabeled data, which we later used to train a classifier on the labeled UWaveGestureLibrary dataset.

### 2. Methodology

#### Data Preparation
The UWaveGestureLibrary dataset, consisting of 320 training samples, 120 validation samples, and 120 test samples, was used for gesture classification. Each sample was a 3-channel time series with 206 time steps. We performed the following steps:
- **Data Loading**: Loaded the data from `.pt` files.
- **Preprocessing**: Converted data to a compatible format (float32) for PyTorch models.

#### Self-Supervised Learning with Autoencoder
To learn meaningful representations, we trained an autoencoder to reconstruct the input time series data:
- **Encoder Architecture**: The encoder consisted of two convolutional layers with ReLU activation functions to capture the features from time series data.
- **Decoder Architecture**: The decoder used transposed convolutional layers to reconstruct the original time series from the encoded features.

The autoencoder was trained over 20 epochs, achieving a reduction in reconstruction loss, indicating that it effectively learned useful representations of the data.

#### Feature Extraction
After training, we used the encoder part of the autoencoder to extract features from the training, validation, and test datasets. These encoded features served as inputs to a downstream classification model.

#### Classification Model
Using the extracted features, we trained a Random Forest classifier to classify gestures in the test dataset:
- **Training**: The classifier was trained on the encoded features from the training set.
- **Evaluation**: We evaluated the model on the test set and reported accuracy, precision, recall, and F1-score.

### 3. Results

The results of the classification model are as follows:

#### Classification Metrics
| Label | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.73      | 0.73   | 0.73     | 15      |
| 1     | 0.80      | 0.80   | 0.80     | 15      |
| 2     | 0.87      | 0.87   | 0.87     | 15      |
| 3     | 0.62      | 0.67   | 0.65     | 15      |
| 4     | 0.40      | 0.13   | 0.20     | 15      |
| 5     | 0.17      | 0.20   | 0.18     | 15      |
| 6     | 0.78      | 0.93   | 0.85     | 15      |
| 7     | 0.78      | 0.93   | 0.85     | 15      |

**Overall Accuracy**: 66%

- **Macro Avg**: Precision = 0.64, Recall = 0.66, F1-score = 0.64
- **Weighted Avg**: Precision = 0.64, Recall = 0.66, F1-score = 0.64

#### Confusion Matrix

|   | Predicted 0 | Predicted 1 | Predicted 2 | Predicted 3 | Predicted 4 | Predicted 5 | Predicted 6 | Predicted 7 |
|---|--------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|
| **Actual 0** | 11 | 1 | 0 | 0 | 0 | 2 | 0 | 1 |
| **Actual 1** | 0 | 12 | 0 | 0 | 1 | 0 | 2 | 0 |
| **Actual 2** | 0 | 0 | 13 | 0 | 0 | 2 | 0 | 0 |
| **Actual 3** | 1 | 0 | 0 | 10 | 0 | 3 | 0 | 1 |
| **Actual 4** | 3 | 0 | 0 | 3 | 2 | 7 | 0 | 0 |
| **Actual 5** | 0 | 2 | 2 | 3 | 2 | 3 | 1 | 2 |
| **Actual 6** | 0 | 0 | 0 | 0 | 0 | 1 | 14 | 0 |
| **Actual 7** | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 14 |

### 4. Discussion and Analysis

- **High Precision and Recall for Certain Classes**: Classes 2, 6, and 7 achieved higher precision and recall, indicating the model's effectiveness for these gestures.
- **Low Performance for Other Classes**: Class 4 and Class 5 had low recall and precision, suggesting difficulties in distinguishing these gestures, potentially due to insufficient distinctive features in the self-supervised representations.
- **Overall Accuracy**: An accuracy of 66% was achieved. While this demonstrates a reasonable degree of classification capability, further tuning of the autoencoder or using a more complex classifier may improve performance.

### 5. Future Improvements
- **Experiment with Different Self-Supervised Models**: Try other self-supervised architectures like contrastive learning or time-series transformers.
- **Hyperparameter Tuning**: Adjusting hyperparameters for both the autoencoder and the Random Forest classifier may yield better results.
- **Data Augmentation**: Implement data augmentation techniques to improve the model's robustness and potentially increase classification accuracy for underperforming classes.

### 6. Conclusion
This project demonstrated the effectiveness of self-supervised learning in extracting features from time-series data for gesture classification. While promising, there is room for improvement to achieve higher accuracy and more balanced performance across all gesture classes.

