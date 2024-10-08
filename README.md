# AdaBoost-Boosting_ML_Algorithm_for_Enhanced_Classification

# Problem Statement:
Implementing the AdaBoost Algorithm with Decision Stumps as Base Learners for Binary Classification on the Breast Cancer Wisconsin Dataset

# Description
This project involves building a binary classification model using the AdaBoost algorithm, where the base learners are decision stumps (single-split decision trees). The goal is to combine multiple weak learners to create a strong classifier that can accurately distinguish between benign and malignant tumors in the Breast Cancer Wisconsin dataset. The implemented model iteratively trains decision stumps on different weighted distributions of the training data, adjusting the weights based on misclassification errors, and aggregates the predictions from each weak learner to produce the final classification decision. The effectiveness of the model is evaluated by experimenting with different numbers of stumps (weak classifiers) and learning rates to identify the best-performing configuration for classifying the breast cancer dataset.

# Implementation:
1. Data Loading and Preprocessing: The dataset is loaded, and the id column is removed since it is not useful for the model.
Features (X) are stored as a numpy array, while the target variable (y) is converted into numerical labels: 1 for malignant (M) and -1 for benign (B).
2. Manual Train-Test Split: The data is split into 70% training and 30% testing. This split is controlled by setting a random seed (np.random.seed(42)) to ensure reproducibility.
3. Decision Stump Implementation: Decision Stumps are simple classifiers that split data on a single feature using a threshold. This function iterates through each feature and tries various thresholds (lt for less than and gt for greater than) to find the best split based on the current weights (w) of the samples.
4. AdaBoost Algorithm: The main AdaBoost function repeatedly calls the decision_stump function n_estimators times.
The weights of incorrectly classified samples are increased, making the model focus more on these "hard" samples in the next iteration.
5. Prediction Function: The predict function aggregates the predictions from all weak classifiers using their respective weights (alpha). The final decision is made based on the sign of the combined prediction.
6. Hyperparameter Tuning: Various combinations of n_estimators and learning_rate are tested to find the optimal set of hyperparameters that maximize the model's performance on the test set.
7. Model Evaluation and Visualization: After training the model with the best hyperparameters, the final accuracy is calculated.
Feature importance is visualized based on the cumulative weights (alpha) assigned to each feature.

# Conclusion:
In conclusion, through the application of the AdaBoost algorithm on the Breast Cancer Wisconsin dataset, I successfully identified key features that significantly contribute to the classification of malignant and benign tumors. By leveraging AdaBoost's ensemble learning capabilities, which combine multiple weak learners into a robust model, I was able to evaluate the importance of each feature in predicting breast cancer.

The analysis revealed that certain features, such as texture_mean, exhibited higher importance scores, indicating their critical role in distinguishing between cancerous and non-cancerous conditions. This insight not only enhances our understanding of the factors influencing breast cancer detection but also underscores the potential of machine learning techniques like AdaBoost in improving diagnostic accuracy in healthcare. Overall, this project illustrates the effectiveness of ensemble methods in feature selection and classification tasks, paving the way for further exploration and application in medical data analytics.


