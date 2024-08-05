# Customer Churn Prediction using Logistic Regression
This project aims to predict customer churn using logistic regression, which is a common task in machine learning and data analysis. Customer churn refers to when customers stop doing business with a company. Predicting churn helps companies take proactive measures to retain customers. This project demonstrates how to build a logistic regression model for churn prediction.

# Table of Contents
Introduction
Dataset
Installation
Usage
Project Workflow
Results
Contributing
License

# Introduction
Customer churn prediction is critical for businesses aiming to retain their customers. By identifying customers likely to churn, companies can implement strategies to retain them. Logistic regression, a statistical method for binary classification problems, is used in this project to predict whether a customer will churn based on various features.

# Dataset
The dataset used for this project is sourced from IBM and is publicly available. It contains attributes such as tenure, age, address, income, education, employment, equipment, and whether the customer has a call card or wireless service. The target variable is churn, indicating whether a customer has churned.

# Installation
To run this project, ensure you have Python 3.x installed along with the necessary libraries:
pandas
numpy
matplotlib
scikit-learn
scipy

# Project Workflow
1. Data Loading and Preprocessing
The first step is to load the dataset and select the relevant features for analysis. The target variable is also specified, and the data is standardized to ensure that all features have a mean of 0 and a standard deviation of 1. This helps in improving the performance of the machine learning model.

2. Splitting Data into Training and Testing Sets
The dataset is split into training and testing sets. This is crucial for evaluating the model's performance on unseen data. Typically, 80% of the data is used for training, and 20% is reserved for testing.

3. Model Training and Prediction
A logistic regression model is trained using the training data. The model learns the relationship between the features and the target variable (churn). After training, the model is used to predict churn on the test data.

4. Model Evaluation
The model's performance is evaluated using several metrics:
Jaccard Score: Measures the similarity between predicted and actual labels.
Confusion Matrix: A table that describes the performance of a classification model by displaying true positives, true negatives, false positives, and false negatives.
Classification Report: Provides precision, recall, and F1-score for each class.
Log Loss: Measures the performance of a classification model with probabilistic predictions.
Logistic Regression with Different Solver
An additional logistic regression model is trained using a different solver to compare performance. The log loss is computed to evaluate this model's performance.

5. Results

The logistic regression model achieved an accuracy of 75% on the test set. The confusion matrix indicated that the model correctly predicted most of the churn cases but had some false positives and false negatives. The classification report showed a higher precision for the non-churn class compared to the churn class. The log loss for the model was 0.61, indicating the model's probabilistic predictions were reasonably accurate.
