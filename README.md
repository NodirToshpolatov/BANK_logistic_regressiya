# BANK_logistic_regressiya
Project Overview
This project focuses on predicting customer churn (whether a customer will exit the service) using a machine learning model based on the Stacking Classifier technique. The dataset includes various customer attributes such as age, credit score, balance, geography, and more. The goal is to use these features to predict the likelihood of churn (customer exit) for future customers.

Dataset
The project uses two datasets:

Train Dataset: Used to train and evaluate the model.
Test Dataset: Used for predicting customer churn and submitting results.
Key Columns
CustomerId: Unique identifier for each customer.
Surname: Customer's surname (used only for validation, not modeling).
CreditScore: Customer's credit score.
Geography: The country of the customer (One-hot encoded for modeling).
Gender: Gender of the customer (Label encoded for modeling).
Age: Age of the customer.
Balance: Customer's bank balance.
NumOfProducts: Number of products a customer holds with the bank.
IsActiveMember: Whether the customer is an active member.
EstimatedSalary: Customer's estimated salary.
Exited: Target variable (1 if the customer exited, 0 otherwise).
Feature Engineering
Various new features were created to enhance the model's predictive power:

Ratios and Interactions: Features like Age_to_NumOfProducts, Balance_to_CreditScore, and Tenure_to_Age were created to capture important interactions between variables.
Polynomial Features: Additional polynomial features were generated using CreditScore, Age, and Balance to capture higher-order interactions.
Model
We employ a Stacking Classifier, which combines predictions from two base classifiers:

Logistic Regression: A simple linear model for binary classification.
Ridge Classifier: A robust classifier that uses L2 regularization to avoid overfitting.
These two models are stacked, and their predictions are used by a final Logistic Regression model to generate the final output.

Steps
Data Preprocessing:

Categorical columns like Gender and Geography were encoded.
Feature scaling was applied using StandardScaler to normalize the data.
Model Training:

The training dataset was split into a training set (70%) and a test set (30%) for model evaluation.
The stacking model was trained using logistic regression and ridge classifiers as base estimators.
Evaluation:

The model performance was evaluated using the ROC-AUC score, a robust metric for classification tasks that involve imbalanced datasets.
Prediction:

The trained model was applied to the test dataset, and the predicted probabilities were saved for submission.
Usage
Training the Model
Ensure all required libraries are installed:

bash
Копировать код
pip install pandas numpy scikit-learn
Load and preprocess the data using the provided code. This includes encoding categorical features, scaling numerical features, and engineering new features.

Train the model using the following code:

python
Копировать код
stacking_model.fit(X_train, y_train)
Evaluate the model:

python
Копировать код
roc_auc = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC: {roc_auc}")
Prediction on Test Data
Load the test dataset and apply the same preprocessing steps.
Use the trained stacking model to predict probabilities for customer churn.
python
Копировать код
y_test_prob = stacking_model.predict_proba(X_test_scaled)[:, 1]
Save the predictions to a CSV file for submission.
Results
The model was evaluated using the ROC-AUC metric, which balances sensitivity and specificity. The final submission file contains the predicted probabilities for each customer in the test set, indicating the likelihood of churn.

File Structure
train.csv: Training dataset used for model development.
test.csv: Test dataset used for making predictions.
sample_submission.csv: Template for submitting the predictions.
submission5.csv: Final submission file with predicted probabilities.
train[1].csv: The pre-processed training dataset with engineered features.
Dependencies
Python 3.7+
Pandas: Data manipulation and preprocessing.
NumPy: Numerical operations.
Scikit-learn: Machine learning model building, evaluation, and feature preprocessing.
Conclusion
This project demonstrates the application of stacking classifiers to predict customer churn. By leveraging engineered features and polynomial interactions, we were able to improve the model's performance. The stacking technique allowed us to combine the strengths of multiple models, leading to robust predictions.
