# Module-12-Challenge

For this project, I am utilizing Python and and Jupyter Lab to evaluate models with imbalanced classes. Given historical data of lending activity, I created a logistic regression model then resampled the data and compared the two datasets. I then counted the target classes, calculated the balanced accuracy score, generated a confusion matrix, and printed a classification report. 

---

## Technologies

This project uses pandas programming language and utilizes Anaconda, Python, Git Bash, Jupyter Lab, and Github. This project also uses scikit-learn and imbalanced-learn.

---

## Installation Guide

These are the required libraries and dependencies:

import numpy as np

import pandas as pd

from pathlib import Path

from sklearn.metrics import balanced_accuracy_score

from sklearn.metrics import confusion_matrix

from imblearn.metrics import classification_report_imbalanced

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from imblearn.over_sampling import RandomOverSampler


You must also install the imbalanced-learn in your conda dev environment using the code:

    conda install -c conda-forge imbalanced-learn

---

## Usage

### Module 12 Report

### Overview of the Analysis

In this section, I'll describe the analysis I completed for the machine learning models used in this Challenge.

* Explain the purpose of the analysis.
Given historical data of lending activity, I created a logistic regression model then resampled the data and compared the two datasets. I then counted the target classes, calculated the balanced accuracy score, generated a confusion matrix, and printed a classification report. 

* Explain what financial information the data was on, and what you needed to predict.
A few important aspects of the data that was used was loan size, interest rate, borrower income, derogatory marks, and total debt.

* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).
 Counts the number of each loan type:
     value_counts() 
     
 Calculates the accuracy between our predictions and the actual targets:
     balanced_accuracy_score(y_test, predictions)
     
 Shows the number of observations the model correctly classified:
     confusion_matrix(y_test, predictions)
     
 Calculates the accuracy, precision, recall, and F1 scores. This breaks down how well the model performed for each class:
     print(classification_report_imbalanced(y_test, predictions))
     
* Describe the stages of the machine learning process you went through as part of this analysis.
 First we gathered and prepared the data then created a logistic regression model with the original data. I then fit the model with training data and evaluated the performace of the model. Next I predicted a logistic regression model with resampled training data then evaluted the performance of that model.

* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).
 We use Logistic Regression to assess and analyze data to predict outcomes and make decisions. We then introduce new sample data to the model and it determines the probabality that data belongs to a specific class.
 
 With imabalanced classes, we can use oversampling or undersampling methods to improve the predictions of the model.

### Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
 Accuracy: 0.9520479254722232
 Precision:
 Recall score:



* Machine Learning Model 2:
 Accuracy:
 Precision:
 Recall scores:

### Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )


---

## Contributors

Allyssa Carmin

---

## License

SMU Fintech Course