# Module 12 Report Template

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
 
 Precision: healthy loan- 100%, high-risk loan- 85%
 
 Recall score: healthy loan- 99%, high-risk loan- 91%



* Machine Learning Model 2:

 Accuracy: 0.9936781215845847
 
 Precision: healthy loan- 100%, high-risk loan- 84%
 
 Recall scores: healthy loan- 99%, high-risk loan- 99%

### Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:

* Which one seems to perform best? How do you know it performs best?

 Machine Learning Model 2 seems to perform slightly better than model 1. The overall accuracy and recall scores are better. The only metric that is lower on model 2 is the precision on high-risk loans, which is only worse by 1%, going from 85% to 84%.

* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

 How well your model performs, will depend on which problem you are trying to solve. For our model, the metrics proved better results for healthy loans(the '0's).