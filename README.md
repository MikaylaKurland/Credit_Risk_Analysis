# Credit_Risk_Analysis

## Overview 

The purpose of this analysis was to analyze the loan statistics data set and utilize different machine learning models to predict risk to the lender.

## Results:

### - Naive Random Oversampling
This model's precision score was 0.01, and the recall score was 0.69.

```
# Calculated the balanced accuracy score
balanced_accuracy_score(y_test, y_pred)
```
0.6503524738582371
```
# Display the confusion matrix
confusion_matrix(y_test, y_pred)
```
array([[   70,    31],
       [ 6711, 10393]], dtype=int64)


### - SMOTE Oversampling
This model's precision score was 0.01, and the recall score was 0.63.

```
# Calculated the balanced accuracy score
balanced_accuracy_score(y_test, y_pred)
```
0.6621602612787003

```
# Display the confusion matrix
confusion_matrix(y_test, y_pred)
```
array([[   64,    37],
       [ 5291, 11813]], dtype=int64)
       
       
### - Cluster Centroid Undersampling
This model's precision score was 0.01, and the recall score was 0.68.

```
# Calculated the balanced accuracy score
balanced_accuracy_score(y_test, y_pred)
```
0.5469747103335216

```
# Display the confusion matrix
confusion_matrix(y_test, y_pred)
```

array([[   69,    32],
       [10078,  7026]], dtype=int64)
       

### - Combination Sampling
This model's precision score was 0.01, and the recall score was 0.72.

```
# Calculated the balanced accuracy score
balanced_accuracy_score(y_test, y_pred)
```
0.644711676499736

```
# Display the confusion matrix
confusion_matrix(y_test, y_pred)
```
array([[  73,   28],
       [7412, 9692]], dtype=int64)
       

### - Balanced Random Forest Classifier
This model's precision score was 0.04, and the recall score was 0.67.

```
# Calculated the balanced accuracy score
balanced_accuracy_score(y_test, y_pred)
```
0.7877672625306695

```
# Display the confusion matrix
confusion_matrix(y_test, y_pred)
```
array([[   58,    29],
       [ 1560, 15558]], dtype=int64)


### - Easy Ensemble AdaBoost Classifier
This model's precision score was 0.07, and the recall score was 0.91.

```
# Calculated the balanced accuracy score
balanced_accuracy_score(y_test, y_pred)
```
0.925427358175101

```
# Display the confusion matrix
confusion_matrix(y_test, y_pred)
```
array([[   79,     8],
       [  979, 16139]], dtype=int64)

## Summary:

The ensemble classifier models vastly outperformed all of the other models in the over and under-sampling categories, which performed about equally.  While both of the ensemble models have higher precision, recall, and accuracy scores than the other four models due to the gact that they aggregate predictions from multiple models to achieve their results, the AdaBoost Classifier Model's accuracy score is ~14% more accurate than the Balanced Random Forest, and therefore I would recommend the Easy Ensemble AdaBoost model to analyze this data.
