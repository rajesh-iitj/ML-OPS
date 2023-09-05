"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import metrics
from utils import preprocess_data, split_data, train_model, read_digits, split_train_dev_test, predict_and_eval


#1. Get the digits data set with images and targets
X, y = read_digits();

#2. data splitting 
# Split data into 70% train, 15% dev and 15% test subsets
X_train, X_dev, X_test, y_train, y_dev, y_test = split_train_dev_test(X, y, test_sz = 0.15, dev_sz = 0.15)

#3. Data preprocessing
# flatten the images
X_train = preprocess_data(X_train)
X_dev   = preprocess_data(X_dev)
X_test  = preprocess_data(X_test)
    
#4. Model training
model = train_model(X_train, y_train, {'gamma': 0.001}, model_type="svm")

#5 predict and evalulate on dev set
print("**********Prediction and evulation on dev set***********")
predict_and_eval(model, X_dev, y_dev)


#6 predict and evalulate on test 
print("**********Prediction and evulation on test set***********")
predict_and_eval(model, X_test, y_test)


