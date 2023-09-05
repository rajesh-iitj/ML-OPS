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
gamma_ranges = [0.001, 0.01, 0.1, 1, 10, 100]
C_ranges = [0.1, 1, 2, 5, 10]

#1. Get the digits data set with images and targets
X, y = read_digits();

#2. data splitting 
# Split data into 70% train, 15% dev and 15% test subsets
X_train, X_dev, X_test, y_train, y_dev, y_test = split_train_dev_test(X, y, test_sz = 0.30, dev_sz = 0.20)

#3. Data preprocessing
# flatten the images
X_train = preprocess_data(X_train)
X_dev   = preprocess_data(X_dev)
X_test  = preprocess_data(X_test)
    
#4. Hyper paramter tuning
#- take all combinations of gamma and C
best_acc_so_far = -1
best_model = None
for cur_gamma in gamma_ranges:
    for cur_C in C_ranges:
        #- train the model with curr_gamma and cur_C
        #print("Running for gamma={} c={}".format(cur_gamma, cur_C))
        cur_model = train_model(X_train, y_train, {'gamma': cur_gamma, 'C': cur_C}, model_type="svm")
        # - get some performance metric on DEV set
        cur_accuracy = predict_and_eval(cur_model, X_dev, y_dev)

        if cur_accuracy > best_acc_so_far:
            print("New best accuracy: ", cur_accuracy)
            best_acc_so_far = cur_accuracy
            optimal_gamma = cur_gamma
            optimal_C = cur_C
            best_model = cur_model

print("Optimal paramters gamma: ", optimal_gamma, "C: ", optimal_C)


#5. Get model prediction on test set
#6. Qualitative sanity check of the prediction
#7. Evaluation
test_acc = predict_and_eval(best_model, X_test, y_test)
print("Test accuracy: ", test_acc)



