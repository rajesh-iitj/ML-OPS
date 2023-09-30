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
from sklearn import metrics, svm
from utils import preprocess_data, split_data, train_model, read_digits, split_train_dev_test, predict_and_eval, create_combinations_dict_from_lists, tune_hparams
import decimal
from joblib import dump, load

gamma_ranges = [0.001, 0.01, 0.1, 1, 10, 100]
C_ranges = [0.1, 1, 2, 5, 10]

#1. Get the digits data set with images and targets
X, y = read_digits();

#2. data splitting 
# Split data into all possbile combination of test_size_list and dev_size_list

test_size_list = [0.1, 0.2, 0.3]
dev_size_list  = [0.1, 0.2, 0.3]

list_of_all_test_dev_combination_dictionaries = create_combinations_dict_from_lists(test_size_list, dev_size_list)

for key, value in list_of_all_test_dev_combination_dictionaries.items():
    #split the train, dev and test
    test_size = value[0]
    dev_size  = value[1]
    train_size = 1 - test_size - dev_size
    train_size = round(train_size, 2)
    X_train, X_dev, X_test, y_train, y_dev, y_test = split_train_dev_test(X, y, test_sz = test_size, dev_sz = dev_size)
    #3. Data preprocessing
    # flatten the images
    X_train = preprocess_data(X_train)
    X_dev   = preprocess_data(X_dev)
    X_test  = preprocess_data(X_test)
    
    #4. Hyper paramter tuning
    #- take all combinations of gamma and C
    best_model = None

    list_of_all_param_combination_dictionaries = create_combinations_dict_from_lists(gamma_ranges, C_ranges)
    best_hparams, best_model_path, best_accuracy    = tune_hparams(X_train, y_train, X_dev, y_dev, list_of_all_param_combination_dictionaries)

    # saving the best model

    # delete the model

    # loading of the model
    best_model = load(best_model_path)

    #5. Get model prediction on test set
    #6. Qualitative sanity check of the prediction
    #7. Evaluation
    
    train_acc = predict_and_eval(best_model, X_train, y_train)
    dev_acc   = best_accuracy
    test_acc  = predict_and_eval(best_model, X_test, y_test)
    
    print("test_size = ", test_size, "dev_size = ", dev_size, "train_size = ", train_size, "train_acc = ", train_acc, "dev_acc = ", dev_acc, "test_acc = ", test_acc)
    optimal_gamma = best_hparams[0]
    optimal_C = best_hparams[1]
    print("best_hparams: ", "gamma: ", optimal_gamma, "C:", optimal_C)



