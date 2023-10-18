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
from utils import preprocess_data, split_data, train_model, read_digits, split_train_dev_test, predict_and_eval, create_combinations_dict_from_lists, tune_hparams, get_hyperparameter_combinations
import decimal
from joblib import dump, load
import pdb
import pandas as pd


#1. Get the digits data set with images and targets
X, y = read_digits();

#2. Hyperparamter combinations
classifier_param_dict = {}
#2.1 SVM
gamma_ranges = [0.0001, 0.0005, 0.001, 0.01, 0.1, 1]
C_ranges = [0.1, 1, 2, 5, 10]
h_params = {}
h_params['gamma'] = gamma_ranges
h_params['C'] = C_ranges
h_params_combinations = get_hyperparameter_combinations(h_params) 
classifier_param_dict['svm'] = h_params_combinations
#2.2 Decission trees
max_depth_list = [5, 10, 15, 20, 50, 100]
h_params_tree = {}
h_params_tree['max_depth'] = max_depth_list
h_params_combinations = get_hyperparameter_combinations(h_params_tree) 
classifier_param_dict['tree'] = h_params_combinations
#2. data splitting 
# Split data into all possbile combination of test_size_list and dev_size_list

test_size_list = [0.2]      #[0.1, 0.2, 0.3]
dev_size_list  = [0.2]      #[0.1, 0.2, 0.3]

list_of_all_test_dev_combination_dictionaries = create_combinations_dict_from_lists(test_size_list, dev_size_list)

num_runs = 5

results = []

for curr_run_i in range(num_runs):
    cur_run_results = {}
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
        

        for model_type in classifier_param_dict:
            #breakpoint()
            #4. Hyper paramter tuning
            #- take all combinations of gamma and C
            current_hparams = classifier_param_dict[model_type]
            best_model = None

            best_hparams, best_model_path, best_accuracy = tune_hparams(X_train, y_train, X_dev, y_dev, current_hparams, model_type)

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
            
            print("{} \t test_size={:.2f} dev_size={:.2f} train_size={:.2f} train_acc={:.2f} dev_acc={:.2f} test_acc={:.2f}".format(model_type, test_size, dev_size, train_size, train_acc, dev_acc, test_acc))
            #print("best_hparams: ", best_hparams)
            cur_run_results= {'model_type':model_type, 'run_index':curr_run_i, 'train_acc':train_acc, 'dev_acc':dev_acc, 'test_acc':test_acc}
            results.append(cur_run_results)

results_df = pd.DataFrame(results)

print(pd.DataFrame(results).groupby('model_type').describe().T)

#breakpoint()



