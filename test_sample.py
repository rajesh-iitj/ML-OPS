#content of test_sample.py

from utils import create_combinations_dict_from_lists, read_digits, split_train_dev_test, preprocess_data, tune_hparams, get_hyperparameter_combinations
import os
import pdb

def inc(x):
    return x + 1


# def test_answer():
#     assert inc(3) == 4

# def test_wrong_answer():
#     assert inc(3) == 5

def test_hparams_combinations_count():
    # a test case to check all possbile combinatons of paramters are indeed generated
    gamma_ranges = [0.001, 0.01, 0.1, 1, 10, 100]
    C_ranges = [0.1, 1, 2, 5, 10]
    h_params_combinations = create_combinations_dict_from_lists( gamma_ranges, C_ranges)
    assert len(h_params_combinations) == len (gamma_ranges) * len(C_ranges)
    
def test_hparams_combinations_values():
    # a test case to check all possbile combinatons of paramters are indeed generated
    gamma_ranges = [0.001]
    C_ranges = [1]
    combo = create_combinations_dict_from_lists( gamma_ranges, C_ranges)
    expected_param_combo_1 = {'(0.001,1)': (0.001, 1)}
    expected_param_combo_2 = {'(0.01,1)': (0.01, 1)}

    bval = ((all(key in combo and combo[key] == value for key, value in expected_param_combo_1.items())) and 
           (all(key in combo and combo[key] == value for key, value in expected_param_combo_1.items())))
    assert(bval)


def test_data_splitting():
    X, y = read_digits()
    X = X[:100,:,:]
    y = y[:100]

    test_size = 0.1
    dev_size = 0.6
    train_size = 1 - dev_size - test_size

    X_train, X_dev, X_test, y_train, y_dev, y_test = split_train_dev_test(X, y, test_sz = test_size, dev_sz = dev_size)

    #import pdb; pdb.set_trace()

    assert (len(X_train)    == int(train_size * len(X))) 
    assert (len(X_dev)      == int(dev_size * len(X))) 
    assert (len(X_test)     == int(test_size * len(X)))

    #X = X[:100, :, :]
    #y = y[:100]
    #X_trai


#test case for machine learning
    # - turning the hparams
    # - test teh accuracy based on metrics. 
    #       Regression test->after training get infeerences (predictions) on the regression tests
    #       when you deploy the model - there should be a check as part of the deployment pipeline inferences on the model should be 
    #       exactly same as from the previous step
    #           deployment pipeline (model, regression)set, expected_prediction
    #           inference code differece between the training repo and the deployment repo
    #           repo - the deployment candidate mdoel is the same as epxected (based on the model selection) 
    #                 
    # - generalizablity of the model
    # - data quality - is feature encoding as per requirment or not.
    # - shape of data frames
    # - overfitting, underfitting - test case would check isOverfitting(model, benchmark_data) == true
    #       is my code good enough to make the model learn something
    #           if my data is very small and clean , can the model atleast overfitt on it
    #           evalaute(model, training_module(small_training_set) == near perfect accuracy / some metric
    # - is the model getting saved or not
    # - is the model we save is right or not

    
    # - functionaliy change :
    # - once training is done, model must be saved
    # - 

    # - anomility detection - valid parts of data curating / cleaning


def create_dummy_hyperparamters():
    gamma_list = [0.001, 0.01]
    C_list = [1]

    h_params= {}
    h_params['gamma'] = gamma_list
    h_params['C'] = C_list
    h_params_combinations = get_hyperparameter_combinations(h_params) 
    #breakpoint()
    return h_params_combinations
def create_dummy_data():
    X, y = read_digits()

    X_train = X[:100,:,:]
    y_train = y[:100]
    X_dev   = X[:50,:,:]
    y_dev   = y[:50]


    X_train = preprocess_data(X_train)
    X_dev   = preprocess_data(X_dev)

    return X_train, y_train, X_dev, y_dev

def test_model_saving():
    X_train, y_train, X_dev, y_dev = create_dummy_data()
    h_params_combination = create_dummy_hyperparamters()

    _, best_model_path, _ = tune_hparams(X_train, y_train, X_dev, y_dev, h_params_combination, model_type='svm')

    #assert os.path.exists(best_model_path)
