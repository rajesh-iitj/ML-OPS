#content of test_sample.py

from utils import create_combinations_dict_from_lists, read_digits, split_train_dev_test, preprocess_data, tune_hparams, get_hyperparameter_combinations
import os
from api.app import app
import pdb
import pytest
import json
from sklearn import datasets
from joblib import load
from sklearn.linear_model import LogisticRegression

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

    assert os.path.exists(best_model_path)

def test_get_root():
    response = app.test_client().get("/")
    assert response.status_code == 200
    assert response.get_data() == b"<p>Hello, World!</p>"

def test_post_root():
    suffix = "post suffix"
    response = app.test_client().post("/", json={"suffix":suffix})
    #breakpoint()
    assert response.status_code == 200    
    assert response.get_json()['op'] == "Hello, World POST "+suffix

digit_0_data =[[ 0.,  0.,  5., 13.,  9.,  1.,  0.,  0.],
       [ 0.,  0., 13., 15., 10., 15.,  5.,  0.],
       [ 0.,  3., 15.,  2.,  0., 11.,  8.,  0.],
       [ 0.,  4., 12.,  0.,  0.,  8.,  8.,  0.],
       [ 0.,  5.,  8.,  0.,  0.,  9.,  8.,  0.],
       [ 0.,  4., 11.,  0.,  1., 12.,  7.,  0.],
       [ 0.,  2., 14.,  5., 10., 12.,  0.,  0.],
       [ 0.,  0.,  6., 13., 10.,  0.,  0.,  0.]]

lst=[0, 0, 0 , 0 , 0, 0, 0, 0 , 0 , 0]

def get_digit_lable():
    digits = datasets.load_digits();
    X = digits.images
    y = digits.target
    
    noSamples, height, width = digits.images.shape


    count = 0
    for i in range(0, len(y)):
        if lst[y[i]] == 0:
            lst[y[i]] = 1
            print("lable:", y[i])
            x = [element for row in X[i] for element in row]
            count = count + 1
        if (count == 10):
            break
       
        i = i + 1



    print ("Total Samples ", noSamples)
    print ("Images width: ", width, "Height: ", height)

    return X, y


def eval_label_digit(image_label, one_d_list):
    response = app.test_client().post("/predict", json={"image": one_d_list})
    assert response.status_code == 200
    response_data = (response.get_data(as_text=True))
    predicted_digit = int(response_data.strip('[]'))
    #assert image_label == predicted_digit
    if image_label != predicted_digit:
        print("image_label: ", image_label, "predicted_digit: ", predicted_digit)

def test_post_predict():
    digits = datasets.load_digits();
    X = digits.images
    y = digits.target
    
    noSamples, height, width = digits.images.shape
    count = 0
    for i in range(0, len(y)):
        if lst[y[i]] == 0:
            lst[y[i]] = 1
            print("label:", y[i])
            x = [element for row in X[i] for element in row]  
            eval_label_digit(y[i], x)
            count = count + 1
        if (count == 10):
            break
        i = i + 1




def test_lr_model_type():
    model_path = "./models/m22aie221_lr_solver_lbfgs.joblib"


    loaded_model = load(model_path)

    assert isinstance(loaded_model, LogisticRegression)

def test_solver_name():
    model_path = "./models/m22aie221_lr_solver_lbfgs.joblib"

    parts = model_path.split("_")
    solver_name_from_filename = parts[-1].split(".")[0]

    # Load the model from the file
    loaded_model = load(model_path)
    #breakpoint()
    # Check that the solver name in the file name matches the solver used in the model
    assert solver_name_from_filename == loaded_model.get_params(deep=True)['solver'], "Solver name in the file name does not match the solver used in the model."
