#content of test_sample.py

from utils import create_combinations_dict_from_lists, read_digits, split_train_dev_test

def inc(x):
    return x + 1


def test_answer():
    assert inc(3) == 4

def test_wrong_answer():
    assert inc(3) == 5

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


# def test_data_splitting():
#     X, y = read_digits()
#     X = X[:100, :, :]
#     y = y[:100]

#     X_trai