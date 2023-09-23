from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
from itertools import product

def preprocess_data(data):
    n_samples = len(data)
    data = data.reshape((n_samples, -1))
    return data

def split_data(x, y, test_size, random_state):
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.5,random_state = random_state)
    return X_train, X_test, y_train, y_test


def train_model(x,y, model_paramters, model_type="svm"):
    if model_type == "svm":
        # Create a classifier: a support vector classifier
        clf = svm.SVC
    model = clf(**model_paramters)
    # train the model
    model.fit(x, y)
    return model

def read_digits():
    digits = datasets.load_digits();
    X = digits.images
    y = digits.target

    noSamples, height, width = digits.images.shape

    print ("Total Samples ", noSamples)
    print ("Images width: ", width, "Height: ", height)

    return X, y

def split_train_dev_test(x, y, test_sz, dev_sz):
    X_train, X_dev_test, y_train, y_dev_test = train_test_split(x, y, train_size = 1 - (test_sz + dev_sz), random_state = False)
    new_dev_sz = dev_sz / (dev_sz + test_sz)
    X_dev, X_test, y_dev, y_test = train_test_split(X_dev_test, y_dev_test, train_size = new_dev_sz, random_state = False)
    return X_train, X_dev, X_test, y_train, y_dev, y_test


def predict_and_eval(model, X_test, y_test):
    predicted = model.predict(X_test)
    return metrics.accuracy_score(y_test, predicted)

def predict_and_eval2(model, X_test, y_test):
    predicted = model.predict(X_test)
    print(
    f"Classification report for classifier on data set:\n"
    f"{metrics.classification_report(y_test, predicted)}\n")
    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")
    # The ground truth and predicted lists
    y_true = []
    y_pred = []
    cm = disp.confusion_matrix

    # For each cell in the confusion matrix, add the corresponding ground truths
    # and predictions to the lists
    for gt in range(len(cm)):
        for pred in range(len(cm)):
            y_true += [gt] * cm[gt][pred]
            y_pred += [pred] * cm[gt][pred]

    print(
        "Classification report rebuilt from confusion matrix:\n"
        f"{metrics.classification_report(y_true, y_pred)}\n")


def create_combinations_dict_from_lists(listA, listB):
    comb = list(product(listA, listB))
    comb_dict = {f"({x},{y})": (x, y) for x, y in comb}
    return comb_dict

def tune_hparams(X_train, y_train, X_dev, y_dev, list_of_all_param_combination_dictionaries):

    best_accuracy = -1
    for key, value in list_of_all_param_combination_dictionaries.items():
        cur_gamma = value[0]
        cur_C = value[1]
        cur_model = train_model(X_train, y_train, {'gamma': cur_gamma, 'C': cur_C}, model_type="svm")
        cur_accuracy = predict_and_eval(cur_model, X_dev, y_dev)
        if cur_accuracy > best_accuracy:
            #print("New best accuracy: ", cur_accuracy)
            best_accuracy = cur_accuracy
            best_model = cur_model
            best_hparams = (cur_gamma, cur_C)

    #print("Optimal paramters gamma: ", best_hparams[0], "C: ", best_hparams[1])
    return best_hparams, best_model, best_accuracy