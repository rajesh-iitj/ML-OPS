from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split

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
    return X, y

def split_train_dev_test(x, y, test_sz, dev_sz):
    X_train, X_dev_test, y_train, y_dev_test = train_test_split(x, y, test_size = (test_sz + dev_sz), random_state = False)
    X_dev, X_test, y_dev, y_test = train_test_split(X_dev_test, y_dev_test, test_size = dev_sz, random_state = False)
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

