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

# def train_test_dev_split():
#         return X_train, X_test, y_train, y_test, X_dev, y_dev


# def predict_and_eval():
#      #prediction
#      #report metrics
