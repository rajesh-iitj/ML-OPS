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
from utils import preprocess_data, split_data, train_model, read_digits


#1. Get the dataset
X, y = read_digits();

#2. data splitting 
# Split data into 50% train and 50% test subsets
X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.3, random_state = False)

#3. Data preprocessing
# flatten the images
X_train = preprocess_data(X_train)
X_test  = preprocess_data(X_test)
        
#4. Model training
model = train_model(X_train, y_train, {'gamma': 0.001}, model_type="svm")


#5. Getting model prediction on test set
# Predict the value of the digit on the test subset
predicted = model.predict(X_test)

#6 Qualitative sanity check of the prediction
#_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
#for ax, image, prediction in zip(axes, X_test, predicted):
#    ax.set_axis_off()
#    image = image.reshape(8, 8)
#    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
#    ax.set_title(f"Prediction: {prediction}")

#7. Model evaluation
###############################################################################
# :func:`~sklearn.metrics.classification_report` builds a text report showing
# the main classification metrics.

print(
    f"Classification report for classifier :\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)

###############################################################################
# We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
# true digit values and the predicted digit values.

disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()

###############################################################################
# If the results from evaluating a classifier are stored in the form of a
# :ref:`confusion matrix <confusion_matrix>` and not in terms of `y_true` and
# `y_pred`, one can still build a :func:`~sklearn.metrics.classification_report`
# as follows:

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
    f"{metrics.classification_report(y_true, y_pred)}\n"
)
