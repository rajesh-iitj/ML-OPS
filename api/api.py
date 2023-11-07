from flask import Flask, request

from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from itertools import product
import os
from joblib import dump, load
import pdb
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import numpy as np

app = Flask(__name__)

@app.route("/hello/<val>")
def hello_world(val):
    return "<p>Hello, World!</p>" + val


@app.route("/sum/<x>/<y>")
def sum_num(x,y):
    return str(int(x) + int (y))

@app.route("/models", methods = ['POST'])
def pred_model():
    js = request.get_json( )
    x = js['x']
    y = js['y']
    return x + y

@app.route("/predict", methods = ['POST'])
def pred_image():
    js = request.get_json( )
    image1 = js['image1']
    image2 = js['image2']
    best_model = load('../models/svm_gamma:0.001_C:1.joblib')

    n_samples = len(image1)
    image1_2d = np.array(image1)
    image1_2d = image1_2d.reshape((n_samples, -1))

    n_samples = len(image2)
    image2_2d = np.array(image2)
    image2_2d = image2_2d.reshape((n_samples, -1))

    predicted1 = best_model.predict(image1_2d)
    predicted2 = best_model.predict(image2_2d)

    return str(predicted1 == predicted2)


    return str(x==y)