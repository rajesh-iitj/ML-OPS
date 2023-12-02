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
    image1 = js['image']
    best_model = load('./models/m22aie221_svm_gamma_0.001C_1.joblib')
    image1_1d = np.array(image1).reshape(1, -1)
    predicted1 = best_model.predict(image1_1d)
    print(predicted1)
    return str(predicted1)


@app.route("/compare", methods = ['POST'])
def comp_image():
    js = request.get_json( )
    image1 = js['image1']
    image2 = js['image2']
    best_model = load('./models/m22aie221_svm_gamma_0.001C_1.joblib')
    image1_1d = np.array(image1).reshape(1, -1)
    image2_1d = np.array(image2).reshape(1, -1)
    predicted1 = best_model.predict(image1_1d)
    predicted2 = best_model.predict(image2_1d)
    return str(predicted1 == predicted2)


@app.route("/")
def hello_world1():
    return "<p>Hello, World!</p>"

@app.route("/", methods=["POST"])
def hello_world_post():    
    return {"op" : "Hello, World POST " + request.json["suffix"]}


def load_model(model_type):
    if model_type == 'svm':
        model = load('./models/m22aie221_svm_gamma_0.001C_1.joblib')
    elif model_type == 'tree':
        model = load('./models/m22aie221_dtree_max_depth_5.joblib')
    elif model_type == 'lr':
        model = load('./models/m22aie221_lr_solver_lbfgs.joblib')
    return model

@app.route("/predict/<model_type>", methods = ['POST'])
def pred_image_model_type(model_type):
    js = request.get_json( )
    image1 = js['image']
    model = load_model(model_type)
    image1_1d = np.array(image1).reshape(1, -1)
    predicted1 = model.predict(image1_1d)
    print(predicted1)
    return str(predicted1)
