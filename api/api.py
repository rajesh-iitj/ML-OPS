from flask import Flask, request

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
