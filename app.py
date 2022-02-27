from logging import debug
from xml.etree.ElementTree import SubElement
from flask import Flask, render_template,url_for, request
import pickle
import joblib
import numpy as np

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    rf_model = open('data/random_forest.pkl', 'rb')
    rf = joblib.load(rf_model)

    if request.method == 'POST':
        ph = request.form['ph']
        Trihalomethanes= request.form['trihalomethanes']
        Hardness = request.form['hardness']
        Sulfate= request.form['sulfate']
        Solids= request.form['solids']
        Organic_carbon=request.form['organic_carbon']

        arr=[ph, Trihalomethanes, Hardness, Sulfate, Solids, Organic_carbon]
        arr_new=[float(i) for i in arr]
        arr2=np.array(arr_new).reshape(1,-1)

        p = rf.predict(arr2)
        if p == 1:
            return render_template("unsafe.html")
        else:
            return render_template("safe.html")


if __name__ == '__main__':
    app.run(debug=True)

    