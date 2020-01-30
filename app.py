# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 00:24:21 2020

@author: avnis
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('outputfile.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    predicted_value = model.predict(final_features)

    #output = round(prediction[0], 2)
    output=""
    for value in predicted_value:
        if(value[0]>value[1]):
            output="Normal"
        else:
            output="PothHole"

    return render_template('index.html', prediction_text='The image is : {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)