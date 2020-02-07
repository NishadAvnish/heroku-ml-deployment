# keras_server.py 
  
# Python program to expose a ML model as flask REST API 
  
# import the necessary modules 
from PIL import Image 
import numpy as np 
import flask 
import io 
from keras.models import load_model
import joblib
from flask import render_template
  
# Create Flask application and initialize Keras model 
app = flask.Flask(__name__) 
model = None
image1=None
# Function to Load the model  
def load_models(): 
      
    # global variables, to be used in another function 
    global model      
    model= joblib.load('outputfile.pkl') 
  
# Every ML/DL model has a specific format 
# of taking input. Before we can predict on 
# the input image, we first need to preprocess it. 
def prepare_image(image, target): 
    #if image.mode != "RGB": 
    #    image = image.convert("RGB") 
    
    # Resize the image to the target dimensions 
    image = image.resize(target)  
    # PIL Image to Numpy array
    global image1
    image1=np.array(image)  
    
    # Expand the shape of an array, 
    # as required by the Model 
    image1.resize(1,128,128,3)
    
@app.route("/",methods=["GET"])
def home():
     return "HELLO, go to http://127.0.0.1:5000//predict"
  
# Now, we can predict the results. 
@app.route("/predict", methods =["POST","GET"]) 
def predict(): 
    data = {} # dictionary to store result 
    data["success"] = False
  
    # Check if image was properly sent to our endpoint 
    if flask.request.method == "POST": 
        if flask.request.files.get("image"): 
            image = flask.request.files["image"].read() 
            image = Image.open(io.BytesIO(image)) 
            
            # Resize it to 224x224 pixels  
            # (required input dimensions for ResNet) 
            prepare_image(image, target =(128,128)) 
             
            preds = model.predict(image1) 
            
            if(preds[0][0]>preds[0][1]):
                data["pothhole"]=False
            else:
                data["pothhole"]=True
  
  
            data["success"] = True
  
    # return JSON response 
    return flask.jsonify(data) 
  
  
  
if __name__ == "__main__": 
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started")) 
    load_models() 
    app.run(threaded=False) 