app = flask.Flask(__name__)
image1 = None
# Function to Load the model   
model = joblib.load('eww1.pkl')


# Every ML/DL model has a specific format 
# of taking input. Before we can predict on 
# the input image, we first need to preprocess it. 
def prepare_image(image):
    # Resize it to 224x224 pixels  
    # (required input dimensions for ResNet) 
    image = image.resize(224, 224)
    # PIL Image to Numpy array
    global image1
    image1 = np.array(image)
    image1.resize(1, 224, 224, 3)


@app.route("/", methods=["GET"])
def home():
    return "https://keraspothhole.herokuapp.com/predict"


# Now, we can predict the results. 
@app.route("/predict", methods=["POST", "GET"])
def predict():
    data = {"success": False}  # dictionary to store result 

    # Check if image was properly sent to our endpoint 
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            prepare_image(image)

            categories = ["asphalt", "crack", "electronic", "furniture", "kitchen", "pants and flower", "pothhole",
                          "tree", "vehicles", "wall"]

            preds = model.predict(image1)
            for i in range(0, len(preds[0])):
                if preds[0][i] > 0:
                    data[categories[i]] = True

            data["success"] = True

    # return JSON response 
    return flask.jsonify(data)


if __name__ == "__main__":
    app.run(threaded=False)
