from flask import (Flask, render_template, request)
from werkzeug.utils import secure_filename
import os 
import cv2 
import numpy as np 
import pickle

UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def hello():
    return render_template("index.html")

def run_model(filename):
    def preprocess_image(filename):
        image_path = os.path.join("static/uploads", filename)
        img = cv2.imread(image_path, 1)
        # Resizing image to 100x100 pixels 
        resized_img = cv2.resize(img, (100, 100))
        # Storing it onto numpy array
        resized_img_arr = np.array(resized_img)
        # Flattening and resizing for model input
        flattened_img_arr = resized_img_arr.flatten().reshape(1, -1)
        return flattened_img_arr 

    def predict(image):
        categories_dict = {
            0: 'clay',
            1: 'loam'
        }
        model = pickle.load(open("models/nusvc_soil.pkl", 'rb'))
        predicted = int(model.predict(image)[0])
        predicted_soil = categories_dict[predicted]
        return predicted_soil
    
    image = preprocess_image(filename)
    predicted_soil = predict(image)

    return predicted_soil

@app.route("/upload", methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        if f.filename != '':
            filename = secure_filename(f.filename)
            f.save(app.config["UPLOAD_FOLDER"] + filename)

        # Instead of returning this, will have to 
        # run the model.py script to take the user 
        # given soil image and output a predicted 
        # soil type, then output a plotly sunburst 
        # visualization.     
        print("File uploaded successfully.")
        return run_model(filename)
    return render_template("upload.html")

if __name__ == "__main__":
    app.run(debug=True)