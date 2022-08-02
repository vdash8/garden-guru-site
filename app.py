from flask import (Flask, render_template, request)
from werkzeug.utils import secure_filename
import os 
import cv2 
import numpy as np 
import pickle
import plotly.express as px
import pandas as pd

UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def hello():
    return render_template("home.html")

def generate_sunburst(df,
                     soil,
                     want_edible=True,
                     want_medicinal=False, 
                     desired_edible_rating=5,
                     desired_medicinal_rating=None,
                     sunburst_path=["Continent", "Family", "Common name"]):
    """
    Plots sunburst chart for soil type SOIL, from main dataframe DF. By default, plots
    plants with the highest edibility rating. Can modify default parameters to plot 
    medicinal plants instead. Uses plotly.express for sunburst plot. 
    """
    
    if soil.lower() == "clay":
        soil_df = df[df['Heavy clay'] == True]
    else: 
        soil_df = df[df['Cultivation details'].str.contains('loam') == True]
        
    if want_edible:
        desired_plant_attribute = "EdibilityRating"
        desired_rating = desired_edible_rating
        sunburst_values="EdibilityRating"
    
    elif want_medicinal: 
        desired_plant_attribute = "MedicinalRating"
        desired_rating = desired_medicinal_rating
        sunburst_values="MedicinalRating"
    
    # Removing all plants with NULL continent data 
    cleaned_soil_df = soil_df[~soil_df["Continent"].isna()]
    # Removing all plants with undesired values
    desired_rating_df = cleaned_soil_df[cleaned_soil_df[desired_plant_attribute] == desired_rating]
    
    # Creating a list of columns needed for sunburst plotting purposes
    cols = list(sunburst_path) + [sunburst_values]
    truncated_soil_df = desired_rating_df[cols]
    
    # Plotting code 
    fig = px.sunburst(truncated_soil_df, 
                      path=sunburst_path, 
                      values=sunburst_values)
    
    return fig   

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
        os.remove(image_path)
        return flattened_img_arr 

    def predict(image):
        categories_dict = {
            0: 'clay',
            1: 'loam'
        }
        model = pickle.load(open("static/models/nusvc_soil.pkl", 'rb'))
        predicted = int(model.predict(image)[0])
        predicted_soil = categories_dict[predicted]
        return predicted_soil
    
    image = preprocess_image(filename)
    predicted_soil = predict(image)

    plants = pd.read_csv("static/data/cleaned_plants.csv")

    sunburst = generate_sunburst(plants, predicted_soil)

    sunburst.show()

    return render_template("home.html")

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

@app.route("/overview")
def get_overview():
    return render_template("overview.html")

@app.route("/home")
def return_home():
    return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True)