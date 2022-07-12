from flask import (Flask, render_template, request)
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def hello():
    return render_template("index.html")

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
        return "File uploaded successfully."
    return render_template("upload.html")

if __name__ == "__main__":
    app.run(debug=True)