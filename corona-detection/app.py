import sys
import os
import numpy as np
import cv2

from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
MODEL_PATH = 'model1.h5'

model = load_model(MODEL_PATH)
print('Model loaded. Start serving...')

print('Check http://127.0.0.1:5000/')

def model_predict(img_path, model):
    img_height,img_width=224,224
    image=cv2.imread(img_path)
    image_resized= cv2.resize(image, (img_height,img_width))
    image=np.expand_dims(image_resized,axis=0)
    preds=model.predict(image)
    return preds

@app.route('/', methods=['GET'])
def index():
    return render_template('home.html')
@app.route('/about.html', methods=['GET'])
def about():
    return render_template('about.html')    


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        preds = model_predict(file_path, model)
        if preds[0][0] == 0:
            prediction = 'Covid-19 POSITIVE'
        else:
            prediction = 'Covid-19 NEGATIVE'
        return prediction
    return None

if __name__ == '__main__':
    app.run(debug=True)

