from flask import Flask,render_template,request
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import requests

model = tf.keras.models.load_model("model.h5")

app = Flask(__name__ , template_folder = 'template')

@app.route("/",methods = ['GET','POST'])
def home():
    if request.method == 'GET':
        return render_template("index.html")
    else:
        url = request.form['url']
        try:
            response = requests.get(url)
            img_bytes = io.BytesIO(response.content)
            img = Image.open(img_bytes)
            img = tf.convert_to_tensor(np.asarray(img))
            if len(img.shape)<3:
                img = tf.expand_dims(img,-1)
                img = tf.image.resize(img,(28,28))
            
            else:
                img = tf.image.resize(img,(28,28))
                img = tf.image.rgb_to_grayscale(img)
            img_tensor = tf.expand_dims(img,0)
            prediction = tf.argmax(model.predict(img_tensor),1)[0].numpy()
            return f"<h1 text_align = 'center'>{prediction}</h1>"

        except:
            return "Something went wrong"
