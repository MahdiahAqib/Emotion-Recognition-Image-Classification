from flask import Flask, render_template, request
import cv2
import numpy as np
import tensorflow as tf

# Load model
model = tf.keras.models.load_model('expression_classifier.keras')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        image = request.files['image']
        # Preprocess the uploaded image
        img = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_COLOR)
        resized_image = cv2.resize(img, (256, 256))
        resized_image = np.expand_dims(resized_image.astype(np.float32) / 255.0, axis=0)
        # Make a prediction using the model
        predictions = model.predict(resized_image)
        probability_class_1 = predictions[0][0]
        if probability_class_1 >= 0.5:
            result = 'Happy'
        else:
            result = 'Sad'

        image_path = None
        return render_template('index.html', result=result, image_path=image_path)


if __name__ == '__main__':
    app.run(debug=True)