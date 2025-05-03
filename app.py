import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Initialize Flask app
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
MODEL_PATH = 'best_model.h5'
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# Class names for display
class_names = [
    'BENGIN CASE (It is the lung tumors are non-cancerous growths in the lungs that typically do not spread or pose serious health risks.)',
    'MALIGNANT CASE (It is the lung tumors are cancerous growths that can spread rapidly to other parts of the body and require immediate medical attention.)',
    'TEST REPORT IS NORMAL'
]

# Utility function to check file type
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home route
@app.route('/')
def home():
    return render_template('index0.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index0.html', prediction='No file uploaded.')

    file = request.files['image']
    if file.filename == '':
        return render_template('index0.html', prediction='No file selected.')

    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        try:
            # Load and preprocess the image
            img = load_img(filepath, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)

            # Predict with model
            preds = model.predict(img_array)
            pred_label = class_names[np.argmax(preds)]

            # Return result
            return render_template('index0.html', prediction=f'Result: {pred_label}')

        except Exception as e:
            return render_template('index0.html', prediction=f'Error during prediction: {str(e)}')

    else:
        return render_template('index0.html', prediction='Invalid file type. Allowed: png, jpg, jpeg')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
