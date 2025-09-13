from flask import Flask, render_template, request, redirect, send_from_directory, url_for
import numpy as np
import json
import uuid
import tensorflow as tf
import requests
import os
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Google Drive direct download link
MODEL_URL = "https://drive.google.com/uc?export=download&id=1WvWNOOGxHHjorlUBmjf2AlyVtTSSF84m"
MODEL_PATH = "models/plant_disease_classification_model.keras"

# Download model if it doesn't exist
def download_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs("models", exist_ok=True)
        print("Downloading model from Google Drive...")
        response = requests.get(MODEL_URL, stream=True)
        
        # Handle Google Drive virus scan warning for large files
        if 'content-disposition' not in response.headers:
            # This might be the virus scan warning page, try to get the actual download link
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    confirm_url = f"{MODEL_URL}&confirm={value}"
                    response = requests.get(confirm_url, stream=True)
                    break
        
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("Model downloaded successfully!")
    
    return tf.keras.models.load_model(MODEL_PATH)

# Load model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded from local storage")
except:
    print("Local model not found, downloading...")
    model = download_model()

label = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Background_without_leaves',
 'Blueberry___healthy',
 'Cherry___Powdery_mildew',
 'Cherry___healthy',
 'Corn___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn___Common_rust',
 'Corn___Northern_Leaf_Blight',
 'Corn___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']

with open("plant_disease.json",'r') as file:
    plant_disease = json.load(file)

@app.route('/uploadimages/<path:filename>')
def uploaded_images(filename):
    return send_from_directory('./uploadimages', filename)

@app.route('/',methods = ['GET'])
def home():
    return render_template('home.html')

def extract_features(image_path):
    image = tf.keras.utils.load_img(image_path, target_size=(160,160))
    feature = tf.keras.utils.img_to_array(image)
    feature = np.array([feature])
    return feature

def model_predict(img_path):
    # Load the image and resize to 224x224
    img = image.load_img(img_path, target_size=(224, 224))  

    # Convert to array
    img_array = image.img_to_array(img)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)  

    # Normalize if your model expects normalized input
    img_array /= 255.0  

    # Make prediction
    prediction = model.predict(img_array)

    # Get predicted label
    prediction_label = plant_disease[prediction.argmax()]

    return prediction_label

@app.route('/upload/',methods = ['POST','GET'])
def uploadimage():
    if request.method == "POST":
        image_file = request.files['img']
        if image_file:
            # Create upload directory if it doesn't exist
            os.makedirs("uploadimages", exist_ok=True)
            
            # Generate unique filename
            filename = f"temp_{uuid.uuid4().hex}_{image_file.filename}"
            filepath = f"uploadimages/{filename}"
            
            # Save the file
            image_file.save(filepath)
            
            # Make prediction
            try:
                prediction = model_predict(filepath)
                return render_template('home.html', result=True, imagepath=f'/uploadimages/{filename}', prediction=prediction)
            except Exception as e:
                return render_template('home.html', error=f"Prediction error: {str(e)}")
    
    return redirect('/')

if __name__ == "__main__":
    app.run(debug=True)
