"""
These code is a backend code for waste classification, Image recognition system.
It classify the uploaded image to cardboard, glass, metal, paper, plastic and trash.
Here are the codes steps
"""

#Import the required libaries
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

#Create a flask app module and load the model
def create_app(test_config=None):

    # Create and configure the app
    app = Flask(__name__, instance_relative_config=True)

    # Configure the app
    app.config.from_mapping(
        SECRET_KEY= os.getenv('SECRET_KEY', 'fallback-secret-key')    
    )
    
    if test_config is None:

        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)

    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass
    
    # Load the TensorFlow Lite model
    interpreter = tf.lite.Interpreter(model_path="converted_waste_model.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    #Names of the class
    class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash'] #Name of the image class

    #Route the home page
    @app.route('/')
    def home():
        return render_template('waste.html')

    #Create an end point to accept user input
    @app.route('/predict', methods=['POST'])
    def predict():
        if 'file' not in request.files:
            return "No file uploaded", 400 #To handle if no file is choosen

        file = request.files['file'] #Accept the input file
        if file.filename == '':
            return "No file selected", 400 #If the chosen file has no name

        """
        Some images like png has 4 channels (RGBA) instead of 3 channels RGB.
        These piece of code convert the 4 channels image like png to 3 channels RGB.
        And it prevents the image recognition system from bringing error.
        """
        try:
            # Read the uploaded image directly (no saving) for production
            img = Image.open(io.BytesIO(file.read())) #Open the image
        
            #Convert all images to RGB (fixes PNG alpha issue)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize((150, 150))

            #Resize the input image, convert to NumPy array and normalize
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array.astype(np.float32)

            #Code to save the uploaded image, perfect for local host
            """
            file_path = os.path.join('static', file.filename)
            file.save(file_path)

            img = image.load_img(file_path, target_size=(150, 150))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            """
        
            # Set tensor input
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()

            # Get predictions
            predictions = interpreter.get_tensor(output_details[0]['index'])[0]
            predicted_class = class_names[np.argmax(predictions)]
            confidence = round(100 * np.max(predictions), 2)

            return render_template('result.html', 
                                prediction=predicted_class, 
                                confidence=confidence,
                                )
        except Exception as e:
            return f"Error processing image: {str(e)}", 500

    return app
