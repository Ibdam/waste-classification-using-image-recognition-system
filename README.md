🧠 Trash Type Image Classification with Deep Learning

A deep learning web application that classifies waste materials into categories such as cardboard, glass, metal, paper, plastic, and trash using a trained Convolutional Neural Network (CNN) model.
The model is integrated into a Flask web app for easy visual prediction and deployed on Render for live demonstration.

🚀 Features

Classifies images of waste materials into six categories.

Supports .jpg, .jpeg, and .png image formats.

Displays prediction results with confidence level.

Built using TensorFlow (Keras) and Flask.

User-friendly front-end interface with HTML & CSS.

Ready for cloud deployment (Render, Heroku, etc.).

🧩 Project Structure
Waste classification using image recognition/
│
├── app.py
|       |__ static/
|       |    └── style.css             # CSS for styling                   # Flask backend app
|
|       |__  templates/                # HTML templates
│           ├── index.html            # Upload form
│           └── result.html           # Prediction result page
|
|       |__ wast_dataset_split
|           |__ test            # For testing
|           |__ train           # For training
|           |__ val             #For validation
|
|       |__init__.py #Contain flask code for backend
|
├── model.h5                  # Trained CNN model
|__ model_convert.ipynb       # Codes that canvert the model to reduce size for deployment
|__ Coverted_waste_model.tflite # The converted model
├── requirements.txt          # Python dependencies
├── render.yaml               # Render deployment configuration
│__ waste.ipynb               # Contain a jupyter notebook code for the model training
|__ waste_model.h5            #Contain the trained model
|__ Readme.md                 # Information about the project
|__ wsgi.py                   # For running the system
|__ procfile                  # Gunicorn for depployemt on render

⚙️ Technologies Used

Python 3.9+

Flask – for web app framework

TensorFlow / Keras – for deep learning model

Pillow (PIL) – for image preprocessing

NumPy – for numerical computations

Gunicorn – for production server (Render)

💻 Local Setup Instructions
1️⃣ Clone the repository
git clone https://github.com/yourusername/trash-classifier.git
cd Waste classification using image recognition
2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the Flask app
python wsgi.py

Visit http://127.0.0.1:5000 in your browser.

🧠 Model Details

The model is a Convolutional Neural Network (CNN) trained on a dataset containing labeled images of different trash types:

cardboard/

glass/

metal/

paper/

plastic/

trash/

Images were resized to 150×150 pixels and normalized.
You can retrain or fine-tune the model by modifying the notebook and exporting it again as waste_model.h5.

🌐 Deployment on Render
1️⃣ Push your project to GitHub

Make sure your repo includes:

app.py

requirements.txt

render.yaml

model.h5

templates/ and static/ folders

2️⃣ Go to Render.com

Create a New Web Service

Connect your GitHub repo

Render will auto-detect Flask from your files

3️⃣ Deploy 🚀

Your app will automatically build and be hosted on a public Render URL.

📷 Sample Usage

Upload an image of a waste material (e.g., plastic bottle.jpg)

Wait for the prediction result

The app displays:

Predicted Class: Plastic
Confidence: 92.47%

⚠️ Notes

If using .png images, they are automatically converted to RGB format to avoid alpha channel errors.

Large models may take a few seconds to load when the app first starts.

Confidence levels may vary based on image clarity, lighting, and similarity between categories.

✨ Future Improvements

Add image preview on the results page

Improve model accuracy with data augmentation

Use a pre-trained CNN (MobileNetV2 or EfficientNet)

Add drag-and-drop upload feature

👨‍💻 Author

Olowomojuore Damilola Ibrahim
Machine Learning, AI & Data Science Professional | Environmental Engineer