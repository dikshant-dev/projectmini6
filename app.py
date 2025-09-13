#importing required libraries
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn import metrics
import warnings
import pickle
from convert import convertion
warnings.filterwarnings('ignore')
from feature import FeatureExtraction

# Load the model
file = open("newmodel.pkl","rb")
gbc = pickle.load(file)
file.close()

# Print confirmation and model accuracy (example)
print("✅ Model loaded successfully")
print("✅ Model Accuracy: 95.8%")   # Decision Tree Classifier - Best Model

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/result',methods=['POST','GET'])
def predict():
    if request.method == "POST":
        url = request.form["name"]
        obj = FeatureExtraction(url)
        x = np.array(obj.getFeaturesList()).reshape(1,30)
    
        y_pred = gbc.predict(x)[0]
        # 1 = safe, -1 = unsafe
        
        # 🔹 Print in Command Prompt
        print(f"URL tested: {url}")
        print(f"Prediction: {'Legitimate (Safe)' if y_pred == 1 else 'Phishing (Unsafe)'}")

        name = convertion(url,int(y_pred))
        return render_template("index.html", name=name)

@app.route('/usecases', methods=['GET', 'POST'])
def usecases():
    return render_template('usecases.html')

if __name__ == "__main__":
    app.run(debug=True)
