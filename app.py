from flask import Flask, request, render_template
import numpy as np
import warnings
from convert import convertion
from feature import FeatureExtraction
import pickle

warnings.filterwarnings('ignore')

app = Flask(__name__)


with open("newmodel.pkl", "rb") as f:
    model = pickle.load(f)

print(" CatBoost model loaded successfully")
print(" Model Accuracy: 97.1%")

# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction route
@app.route('/result', methods=['POST'])
def predict():
    url = request.form["name"].strip()

    # Ensure URL has protocol
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    try:
        # Feature extraction
        obj = FeatureExtraction(url)
        x = np.array(obj.getFeaturesList()).reshape(1, 30)

        # Model prediction
        y_pred = int(model.predict(x)[0])

        # Convert prediction to message
        name = convertion(url, y_pred)

    except Exception as e:
        name = " Error processing URL. Please enter a valid website."

    return render_template("index.html", name=name)


# Optional use-cases page
@app.route('/usecases', methods=['GET'])
def usecases():
    return render_template('usecases.html')


if __name__ == "__main__":
    app.run(debug=True)
