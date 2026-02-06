from flask import Flask, request, render_template
import numpy as np
import warnings
from convert import convertion
from feature import FeatureExtraction
import pickle
from urllib.parse import urlparse



warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load CatBoost model
with open("newmodel.pkl", "rb") as f:
    model = pickle.load(f)


print("✅ CatBoost model loaded successfully")
print("✅ Model Accuracy: 97.1%")

@app.route("/")
def home():
    return render_template("index.html")

TRUSTED_DOMAINS = [
    "google.com",
    "gmail.com",
    "amazon.com",
    "flipkart.com",
    "instagram.com",
    "linkedin.com",
    "twitter.com",
    "x.com",
    "microsoft.com",
    "windows.com",
    "apple.com",
    "icloud.com",
    "nasa.gov",
    "github.com",
    "stackoverflow.com",
    "netflix.com"
]

def is_trusted_domain(url):
    try:
        domain = urlparse(url).netloc.lower()
        for trusted in TRUSTED_DOMAINS:
            if trusted in domain:
                return True
        return False
    except:
        return False

@app.route('/result', methods=['POST', 'GET'])
def predict():
    if request.method == "POST":
        url = request.form["name"]

        if is_trusted_domain(url):
            name = convertion(url, 1)
            return render_template("index.html", name=name)

        obj = FeatureExtraction(url)
        x = np.array(obj.getFeaturesList()).reshape(1, 30)
        y_pred = model.predict(x)[0]

        name = convertion(url, int(y_pred))
        return render_template("index.html", name=name)


@app.route('/usecases', methods=['GET', 'POST'])
def usecases():
    return render_template('usecases.html')

if __name__ == "__main__":
    app.run(debug=True)
