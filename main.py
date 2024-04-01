from flask import Flask, request, render_template
import pandas as pd
from joblib import load
from sklearn.calibration import LabelEncoder

app = Flask(__name__)

model = load("chess.pkl")

@app.route("/")
def home():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)