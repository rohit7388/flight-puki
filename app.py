from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model and scaler
with open("fraud_model.pkl", "rb") as f:
    model = pickle.load(f)

try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    use_scaler = True
except:
    scaler = None
    use_scaler = False

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None

    if request.method == "POST":
        duration = float(request.form["duration"])
        days_left = float(request.form["days_left"])
        price = float(request.form["price"])

        features = np.array([[duration, days_left, price]])
        if use_scaler:
            features = scaler.transform(features)

        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0][1]

        prediction = "Fraudulent Transaction 🚨" if pred == 1 else "Safe Transaction ✅"
        probability = f"{prob:.2f}" if pred == 1 else f"{1 - prob:.2f}"

    return render_template("index.html", prediction=prediction, probability=probability)

if __name__ == "__main__":
    app.run(debug=True)
