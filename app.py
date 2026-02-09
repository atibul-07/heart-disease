from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load("heart_disease_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get form values
    features = [float(x) for x in request.form.values()]
    final_input = np.array([features])

    prediction = model.predict(final_input)[0]

    if prediction == 1:
        result = "High chance of Heart Disease"
    else:
        result = "Low chance of Heart Disease"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
