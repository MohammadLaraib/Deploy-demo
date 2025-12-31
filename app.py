from flask import Flask, request, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model safely
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "placement_predictor_rf.pkl")

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [
            float(request.form.get("IQ")),
            float(request.form.get("CGPA")),
            float(request.form.get("10th_Marks")),
            float(request.form.get("12th_Marks")),
            float(request.form.get("Communication_Skills"))
        ]

        # Shape validation
        final_features = np.array(features).reshape(1, -1)

        # Optional: feature count check
        if hasattr(model, "n_features_in_") and model.n_features_in_ != final_features.shape[1]:
            return render_template(
                "index.html",
                prediction_text="Error: Model feature mismatch."
            )

        prediction = model.predict(final_features)[0]
        output = "Placed" if prediction == 1 else "Not Placed"

        return render_template(
            "index.html",
            prediction_text=f"Prediction: {output}"
        )

    except Exception as e:
        print("Prediction error:", e)
        return render_template(
            "index.html",
            prediction_text="Error: Invalid input. Please enter valid values."
        )

if __name__ == "__main__":
    app.run(debug=True)
