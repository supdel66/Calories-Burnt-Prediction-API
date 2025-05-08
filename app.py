
from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Calories prediction API is running."})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        input_array = np.array([[
            data['age'],
            data['gender'],
            data['height'],
            data['weight'],
            data['duration'],
            data['heart_rate'],
            data['body_temp']
        ]])

        prediction = model.predict(input_array)
        return jsonify({"predicted_calories": f"{prediction[0]} cal"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=3000, debug=True)
