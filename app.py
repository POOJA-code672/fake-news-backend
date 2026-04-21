from flask import Flask, request, jsonify
import pickle
from flask_cors import CORS

app = Flask(__name__)

# ✅ Allow frontend (React) to connect
CORS(app)

print("🔥 API HIT")

# ✅ Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ✅ Home route (for testing)
@app.route("/")
def home():
    return "Backend is running ✅"

# ✅ Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        print("🔥 API HIT")   # DEBUG

        data = request.get_json()
        print("Received data:", data)   # DEBUG

        # ✅ Get text safely
        text = data.get("text", "")

        if text == "":
            return jsonify({"prediction": "⚠️ Please enter some text"})

        # ✅ Transform and predict
        transformed = vectorizer.transform([text])
        prediction = model.predict(transformed)[0]

        # ✅ Result
        result = "Fake News ❌" if prediction == 0 else "Real News ✅"

        return jsonify({"prediction": result})

    except Exception as e:
        print("❌ ERROR:", e)
        return jsonify({"prediction": "Error in backend ❌"})

# ✅ IMPORTANT: allow external connection
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)