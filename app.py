from flask import Flask, request, jsonify
import pickle
from flask_cors import CORS
import re

app = Flask(__name__)
CORS(app)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ✅ HOME ROUTE
@app.route("/")
def home():
    return "Fake News API is running 🚀"


# ✅ PREDICT ROUTE
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")

    # 🔍 1. Detect EMAIL / MESSAGE type
    if re.search(r"(dear|regards|congratulations|sincerely|thank you)", text.lower()):
        return jsonify({
            "prediction": "📧 This looks like an EMAIL, not a news article",
            "confidence": 0
        })

    # 🔍 2. Check for short text
    if len(text.split()) < 30:
        return jsonify({
            "prediction": "⚠️ Not enough content to classify",
            "confidence": 0
        })

    # 🔍 3. Normal ML prediction
    transformed = vectorizer.transform([text])
    prediction = model.predict(transformed)[0]
    prob = model.predict_proba(transformed).max()

    result = "Fake News ❌" if prediction == 0 else "Real News ✅"

    return jsonify({
        "prediction": result,
        "confidence": round(prob * 100, 2)
    })


# ✅ IMPORTANT FOR RENDER
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)