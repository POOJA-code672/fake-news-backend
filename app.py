from flask import Flask, request, jsonify
import pickle
from flask_cors import CORS
import re

app = Flask(__name__)
CORS(app)

# ==============================
# LOAD MODELS
# ==============================
news_model = pickle.load(open("model.pkl", "rb"))
news_vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Email model (NEW)
email_model = pickle.load(open("email_model.pkl", "rb"))
email_vectorizer = pickle.load(open("email_vectorizer.pkl", "rb"))

# ==============================
# HOME ROUTE
# ==============================
@app.route("/")
def home():
    return "🚀 AI Detection API Running (News + Email)"


# ==============================
# HELPER: Detect Email
# ==============================
def is_email(text):
    email_patterns = [
        r"dear", r"regards", r"sincerely", r"thank you",
        r"congratulations", r"click here", r"verify",
        r"account", r"bank", r"lottery"
    ]
    
    text = text.lower()
    matches = sum(1 for p in email_patterns if re.search(p, text))
    
    return matches >= 2   # threshold


# ==============================
# MAIN PREDICTION ROUTE
# ==============================
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "").strip()

    # ❌ Empty input
    if not text:
        return jsonify({
            "prediction": "❌ No input provided",
            "confidence": 0
        })

    # ⚠️ Too short
    if len(text.split()) < 5:
        return jsonify({
            "prediction": "⚠️ Text too short to analyze",
            "confidence": 0
        })

    # ==========================
    # 📧 EMAIL DETECTION
    # ==========================
    if is_email(text):
        vector = email_vectorizer.transform([text])
        pred = email_model.predict(vector)[0]
        prob = email_model.predict_proba(vector).max()

        result = "❌ Fake / Scam Email" if pred == 0 else "✅ Legitimate Email"

        return jsonify({
            "type": "email",
            "prediction": result,
            "confidence": round(prob * 100, 2)
        })

    # ==========================
    # 📰 NEWS DETECTION
    # ==========================
    if len(text.split()) < 30:
        return jsonify({
            "prediction": "⚠️ Not enough content for news classification",
            "confidence": 0
        })

    vector = news_vectorizer.transform([text])
    pred = news_model.predict(vector)[0]
    prob = news_model.predict_proba(vector).max()

    result = "Fake News ❌" if pred == 0 else "Real News ✅"

    return jsonify({
        "type": "news",
        "prediction": result,
        "confidence": round(prob * 100, 2)
    })


# ==============================
# RUN SERVER
# ==============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)