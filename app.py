from flask import Flask, request, jsonify
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ✅ HOME ROUTE (MUST HAVE)
@app.route("/")
def home():
    return "Fake News API is running 🚀"

# ✅ PREDICT ROUTE
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")

    transformed = vectorizer.transform([text])
    prediction = model.predict(transformed)[0]

    result = "Fake News ❌" if prediction == 0 else "Real News ✅"

    return jsonify({"prediction": result})

# ✅ IMPORTANT FOR RENDER
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)