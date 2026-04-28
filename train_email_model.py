import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ✅ Simple custom dataset (you can expand later)
data = {
    "text": [
        "Dear user your account has been hacked click here to reset password",
        "Congratulations you have won a lottery claim now",
        "Urgent your bank account will be blocked verify immediately",
        "Meeting scheduled for tomorrow at 10am",
        "Project submission deadline is extended",
        "Greetings from the organizing committee your paper is accepted",
        "Your salary has been credited successfully",
        "Click this link to win free iphone now",
    ],
    "label": [0,0,0,1,1,1,1,0]  # 0 = Fake/Scam, 1 = Real
}

df = pd.DataFrame(data)

# Vectorize
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["text"])

# Train model
model = LogisticRegression()
model.fit(X, df["label"])

# Save
pickle.dump(model, open("email_model.pkl", "wb"))
pickle.dump(vectorizer, open("email_vectorizer.pkl", "wb"))

print("✅ Email model trained!")