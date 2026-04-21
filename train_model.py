import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# STEP 1: Load dataset
data = pd.read_csv("Fake_Dataset.csv", encoding='latin-1', low_memory=False)

# STEP 2: Use only needed columns
data = data[['title']]   # take only title column

data = data.head(2000)  # take small data

# Create fake labels (split into 2 groups)
data['label'] = [0 if i < 1000 else 1 for i in range(len(data))]

# STEP 4: Remove missing values
data['title'] = data['title'].fillna('')

X = data['title']
Y = data['label']

# STEP 5: Split data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2
)

# STEP 6: Convert text to numbers
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)

# STEP 7: Train model
model = LogisticRegression()
model.fit(X_train, Y_train)

# STEP 8: Save files
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ DONE! model.pkl created")