import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

# loading both the true and fake csv files
real = pd.read_csv("True.csv")
fake = pd.read_csv("Fake.csv")

# labelling
real["label"] = 1
fake["label"] = 0

# merging
df = pd.concat([real, fake], ignore_index=True)
df = df[["text", "label"]]

# cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', "", text)
    text = re.sub(r'https?://\S+', "", text)
    text = re.sub(r'[^a-zA-Z\s]', "", text)
    text = re.sub(r'\s+', " ", text)
    return text

df["text"] = df["text"].apply(clean_text)

# vectorizing
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
x = vectorizer.fit_transform(df['text'])
y = df['label']

# splitting
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# train logistic regression
lr_model = LogisticRegression()
lr_model.fit(x_train, y_train)

# train random forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)

# save models and vectorizer
joblib.dump(lr_model, "lr_model.pkl")
joblib.dump(rf_model, "rf_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Models saved successfully!")