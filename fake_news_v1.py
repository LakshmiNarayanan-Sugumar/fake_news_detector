import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# loading both the true and fake csv files
real=pd.read_csv("True.csv")
fake=pd.read_csv("Fake.csv")

# labelling the real datasets as 1 and fake datasets as 0
real["label"]=1
fake["label"]=0

#merging both the dataframes into a single dataframe.
df=pd.concat([real,fake], ignore_index=True)

df=df[["text","label"]]

# cleaning of data, removing special charecters,numericals...
def clean_text(text):
    text=text.lower()
    text=re.sub(r'\[.*?\]', "",text)
    text=re.sub(r'https?://\S+', "",text)
    text=re.sub(r'[^a-zA-Z\s]', "",text)
    text=re.sub(r'\s+', " ",text)
    return text

df["text"]=df["text"].apply(clean_text)

# tf=idf which converts each article to vectors
vectorizer=TfidfVectorizer(max_features=5000, stop_words='english')
x=vectorizer.fit_transform(df['text'])
y=df['label']

#splitting the data for test and validation
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=42)

#logistic regression
model=LogisticRegression()
model.fit(x_train,y_train)
predictions=model.predict(x_test)

print(" Logistic Regression Results ")
print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))

# comparing both logistic regression and random forest

#random forest 
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)
rf_predictions = rf_model.predict(x_test)

print(" Random Forest Results ")
print("Accuracy:", accuracy_score(y_test, rf_predictions))
print(classification_report(y_test, rf_predictions))