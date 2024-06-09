import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv("depression_dataset_reddit_cleaned.csv", sep=",")

x = df["clean_text"]
y = df["is_depression"]

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2)

tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_tfidf, Y_train)

y_predicted = model.predict(X_test_tfidf)

print(classification_report(Y_test, y_predicted))
