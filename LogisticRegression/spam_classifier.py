import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
import pickle
df = pd.read_csv('csv/email.csv')
X = df['Message']
y = df['Category'].map({'ham': 0, 'spam': 1})
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


vectorizer = TfidfVectorizer()
model = LogisticRegression(max_iter=1000);
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

file_name = 'spam_classifier'
with open('spam_classifier', 'wb') as file:
    pickle.dump((model, vectorizer), file)