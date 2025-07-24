import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score



file_name = 'spam_classifier'

# Load vectorizer và model
with open(file_name, 'rb') as file:
    model, vectorizer = pickle.load(file)

new_message = ["You won a $1000 Walmart gift card. Go to http://bit.ly/123456 to claim now."]

# Dự đoán message mới
new_message_vec = vectorizer.transform(new_message)

# 2. Dự đoán label (spam hoặc ham)
prediction = model.predict(new_message_vec)

# 3. In kết quả
print("Label:", prediction[0])
