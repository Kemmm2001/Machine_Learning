import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Đọc dữ liệu
df = pd.read_csv('csv/Score.csv')
X = df[['Hours']]
y = df['Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Dự đoán toàn bộ X_test
y_pred = model.predict(X_test)

print("Dự đoán điểm theo giờ:", y_pred)
print("MSE:", mean_squared_error(y_test, y_pred))
