import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Đọc dữ liệu
df = pd.read_csv('csv/HousePrice.csv')
X = df[['area']]
y = df['price']
plt.scatter(X, y, color='blue', label='Dữ liệu thực tế')
plt.xlabel('Diện tích (area)')
plt.ylabel('Giá nhà (price)')
plt.title('Biểu đồ giá nhà theo diện tích')
plt.legend()
plt.grid(True)

# plt.show()
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict([[3900]])
print("Dự đoán giá nhà cho diện tích 3900:", y_pred)
print("Hệ số (slope):", model.coef_[0])
print("Độ lệch (intercept):", model.intercept_)
