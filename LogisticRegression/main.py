import pandas as pd
from sklearn.linear_model import LogisticRegression,LinearRegression
df = pd.read_csv('csv/hoursOfStudy.csv')
X = df.drop(['Pass'], axis="columns")
Y = df['Pass']
model = LogisticRegression();
model.fit(X,Y);
result = model.predict(pd.DataFrame([[2]], columns=['hours of study']))
print(result)
