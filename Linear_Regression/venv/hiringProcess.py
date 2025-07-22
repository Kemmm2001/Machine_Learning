import pandas as pd
import numpy as np
from sklearn import linear_model
import math
import pickle
from joblib import dump,load

df = pd.read_csv('csv/HiringProcess.csv')
print(df.columns)

model = linear_model.LinearRegression()
model.fit(df[['Test score ', 'IQ test score', 'English score', 'Interview score','Years of experiences']], df['Salary($) per month'])
result = model.predict([[5,5,3,5,2]])
print(result)
file_name = 'HiringProcess'
with open (file_name, 'wb') as file:
    pickle.dump(model,file)