
import pickle

file_name = 'HiringProcess'
with open(file_name, 'rb') as file:
     model2 = pickle.load(file)

result = model2.predict([[5,5,3,5,2]])
print(result)
