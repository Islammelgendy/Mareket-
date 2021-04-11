import pandas as pd
import pickle
model = pickle.load(open('marketModel.pkl', 'rb'))
data = pd.read_csv('test.csv')

print(model.predict(data))
