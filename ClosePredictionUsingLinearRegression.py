from idlelib.iomenu import errors

import pandas as pd
import numpy as np
import scipy
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.metrics import mean_absolute_error,r2_score


data = pd.read_csv(r"C:\Users\SAMAY\Desktop\ml\COALINDIA.csv")
df = pd.read_csv(r"C:\Users\SAMAY\Desktop\ml\COALINDIA.csv")
features = ['Open' , 'High' , 'Low' , 'Volume']
target = ['Close']
X = df[features]
y = df[target]

df['Date'] = pd.to_datetime(df['Date'] , format='%d-%m-%Y')
df.fillna(df[['Open' , 'High' , 'Volume' , 'Low' , 'Close']].mean() , inplace = True)
df = df.apply(pd.to_numeric, errors = 'coerce')
df['Open'] = pd.to_numeric(df['Open'] , errors= 'coerce')
df['Close']=pd.to_numeric(df['Close'] , errors='coerce')
df['High']=pd.to_numeric(df['High'] , errors='coerce')
df['Low']=pd.to_numeric(df['Low'] , errors='coerce')
df['Volume']=pd.to_numeric(df['Volume'] , errors='coerce')


df.fillna(df[['Open' , 'High' , 'Low' , 'Close' , 'Volume']].mean() , inplace = True)




X_train , X_test , y_train , y_test = train_test_split(X , y , test_size= 0.2 ,random_state=42)
'''print("X_train: " , X_train.shape )
print("X_test: " , X_test.shape )
print("y_train: " , y_train.shape )
print("y_test: " , y_test.shape )'''

#print(df.isna().sum())
model = LinearRegression()
model.fit(X_train , y_train)
y_pred = model.predict(X_test)

y_test = y_test.values.flatten()
y_pred = y_pred.flatten()
'''for i in range (10):
    print(f"Actual Close: , {y_test[i]}, Predicted Close: , {y_pred[i]:.2f}")'''


mape = np.mean(np.abs((y_test - y_pred) / (y_pred))) * 100
accuracy = 100-mape

print(accuracy:2f%)