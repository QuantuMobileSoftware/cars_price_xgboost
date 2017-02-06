from sklearn import linear_model
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

input_file = "autos.csv"
df = pd.read_csv(input_file, header=0).dropna()
df = df._get_numeric_data()

x_train = df["yearOfRegistration"][:-20]
x_test = df["yearOfRegistration"][-20:]

y_train = df['price'][:-20]
y_test = df['price'][-20:]

reg = linear_model.LinearRegression()
reg.fit(np.transpose(np.matrix(x_train)), np.transpose(np.matrix(y_train)))

y_pred = reg.predict(np.transpose(np.matrix(x_test)))

print r2_score(y_test, y_pred)
