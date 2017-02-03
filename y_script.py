from sklearn import preprocessing, linear_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


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

sumu = 0
su = 0
for i in range(0, len(y_pred)):
    sumu += abs(y_test.iloc[i]-y_pred[i])
    su += y_test.iloc[i]
sumu /= len(y_test)
su /= len(y_test)

print sumu
print su
