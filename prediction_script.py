import pandas as pd
from sklearn import preprocessing
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

dataset = pd.read_csv("autos.csv", header=0, sep=',')
numeric_columns = [4, 7, 9, 11, 12]
string_columns = [2, 3, 5, 8, 15]

#
# preprocessing
#

# rename columns
num = len(dataset.columns)
lst = []
for i in range(0, num):
    lst.append(i)

dataset.columns = lst
#

# process missing values
dataset = dataset.dropna()
#

# remove unused data
dataset = dataset.drop([0, 1, 10, 14, 16, 17, 18, 19], 1)
#

# process text data
for column in string_columns:
    le = preprocessing.LabelEncoder()
    le.fit(dataset[column])
    dataset[column] = le.transform(dataset[column])
#

# preprocessing of categorical data
dataset = pd.get_dummies(dataset)

#
# split our dataset to 1. data and target 2. trainee and test
#

dataset = pd.DataFrame(dataset[:][:4000])

# kill the outliers
mean = dataset[4].mean()
std = dataset[4].std()

inliers = (abs((dataset[4] - mean)) <= 3*std) & (dataset[4] != 0)
dataset = dataset[inliers]
#

target = dataset[4].copy()
data = dataset.drop([4], 1).copy()

train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)

#
# run xgboost
#

for x in range(3, 4):
    for gam in range(0, 5, 2):
        gbm = xgb.XGBClassifier(max_depth=x, n_estimators=300, learning_rate=0.05, gamma=float(gam) / 10.0).fit(
            train_data, train_target)

        preds = gbm.predict(test_data)

        print "for depth %d on 4 000 examples accuracy is %f \n" % (x, r2_score(test_target, preds))
