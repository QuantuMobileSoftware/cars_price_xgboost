**Prediction the sale price of the car using XGBoost**

This repo contains script for prediction sale price of the car.
It includes next steps:

Preprocessing:
*renaming column;
*processing missing values (full drop of object if any value not exists);
*feature reduction;
*convert text data in appropriate format;
*removing outliers;
*splitting our dataset in train and test data;

Running XGBoost:
*running XGBoost in 2 loops, which change parameters "gamma" and "max_depth";
*printing score of model.

Best result for model - 71% on max_depth=2 and gamma=0.
In comparing with linear regression, which gives us 32% accuracy, this result is much better).

All necessary information about packages consists in file "requirements.txt".