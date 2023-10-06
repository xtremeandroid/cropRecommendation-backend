import numpy as np
import pandas as pd

import xgboost as xgb
import joblib

from sklearn import metrics
from sklearn.metrics import classification_report, mean_absolute_error
from sklearn.model_selection import train_test_split

DATASET_PATH = "./dataset/crop_recommendation.csv"

dataset = pd.read_csv(DATASET_PATH)

features = dataset[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
target = dataset["label"]

Xtrain, Xtest, Ytrain, Ytest = train_test_split(features, target, test_size=0.2, random_state=2)

model = xgb.XGBClassifier()
model.fit(Xtrain, Ytrain)
# predicted_values = model.predict(Xtest)
# xg = metrics.accuracy_score(Ytest, predicted_values)
# print("Accuracy : ", xg)
# print(classification_report(Y_test, predicted_values))

joblib.dump(model, "model.pkl") # Save model
