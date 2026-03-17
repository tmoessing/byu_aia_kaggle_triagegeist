import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

data = pd.read_csv("data/train.csv")

X = data.drop(columns="triage_acuity")
y = data[["triage_acuity"]]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.20)
