import missingno as msno
import matplotlib.pyplot as plt
import pandas as pd

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

msno.matrix(train)
plt.show()