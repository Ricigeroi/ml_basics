import numpy as np
import pandas as pd
from matplotlib.pyplot import figure
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# loading data from .csv into pandas dataframe
dataset = pd.read_csv('datasets/car details v3.csv')

# looking for empty entries
print(dataset.isnull().sum())

# dividing features from target variable
x = dataset.drop(columns=['selling_price'])
y = dataset['selling_price']

# dividing dataframe to train and test parts
x_train, y_train, x_test, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
