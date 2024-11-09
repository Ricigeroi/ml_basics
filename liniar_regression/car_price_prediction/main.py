import numpy as np
import pandas as pd
from matplotlib.pyplot import figure
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


# loading data from .csv into pandas dataframe
dataset = pd.read_csv('datasets/car details v3.csv')


# looking for empty entries
# print(train_df.isnull().sum())

# remove uncompleted rows (about 230 from total of 7900)
dataset = dataset.dropna()


def preprocess_dataframe(df):
    df = df.rename(columns={'mileage': 'consumption'})

    # ordinal_encoding owner info
    owner_encoder = OrdinalEncoder(categories=[
        ['Test Drive Car', 'First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner']
    ])
    df['owner'] = owner_encoder.fit_transform(df[['owner']].values)

    return df

dataset = preprocess_dataframe(dataset)

# dividing features from target variable
x = dataset.drop(columns=['selling_price'])
y = dataset['selling_price']

# dividing dataframe to train and test parts
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)




print(x_train.columns)
print(x_train[['year', 'km_driven', 'consumption', 'engine', 'owner']].head(20))
