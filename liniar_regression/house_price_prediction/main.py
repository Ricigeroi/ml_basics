import numpy as np
import pandas as pd
from matplotlib.pyplot import figure
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# loading data from .csv into pandas dataframe
train_df = pd.read_csv('datasets/df_train.csv')
test_df = pd.read_csv('datasets/df_test.csv')

"""
    We have 14 columns in the dataset, as it follows:
        date: Date of the home sale
        price: Price of each home sold
        bedrooms: Number of bedrooms
        bathrooms: Number of bathrooms
        living_in_m2: Square meters of the apartments interior living space
        nice_view: A flag that indicates the view's quality of a property
        perfect_condition: A flag that indicates the maximum index of the apartment condition
        grade: An index from 1 to 5, where 1 falls short of quality level and 5 have a high quality level of construction and design
        has_basement: A flag indicating whether or not a property has a basement
        renovated: A flag if the property was renovated
        has_lavatory: Check for the presence of these incomplete/secondary bathrooms (bathtub, sink, toilet)
        single_floor: A flag indicating whether the property had only one floor
        month: The month of the home sale
        quartile_zone: A quartile distribution index of the most expensive zip codes, where 1 means less expansive and 4 most expansive.
"""

# looking for empty entries
# print(train_df.isnull().sum())

def preprocess_dataframe(df):
    # simplify date column to just year
    df['date'] = df['date'].str[:4]
    df['date'] = df['date'].astype('int')

    # change boolean dtype to int
    df = df.astype({col: 'int' for col in df.select_dtypes('bool').columns})

    # scaling

    # year = year - 2010, so instead of 2014 we get 4
    df['date'] = df['date'] - 2010

    # using MinMax scaling from sklearn to scale living area to [0, 1] format
    min_max_scaler = MinMaxScaler()
    df['living_in_m2'] = min_max_scaler.fit_transform(df[['living_in_m2']])

    return df

train_df = preprocess_dataframe(train_df)

# visualization of features

fig, axs = plt.subplots(2, 2)

train_df.plot(kind='scatter', s=0.2, x='living_in_m2', y='price', ax=axs[0, 0], title='living_in_m2 vs price')
train_df.plot(kind='scatter', s=0.2, x='bedrooms', y='price', ax=axs[0, 1], title='bedrooms vs price')
train_df.plot(kind='scatter', s=0.2, x='grade', y='price', ax=axs[1, 0], title='grade vs price')
train_df.plot(kind='scatter', s=0.2, x='quartile_zone', y='price', ax=axs[1, 1], title='quartile_zone vs price')
plt.tight_layout()
plt.show()

# building model

model = LinearRegression()

# all columns except price
x = train_df.drop(columns=['price'])
# price column
y = train_df['price']

model.fit(x, y)


# Testing model
test_df = preprocess_dataframe(test_df)
x_test = test_df.drop(columns=['price'])
y_test = test_df['price']

y_pred = model.predict(x_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R-squared (R2):", r2)

