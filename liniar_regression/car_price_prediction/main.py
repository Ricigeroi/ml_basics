import numpy as np
import pandas as pd
from matplotlib.pyplot import figure
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
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

    # one-hot encoding for fuel type
    fuel_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    fuel_encoded = fuel_encoder.fit_transform(df[['fuel']])
    fuel_encoded_df = pd.DataFrame(fuel_encoded, columns=fuel_encoder.get_feature_names_out(['fuel']))

    # reset indexes -- very important thing! I spent fucking hour fighting with this problem
    df.reset_index(drop=True, inplace=True)
    fuel_encoded_df.reset_index(drop=True, inplace=True)
    df = pd.concat([df, fuel_encoded_df], axis=1).drop(columns=['fuel'])

    # one-hot encoding for transmission
    gearbox_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    gearbox_encoded = gearbox_encoder.fit_transform(df[['transmission']])
    gearbox_encoded_df = pd.DataFrame(gearbox_encoded, columns=gearbox_encoder.get_feature_names_out(['transmission']))
    df.reset_index(drop=True, inplace=True)
    gearbox_encoded_df.reset_index(drop=True, inplace=True)
    df = pd.concat([df, gearbox_encoded_df], axis=1).drop(columns=['transmission'])

    # one-hot encoding for seller type
    seller_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    seller_encoded = seller_encoder.fit_transform(df[['seller_type']])
    seller_encoded_df = pd.DataFrame(seller_encoded, columns=seller_encoder.get_feature_names_out(['seller_type']))
    df.reset_index(drop=True, inplace=True)
    seller_encoded_df.reset_index(drop=True, inplace=True)
    df = pd.concat([df, seller_encoded_df], axis=1).drop(columns=['seller_type'])

    # format max_power
    df['max_power'] = df['max_power'].str.split(' ').str[0]

    # format torque
    # df['torque'] = df['torque'].str.split(r'(?i)nm').str[0].str.strip()

    # format name -> make + model
    df['make'] = df['name'].str.split(' ').str[0]
    df['model'] = df['name'].str.split(' ').str[1]

    # df['trim'] = df['name'].str.split(' ').str[-1]

    # format consumption
    df['consumption'] = df['consumption'].str.split(' ').str[0]

    # format engine
    df['engine'] = df['engine'].str.split(' ').str[0]

    # scaling

    # scale year = year - 2010; example 2014 = 4, 2020 = 10
    df['year'] = df['year'] - 2010

    consumption_scaler = MinMaxScaler()
    df['consumption'] = consumption_scaler.fit_transform(df[['consumption']])

    km_scaler = MinMaxScaler()
    df['km_driven'] = km_scaler.fit_transform(df[['km_driven']])

    engine_scaler = MinMaxScaler()
    df['engine'] = engine_scaler.fit_transform(df[['engine']])

    # one-hot encoding for make
    make_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    make_encoded = make_encoder.fit_transform(df[['make']])
    make_encoded_df = pd.DataFrame(make_encoded, columns=make_encoder.get_feature_names_out(['make']))
    df.reset_index(drop=True, inplace=True)
    make_encoded_df.reset_index(drop=True, inplace=True)
    df = pd.concat([df, make_encoded_df], axis=1).drop(columns=['make'])

    # one-hot encode for model
    model_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    model_encoded = model_encoder.fit_transform(df[['model']])
    model_encoded_df = pd.DataFrame(model_encoded, columns=model_encoder.get_feature_names_out(['model']))
    df.reset_index(drop=True, inplace=True)
    model_encoded_df.reset_index(drop=True, inplace=True)
    # df = pd.concat([df, model_encoded_df], axis=1).drop(columns=['model'])


    return df.drop(columns=['torque', 'name', 'model'])

def scale_price(df):
    df = df / 1000
    return df

dataset = preprocess_dataframe(dataset)

# dividing features from target variable
x = dataset.drop(columns=['selling_price'])
y = dataset['selling_price']

# dividing dataframe to train and test parts
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# divide price by 1000
y_train, y_test = scale_price(y_train), scale_price(y_test)

print(x_train.columns)

# model building
model = LinearRegression()
model.fit(x_train, y_train)

# noinspection DuplicatedCode
y_pred = model.predict(x_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error (MAE): {mae:.2f}k')
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'R-squared (R2): {r2:.4f}')

