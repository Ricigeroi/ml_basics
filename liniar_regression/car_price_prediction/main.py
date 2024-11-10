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
    df['torque'] = df['torque'].str.split(r'(?i)nm').str[0].str.strip()

    # format name -> make, model, trim (комлпектация)
    df['make'] = df['name'].str.split(' ').str[0]
    df['model'] = df['name'].str.split(' ').str[1]
    df['trim'] = df['name'].str.split(' ').str[-1]
    df = df.drop(columns=['name'])

    # scaling


    return df

dataset = preprocess_dataframe(dataset)

# dividing features from target variable
x = dataset.drop(columns=['selling_price'])
y = dataset['selling_price']

# dividing dataframe to train and test parts
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)




print(x_train.columns)

print(x_test[['make', 'model', 'trim']])

