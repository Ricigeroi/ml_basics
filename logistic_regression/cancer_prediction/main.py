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
dataset = pd.read_csv('datasets/Breast_Cancer.csv')

# looking for empty entries
# print(dataset.isnull().sum())

def preprocess_dataframe(df):

    # one-hot race encoder
    race_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    race_encoded = race_encoder.fit_transform(df[['Race']])
    race_encoded_df = pd.DataFrame(race_encoded, columns=race_encoder.get_feature_names_out(['Race']))

    # one-hot marital status encoder
    marital_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    marital_encoded = marital_encoder.fit_transform(df[['Marital Status']])
    marital_encoded_df = pd.DataFrame(
        marital_encoded, columns=marital_encoder.get_feature_names_out(['Marital Status'])
    )

    # ordinal encoding t stage
    t_stage_encoder = OrdinalEncoder(categories=[['T1', 'T2', 'T3', 'T4']])
    df['T Stage '] = t_stage_encoder.fit_transform(df[['T Stage ']].values)

    # ordinal encoding n stage
    n_stage_encoder = OrdinalEncoder(categories=[['N1', 'N2', 'N3']])
    df['N Stage'] = n_stage_encoder.fit_transform(df[['N Stage']].values)

    # ordinal encoding 6th stage
    sixth_stage_encoder = OrdinalEncoder(categories=[['IIA', 'IIB', 'IIIA', 'IIIB', 'IIIC']])
    df['6th Stage'] = sixth_stage_encoder.fit_transform(df[['6th Stage']].values)

    # ordinal differentiate encoding
    differentiate_encoder = OrdinalEncoder(categories=[
        ['Undifferentiated', 'Poorly differentiated', 'Moderately differentiated', 'Well differentiated']
    ])
    df['differentiate'] = differentiate_encoder.fit_transform(df[['differentiate']].values)

    # one-hot encoding for A stage
    a_stage_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    a_stage_encoded = a_stage_encoder.fit_transform(df[['A Stage']])
    a_stage_encoded_df = pd.DataFrame(a_stage_encoded, columns=a_stage_encoder.get_feature_names_out(['A Stage']))

    # one-hot encoding for Estrogen Status
    estrogen_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    estrogen_encoded = estrogen_encoder.fit_transform(df[['Estrogen Status']])
    estrogen_encoded_df = pd.DataFrame(
        estrogen_encoded, columns=estrogen_encoder.get_feature_names_out(['Estrogen Status']))


    # one-hot encoding for Progesterone Status
    progesterone_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    progesterone_encoded = progesterone_encoder.fit_transform(df[['Progesterone Status']])
    progesterone_encoded_df = pd.DataFrame(
        progesterone_encoded, columns=progesterone_encoder.get_feature_names_out(['Progesterone Status'])
    )

    # combining one-hot encoded data to df
    df.reset_index(drop=True, inplace=True)
    race_encoded_df.reset_index(drop=True, inplace=True)
    marital_encoded_df.reset_index(drop=True, inplace=True)
    a_stage_encoded_df.reset_index(drop=True, inplace=True)
    estrogen_encoded_df.reset_index(drop=True, inplace=True)
    progesterone_encoded_df.reset_index(drop=True, inplace=True)
    df = pd.concat(
    [
            df, race_encoded_df,
            marital_encoded_df,
            a_stage_encoded_df,
            estrogen_encoded_df,
            progesterone_encoded_df
        ],
        axis=1
    )
    return df


dataset = preprocess_dataframe(dataset)
print(dataset.columns)
print(dataset[['Estrogen Status', 'Estrogen Status_Positive', 'Estrogen Status_Negative']].head(20))