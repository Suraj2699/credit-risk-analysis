# -*- coding: utf-8 -*-
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder
from scipy.stats import boxcox
import numpy as np


def get_data():
    # fetch dataset
    statlog_german_credit_data = fetch_ucirepo(id=144)
    
    # data (as pandas dataframes)
    X = statlog_german_credit_data.data.features
    y = statlog_german_credit_data.data.targets

    ## Concatenate features and target DataFrame for better exploratory analysis.

    df = pd.concat([X, y], axis=1)
    df.head()
    
    # metadata
    return df

def handling_categories(df):
    ## Let's rename the attributes to get a better knowledge about the data.
    df.rename(columns={'Attribute1': 'Existing-Checking-Account-Status', 'Attribute2': 'Duration(Months)', 'Attribute3': 'Credit-History',
                      'Attribute4': 'Purpose', 'Attribute5': 'Credit-Amount', 'Attribute6': 'Savings-Account(Bonds)',
                      'Attribute7': 'Present-Employment-Since', 'Attribute8': 'Installment-Rate(%)', 'Attribute9': 'Personal-Status-Sex',
                      'Attribute10': 'Other-Debtors', 'Attribute11': 'Present-Residence-Since', 'Attribute12': 'Property', 'Attribute13': 'Age',
                      'Attribute14': 'Other-Installment-Plans', 'Attribute15': 'Housing', 'Attribute16': 'Bank-Existing-Credits',
                      'Attribute17': 'Job', 'Attribute18': 'Total-Reliable-People', 'Attribute19': 'Telephone', 'Attribute20': 'Foreign-Worker'}, inplace=True)
    
    categorical_columns = ['Existing-Checking-Account-Status', 'Credit-History', 'Purpose', 'Savings-Account(Bonds)', 'Present-Employment-Since',
                           'Personal-Status-Sex', 'Other-Debtors', 'Property', 'Other-Installment-Plans', 'Housing', 'Job', 'Telephone', 'Foreign-Worker']
    
    ## LabelEncoder to transform categorical data to integers
    encoders = {}

    for col in categorical_columns:
        encoders[col] = LabelEncoder()
        df[col] = encoders[col].fit_transform(df[col])
    
    # For class, 1 is Good, 2 is Bad. Let's change 1->0 & 2->1 since our primary goal is to
    ## catch bad applicants.

    df['class'].replace({1: 0, 2:1}, inplace=True)
    
    return df

def handling_continuous_columns(df):
    
    # Apply Box-Cox transformation to reduce skewness
    df['Credit-Amount'], lambda1 = boxcox(df['Credit-Amount'] + 1)
    df['Duration(Months)'], lambda2 = boxcox(df['Duration(Months)'] + 1)
    df['Age'], lambda3 = boxcox(df['Age'] + 1)
    df['Debt_to_Income_Ratio'], lambda4 = boxcox(df['Debt_to_Income_Ratio'] + 1)
    df['Credit_Utilization'], lambda5 = boxcox(df['Credit_Utilization'] + 1)
    
    return df, lambda1, lambda2, lambda3, lambda4, lambda5