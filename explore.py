
# importing librabries
from pydataset import data
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
import env

# Creating function to look at relationships between all features in the dataset
def plot_variable_pairs(df):
   sns.pairplot(df, kind = "reg", plot_kws={'line_kws':{'color':'red'}})

# Creating function to look at relationships between the target variable and all the values
def plot_value_features(df, value, features):
    value = ['taxvaluedollarcnt']
    features = ['bathroomcnt', 'bedroomcnt', 'yearbuilt', 'fips', 'calculatedfinishedsquarefeet']
    #looping through features to create strip plots, box plots and violin plots for each
    for value in value:
        for feature in features:
            _, ax = plt.subplots(1,3,figsize=(16,5))
            p = sns.stripplot(data = df, x=feature, y=value, ax=ax[0], s=1)
            p.axhline(df[value].mean())
            p = sns.boxplot(data = df, x=feature, y = value, ax=ax[1])
            p.axhline(df[value].mean())
            p = sns.violinplot(data = df, x=feature, y=value, hue = feature, ax=ax[2])
            p.axhline(df[value].mean())
            plt.legend('',frameon=False)
            plt.suptitle(f'{value} by {feature}', fontsize = 22)
            plt.show()
