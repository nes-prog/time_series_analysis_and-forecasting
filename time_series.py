# import libraries
import json
from google.cloud import bigquery, storage
import seaborn as sns
import pandas as pd
from matplotlib import dates
import numpy as np
import os, colorama
from colorama import Fore,Style,Back
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime, date 
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime, date 
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math


def load_table_bq(project_id, database_id, table_name):
    """
    connect to biquery and load a specific table
    """
    # connect to dbt_nesrine in bigquery
    client = bigquery.Client()
    dataset_ref = bigquery.DatasetReference(project_id, database_id)
    # load table 
    table_ref = dataset_ref.table(table_name)
    table = client.get_table(table_ref)
    result = client.list_rows(table).to_dataframe()
    
    return result


def visualize_plt(df, x, y, title="", xlabel='', ylabel='', dpi=100, color = ''):
    """
    visualization with matplotlib
    """
    plt.figure(figsize=(15,4), dpi=dpi)
    plt.plot(x, y, color='tab:{}'.format(color))
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()

    
def visualize_sns_plt(df, x, y):
    """
    visualization with seaborn
    """
    plt.rcParams["figure.figsize"] = [10, 4]
    plt.rcParams["figure.autolayout"] = True
    ax = sns.lineplot(data=df, y=y , x=x)
    ax.tick_params(rotation=45)
    plt.show()

    
def decompose(df, column_name):
    """
    A function that returns the trend, seasonality and residual captured by applying both multiplicative and
    additive model.
    df -> DataFrame
    column_name -> column_name for which trend, seasonality is to be captured
    """
    result_mul = seasonal_decompose(df[column_name], model='multiplicative', extrapolate_trend =                                           'freq', period = 30)
    result_add = seasonal_decompose(df[column_name], model = 'additive', extrapolate_trend='freq',                                         period = 30)

    plt.rcParams.update({'figure.figsize': (20, 10)})
    result_mul.plot().suptitle('Multiplicative Decompose', fontsize=30)
    result_add.plot().suptitle('Additive Decompose', fontsize=30)
    plt.show()
    
    return result_mul, result_add  


def visualize_adfuller_results(series, title, ax):
    result = adfuller(series)
    significance_level = 0.05
    adf_stat = result[0]
    p_val = result[1]
    crit_val_1 = result[4]['1%']
    crit_val_5 = result[4]['5%']
    crit_val_10 = result[4]['10%']

    if (p_val < significance_level) & ((adf_stat < crit_val_1)):
        linecolor = 'forestgreen' 
    elif (p_val < significance_level) & (adf_stat < crit_val_5):
        linecolor = 'orange'
    elif (p_val < significance_level) & (adf_stat < crit_val_10):
        linecolor = 'red'
    else:
        linecolor = 'purple'
    sns.lineplot(x=daily_weather_pollutants['con_timestamp'], y=series, ax=ax, color=linecolor)
    ax.set_title(f'ADF Statistic {adf_stat:0.3f},p-value: {p_val:0.3f}\nCritical Values 1%: {crit_val_1:0.3f}, 5%: {crit_val_5:0.3f}, 10%: {crit_val_10:0.3f}', fontsize=14)
    ax.set_ylabel(ylabel=title, fontsize=14)
    
    
def split_data(date, x_transformed, df):
    """
    """
    N_SPLITS = 3
    folds = TimeSeriesSplit(n_splits=N_SPLITS)
    train_size = int(0.85 * len(df))
    test_size = len(df) - train_size
    univariate_df = df[[date, x_transformed]].copy()
    univariate_df.columns = ['ds', 'y']
    train = univariate_df.iloc[:train_size, :]
    x_train, y_train = pd.DataFrame(univariate_df.iloc[:train_size, 0]), pd.DataFrame(univariate_df.iloc[:train_size, 1])
    x_test, y_test = pd.DataFrame(univariate_df.iloc[train_size:, 0]), pd.DataFrame(univariate_df.iloc[train_size:, 1])
    return x_train, y_train, x_test, y_test


def train (y_train, p, d, q):
    """
    """
    # Fit model
    model = ARIMA(y_train, order=(p,d,q))
    model_fit = model.fit()
    print(model_fit.summary())
    return model_fit 


def predict(model_fit, y_test):
    # Prediction with ARIMA
    y_pred = model_fit.forecast(274)
    # Calcuate metrics
    score_mae = mean_absolute_error(y_test, y_pred)
    score_rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    print(Fore.GREEN + 'RMSE: {}'.format(score_rmse))
    return y_pred


def plot_predictions(prediction, cf):
    """
    """
    prediction_series = pd.Series(prediction,index=test.index)
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    ax.plot(data.NO2_avg_ugm3)
    ax.plot(prediction_series)
    ax.fill_between(prediction_series.index,
                    cf[0],
                    cf[1],color='grey',alpha=.3)
    
