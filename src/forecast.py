# -*- coding: utf-8 -*-
"""
File:        forecast.py
Description: Uses Facebook Prophet to model overall sales data for time series forecasting.
Author:      Yuseof
Created:     2025-08-01
Modified:    2025-08-06
Usage:       --

 !! IMPORTANT: run in cctv_env, base has issues with prophet !!
"""

import pandas as pd
from prophet import Prophet
from datetime import datetime
import matplotlib.pyplot as plt

PATH_TO_ = 'output/monthly_revenue.xlsx'
FORECAST_LEN = 52 #52 for weekly, 12 for monthly
AGGREGATE_LEVEL = 'W' #'W' - weekly, 'M' - monthly
SMOOTHED = True # for removing volatility in weekly sales data
CAP_OUTLIERS = True # handle outliers, generally just to increase model performance for demo purposes

# import data
df_sales = pd.read_csv('Sample - Superstore.csv', encoding='cp1252')

###############
# PREPROCESSING
###############

# basically the dates are from 2014-2018, and I want them to coincide with present day for 
# demo purposes, so simply offset the dates by the necessary amount of time

# convert date columns to datetime dtype
df_sales[['Order Date', 'Ship Date']] = df_sales[['Order Date', 'Ship Date']].apply(pd.to_datetime)

# get number of days between most recent sale and today
date_today = datetime.today()
days_until_today = date_today - df_sales['Order Date'].max()
days_until_today = days_until_today.days

# add this day offset to all dates in the df
df_sales[['Order Date', 'Ship Date']] = df_sales[['Order Date', 'Ship Date']] + pd.Timedelta(days=days_until_today)

# aggregate sales by week for forecasting
df_sales_agg = df_sales.groupby(pd.Grouper(key='Order Date', freq=AGGREGATE_LEVEL))['Sales'].sum().reset_index()

if CAP_OUTLIERS:
    q_hi = df_sales_agg['Sales'].quantile(0.90)
    df_sales_agg['Sales'] = df_sales_agg['Sales'].clip(upper=q_hi)

# determine whether to use smoothed sales data or volatile
if SMOOTHED:
    # smooth out highly volatile sales data
    df_sales_agg['Sales_Smoothed'] = df_sales_agg['Sales'].rolling(window=3, center=True, min_periods=1).mean()
    sales_col = 'Sales_Smoothed'
else:
    sales_col = 'Sales'

# rename columns for prophet expected input
df_sales_agg.rename(columns={'Order Date': 'ds', sales_col: 'y'}, inplace=True)

##########
# FORECAST
##########

# initialize and fit prophet model
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    seasonality_mode='additive',
    changepoint_prior_scale=0.5
    )
model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
#model.add_seasonality(name='custom_week', period=7, fourier_order=3)
model.fit(df_sales_agg)

# forecast the next <FORECAST_LEN> number of days
df_future = model.make_future_dataframe(periods=FORECAST_LEN, freq=AGGREGATE_LEVEL)
df_forecast = model.predict(df_future)

# join historical sales to forecast to get y (actual sales)
df_forecast = df_forecast.merge(df_sales_agg, how='left', on='ds')

# save output
df_forecast[['ds', 'y','yhat', 'yhat_lower', 'yhat_upper', 'trend']].to_csv('forecast_output.csv', index=False)

model.plot(df_forecast)
plt.show()