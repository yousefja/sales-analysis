# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 16:48:59 2025

Scenario: A client wants a dashboard to understand their product sales over time.

Dataset: Superstore Sales Dataset (Kaggle)

Prompt:
    "Create a dashboard that shows monthly revenue, top 10 selling products, and profit by region. I want to filter by category and sub-category."

Skills:

    Data wrangling (Pandas)
    
    Time series aggregation
    
    Dashboard creation (Tableau, Power BI, or plotly/dash)
    
    Data storytelling

@author: youse
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# import data
df_sales = pd.read_csv('Sample - Superstore.csv', encoding='cp1252')

PATH_TO_MONTHLY_REVENUE_DATA = 'output/monthly_revenue.xlsx'
PATH_TO_CATEGORIES = 'output/categories.xlsx'
PATH_TO_PRODUCTS_DATA = 'output/products.xlsx'
PATH_TO_REGIONAL_DATA = 'output/regional_sales.xlsx'

######
# EDA
######

#In this section, I take a look at what fields are available, what data types are present, and 
# overall what we have to work with. I also make sure data looks accurate and fix any issues found


# which columns/fields are available
df_sales.dtypes
'''
Row ID             int64
Order ID          object
Order Date        object 
Ship Date         object 
Ship Mode         object
Customer ID       object
Customer Name     object
Segment           object
Country           object
City              object
State             object
Postal Code        int64
Region            object
Product ID        object
Category          object
Sub-Category      object
Product Name      object
Sales            float64
Quantity           int64
Discount         float64
Profit           float64
'''

# drop unneeded cols
df_sales.drop(columns=['Row ID'], inplace=True)

# what is the spread of the numerical data
df_summary = df_sales.describe()

TODO
# do a brief statistical analysis of sales and profit
# whats the shape of the bell curve
# ask chatgpt to walk you through this, how to interpret it, what is normalizing?, what is a std dev EXACTLY?

# for categorical variables, how many unique values?
print("Column Name : # Unique Values")
print('-----------------------------')
df_sales_obj = df_sales.select_dtypes(include=['object']).copy()
for col in df_sales_obj.columns:
    print(f'{col}: {df_sales[col].nunique()}')
'''
Column Name : # Unique Values
-----------------------------
Order ID: 5009
Order Date: 1237
Ship Date: 1334
Ship Mode: 4
Customer ID: 793
Customer Name: 793
Segment: 3
Country: 1
City: 531
State: 49
Region: 4
Product ID: 1862
Category: 3
Sub-Category: 17
Product Name: 1850
'''

'''
dates: what is the spread of dates
any missing values?
categorical: how many values are there?
geographic spread of sales

products:
    - types of categories
'''

#################
# CATEGORY FILTER
#################

# a seperate category/sub-category table is created so that we can create relationships 
# between this table, and the corresponding fields in the monthly revenue and products tables

df_categories = df_sales[['Category', 'Sub-Category']].drop_duplicates()
df_categories.to_excel(PATH_TO_CATEGORIES, index=False)


#################
# MONTHLY REVENUE
#################

# In this section, I perform some data manipulation so that we can calculate
# and later visualize how much revenue the company is generating on a monthly basis

# convert date columns to datetime dtype
df_sales[['Order Date', 'Ship Date']] = df_sales[['Order Date', 'Ship Date']].apply(pd.to_datetime)

# what range of date does the sales data cover?
df_date_dist = df_sales[['Order Date', 'Ship Date']].describe()

# create seperate year and month column for yearmo aggregates
df_sales['Order Month'] = df_sales['Order Date'].dt.month
df_sales['Order Year'] = df_sales['Order Date'].dt.year

# monthly sales over time (aggregate on yearmo), broken down by category and sub-category
df_sales_yearmo_agg_filterable = df_sales.groupby(['Order Year', 'Order Month', 'Category', 'Sub-Category'])['Sales'].sum().reset_index()

# convert yearmo to date for plotting
df_sales_yearmo_agg_filterable.rename(columns={'Order Year': 'year', 'Order Month': 'month'}, inplace=True)
df_sales_yearmo_agg_filterable['date'] = pd.to_datetime(df_sales_yearmo_agg_filterable[['year', 'month']].assign(day=1))
df_sales_yearmo_agg_filterable = df_sales_yearmo_agg_filterable.sort_values(by='date')

# monthly sales over time (aggregate on yearmo)
df_sales_yearmo_agg = df_sales.groupby(['Order Year', 'Order Month'])['Sales'].sum().reset_index()

# convert yearmo to date for plotting
df_sales_yearmo_agg.rename(columns={'Order Year': 'year', 'Order Month': 'month'}, inplace=True)
df_sales_yearmo_agg['date'] = pd.to_datetime(df_sales_yearmo_agg[['year', 'month']].assign(day=1))
df_sales_yearmo_agg = df_sales_yearmo_agg.sort_values(by='date')

# set seaborn style for modern look
sns.set_style("whitegrid")

# create the plot
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(df_sales_yearmo_agg['date']
        , df_sales_yearmo_agg['Sales']
        , marker='o'
        , linestyle='-'
        , color='#007acc'
        , linewidth=2)                           

# format x-axis with month and year
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.xticks(rotation=90)

# add labels and title
ax.set_title('Monthly Sales Over Time', fontsize=16, weight='bold')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Sales ($)', fontsize=12)

# Tight layout for clean spacing
plt.tight_layout()

# Show the plot
plt.show()

# export monthly revenue data for visualization
df_sales_yearmo_agg_filterable.to_excel(PATH_TO_MONTHLY_REVENUE_DATA, index=False)

PATH_TO_MONTHLY_REVENUE_DATA = 'output/monthly_revenue.xlsx'
PATH_TO_PRODUCTS_DATA = 'output/products.xlsx'


######################
# TOP SELLING PRODUCTS
######################

# In this section, we aggregate product data to determine sale totals and units sold per product

# get product sales info, broken down by category and sub-category
df_products = df_sales.groupby()
df_products = df_sales.groupby(['Category', 'Sub-Category', 'Product ID', 'Product Name'])['Sales'].agg(['sum', 'count']).reset_index()

# export product sales data for visualization
df_products.to_excel(PATH_TO_PRODUCTS_DATA, index=False)

# see breakdown of WHEN specific product are being sold (should we be advertising selling phones in december?)


##################
# PROFIT BY REGION
##################

df_regional_sales = df_sales.groupby(['Category', 'Sub-Category', 'Country', 'Region', 'State', 'City'])['Sales'].agg(['sum', 'count']).reset_index()

# export regional sales data for visualization
df_regional_sales.to_excel(PATH_TO_REGIONAL_DATA, index=False)



