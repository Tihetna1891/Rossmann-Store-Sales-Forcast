import logging
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(filename='eda.log', level=logging.INFO)

logging.info('Loading data...')
train = pd.read_csv('C:/Users/dell/Rossmann-Store-Sales-Forcast/dataset/train.csv')
test = pd.read_csv('C:/Users/dell/Rossmann-Store-Sales-Forcast/dataset/test.csv')
store = pd.read_csv('C:/Users/dell/Rossmann-Store-Sales-Forcast/dataset/store.csv')
logging.info('Data loaded successfully.')

#EDA
st.title('Exploratory Data Analysis on Customer Purchasing Behavior')

# Distribution of promotions
st.header('Distribution of Promotions in Training and Test Sets')
promo_train = train['Promo'].value_counts()
promo_test = test['Promo'].value_counts()

fig, ax = plt.subplots()
sns.barplot(x=promo_train.index, y=promo_train.values, ax=ax)
ax.set_title('Promotion Distribution in Train Set')
st.pyplot(fig)

# Merge train and store data
data = pd.merge(train, store, on='Store')
data['Date'] = pd.to_datetime(data['Date'])
# st.write(data)

# Sales behavior around holidays:
# Sales before, during, and after holidays
st.header('Sales Behavior Around Holidays')

# Filter holidays
holiday_sales = data[data['StateHoliday'] != '0']

# Plot sales around holidays
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(x='Date', y='Sales', data=holiday_sales, ax=ax, hue='StateHoliday')
ax.set_title('Sales Around Holidays (Public, Easter, Christmas)')
st.pyplot(fig)


# Seasonal trends:

st.header('Seasonal Sales Trends')

# Extract month and year from Date for seasonal trends
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year

# Group by month to see seasonal trends
seasonal_sales = data.groupby('Month')['Sales'].mean().reset_index()

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(x='Month', y='Sales', data=seasonal_sales, marker='o')
ax.set_title('Average Monthly Sales')
st.pyplot(fig)


# Correlation between sales and number of customers:
st.header('Correlation Between Sales and Number of Customers')

# Scatter plot
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='Customers', y='Sales', data=data, ax=ax)
ax.set_title('Sales vs Customers')
st.pyplot(fig)

# Calculate correlation
corr_sales_customers = data[['Sales', 'Customers']].corr()
st.write('Correlation between Sales and Customers:', corr_sales_customers.loc['Sales', 'Customers'])

# Effect of promotions on sales:
st.header('Effect of Promotions on Sales')

# Group by Promo to see sales behavior with and without promotions
promo_sales = data.groupby('Promo')['Sales'].mean().reset_index()

# Bar plot
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='Promo', y='Sales', data=promo_sales, ax=ax)
ax.set_title('Average Sales with and without Promo')
st.pyplot(fig)


# Promo effectiveness across stores:
st.header('Promo Effectiveness Across Stores')

# Group sales by Store and Promo
promo_sales_by_store = data.groupby(['Store', 'Promo'])['Sales'].mean().reset_index()

# Plot sales for stores with and without promotions
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(x='Store', y='Sales', hue='Promo', data=promo_sales_by_store)
ax.set_title('Promo Effectiveness Across Stores')
st.pyplot(fig)


# Customer behavior during store open/close times:
st.header('Customer Behavior During Store Open/Close Times')

# Filter for stores that are open and closed
open_sales = data[data['Open'] == 1]
closed_sales = data[data['Open'] == 0]

# Compare sales for open and closed stores
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(open_sales['Sales'], label='Open', color='green', kde=True, ax=ax)
sns.histplot(closed_sales['Sales'], label='Closed', color='red', kde=True, ax=ax)
ax.set_title('Sales Distribution During Store Open/Close Times')
ax.legend()
st.pyplot(fig)

# Weekday vs. Weekend Sales:
st.header('Weekday vs Weekend Sales')

# Extract day of the week
data['DayOfWeek'] = data['Date'].dt.dayofweek
data['IsWeekend'] = data['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

# Group by weekday/weekend
weekend_sales = data.groupby('IsWeekend')['Sales'].mean().reset_index()

# Bar plot
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='IsWeekend', y='Sales', data=weekend_sales, ax=ax)
ax.set_xticklabels(['Weekday', 'Weekend'])
ax.set_title('Sales: Weekdays vs Weekends')
st.pyplot(fig)


# Effect of assortment type:
st.header('Effect of Assortment Type on Sales')

# Group by assortment type
assortment_sales = data.groupby('Assortment')['Sales'].mean().reset_index()

# Bar plot
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='Assortment', y='Sales', data=assortment_sales, ax=ax)
ax.set_title('Average Sales by Assortment Type')
st.pyplot(fig)


# Distance to competitor:
st.header('Effect of Distance to Competitor on Sales')

# Scatter plot
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='CompetitionDistance', y='Sales', data=data, ax=ax)
ax.set_title('Sales vs Distance to Competitor')
st.pyplot(fig)

# Correlation between sales and competitor distance
corr_sales_competition = data[['Sales', 'CompetitionDistance']].corr()
st.write('Correlation between Sales and Competition Distance:', corr_sales_competition.loc['Sales', 'CompetitionDistance'])


# Effect of new competitors:
st.header('Effect of New Competitors on Sales')

# Filter for stores with and without competitors
has_competitor = data[data['CompetitionDistance'].notna()]
no_competitor = data[data['CompetitionDistance'].isna()]

# Compare sales for stores with and without competitors
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(has_competitor['Sales'], label='With Competitor', color='blue', kde=True, ax=ax)
sns.histplot(no_competitor['Sales'], label='Without Competitor', color='orange', kde=True, ax=ax)
ax.set_title('Sales Distribution: With and Without Competitors')
ax.legend()
st.pyplot(fig)


st.header('Sales Summary by Store')
sales_summary = train.groupby('Store')['Sales'].describe()
st.dataframe(sales_summary)

logging.info('Checking distribution of promotions...')
# Add logs when detecting missing data
missing_train = train.isnull().sum()
missing_store = store.isnull().sum()
logging.info(f"Missing data in training set: {missing_train}")
logging.info(f"Missing data in store set: {missing_store}")

# Log the results of outlier detection, correlations, etc.
logging.info('Completed outlier detection for sales and customer data.')
logging.info('Correlation analysis between Sales and Customers completed.')

