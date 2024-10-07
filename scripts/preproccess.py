# Step 2.1: Preprocessing
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging

logging.basicConfig(filename='eda.log', level=logging.INFO)

logging.info('Loading data...')
train = pd.read_csv('C:/Users/dell/Rossmann-Store-Sales-Forcast/dataset/train.csv')
test = pd.read_csv('C:/Users/dell/Rossmann-Store-Sales-Forcast/dataset/test.csv')
store = pd.read_csv('C:/Users/dell/Rossmann-Store-Sales-Forcast/dataset/store.csv')
logging.info('Data loaded successfully.')

# Assuming your data is in a Pandas DataFrame called `train`
# Merging store info if not done already
data = pd.merge(train, store, on='Store')
data['Date'] = pd.to_datetime(data['Date'])

# Fill missing values for CompetitionDistance
data['CompetitionDistance'] = data['CompetitionDistance'].fillna(data['CompetitionDistance'].median())

# Fill missing values for CompetitionOpenSinceMonth and CompetitionOpenSinceYear
data['CompetitionOpenSinceMonth'] = data['CompetitionOpenSinceMonth'].fillna(0)
data['CompetitionOpenSinceYear'] = data['CompetitionOpenSinceYear'].fillna(0)

# Fill missing values for Promo2SinceWeek and Promo2SinceYear
data['Promo2SinceWeek'] = data['Promo2SinceWeek'].fillna(0)
data['Promo2SinceYear'] = data['Promo2SinceYear'].fillna(0)

# Fill missing values for PromoInterval
data['PromoInterval'] = data['PromoInterval'].fillna('None')

# Convert categorical variables to numeric
label_encoder = LabelEncoder()

# StateHoliday
data['StateHoliday'] = label_encoder.fit_transform(data['StateHoliday'])

# StoreType, Assortment, PromoInterval
data['StoreType'] = label_encoder.fit_transform(data['StoreType'])
data['Assortment'] = label_encoder.fit_transform(data['Assortment'])
data['PromoInterval'] = label_encoder.fit_transform(data['PromoInterval'])

# Extract Weekday and Weekend
data['DayOfWeek'] = data['Date'].dt.dayofweek
data['IsWeekend'] = data['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

# Extract beginning, mid, and end of the month
data['Day'] = data['Date'].dt.day
data['MonthPeriod'] = data['Day'].apply(lambda x: 'Beginning' if x <= 10 else ('Mid' if x <= 20 else 'End'))

# Extract Month and Year
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year

# Days to nearest holiday (assuming holidays are marked in the StateHoliday column)
data['DaysToHoliday'] = data.groupby('Store')['StateHoliday'].apply(lambda x: x.shift(-1).fillna(0))

# Days after last holiday
data['DaysAfterHoliday'] = data.groupby('Store')['StateHoliday'].apply(lambda x: x.shift(1).fillna(0))

# New Feature: Whether it's the start of a promo or not
data['IsPromoStart'] = data['Promo2SinceWeek'] == data['Date'].dt.isocalendar().week

# New Feature: Time since competition opened
data['CompetitionOpenSince'] = (data['Year'] - data['CompetitionOpenSinceYear']) * 12 + (data['Month'] - data['CompetitionOpenSinceMonth'])

# Fill missing competition time since with 0
data['CompetitionOpenSince'] = data['CompetitionOpenSince'].fillna(0)

# Select numerical columns for scaling
numeric_columns = ['Sales', 'Customers', 'CompetitionDistance', 'Promo2SinceWeek', 'Promo2SinceYear', 
                   'DaysToHoliday', 'DaysAfterHoliday', 'CompetitionOpenSince']

# Instantiate StandardScaler
scaler = StandardScaler()

# Scale numerical features
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

# Select features and target variable
X = data.drop(columns=['Sales', 'Date', 'Id'])  # Features
y = data['Sales']  # Target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


