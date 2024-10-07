# Step 2.2: Building Models with Sklearn Pipelines
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import joblib
import datetime

# Define the pipeline
pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),  # First step: Scale the data
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))  # Second step: Random Forest
])

# Split the data into training and testing sets (assuming `X` and `y` are defined from preprocessing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Predict on the test data
y_pred = pipeline.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Step 2.3: Choose a Loss Function
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Evaluate model with MAE
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error (MAE): {mae}')

# Evaluate model with MSE (for comparison)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error (MSE): {mse}')


# Step 2.4: Post Prediction Analysis
# Extracting feature importances from the Random Forest
importances = pipeline.named_steps['model'].feature_importances_
features = X.columns

# Display feature importance
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print(feature_importance)

# Confidence Interval of Predictions

# Function to generate bootstrapped predictions for confidence intervals
def bootstrap_predictions(model, X_test, n_iterations=1000):
    preds = []
    for i in range(n_iterations):
        # Resample data with replacement
        X_resampled = X_test.sample(frac=1, replace=True)
        preds.append(model.predict(X_resampled))
    preds = np.array(preds)
    lower_bound = np.percentile(preds, 2.5, axis=0)
    upper_bound = np.percentile(preds, 97.5, axis=0)
    return lower_bound, upper_bound

# Generate confidence intervals
lower, upper = bootstrap_predictions(pipeline.named_steps['model'], X_test)

# Display confidence intervals alongside predictions
prediction_df = pd.DataFrame({
    'Predicted': y_pred,
    'Lower Bound': lower,
    'Upper Bound': upper
})
print(prediction_df)

# Step 2.5: Serialize Models
# Get the current timestamp
timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

# Serialize the model with the timestamp
filename = f'store_sales_model_{timestamp}.pkl'
joblib.dump(pipeline, filename)

print(f"Model saved as {filename}")

# 
# Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error

# Load the dataset
# Assuming 'train.csv' is the dataset containing sales information
data = pd.read_csv('train.csv', parse_dates=['Date'], index_col='Date')
store_sales = data[['Sales']].resample('D').sum()  # Resample daily data

# 1. Check for missing values
store_sales = store_sales.fillna(store_sales.mean())

# 2. Check if Time Series is Stationary
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    if result[1] > 0.05:
        print("Series is not stationary")
    else:
        print("Series is stationary")

check_stationarity(store_sales['Sales'])

# 3. Differencing the data if not stationary
store_sales_diff = store_sales.diff().dropna()
check_stationarity(store_sales_diff['Sales'])

# 4. Autocorrelation and Partial Autocorrelation Plots
plot_acf(store_sales_diff)
plot_pacf(store_sales_diff)
plt.show()

# 5. Sliding Window for Time Series
def create_supervised_data(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size), 0])
        y.append(data[i + window_size, 0])
    return np.array(X), np.array(y)

window_size = 30  # Days
scaled_sales = MinMaxScaler(feature_range=(-1, 1))
store_sales_scaled = scaled_sales.fit_transform(store_sales)

X, y = create_supervised_data(store_sales_scaled, window_size)
X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshape to [samples, time_steps, features]

# 6. Build LSTM Model
model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(window_size, 1)))
model.add(Dropout(0.2))  # To prevent overfitting
model.add(LSTM(100, activation='relu'))
model.add(Dense(1))  # Output layer for regression

model.compile(optimizer='adam', loss='mean_squared_error')

# 7. Train the model
history = model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

# 8. Predict the sales for the next day
y_pred = model.predict(X[-1].reshape(1, window_size, 1))
predicted_sales = scaled_sales.inverse_transform(y_pred)

print(f"Predicted Sales: {predicted_sales}")

# 9. Evaluate the model
y_train_pred = model.predict(X)
train_rmse = np.sqrt(mean_squared_error(y, y_train_pred))
print(f"Train RMSE: {train_rmse}")

# 10. Visualize the results
plt.plot(store_sales.index[-len(y):], scaled_sales.inverse_transform(y_train_pred), label='Predicted Sales')
plt.plot(store_sales.index[-len(y):], scaled_sales.inverse_transform(y.reshape(-1, 1)), label='True Sales')
plt.legend()
plt.show()
