# Ex.No: 08     MOVINTG AVERAGE MODEL AND EXPONENTIAL SMOOTHING

### AIM:
To implement Moving Average Model and Exponential smoothing Using Python.

### ALGORITHM:
1. Import necessary libraries
2. Read the electricity time series data from a CSV file,Display the shape and the first 20 rows of
the dataset
3. Set the figure size for plots
4. Suppress warnings
5. Plot the first 50 values of the 'Value' column
6. Perform rolling average transformation with a window size of 5
7. Display the first 10 values of the rolling mean
8. Perform rolling average transformation with a window size of 10
9. Create a new figure for plotting,Plot the original data and fitted value
10. Show the plot
11. Also perform exponential smoothing and plot the graph

### PROGRAM:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

data = sm.datasets.sunspots.load_pandas().data
data['YEAR'] = pd.to_datetime(data['YEAR'], format='%Y')
data.set_index('YEAR', inplace=True)
sunspots_data = data['SUNACTIVITY']

print("Shape:", sunspots_data.shape)
print(sunspots_data.head())

plt.figure(figsize=(12,6))
plt.plot(sunspots_data)
plt.title('Sunspot Activity Over Years')
plt.xlabel('Year')
plt.ylabel('Sunspots')
plt.grid()
plt.show()

rolling_mean_5 = sunspots_data.rolling(window=5).mean()
rolling_mean_10 = sunspots_data.rolling(window=10).mean()

plt.figure(figsize=(12,6))
plt.plot(sunspots_data, label='Original Data')
plt.plot(rolling_mean_5, label='MA window=5')
plt.plot(rolling_mean_10, label='MA window=10')
plt.legend()
plt.title('Moving Average - Sunspots')
plt.grid()
plt.show()

x = int(len(scaled_data) * 0.8)
train_data = scaled_data[:x]
test_data = scaled_data[x:]

model = ExponentialSmoothing(train_data, trend='add', seasonal='mul', seasonal_periods=11).fit()
test_predictions = model.forecast(len(test_data))

plt.figure(figsize=(12,6))
plt.plot(train_data, label="Train Data")
plt.plot(test_data, label="Test Data")
plt.plot(test_predictions, label="Predictions")
plt.legend()
plt.title('Forecasting on Sunspots Data')
plt.show()

print("RMSE:", np.sqrt(mean_squared_error(test_data, test_predictions)))
```
### OUTPUT:

Moving Average

<img width="1012" height="534" alt="image" src="https://github.com/user-attachments/assets/82b0d567-cf46-4893-b94c-c5b13889d5f8" />

Exponential Smoothing

<img width="993" height="524" alt="image" src="https://github.com/user-attachments/assets/ed5e56a4-0220-4eb1-abf0-038abaf9b353" />

### RESULT:
Thus we have successfully implemented the Moving Average Model and Exponential smoothing using python.
