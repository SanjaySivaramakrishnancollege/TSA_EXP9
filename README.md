# EX.NO.09        A project on Time series analysis on weather forecasting using ARIMA model 
### Date: 25-05-2026

### AIM:
To Create a project on Time series analysis on weather forecasting using ARIMA model in  Python and compare with other models.
### ALGORITHM:
1. Explore the dataset of weather 
2. Check for stationarity of time series time series plot
   ACF plot and PACF plot
   ADF test
   Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
7. Evaluate model predictions
### PROGRAM:
```py
# Import the necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv("/content/Walmart_Sales.csv")

# Convert 'Date' column to datetime format
data=data.head(200)
data['Date'] = pd.to_datetime(data['Date'], format="%d-%m-%Y")

# Set 'Date' column as index
data.set_index('Date', inplace=True)

# Sort the index to ensure it's monotonic
data.sort_index(inplace=True)

# ARIMA Model
def arima_model(data, target_variable, order):
    train_size = int(len(data) * 0.8)
    train_data, test_data = data[:train_size], data[train_size:]

    model = ARIMA(train_data[target_variable], order=order)
    fitted_model = model.fit()

    forecast = fitted_model.forecast(steps=len(test_data))

    rmse = np.sqrt(mean_squared_error(test_data[target_variable], forecast))

    plt.figure(figsize=(10, 6))
    plt.plot(train_data.index, train_data[target_variable], label='Training Data')
    plt.plot(test_data.index, test_data[target_variable], label='Testing Data')
    plt.plot(test_data.index, forecast, label='Forecasted Data')

    plt.xlabel('Date')
    plt.ylabel('Weekly Sales')
    plt.title('ARIMA Forecasting for Weekly Sales')

    plt.legend()
    plt.show()

    print("Root Mean Squared Error (RMSE):", rmse)

# Call the function
arima_model(data, 'Weekly_Sales', order=(5,1,0))
```
### OUTPUT:
<img width="1155" height="708" alt="image" src="https://github.com/user-attachments/assets/1be0c15f-0b3d-4390-ad4a-d65d25a199dd" />


### RESULT:
Thus the program run successfully based on the ARIMA model using python.
