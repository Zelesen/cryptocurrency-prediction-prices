import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load historical cryptocurrency data
# You can get data from various sources like Yahoo Finance, Crypto exchanges, etc.
# For simplicity, I'll use random data for this example
data = {
    'Date': pd.date_range('2022-01-01', '2024-01-01', freq='D'),
    'Price': [100 + i * 2 + 10 * (i % 5) + 5 * (i % 10) + 30 * (i % 30) for i in range(730)]
}

df = pd.DataFrame(data)

# Feature engineering
df['Day'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

# Prepare data for training
X = df[['Day', 'Month', 'Year']]
y = df['Price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Visualize predictions
plt.scatter(X_test['Date'], y_test, color='black', label='Actual Price')
plt.plot(X_test['Date'], y_pred, color='blue', linewidth=3, label='Predicted Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Cryptocurrency Price Prediction')
plt.legend()
plt.show()