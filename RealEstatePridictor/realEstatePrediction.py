from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the dataset
# replace the path with your path
real_estate_data = pd.read_csv(
    "C:/Users/uday.kiran.ramaiah/OneDrive - Accenture/Documents/Personal Projects/Real Estate Price Prediction/Real_Estate.csv")


# Selecting features and target variable
features = ['Distance to the nearest MRT station',
            'Number of convenience stores', 'Latitude', 'Longitude']
target = 'House price of unit area'

X = real_estate_data[features]
y = real_estate_data[target]

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Model initialization
model = LinearRegression()

# Training the model
model.fit(X_train, y_train)
