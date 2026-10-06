import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("housing1.csv")

print("Shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nTotal missing values:")
print(df.isnull().sum().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

df = df.fillna(df.mean(numeric_only=True))

df = df.drop_duplicates()

X = df[
    [
        "longitude",
        "latitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income"
    ]
].values

y = df["median_house_value"].values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

model = LinearRegression()

model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)

y_test_pred = model.predict(X_test)

train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_r2 = r2_score(y_train, y_train_pred)

test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, y_test_pred)

print("\nIntercept:")
print(model.intercept_)

print("\nCoefficients:")
print(model.coef_)

print("\n----- Training Results -----")
print("MSE:", train_mse)
print("RMSE:", train_rmse)
print("R2:", train_r2)

print("\n----- Testing Results -----")
print("MSE:", test_mse)
print("RMSE:", test_rmse)
print("R2:", test_r2)