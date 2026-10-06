import numpy as np
import pandas as pd

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

split = int(0.8 * len(X))

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

beta = np.linalg.inv(
    X_train.T @ X_train
) @ X_train.T @ y_train

y_train_pred = X_train @ beta
y_test_pred = X_test @ beta

train_mse = np.mean(
    (y_train - y_train_pred) ** 2
)

train_rmse = np.sqrt(train_mse)

train_r2 = 1 - (
    np.sum((y_train - y_train_pred) ** 2)
    /
    np.sum((y_train - np.mean(y_train)) ** 2)
)

test_mse = np.mean(
    (y_test - y_test_pred) ** 2
)

test_rmse = np.sqrt(test_mse)

test_r2 = 1 - (
    np.sum((y_test - y_test_pred) ** 2)
    /
    np.sum((y_test - np.mean(y_test)) ** 2)
)

print("\nCoefficients:")
print(beta)

print("\n----- Training Results -----")
print("MSE:", train_mse)
print("RMSE:", train_rmse)
print("R2:", train_r2)

print("\n----- Testing Results -----")
print("MSE:", test_mse)
print("RMSE:", test_rmse)
print("R2:", test_r2)