import numpy as np
import pandas as pd

# Define the estimatePrice function
def estimatePrice(theta0, theta1, mileage):
    return theta0 + (theta1 * mileage)
def estimatePriceDenormalized(theta0, theta1, mileage, linear_regression):
    X_mean = np.mean(linear_regression.X)
    X_std = np.std(linear_regression.X)
    # theta0_denormalized = theta0 * X_std + X_mean - (theta1 * X_mean * X_std)
    # theta1_denormalized = theta1 * X_std
    theta0_denormalized = theta0 - (theta1 * X_mean / X_std)
    theta1_denormalized = theta1 / X_std
    return theta0_denormalized + (theta1_denormalized * mileage)
def denormalize(theta0, theta1, X):
    X_mean = np.mean(X)
    X_std = np.std(X)
    # theta0_denormalized = theta0 * X_std + X_mean - (theta1 * X_mean * X_std)
    # theta1_denormalized = theta1 * X_std
    theta0_denormalized = theta0 - (theta1 * X_mean / X_std)
    theta1_denormalized = theta1 / X_std
    return theta0_denormalized, theta1_denormalized

# Load the model
model = np.load('model.npz')
theta0 = model['theta0']
theta1 = model['theta1']

# Load the data
data = pd.read_csv('data.csv')
X = data.iloc[:, 0]



if __name__ == '__main__':
    while True:
        mileage_input = input("Enter the mileage (type 'exit' to quit): ")

        if mileage_input.lower() == 'exit':
            break

        try:
            mileage = int(mileage_input)
            price = estimatePrice(theta0, theta1, mileage)
            print("Estimated price:", price)
        except ValueError:
            print("Please enter a valid number or type 'exit' to quit.")