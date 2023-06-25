import numpy as np
import pandas as pd

# Define the estimatePrice function
def estimatePrice(theta0, theta1, mileage):
    return theta0 + (theta1 * mileage)

# Load the model
model = np.load('model.npz')
theta0 = model['theta0']
theta1 = model['theta1']

# Load the data
data = pd.read_csv('data.csv')
X = data.iloc[:, 0]

while True:
    mileage_input = input("Enter the mileage (type 'exit' to quit): ")
    if mileage_input.lower() == 'exit':
        break

    try:
        mileage = float(mileage_input)
        price = estimatePrice(theta0, theta1, mileage)
        print("Estimated price:", price)
    except ValueError:
        print("Invalid mileage input. Please enter a valid number or type 'exit' to quit.")
