import numpy as np
from linear_regression import LinearRegression
from estimate_price import estimatePrice

data_file = 'data.csv'
learningRate = 0.01
epochs = 10000

linear_regression = LinearRegression(data_file)
estimated_prices, precisions, formulas, theta0, theta1 = linear_regression.train(learningRate, epochs)
np.savez('model.npz', theta0=theta0, theta1=theta1)

while True:
    mileage_input = input("Enter the mileage (type 'exit' to quit): ")

    # Check if the user entered 'exit'
    if mileage_input.lower() == 'exit':
        break

    # Try to convert the user input to an integer and estimate the price
    try:
        mileage = int(mileage_input)
        price = estimatePrice(theta0, theta1, mileage)
        print("Estimated price:", price)
    except ValueError:
        print("Please enter a valid number or type 'exit' to quit.")



linear_regression.plotAnimatedGraph()
