import numpy as np
from linear_regression import LinearRegression
from estimate_price import estimatePrice

data_file = 'data.csv'
learningRate = 0.1
epochs = 100

linear_regression = LinearRegression(data_file)
estimated_prices, precisions, formulas, theta0, theta1 = linear_regression.train(learningRate, epochs)
np.savez('model.npz', theta0=theta0, theta1=theta1)

while True:
    mileage_input = input("Enter the mileage (type 'exit' to quit): ")
    if mileage_input.lower() == 'exit':
        break

    try:
        mileage = float(mileage_input)
        price = linear_regression.estimatePrice(theta0, theta1, mileage)
        print("Estimated price:", price)
    except ValueError:
        print("Invalid mileage input. Please enter a valid number or type 'exit' to quit.")

linear_regression.plotAnimatedGraph()