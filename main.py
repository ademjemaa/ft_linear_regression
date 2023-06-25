import numpy as np
from linear_regression import LinearRegression
from estimate_price import estimatePrice

learningRate = 0.01
epochs = 1250

linear_regression = LinearRegression()

model = np.load("model.npz")
theta0 = model["theta0"]
theta1 = model["theta1"]


while True:
    mileage_input = input("Enter the mileage (type 'exit' to quit): ")
    if mileage_input.lower() == "exit":
        break
    try:
        mileage = int(mileage_input)
        price = estimatePrice(theta0, theta1, mileage)
        print("Estimated price:", price)
    except ValueError:
        print("Please enter a valid number or type 'exit' to quit.")

print("Linear regression Animation compiling")
linear_regression.plotAnimatedGraph()