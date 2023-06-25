import numpy as np
import pandas as pd
from estimate_price import estimatePrice
from estimate_price import estimatePriceDenormalized
from estimate_price import denormalize


class LinearRegression:
    def __init__(self, data_file='data.csv'):
        self.data = pd.read_csv(data_file)
        self.X = self.data.iloc[:, 0]
        self.Y = self.data.iloc[:, 1]
        self.X_normalized = (self.X - np.mean(self.X)) / np.std(self.X)
        self.m = len(self.X)
        self.theta0 = 0
        self.theta1 = 0
        self.train(learningRate=0.01, epochs=1000)  # Call train with default values
        self.save_model()

    def precision(self):
        total = 0
        for i in range(self.m):
            total += abs(abs(self.Y[i] / estimatePrice(self.theta0, self.theta1, self.X_normalized[i])) - 1)
        return (total / self.m)

    def train(self, learningRate, epochs):
        estimated_prices = []
        precisions = []
        formulas = []
        for _ in range(epochs):
            tmp0 = 0
            tmp1 = 0
            for i in range(self.m):
                error = estimatePrice(self.theta0, self.theta1, self.X_normalized[i]) - self.Y[i]
                tmp0 += error
                tmp1 += error * self.X_normalized[i]
            self.theta0 -= (learningRate * tmp0) / self.m
            self.theta1 -= (learningRate * tmp1) / self.m
            estimate_price_new_serie = pd.concat([pd.Series([estimatePriceDenormalized(self.theta0, self.theta1, 0, self)]) ,estimatePriceDenormalized(self.theta0, self.theta1, self.X, self)])
            print(estimate_price_new_serie)
            estimated_prices.append(estimate_price_new_serie)
            # estimated_prices.insert(0, estimatePriceDenormalized(self.theta0, self.theta1, 0, self))
            precisions.append(self.precision() * 100)
            formulas.append(
                f"θ0({denormalize(self.theta0, self.theta1, self.X)[0]:.2f}) + (θ1({denormalize(self.theta0, self.theta1, self.X)[1]:.2f}) * Mileage)"
            )
        self.estimated_prices = estimated_prices
        self.precisions = precisions
        self.formulas = formulas
        return estimated_prices, precisions, formulas, (self.theta0 - (self.theta1 * np.mean(self.X) / np.std(self.X))), self.theta1 / np.std(self.X)


    def save_model(self):
        X_mean = np.mean(self.X)
        X_std = np.std(self.X)
        print(self.theta0, self.theta1)
        theta0_denormalized = self.theta0 - (self.theta1 * X_mean / X_std)
        theta1_denormalized = self.theta1 / X_std
        # theta0_denormalized = self.theta0 * X_std + X_mean - (self.theta1 * X_mean * X_std)
        # theta1_denormalized = self.theta1 * X_std
        print(theta0_denormalized, theta1_denormalized)
        # print(theta0_denormalizedBeta, theta1_denormalizedBeta)
        np.savez('model.npz', theta0=theta0_denormalized, theta1=theta1_denormalized)

    def plotData(self):
        import matplotlib.pyplot as plt
        plt.scatter(self.X, self.Y)
        plt.xlabel("Mileage")
        plt.ylabel("Price")
        plt.show()

    def plotAnimatedGraph(self):
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        fig, ax = plt.subplots()
        ax.scatter(self.X, self.Y)
        line, = ax.plot([], [], color='blue', alpha=0.2)
        lines_actual = [ax.plot([], [], color='red', alpha=0.2)[0] for _ in range(self.m)]
        ax.set_xlabel("Mileage")
        ax.set_ylabel("Price")
        precision_text = ax.text(0.05, 0.95, "", transform=ax.transAxes)
        formula_text = ax.text(0.50, 0.90, "", transform=ax.transAxes)

        def animate(frame):
            tmp_X = pd.concat([pd.Series([0]), self.X])
            print(tmp_X)
            line.set_data(tmp_X, self.estimated_prices[frame])
            precision_text.set_text(f"Deviation: {self.precisions[frame]:.2f}%")
            formula_text.set_text(self.formulas[frame])
            return line, precision_text, formula_text

        animation = FuncAnimation(fig, animate, frames=len(self.estimated_prices), interval=10, blit=True)
        plt.show()


if __name__ == '__main__':
    linear_regression = LinearRegression()
    linear_regression.plotAnimatedGraph()
