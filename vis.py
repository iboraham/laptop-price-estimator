import matplotlib.pyplot as plt


def plot(actual_prices, predicted_prices, r_squared, mae):
    # Create a scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(
        actual_prices,
        predicted_prices,
        color="blue",
        label="Actual vs. Predicted Prices",
    )
    plt.plot(
        actual_prices,
        actual_prices,
        color="red",
        linestyle="--",
        label="Perfect Prediction",
    )

    # Add R-squared and MAE to the plot
    plt.text(3000, 4600, f"R-squared = {r_squared:.2%}", fontsize=12)
    plt.text(3000, 4400, f"MAE = ${mae:.2f}", fontsize=12)

    # Customize the plot
    plt.title("Laptop Price Estimator Accuracy")
    plt.xlabel("Actual Prices ($)")
    plt.ylabel("Predicted Prices ($)")
    plt.legend()
    plt.grid(False)

    # Show the plot
    plt.savefig("docs/scores.png")
