import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# **********CREATE THE MODEL**********

# Read the dataset
data = pd.read_csv("part2-training-testing-data/blood_pressure_data.csv")
x = data["Age"].values.reshape(-1, 1)  # Ensure x is a 2D array for sklearn
y = data["Blood Pressure"].values

# Split the data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression().fit(xtrain, ytrain)

# Get the model coefficients and R-squared value
coef = round(float(model.coef_[0]), 2)
intercept = round(float(model.intercept_), 2)
r_squared = model.score(xtrain, ytrain)  # Evaluate the model on the training data

# Print the model's equation and R-squared value
print(f"Model's Linear Equation: y = {coef}x + {intercept}")
print(f"R Squared value: {r_squared}")

# **********TEST THE MODEL**********

# Reshape xtest to match the expected input format
xtest = xtest.reshape(-1, 1)

# Make predictions using the test data
predict = model.predict(xtest)

# Round the predicted values for clarity
predict = np.around(predict, 2)

# Print the predictions and actual values
print("\nTesting Linear Model with Testing Data:")
for index in range(len(xtest)):
    actual = ytest[index]
    predicted_y = predict[index]
    x_coord = xtest[index]
    print(f"x value: {float(x_coord[0]):.2f}, Predicted y value: {predicted_y}, Actual y value: {actual}")

# **********CREATE A VISUAL OF THE RESULTS**********

# Create a plot to visualize the data and predictions
plt.figure(figsize=(8, 6))

# Scatter plot for training and testing data
plt.scatter(xtrain, ytrain, c="purple", label="Training Data", alpha=0.6)
plt.scatter(xtest, ytest, c="blue", label="Testing Data", alpha=0.6)

# Plot predictions for testing data
plt.scatter(xtest, predict, c="red", label="Predictions", marker='x')

# Plot the line of best fit
plt.plot(x, coef*x + intercept, c="r", label="Line of Best Fit")

# Add labels and title
plt.xlabel("Age")
plt.ylabel("Blood Pressure")
plt.title("Blood Pressure by Age")

# Add legend
plt.legend()

# Show the plot
plt.show()