import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# Folder Path

data_dir = Path(
    "C:/Users/johnn/Documents/Python/Predictive_Maintenance_Project/data/raw/CMAPSSData"
)


# Generating a list of all the column names

my_columns_names = [
    "unit number",
    "time in cycles",
    "operational setting 1",
    "operational setting 2",
    "operational setting 3"
]

for i in range(21):
    name = f"sensor measurement {i+1}"
    my_columns_names.append(name)


# Get a list of all the txt files in the folder

file_paths = [str(file) for file in data_dir.glob("*.txt")]

file_paths.remove(
    r"C:\Users\johnn\Documents\Python\Predictive_Maintenance_Project\data\raw\CMAPSSData\readme.txt"
)
# print(file_paths)




# Read in the training data

df_train = pd.read_csv(
    file_paths[8],
    sep=r"\s+",
    header=None,
    names=my_columns_names
)


# Read in the test data

df_test = pd.read_csv(
    file_paths[4],
    sep=r"\s+",
    header=None,
    names=my_columns_names
)


# --------------------------------------------------
# Training data preparation
# --------------------------------------------------

df_reduced = df_train[
    ["unit number", "time in cycles", "sensor measurement 4"]
].copy()


# Transform sensor measurement 4

meas_4_max = df_reduced["sensor measurement 4"].max()

df_reduced["positive measurement"] = abs(
    df_reduced["sensor measurement 4"] - meas_4_max
)


# Calculate rolling mean separately for each machine

df_reduced["rolling measurement"] = (
    df_reduced
    .groupby("unit number")["positive measurement"]
    .transform(lambda x: x.rolling(20).mean())
)

# Add a gradient feature
df_reduced["gradient"] = (-df_reduced
    .groupby("unit number")["rolling measurement"]
    .transform(lambda x: x - x.shift(10))
)

# print(df_reduced["gradient"].apply(lambda x: (x <= 0)).value_counts())

# Calculate actual RUL for each training observation

df_reduced["times remaining"] = (
    df_reduced
    .groupby("unit number")["time in cycles"]
    .transform(lambda x: x.max() - x)
)


# --------------------------------------------------
# Prepare X and y
# --------------------------------------------------

# X = df_reduced["rolling measurement"]
# y = df_reduced["times remaining"]

# Z = pd.concat([X, y], axis=1)

# # Remove rows where rolling mean doesn't exist yet
# Z = Z.dropna()

# X = Z["rolling measurement"].values
# y = Z["times remaining"].values

# X = X.reshape(-1, 1)
# y = y.reshape(-1, 1)


# --------------------------------------------------
# Fit linear model
# --------------------------------------------------

# LR = LinearRegression()

# reg = LR.fit(X, y)

# print(f"Gradient of the fitted line is {reg.coef_}")
# print(f"y-intercept of the fitted line is {reg.intercept_}")
# print(f"R² = {LR.score(X, y)}")


# --------------------------------------------------
# Create smooth x-values for plotting
# --------------------------------------------------

# x_plot = np.linspace(
#     X.min(),
#     X.max(),
#     500
# ).reshape(-1, 1)

# y_pred = LR.predict(x_plot)


# --------------------------------------------------
# Plot fitted line
# --------------------------------------------------

# plt.plot(
#     x_plot,
#     y_pred,
#     label="Linear Fit"
# )

# plt.xlabel("Rolling Measurement")
# plt.ylabel("Time Until Failure")
# plt.legend()

# plt.show()

# --------------------------------------------------
# Test data preparation
# --------------------------------------------------

# df_reduced = df_test[
#     ["unit number", "time in cycles", "sensor measurement 4"]
# ].copy()


# # Apply the same transformation used on the training data

# df_reduced["positive measurement"] = abs(
#     df_reduced["sensor measurement 4"] - meas_4_max
# )

# # Calculate rolling mean separately for each machine

# df_reduced["rolling measurement"] = (
#     df_reduced
#     .groupby("unit number")["positive measurement"]
#     .transform(lambda x: x.rolling(20).mean())



# --------------------------------------------------
# Prepare Xtest and ytest
# --------------------------------------------------

# Take the final available measurement for each machine

# Xtest = (
#     df_reduced
#     .groupby("unit number")["rolling measurement"]
#     .last()
# )

# # Convert to NumPy array for sklearn

# Xtest = Xtest.values.reshape(-1, 1)


# Read the true RUL values

# df_RUL = pd.read_csv(
#     file_paths[0],
#     sep=r"\s+",
#     header=None
# )

# ytest = df_RUL.values.reshape(-1, 1)


# --------------------------------------------------
# Predict
# --------------------------------------------------

# y_pred_test = LR.predict(Xtest)
# score = LR.score(Xtest,ytest)
# print(score)


# --------------------------------------------------
# Plot
# --------------------------------------------------

# plt.scatter(Xtest, ytest, s=5, color="r", label="True Values")
# plt.scatter(Xtest, y_pred_test, s=5, color="b", label="Predicted Values")

# plt.xlabel("Rolling Measurement")
# plt.ylabel("Remaining Useful Life")

# plt.legend()
# # plt.savefig("C:/Users/johnn/Documents/Python/Predictive_Maintenance_Project/output/LR_Rolling_Avg_Results.png")
# plt.show()












# # Plot the fitted line with the actual data
# plt.plot(df_reduced["rolling measurement"], df_reduced["times remaining"], label="Actual Data")
# plt.plot(X,y_pred,label="Fitted Line")
# plt.xlabel("Time Until Failure")
# plt.ylabel("Rolling Measurement")
# plt.show()

# # Fitting a quadratic model

# poly = PolynomialFeatures(degree=2, include_bias=False)
# poly_features = poly.fit_transform(X)

# LR2 = LinearRegression()
# quad = LR2.fit(poly_features, y)

# print(f"Fitted quadratic coefficients are: {quad.coef_}")
# print(f"Intercept of the quadratic model is: {quad.intercept_}")
# print(f"Fitting Score is {LR2.score(poly_features,y)}")


# # Create smooth x-values for plotting

# x_plot = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)

# x_plot_poly = poly.transform(x_plot)

# y_plot = quad.predict(x_plot_poly)


# # Plot data and fitted quadratic

# # plt.scatter(X, y, s=5, label="Actual Data")
# plt.plot(x_plot, y_plot, label="Quadratic Fit")

# plt.xlabel("Rolling Measurement")
# plt.ylabel("Time Until Failure")
# plt.legend()
# plt.show()


