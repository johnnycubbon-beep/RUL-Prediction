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

df_reduced["distance_from_max"] = abs(
    df_reduced["sensor measurement 4"] - meas_4_max
)


# Calculate rolling mean separately for each machine

df_reduced["rolling measurement"] = (
    df_reduced
    .groupby("unit number")["distance_from_max"]
    .transform(lambda x: x.rolling(20).mean())
)

# Add a gradient feature
df_reduced["gradient"] = (-df_reduced
    .groupby("unit number")["rolling measurement"]
    .transform(lambda x: x - x.shift(20))
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

X = df_reduced[["rolling measurement","gradient"]]
y = df_reduced["times remaining"]

Z = pd.concat([X, y], axis=1)

# Remove rows where rolling mean doesn't exist yet
Z = Z.dropna()

X = Z[["rolling measurement","gradient"]].values
y = Z["times remaining"].values


X = X.reshape(-1, 2)
y = y.reshape(-1, 1)

print(X[:10])
print(y[:10])


# --------------------------------------------------
# Fit linear model
# --------------------------------------------------

LR = LinearRegression()

reg = LR.fit(X, y)

print(f"Regression coefficients of the fitted line are {reg.coef_}")
print(f"Intercept of the regression is {reg.intercept_}")
print(f"R² = {LR.score(X, y)}")




# --------------------------------------------------
# Test data preparation
# --------------------------------------------------

df_reduced = df_test[
    ["unit number", "time in cycles", "sensor measurement 4"]
].copy()


# Apply the same transformation used on the training data

df_reduced["distance_from_max"] = abs(
    df_reduced["sensor measurement 4"] - meas_4_max
)

# Calculate rolling mean separately for each machine

df_reduced["rolling measurement"] = (
    df_reduced
    .groupby("unit number")["distance_from_max"]
    .transform(lambda x: x.rolling(20).mean())
)

# Add a gradient feature
df_reduced["gradient"] = (-df_reduced
    .groupby("unit number")["rolling measurement"]
    .transform(lambda x: x - x.shift(20))
)





# --------------------------------------------------
# Prepare Xtest and ytest
# --------------------------------------------------

# Take the final available measurement for each machine


# Read the true RUL values
df_RUL = pd.read_csv(
    file_paths[0],
    sep=r"\s+",
    header=None
)

Xtest = (
    df_reduced
    .groupby("unit number")[["rolling measurement", "gradient"]]
    .last()
)

valid = Xtest.notna().all(axis=1)

Xtest = Xtest.loc[valid]

ytest = df_RUL.iloc[Xtest.index - 1, 0].values

y_pred_test = LR.predict(Xtest.values)

score = LR.score(Xtest.values, ytest)

print(score)

# # Crude estimate of remaining life 
# RUL_physical = abs(Xtest["rolling measurement"]/Xtest["gradient"])
# print(RUL_physical[:10])

# --------------------------------------------------
# Plot
# --------------------------------------------------

plt.scatter(Xtest["gradient"], ytest, s=5, color="r", label="True Values")
plt.scatter(Xtest["gradient"], y_pred_test, s=5, color="b", label="Predicted Values")
# plt.scatter(Xtest["gradient"], RUL_physical, s=5, color="y", label="Physical Values")

plt.xlabel("Rolling Measurement")
plt.ylabel("Remaining Useful Life")

plt.legend()
# plt.savefig("C:/Users/johnn/Documents/Python/Predictive_Maintenance_Project/output/LR_Rolling_Avg_Results.png")
plt.show()

















