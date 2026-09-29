import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from linear_regressor_refined import create_sensor_features


# Select any sensor measurements to use, for example [4, 7, 11].
SENSOR_NUMBERS = [7]
ROLLING_WINDOW = 20
GRADIENT_WINDOW = 10

data_dir = Path(
    "C:/Users/johnn/Documents/Python/Predictive_Maintenance_Project/data/raw/CMAPSSData"
)

column_names = [
    "unit number",
    "time in cycles",
    "operational setting 1",
    "operational setting 2",
    "operational setting 3",
    *[f"sensor measurement {i}" for i in range(1, 22)],
]

# Keep the existing train/test dataset selection.
file_paths = [
    str(path) for path in data_dir.glob("*.txt") if path.name.lower() != "readme.txt"
]
train_path = file_paths[8]
test_path = file_paths[4]
rul_path = file_paths[0]

df_train = pd.read_csv(train_path, sep=r"\s+", header=None, names=column_names)
df_test = pd.read_csv(test_path, sep=r"\s+", header=None, names=column_names)
df_RUL = pd.read_csv(rul_path, sep=r"\s+", header=None)


# Training rows have known end-of-life cycles, so derive RUL within each unit.
train_features = create_sensor_features(
    df_train, SENSOR_NUMBERS, ROLLING_WINDOW, GRADIENT_WINDOW
)
train_features["RUL"] = train_features["unit number"].map(
    df_train.groupby("unit number")["time in cycles"].max()
) - train_features["time in cycles"]

feature_columns = [
    name
    for sensor_num in SENSOR_NUMBERS
    for name in (
        f"sensor_{sensor_num}_raw",
        f"sensor_{sensor_num}_rolling_mean",
        f"sensor_{sensor_num}_gradient",
    )
]
train_data = train_features.dropna(subset=feature_columns)
X_train = train_data[feature_columns]
y_train = train_data["RUL"]


# Fit using training data only.
model = LinearRegression()
model.fit(X_train, y_train)


# Test files contain truncated trajectories. Score the final available row for
# each unit against its separately supplied true RUL value.
test_features = create_sensor_features(
    df_test, SENSOR_NUMBERS, ROLLING_WINDOW, GRADIENT_WINDOW
)
X_test = test_features.groupby("unit number")[feature_columns].last()
X_test = X_test.dropna(subset=feature_columns)
y_test = df_RUL.iloc[X_test.index.to_numpy(dtype=int) - 1, 0].to_numpy()
y_pred = model.predict(X_test[feature_columns])

print(f"Test R²: {r2_score(y_test, y_pred):.4f}")
print(f"Test MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"Test MAE: {mean_absolute_error(y_test, y_pred):.4f}")
