import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
import time
from linear_regressor_refined import create_sensor_features


# Start Time
start = time.time()


# Select any sensor measurements to use, for example [4, 7, 11].
# Rolling window is the amount of past data used for calculating rolling means
# Gradient window is the past point used to estimate the gradient of the trajectory
SENSOR_NUMBERS = [2,4,7,11,12,14]
ROLLING_WINDOW = 100
GRADIENT_WINDOW = 100

# Path to the folder containing all of the relevant txt files with the data
data_dir = Path(
    "C:/Users/johnn/Documents/Python/Predictive_Maintenance_Project/data/raw/CMAPSSData"
)

# Initialising names of columns for our dataframes
column_names = [
    "unit number",
    "time in cycles",
    "operational setting 1",
    "operational setting 2",
    "operational setting 3",
    *[f"sensor measurement {i}" for i in range(1, 22)],
]

# Keep the existing training dataset selection. The test data is not loaded or
# used during feature selection or validation.
file_paths = [
    str(path) for path in data_dir.glob("*.txt") if path.name.lower() != "readme.txt"
]
train_path = file_paths[8]

df_train = pd.read_csv(train_path, sep=r"\s+", header=None, names=column_names)

feature_columns = [
    name
    for sensor_num in SENSOR_NUMBERS
    for name in (
        f"sensor_{sensor_num}_raw",
        f"sensor_{sensor_num}_rolling_mean",
        f"sensor_{sensor_num}_gradient",
    )
]

# GroupKFold keeps every row from a unit in the same fold. Feature construction
# is repeated within each fold, and the rolling calculations are unit-local.
groups = df_train["unit number"]
cross_validator = GroupKFold(n_splits=5)
fold_scores = []

for fold_number, (train_indices, validation_indices) in enumerate(
    cross_validator.split(df_train, groups=groups), start=1
):
    df_fit = df_train.iloc[train_indices].copy()
    df_validation = df_train.iloc[validation_indices].copy()

    train_features = create_sensor_features(
        df_fit, SENSOR_NUMBERS, ROLLING_WINDOW, GRADIENT_WINDOW
    )
    train_features["RUL"] = train_features["unit number"].map(
        df_fit.groupby("unit number")["time in cycles"].max()
    ) - train_features["time in cycles"]

    validation_features = create_sensor_features(
        df_validation, SENSOR_NUMBERS, ROLLING_WINDOW, GRADIENT_WINDOW
    )
    validation_features["RUL"] = validation_features["unit number"].map(
        df_validation.groupby("unit number")["time in cycles"].max()
    ) - validation_features["time in cycles"]

    train_data = train_features.dropna(subset=feature_columns)
    validation_data = validation_features.dropna(subset=feature_columns)

    X_train = train_data[feature_columns]
    y_train = train_data["RUL"]
    X_validation = validation_data[feature_columns]
    y_validation = validation_data["RUL"]

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_validation_pred = model.predict(X_validation)

    scores = {
        "R-squared": r2_score(y_validation, y_validation_pred),
        "MSE": mean_squared_error(y_validation, y_validation_pred),
        "MAE": mean_absolute_error(y_validation, y_validation_pred),
    }
    fold_scores.append(scores)

    train_units = df_fit["unit number"].nunique()
    validation_units = df_validation["unit number"].nunique()
    print(
        f"Fold {fold_number} ({train_units} train units, "
        f"{validation_units} validation units): "
        f"R-squared={scores['R-squared']:.4f}, "
        f"MSE={scores['MSE']:.4f}, MAE={scores['MAE']:.4f}"
    )

score_summary = pd.DataFrame(fold_scores)
print("\nCross-validation summary (mean and standard deviation):")
for metric in ("R-squared", "MSE", "MAE"):
    print(
        f"{metric}: mean={score_summary[metric].mean():.4f}, "
        f"{score_summary[metric].std():.4f}"
    )

# End Time
end = time.time()

# Print the time taken for program to run
print(f"Time taken for the program to run is {end - start} seconds.")

