import time
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from linear_regressor_refined import create_sensor_features


start = time.time()

# Keep the sensor set fixed while comparing window sizes.
SENSOR_NUMBERS = [4,15]
# WINDOW_VALUES = range(110, 141, 10)
WINDOW_VALUES = [140,150]
N_FOLDS = 5

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

# Use only the existing training dataset; the final test data is not loaded.
file_paths = [
    str(path) for path in data_dir.glob("*.txt") if path.name.lower() != "readme.txt"
]
train_path = file_paths[8]
df_train = pd.read_csv(train_path, sep=r"\s+", header=None, names=column_names)


def get_common_validation_indices(df, sensor_numbers, folds):
    """Find rows with valid features for every candidate window pair."""
    largest_window = max(WINDOW_VALUES)
    feature_columns = [
        name
        for sensor_num in sensor_numbers
        for name in (
            f"sensor_{sensor_num}_raw",
            f"sensor_{sensor_num}_rolling_mean",
            f"sensor_{sensor_num}_gradient",
        )
    ]
    common_indices_by_fold = []

    for _, validation_indices in folds:
        df_validation = df.iloc[validation_indices]
        strictest_features = create_sensor_features(
            df_validation, sensor_numbers, largest_window, largest_window
        )
        valid_indices = strictest_features.dropna(subset=feature_columns).index
        common_indices_by_fold.append(valid_indices)

    return common_indices_by_fold


def cross_validate_windows(
    df, sensor_numbers, rolling_window, gradient_window, folds,
    common_validation_indices, return_predictions=False,
):
    """Return fold scores, and optionally actual/predicted values for the last fold."""
    feature_columns = [
        name
        for sensor_num in sensor_numbers
        for name in (
            # f"sensor_{sensor_num}_raw",
            f"sensor_{sensor_num}_rolling_mean",
            f"sensor_{sensor_num}_gradient",
        )
    ]
    fold_scores = []
    last_fold_values = None

    for (train_indices, validation_indices), common_indices in zip(
        folds, common_validation_indices
    ):
        df_fit = df.iloc[train_indices].copy()
        df_validation = df.iloc[validation_indices].copy()

        train_features = create_sensor_features(
            df_fit, sensor_numbers, rolling_window, gradient_window
        )
        train_features["RUL"] = train_features["unit number"].map(
            df_fit.groupby("unit number")["time in cycles"].max()
        ) - train_features["time in cycles"]

        validation_features = create_sensor_features(
            df_validation, sensor_numbers, rolling_window, gradient_window
        )
        validation_features["RUL"] = validation_features["unit number"].map(
            df_validation.groupby("unit number")["time in cycles"].max()
        ) - validation_features["time in cycles"]

        train_data = train_features.dropna(subset=feature_columns)
        # Score all candidates on the exact same valid rows for this fold.
        validation_data = validation_features.loc[common_indices]
        X_train = train_data[feature_columns]
        y_train = train_data["RUL"]
        X_validation = validation_data[feature_columns]
        y_validation = validation_data["RUL"]

        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_validation)
        last_fold_values = (y_validation.to_numpy(), predictions)

        fold_scores.append(
            {
                "R-squared": r2_score(y_validation, predictions),
                "MSE": mean_squared_error(y_validation, predictions),
                "MAE": mean_absolute_error(y_validation, predictions),
            }
        )

    score_frame = pd.DataFrame(fold_scores)
    if return_predictions:
        return score_frame, last_fold_values
    return score_frame


# Reuse the same trajectory-level folds for every window combination.
cross_validator = GroupKFold(n_splits=N_FOLDS)
folds = list(
    cross_validator.split(df_train, groups=df_train["unit number"])
)
common_validation_indices = get_common_validation_indices(
    df_train, SENSOR_NUMBERS, folds
)
for fold_number, valid_indices in enumerate(common_validation_indices, start=1):
    print(
        f"Fold {fold_number}: {len(valid_indices)} common validation observations"
    )

candidate_results = []
for rolling_window in WINDOW_VALUES:
    for gradient_window in WINDOW_VALUES:
        scores = cross_validate_windows(
            df_train,
            SENSOR_NUMBERS,
            rolling_window,
            gradient_window,
            folds,
            common_validation_indices,
        )
        result = {
            "rolling_window": rolling_window,
            "gradient_window": gradient_window,
        }
        for metric in ("R-squared", "MSE", "MAE"):
            result[f"mean_{metric}"] = scores[metric].mean()
            result[f"std_{metric}"] = scores[metric].std()
        candidate_results.append(result)

results = pd.DataFrame(candidate_results)
print("Window candidates (mean and standard deviation across 5 folds):")
print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

best_result = results.loc[results["mean_MAE"].idxmin()]
print("\nBest window pair by mean validation MAE:")
print(
    f"ROLLING_WINDOW={int(best_result['rolling_window'])}, "
    f"GRADIENT_WINDOW={int(best_result['gradient_window'])}, "
    f"MAE={best_result['mean_MAE']:.4f} "
    f"(std={best_result['std_MAE']:.4f})"
)

# Plot validation predictions from the last fold using the best window pair.
# Make sure you understand this code!!!
_, (actual_rul, predicted_rul) = cross_validate_windows(
    df_train,
    SENSOR_NUMBERS,
    int(best_result["rolling_window"]),
    int(best_result["gradient_window"]),
    folds[-1:],
    common_validation_indices[-1:],
    return_predictions=True,
)

observation_numbers = range(1, len(actual_rul) + 1)
plt.scatter(observation_numbers, actual_rul, color="blue", s=10, label="Actual RUL")
plt.scatter(
    observation_numbers, predicted_rul, color="red", s=10, label="Predicted RUL"
)
plt.xlabel("Validation observation")
plt.ylabel("Remaining Useful Life (cycles)")
plt.title("Actual and Predicted RUL: Last Validation Fold")
plt.legend()
plt.tight_layout()
plt.show()

end = time.time()
print(f"Time taken for the program to run is {end - start} seconds.")
