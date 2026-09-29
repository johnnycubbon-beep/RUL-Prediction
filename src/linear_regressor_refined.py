import pandas as pd


def create_sensor_features(df, sensor_numbers, rolling_window=20, gradient_window=10):
    """Return raw, rolling-mean, and gradient features for selected sensors.

    Rolling means and gradients are calculated independently for each unit.
    The gradient is the rolling-mean change over ``gradient_window`` cycles,
    divided by that window to express change per cycle.
    """
    feature_frame = df[["unit number", "time in cycles"]].copy()

    for sensor_num in sensor_numbers:
        sensor = f"sensor measurement {sensor_num}"
        if sensor not in df.columns:
            raise ValueError(f"Sensor column not found: {sensor}")

        raw_name = f"sensor_{sensor_num}_raw"
        rolling_name = f"sensor_{sensor_num}_rolling_mean"
        gradient_name = f"sensor_{sensor_num}_gradient"

        feature_frame[raw_name] = df[sensor]
        feature_frame[rolling_name] = (
            df.groupby("unit number")[sensor]
            .transform(lambda values: values.rolling(rolling_window).mean())
        )
        feature_frame[gradient_name] = (
            feature_frame.groupby("unit number")[rolling_name]
            .transform(
                lambda values: (values - values.shift(gradient_window))
                / gradient_window
            )
        )

    return feature_frame
