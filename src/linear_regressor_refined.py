import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def create_sensor_features(df, sensor_num, rolling_window=20, gradient_window=10):
    
    # f string for the sensor number
    sensor = f"sensor measurement {sensor_num}"

    # Restricting dataframe to a few columns
    df_reduced = df[["unit number", "time in cycles", sensor]].copy()

    # Maximum value of this feature in training data to make numbers more sensible
    max_measurement = df_reduced[sensor].max()

    # Make numbers easier to interpret
    df_reduced["distance from max"] = max_measurement - df_reduced[sensor]

    # Create the rolling measurement feature
    df_reduced["rolling measurement"] = (df_reduced
                                         .groupby("unit number")[sensor]
                                         .transform(lambda x: x.rolling(rolling_window).mean()))

    df_reduced["gradient"] = (df_reduced
    .groupby("unit number")["rolling measurement"]
    .transform(lambda x: (x - x.shift(gradient_window))/gradient_window)
    )

