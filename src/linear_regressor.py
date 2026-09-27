import pandas as pd
import numpy as np
import matplotlib .pyplot as plt
from pathlib import Path

# Folder Path
data_dir = Path("C:/Users/johnn/Documents/Python/Predictive_Maintenance_Project/data/raw/CMAPSSData")

# Generating a list of all the column names
my_columns_names = ["unit number", "time in cycles","operational setting 1","operational setting 2", "operational setting 3"]
for i in range(21):
    name = f"sensor measurement {i+1}"
    my_columns_names.append(name)

# Get a list of all the txt files in the folder 
file_paths = [str(file) for file in data_dir.glob('*.txt')]
file_paths.remove(r"C:\Users\johnn\Documents\Python\Predictive_Maintenance_Project\data\raw\CMAPSSData\readme.txt")
file_paths = file_paths[4:]


# Read in the text file and convert to csv
df = pd.read_csv(
    file_paths[4],
    sep=r"\s+",
    header=None, 
    names=my_columns_names
    )

df_reduced = df[
    ["unit number", "time in cycles", "sensor measurement 4"]
].copy()

meas_4_max = df_reduced["sensor measurement 4"].max()

df_reduced["positive measurement"] = abs(
    df_reduced["sensor measurement 4"] - meas_4_max
)

df_reduced["rolling measurement"] = (
    df_reduced.groupby("unit number")["sensor measurement 4"]
    .transform(lambda x: x.rolling(20).mean())
)

df_reduced["times remaining"] = df_reduced.groupby("unit number")["time in cycles"].transform(lambda x: x.max() - x)

plt.plot(df_reduced["times remaining"], df_reduced["rolling measurement"].transform(lambda x: x.max() - x))
plt.show()