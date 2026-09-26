from pathlib import Path 
import zipfile
import pandas as pd
import numpy as np
import matplotlib .pyplot as plt

# Folder Path
data_dir = Path("C:/Users/johnn/Documents/Python/Predictive_Maintenance_Project/data/raw/CMAPSSData")

# Generating a list of all the column names
my_columns_names = ["unit number", "time in cycles","operational setting 1","operational setting 2", "operational setting 3"]
for i in range(21):
    name = f"sensor measurement {i+1}"
    my_columns_names.append(name)
# print(my_columns_names)

# Get a list of all the txt files in the folder 
file_paths = [str(file) for file in data_dir.glob('*.txt')]
file_paths.remove(r"C:\Users\johnn\Documents\Python\Predictive_Maintenance_Project\data\raw\CMAPSSData\readme.txt")
file_paths = file_paths[4:]
# print(len(file_paths))
# print(file_paths[4])
# print(file_paths)
# print([str(file) for file in data_dir.glob('*.pdf')])




# with open(file_paths[0], "r") as file:
#     line = file.readline()

# print(repr(line))


df = pd.read_csv(
    file_paths[4],
    sep=r"\s+",
    header=None, 
    names=my_columns_names
    )
print(df.shape)
# print(df.isna().sum())
# print(df.head()[["sensor measurement 1","sensor measurement 2","sensor measurement 3"]])
# print(df.nunique())



meas_num = 3
max_time = max(df["unit number"].value_counts())
total_trajs = df["unit number"].nunique()
print(total_trajs)
time = np.arange(max_time)
fig = plt.figure()
plt.xlabel("Time")
plt.ylabel(f"Sensor Measurement {meas_num}")
for traj_num in np.linspace(1,total_trajs,5).astype(int):
    traj_data = df[df["unit number"] == traj_num]
    plt.plot(traj_data["time in cycles"], traj_data[f"sensor measurement {meas_num}"])
    
plt.show() 

