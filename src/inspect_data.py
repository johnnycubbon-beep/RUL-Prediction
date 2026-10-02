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


# print(len(file_paths))
# print(file_paths[4])
# print(file_paths)
# print([str(file) for file in data_dir.glob('*.pdf')])




df = pd.read_csv(
    file_paths[8],
    sep=r"\s+",
    header=None, 
    names=my_columns_names
    )

df["RUL"] = df["unit number"].map(df.groupby("unit number")["time in cycles"].max()) - df["time in cycles"]


# print(df["sensor measurement 17"].unique())
# print(df.shape)
# print(df.isna().sum())
# print(df.head()[["sensor measurement 1","sensor measurement 2","sensor measurement 3"]])
# print(df.index[(df.nunique()==1).values])


# Removing columns with the same values for all rows
useful_cols = df.columns[(df.nunique()>5).values]
df = df[useful_cols]
# print(useful_cols)
# print(df.nunique(axis=0))





# # Plotting the evolution of a certain feature over time for multiple different machines
# meas_num = 3
# max_time = max(df["unit number"].value_counts())
# total_trajs = df["unit number"].nunique()
# print(total_trajs)
# time = np.arange(max_time)
# fig = plt.figure()
# plt.xlabel("Time")
# plt.ylabel(f"Sensor Measurement {meas_num}")
# for traj_num in np.linspace(1,total_trajs,3).astype(int):
#     traj_data = df[df["unit number"] == traj_num]
#     plt.plot(traj_data["time in cycles"], traj_data[f"sensor measurement {meas_num}"])
    
# plt.show() 

# Plotting the evolution of the rolling mean of a certain feature over time for multiple different machines
meas_num = 15
window = 100
gradient_window = 30
max_time = max(df["unit number"].value_counts())
total_trajs = df["unit number"].nunique()
time = np.arange(max_time)
fig = plt.figure()
plt.xlabel("Log RUL")
plt.ylabel("Rolling Measurement")
# plt.ylabel(f"Sensor Measurement {meas_num}")
for traj_num in np.linspace(1,total_trajs,8).astype(int):
    traj_data = df[df["unit number"] == traj_num]
    rolling_avg = traj_data[f"sensor measurement {meas_num}"].rolling(window).mean()[window:]
    gradient = rolling_avg.transform(lambda x: (x - x.shift(gradient_window))/gradient_window)
        # print(len(gradient),len(traj_data["RUL"]))
    plt.plot((traj_data["RUL"][window:]), rolling_avg)
    
plt.show() 

# # Plotting the evolution of several features over time for one machine
# unit_1 = df[df["unit number"] == 1]
# time = np.arange(len(unit_1))
# fig = plt.figure()
# plt.xlabel("Time")
# plt.ylabel("Measurement")
# meas_inds = [2,4,15,8,13,17]
# colors = ['r','b','g','c','y']
# for ind, col in zip(meas_inds,colors):
#     meas = f"sensor measurement {ind}"
#     devs = unit_1[meas] - unit_1[meas].mean() 
#     devs = devs/max(abs(devs))
#     plt.plot(time, devs, color=col, label=str(ind))
# plt.legend()
# plt.show()
