import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("../../src/features/")
from DataTransformation import LowPassFilter
from scipy.signal import argrelextrema
from sklearn.metrics import mean_absolute_error

pd.options.mode.chained_assignment = None


# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("../../data/interim/01_data_processed.pkl")
df=df[df['label'] !='rest']

acc_r = df["acc_x"] **2 + df["acc_y"] **2 + df["acc_z"] **2
gyr_r = df["gyr_x"] **2 + df["gyr_y"] **2 + df["gyr_z"] **2
df["acc_r"] = np.sqrt(acc_r)
df["gyr_r"] = np.sqrt(gyr_r)
# --------------------------------------------------------------
# Split data
# --------------------------------------------------------------
bench_df = df[df["label"] == "bench"]
squat_df = df[df["label"] == "squat"]
dead_df = df[df["label"] == "dead"]
ohp_df = df[df["label"] == "ohp"]
row_df = df[df["label"] == "row"]

# --------------------------------------------------------------
# Visualize data to identify patterns
# --------------------------------------------------------------
plot_df = bench_df

plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_x"].plot(figsize=(20, 5), title="Bench Press", legend=True)
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_y"].plot(figsize=(20, 5), title="Bench Press", legend=True)
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_z"].plot(figsize=(20, 5), title="Bench Press", legend=True)
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_r"].plot(figsize=(20, 5), title="Bench Press", legend=True)

plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_x"].plot(figsize=(20, 5), title="Bench Press", legend=True)
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_y"].plot(figsize=(20, 5), title="Bench Press", legend=True)
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_z"].plot(figsize=(20, 5), title="Bench Press", legend=True)
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_r"].plot(figsize=(20, 5), title="Bench Press", legend=True)

# --------------------------------------------------------------
# Configure LowPassFilter
# --------------------------------------------------------------
fs = 1000/200
lowpass = LowPassFilter()
# --------------------------------------------------------------
# Apply and tweak LowPassFilter
# --------------------------------------------------------------
bench_set = bench_df[bench_df["set"] == bench_df["set"].unique()[0]]
squat_set = squat_df[squat_df["set"] == squat_df["set"].unique()[0]]
dead_set = dead_df[dead_df["set"] == dead_df["set"].unique()[0]]
ohp_set = ohp_df[ohp_df["set"] == ohp_df["set"].unique()[0]]
row_set = row_df[row_df["set"] == row_df["set"].unique()[0]]

column = "acc_r"
lowpass.low_pass_filter(
    bench_set, col=column, sampling_frequency=fs, cutoff_frequency=0.4, order=5
)[column + "_lowpass"].plot(figsize=(20, 5), legend=True)

# --------------------------------------------------------------
# Create function to count repetitions
# --------------------------------------------------------------
def count_reps(dataset, cutoff=0.4, order=10, column="acc_r"):
    data = lowpass.low_pass_filter(
        dataset, col=column, sampling_frequency=fs, cutoff_frequency=cutoff, order=order
    )
    indexes = argrelextrema(data[column + "_lowpass"].values, np.greater)
    peaks = data.iloc[indexes]

    fig, ax = plt.subplots()
    plt.plot(dataset[f"{column}_lowpass"])
    plt.plot(peaks[column + "_lowpass"], "o", color="red")
    ax.set_ylabel(f"{column}_lowpass")
    exercise = dataset["label"].iloc[0].title()
    category = dataset["category"].iloc[0].title()
    plt.title(f"{category} {exercise}: {len(peaks)} Repetitions")
    plt.show()

    return len(peaks)

count_reps(bench_set, cutoff=0.4)
count_reps(squat_set, cutoff=0.35)
count_reps(dead_set, cutoff=0.4)
count_reps(ohp_set, cutoff=0.35)
count_reps(row_set, cutoff=0.65, column="gyr_x")
# --------------------------------------------------------------
# Create benchmark dataframe
# --------------------------------------------------------------
df["reps"] = df["category"].apply(lambda x:5 if x == "heavy" else 10)
rep_df = df.groupby(["label","category", "set"])["reps"].max().reset_index()
rep_df["reps_predict"] = 0

for s in df["set"].unique():
    subset = df[df["set"] == s]

    column = "acc_r"
    cutoff = 0.4

    if subset["label"].iloc[0] == "squat":
        cutoff = 0.35
    if subset["label"].iloc[0] == "row":
        cutoff = 0.65
        col = "gyr_x"
    if subset["label"].iloc[0] == "ohp":
        cutoff = 0.35
    
    reps = count_reps(subset, cutoff=cutoff, column=column)
    rep_df.loc[(rep_df["set"] == s), "reps_predict"] = reps

rep_df

# --------------------------------------------------------------
# Evaluate the results
# --------------------------------------------------------------

error = mean_absolute_error(rep_df["reps"], rep_df["reps_predict"])
print(f"Mean Absolute Error: {error}")

# --------------------------------------------------------------
#Plot the results
# --------------------------------------------------------------
rep_df.groupby(["label","category"])[["reps", "reps_predict"]].mean().plot(kind="line", figsize=(20, 5), title="Repetitions per Exercise")