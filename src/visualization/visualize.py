import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import display
# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df=pd.read_pickle("../../data/interim/01_data_processed.pkl")

# --------------------------------------------------------------
# Plot single columns
# --------------------------------------------------------------
set_df = df[df["set"]==1]

#plt.plot(set_df["acc_y"]) #this plot shown duration

plt.plot(set_df["acc_y"].reset_index(drop=True)) #this plot shown numbers of sample
plt.title("Accelerometer y-axis")
# --------------------------------------------------------------
# Plot all exercises
# --------------------------------------------------------------
for label in df["label"].unique():
    subset = df[df["label"] == label]
    #fig, ax =plt.subplots()
    plt.figure(figsize=(20,5))
    plt.plot(subset[:100]["acc_y"].reset_index(drop=True), label=label)
    plt.legend()
    plt.show()

# --------------------------------------------------------------
# Adjust plot settings
# --------------------------------------------------------------

# --------------------------------------------------------------
# Compare medium vs. heavy sets
# --------------------------------------------------------------
category_df=df.query("label == 'squat'").query("participants == 'A'").reset_index()

plt.figure(figsize=(20,5))
category_df.groupby(["category"])["acc_y"].plot()
plt.title("Squat")
plt.ylabel("acc_y")
plt.xlabel("samples")
plt.legend()

# --------------------------------------------------------------
# Compare participants
# --------------------------------------------------------------
participants_df=df.query("label == 'squat'").sort_values("participants").reset_index()

plt.figure(figsize=(20,5))
participants_df.groupby(["participants"])["acc_y"].plot()
plt.title("Squat")
plt.ylabel("acc_y")
plt.xlabel("samples")
plt.legend()

# --------------------------------------------------------------
# Plot multiple axis
# --------------------------------------------------------------
label="squat"
participants="A"
all_axis_df=df.query(f"label == '{label}'").query(f"participants == '{participants}'").reset_index()

plt.figure(figsize=(20,5))
all_axis_df[["acc_x","acc_y","acc_z"]].plot()
plt.title("Squat")
plt.xlabel("samples")
plt.legend()

# --------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# --------------------------------------------------------------
labels=df['label'].unique()
participant=df['participants'].unique()

for label in labels:
    for participants in participant:
        all_axis_df = (df.query(f"label == '{label}'").query(f"participants == '{participants}'").reset_index())

        if len(all_axis_df)>0:
            plt.figure(figsize=(20,5))
            all_axis_df[["acc_x","acc_y","acc_z"]].plot()
            plt.title(f"{label} ({participants})".title())
            plt.xlabel("samples")
            plt.ylabel("accelerometer")
            plt.legend()
#Create a loop to plot all combination per sensor gyroscope
for label in labels:
    for participants in participant:
        all_axis_df = (df.query(f"label == '{label}'").query(f"participants == '{participants}'").reset_index())

        if len(all_axis_df)>0:
            plt.figure(figsize=(20,5))
            all_axis_df[["gyr_x","gyr_y","gyr_z"]].plot()
            plt.title(f"{label} ({participants})".title())
            plt.xlabel("samples")
            plt.ylabel("accelerometer")
            plt.legend()
# --------------------------------------------------------------
# Combine plots in one figure
# --------------------------------------------------------------
label="squat"
participants="A"
combined_plot_df=(df.query(f"label == '{label}'").query(f"participants == '{participants}'").reset_index(drop=True))

fig, ax=plt.subplots(nrows=2, sharex=True, figsize=(20,10))
combined_plot_df[["acc_x","acc_y","acc_z"]].plot(ax=ax[0])
combined_plot_df[["gyr_x","gyr_y","gyr_z"]].plot(ax=ax[1])

ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15),ncol=3, fancybox=True, shadow=True)
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15),ncol=3, fancybox=True, shadow=True)
ax[1].set_xlabel("samples")

# --------------------------------------------------------------
# Loop over all combinations and export for both sensors
# --------------------------------------------------------------
labels=df['label'].unique()
participant=df['participants'].unique()

for label in labels:
    for participants in participant:
        combined_plot_df = (df.query(f"label == '{label}'").query(f"participants == '{participants}'").reset_index())

        if len(combined_plot_df)>0:
            fig, ax=plt.subplots(nrows=2, sharex=True, figsize=(20,10))
            combined_plot_df[["acc_x","acc_y","acc_z"]].plot(ax=ax[0])
            combined_plot_df[["gyr_x","gyr_y","gyr_z"]].plot(ax=ax[1])

            ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15),ncol=3, fancybox=True, shadow=True)
            ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15),ncol=3, fancybox=True, shadow=True)
            ax[1].set_xlabel("samples")
            plt.savefig(f"../../reports/figures/{label.title()} ({participants}).png")
            plt.show()

#%%
