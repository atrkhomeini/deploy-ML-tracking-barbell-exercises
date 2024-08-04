import pandas as pd
from glob import glob
import os
from kaggle.api.kaggle_api_extended import KaggleApi
os.chdir('../')

# Entity
from dataclass import dataclass
from pathlib import Path

@dataclass(frozen=True)

#set API credentials
api = KaggleApi()
api.authenticate()
dataset_name = 'atrkhomeini/metamotion'
download_path = '../../data/raw/MetaMotion'
#check if the file already exists
if not os.path.exists(download_path):
    api.dataset_download_files(dataset_name, path=download_path)
    print("Download completed.")
else:
    print("The file already exists.")
# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------
single_read_acc=pd.read_csv("../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv")
single_read_gyr=pd.read_csv("../../data/raw/MetaMotion/A-bench-heavy_MetaWear_2019-01-14T14.22.49.165_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv")
# --------------------------------------------------------------
# List all data in data/raw/MetaMotion
# --------------------------------------------------------------
file=glob("../../data/raw/MetaMotion/*.csv")
len(file)
# --------------------------------------------------------------
# Extract features from filename
# --------------------------------------------------------------
data_path="../../data/raw/MetaMotion/*.csv"
f=file[0]

participants=f.split("-")[0].replace("../../data/raw/MetaMotion\\","")
label=f.split("-")[1]
category=f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

df=pd.read_csv(f)

df["participants"]=participants
df["label"]=label
df["category"]=category
# --------------------------------------------------------------
# Read all files
# --------------------------------------------------------------
acc_df=pd.DataFrame()
gyr_df=pd.DataFrame()

acc_set=1
gyr_set=1

for f in file:
    participants=f.split("-")[0].replace("../../data/raw/MetaMotion\\","")
    label=f.split("-")[1]
    category=f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

    df=pd.read_csv(f)

    df["participants"]=participants
    df["label"]=label
    df["category"]=category

    if "Accelerometer" in f:
        df["set"]=acc_set
        acc_set+=1
        acc_df=pd.concat([acc_df, df])
    if "Gyroscope" in f:
        df["set"]=gyr_set
        gyr_set+=1
        gyr_df=pd.concat([gyr_df,df])
# --------------------------------------------------------------
# Working with datetimes
# --------------------------------------------------------------
acc_df.info()

pd.to_datetime(df["epoch (ms)"], unit="ms")

acc_df.index=pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
gyr_df.index=pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")

del acc_df["epoch (ms)"]
del acc_df["time (01:00)"]
del acc_df["elapsed (s)"]

del gyr_df["epoch (ms)"]
del gyr_df["time (01:00)"]
del gyr_df["elapsed (s)"]

# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------
file=glob("../../data/raw/MetaMotion/*.csv")

def read_data_from_file(file):
    acc_df=pd.DataFrame()
    gyr_df=pd.DataFrame()

    acc_set=1
    gyr_set=1
    for f in file:
        participants=f.split("-")[0].replace("../../data/raw/MetaMotion\\","")
        label=f.split("-")[1]
        category=f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

        df=pd.read_csv(f)

        df["participants"]=participants
        df["label"]=label
        df["category"]=category

        if "Accelerometer" in f:
            df["set"]=acc_set
            acc_set+=1
            acc_df=pd.concat([acc_df, df])
        if "Gyroscope" in f:
            df["set"]=gyr_set
            gyr_set+=1
            gyr_df=pd.concat([gyr_df,df])

    acc_df.index=pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
    gyr_df.index=pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")

    del acc_df["epoch (ms)"]
    del acc_df["time (01:00)"]
    del acc_df["elapsed (s)"]

    del gyr_df["epoch (ms)"]
    del gyr_df["time (01:00)"]
    del gyr_df["elapsed (s)"]

    return acc_df, gyr_df
acc_df, gyr_df=read_data_from_file(file)

# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------
data_merged=pd.concat([acc_df.iloc[:,:3], gyr_df], axis=1)

#---------------------------------------------------------------
# Rename columns
#---------------------------------------------------------------

data_merged.columns = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "participants",
    "label",
    "category",
    "set",
]
# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz
sampling= {
    "acc_x": "mean",
    "acc_y": "mean",
    "acc_z": "mean",
    "gyr_x": "mean",
    "gyr_y": "mean",
    "gyr_z": "mean",
    "participants": "last",
    "label": "last",
    "category": "last",
    "set": "last"
}

data_merged[:1000].resample(rule="200ms").apply(sampling)

#split by day

days=[g for n, g in data_merged.groupby(pd.Grouper(freq="D"))]

data_resampled= pd.concat([df.resample(rule="200ms").apply(sampling).dropna() for df in days])

data_resampled["set"]=data_resampled["set"].astype("int")
# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
# data_resampled.to_pickle("../../data/interim/01_data_processed.pkl")
#%%

#%%
