import os
import time
import pickle
import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from loader import load_raw_log

from datetime import datetime, timedelta
from multiprocessing import Pool
from functools import partial
import random

random.seed(42)


def get_label(df, dataname):
    if dataname in ["BGL", "Thunderbird", "Spirit"]:
        return df["Label"].map(lambda x: "-" != x).astype(int)

def round_to_interval(timestamp, interval):
    # extract the minute part from the date-time string
    datetime_str = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(int(timestamp)))

    minute = int(datetime_str[14:16])
    minute = (minute // interval) * interval
    dt_trunc = datetime_str[:14] + "{:02d}".format(minute) + ":00"
    return dt_trunc


def generate_sliding_windows_chunk(df, start_times, window_size, dataname):
    windows = []
    for start_time in tqdm(start_times):
        window = df.loc[
                start_time : start_time
                + pd.Timedelta(minutes=window_size)
                - pd.Timedelta(seconds=1)
            ]
        window["Label"] = get_label(window, dataname)
        windows.append(
            window.to_dict("records")
        )
    return [item for item in windows if len(item) > 0]


def generate_sliding_time_windows(dataname, df, window_size=5, step_size=1, num_processes=4):
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="s")
    df.set_index("Timestamp", inplace=True, drop=False)
    df["Timestamp"] = df["Timestamp"].map(lambda x: x.isoformat())

    start_times = pd.date_range(
        start=df.index.min(),
        end=df.index.max()
        - pd.Timedelta(minutes=window_size)
        + pd.Timedelta(seconds=1),
        freq=f"{step_size}T",
    )

    if num_processes != 1:
        # Split the range of start times into N chunks
        chunk_size = len(start_times) // num_processes
        chunks = [
            start_times[i : i + chunk_size]
            for i in range(0, len(start_times), chunk_size)
        ]

        # Create a process pool with the specified number of processes
        with Pool(num_processes) as pool:
            windows = []

            # Use apply_async to generate the windows for each chunk in parallel
            results = [
                pool.apply_async(
                    generate_sliding_windows_chunk, args=(df, chunk, window_size, dataname)
                )
                for chunk in chunks
            ]

            # Collect the windows from each result object
            for result in results:
                windows.extend(result.get())
    else:
        windows = generate_sliding_windows_chunk(df, start_times, window_size, dataname)
    windows = sorted(windows, key=lambda x: x[0]["Timestamp"])
    return windows


def generate_fixed_windows(dataname, df, window_size):
    # Use the groupby method to perform the sliding window
    df["Label"] = get_label(df, dataname)
    data_list = df.to_dict(orient='records')
    result = []
    for i in range(0, len(data_list), window_size):
        window_data = data_list[i:i+window_size]
        result.append(window_data)
    return result

def get_anomaly_ratio(windows):
    anomaly_window = [window for window in windows if any(item["Label"]==1 for item in window)]
    total_window_num = len(windows)
    anomaly_window_num = len(anomaly_window)
    print("{}/{} ({:.2f})".format(anomaly_window_num, total_window_num, anomaly_window_num/total_window_num))

def process_hpc(
    dataname,
    log_format,
    log_file,
    output_dir,
    window_size,
    step_size,
    train_size,
    chunksize=10000000,  # bytes
    num_processes=4,
    save_type="json",
    content2template_file="",
    window_type="",
    n_samples=n_samples,
):
    os.makedirs(output_dir, exist_ok=True)

    with open(content2template_file, "r") as f:
        content2template = json.load(f)

    df = load_raw_log(
        log_format, log_file, chunksize=chunksize, num_processes=4
    )
    print("Data loading done.")
    df.sort_values(by="Timestamp", ascending=True, inplace=True)
    df["EventTemplate"] = df["Content"].map(lambda x: content2template.get(x, x))
    df["EventTemplate"].fillna("", inplace=True)
    # drop all templates with white space
    df = df[df["EventTemplate"].map(lambda x: x.strip() != "")]
    print("Data processing done, {} lines in total.".format(len(df)))

    n_train = int(len(df) * train_size)

    train_df = df[:n_train]
    test_df = df[n_train:]

    print("num train: {}, num test: {}".format(len(train_df), len(test_df)))

    # train_df.to_csv("{}_train.csv".format(dataname), index=False)
    # test_df.to_csv("{}_test.csv".format(dataname), index=False)

    print(f"Generating [{window_type}] windows.")
    if window_type == "fixed":
        train_window = generate_fixed_windows(dataname, train_df, window_size)
        test_window = generate_fixed_windows(dataname, test_df, window_size)
    elif window_type == "time":
        train_window = generate_sliding_time_windows(dataname, train_df, window_size, step_size, num_processes=num_processes)
        test_window = generate_sliding_time_windows(dataname, test_df, window_size, step_size, num_processes=num_processes)

    print("{} train windows generated.".format(len(train_window)))
    print("{} test windows generated.".format(len(test_window)))
    
    print(f"Saving processed files to {output_dir}.")
    if train_size < 1:
        if save_type == "pkl":
            with open(os.path.join(output_dir, f"test.pkl"), mode="wb") as f:
                pickle.dump(test_window, f)
        else:
            with open(os.path.join(output_dir, "test.json"), "w") as f:
                json.dump(test_window, f)
        print("Testing info:")
        get_anomaly_ratio(test_window)

    if save_type == "pkl":
        with open(os.path.join(output_dir, f"train.pkl"), mode="wb") as f:
            pickle.dump(train_window, f)
    else:
        with open(os.path.join(output_dir, "train.json"), "w") as f:
            json.dump(train_window, f)
    print("Training info:")
    get_anomaly_ratio(train_window)


if __name__ == "__main__":
    window_size = 60
    step_size = 60
    train_size = 0.8
    save_type = "pkl"
    window_type = "time"

    dataname = "BGL"
    # dataname = "Thunderbird"
    
    if window_type == "fixed":
        postfix = f"fixed_ws={window_size}_{train_size}train"
    elif window_type == "time": 
        postfix = f"ws={window_size}m_s={step_size}m_{train_size}train"
    
    if n_samples is not None:
        postfix = "{}_{}".format(postfix, n_samples)

    log_file = {
        "Thunderbird": f"../data/{dataname}/{dataname}_10m.log",
        "BGL": f"../data/{dataname}/{dataname}.log",
    }[dataname]

    output_dir = f"../proceeded_data/{dataname}/{dataname}_{postfix}"
    content2template_file = f"../data/{dataname}/parsing_results/{dataname}.mapping.json"

    print(f"Processing data: [{log_file}].")
    log_format = {
        "BGL": "<Label> <Timestamp> <Date> <Node> <DateTime> <NodeRepeat> <Type> <Component> <Level> <Content>",
        "Thunderbird": "<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?:? <Content>",
    }

    process_hpc(
        dataname, 
        log_format[dataname],
        log_file,
        output_dir,
        window_size=window_size,
        step_size=step_size,
        train_size=train_size,
        num_processes=1,
        save_type=save_type,
        content2template_file=content2template_file,
        window_type=window_type
        n_samples=n_samples,
    )

    print("Processing done.")
