import argparse
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.cluster import DBSCAN
import sys

out_of_range = 300
max_threshold = 8.9
min_threshold = 6.36


def parse_cli_args():
    parser = argparse.ArgumentParser(description='plot collected data')
    parser.add_argument('file', type=str, help='Serial port which is opened.')
    parser.add_argument('--plot_type', '-t', type=str,
                        choices=['asc_desc', 'asc_desc_base', 'base', 'cut', 'cut_base', 'dbscan', 'raw', 'raw_shifted'],
                        default='', help='How data is plotted.')

    return parser.parse_args(args=None if sys.argv[1:] else ["--help"])


def read_values(file_name):
    # Read the CSV file into a pandas DataFrame
    return pd.read_csv(file_name)


def calculate_samples_per_second(df):
    start = df['Time'].head(1).values[0]
    stop = df['Time'].tail(1).values[0]
    samples_per_second = len(df['Value']) / ((stop - start) / 1e9)
    return samples_per_second


def shift_to_zero(df):
    start_time = df['Time'].head(1).values[0]
    # shift start everything relative to start:
    shifted_df = df.copy()
    shifted_df['Time'] = shifted_df['Time'] - start_time
    return shifted_df


def filter_values_out_of_range(values):
    # replace to near or to far with nan
    values[values > out_of_range] = float('NaN')
    return values


def convert_to_ms(time):
    return time / 1e6


def get_asc_desc_frames(df):
    # Initialize empty DataFrames for A and B
    ascending_df = pd.DataFrame(columns=['Time', 'Value'])
    ascending_frames = []
    descending_df = pd.DataFrame(columns=['Time', 'Value'])
    descending_frames = []

    # Initialize condition variable
    ascending = True

    print("starting")
    # Iterate through the DataFrame
    for _, row in df.iterrows():
        value = row['Value']

        if value < min_threshold:
            ascending = True
            if not descending_df.empty:
                descending_frames.append(descending_df)
                descending_df = pd.DataFrame(columns=['Time', 'Value'])
        elif min_threshold <= value <= max_threshold and ascending:
            ascending_df = pd.concat([ascending_df, row.to_frame().T])
        elif value > max_threshold:
            ascending = False
            if not ascending_df.empty:
                ascending_frames.append(ascending_df)
                ascending_df = pd.DataFrame(columns=['Time', 'Value'])
        elif min_threshold <= value <= max_threshold:
            descending_df = pd.concat([descending_df, row.to_frame().T])

    return ascending_frames, descending_frames

def plot_dbscan(df):
    df = df.dropna(subset=['Value'])
    df_filtered = df[(df['Value'] >= min_threshold) & (df['Value'] <= max_threshold)]
    time = df_filtered['Time']
    time_ms = convert_to_ms(time)
    values = df_filtered['Value']
    plt.scatter(time_ms, values, color='blue', label='base')

    df_features = pd.concat([time_ms, df_filtered['Value']], axis=1)

    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=1000, min_samples=100)  # Adjust the parameters as per your requirements
    cluster_labels = dbscan.fit_predict(df_features)

    # Add the cluster labels as a new column in the DataFrame
    df_filtered['Cluster'] = cluster_labels

    # Plot the clusters
    plt.scatter(time_ms, df_filtered['Value'], c=df_filtered['Cluster'], cmap='viridis')
    plt.colorbar(label='Cluster')

    labels = dbscan.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)

def plot_raw(time, values, color):
    plt.plot(time, values, color=color, label='base')


def plot_raw_shifted(time, values, color):
    time_ms = convert_to_ms(time)

    plt.plot(time_ms, values, color=color, label='base')


def plot_base(time, values, color):
    # replace to near or to far with nan
    values[values > out_of_range] = float('NaN')
    # Convert time to milliseconds for better visualization
    time_ms = convert_to_ms(time)

    plt.plot(time_ms, values, color=color, label='base')


def plot_asc_desc(ascending_frames, descending_frames, asc_color, desc_color):
    # ascending values and time
    for asc_df in ascending_frames:
        time_ns_asc = asc_df[['Time']]
        values_asc = asc_df[['Value']]
        time_ms_asc = convert_to_ms(time_ns_asc)
        plt.scatter(time_ms_asc, values_asc, color=asc_color, label='ascending values')
    # descending values and time
    for desc_df in descending_frames:
        time_ns_desc = desc_df[['Time']]
        values_desc = desc_df[['Value']]
        time_ms_desc = convert_to_ms(time_ns_desc)
        plt.scatter(time_ms_desc, values_desc, color=desc_color, label='descending values')


def plot_cut(time, values, color):
    # replace to near or to far with nan
    values[values > max_threshold] = float('NaN')
    values[values < min_threshold] = float('NaN')
    # Convert time to milliseconds for better visualization
    time_ms = convert_to_ms(time)

    plt.plot(time_ms, values, color=color, label='cut')


if __name__ == "__main__":
    args = parse_cli_args()
    file_name = args.file
    plot_type = args.plot_type

    df = read_values(file_name)
    df_shifted = shift_to_zero(df)
    samples_per_second = calculate_samples_per_second(df_shifted)
    if plot_type == "asc_desc":
        ascending_frames, descending_frames = get_asc_desc_frames(df_shifted)
        plot_asc_desc(ascending_frames, descending_frames, 'green', 'red')
    elif plot_type == "asc_desc_base":
        plot_base(df_shifted['Time'], df_shifted['Value'], 'blue')
        ascending_frames, descending_frames = get_asc_desc_frames(df_shifted)
        plot_asc_desc(ascending_frames, descending_frames, 'green', 'red')
    elif plot_type == "base":
        plot_base(df_shifted['Time'], df_shifted['Value'], 'blue')
    elif plot_type == "cut":
        plot_cut(df_shifted['Time'], df_shifted['Value'], 'blue')
    elif plot_type == "cut_base":
        plot_base(df_shifted['Time'], df_shifted['Value'], 'blue')
        plot_cut(df_shifted['Time'], df_shifted['Value'], 'red')
    elif plot_type == "dbscan":
        plot_base(df_shifted['Time'], df_shifted['Value'], 'blue')
        plot_dbscan(df_shifted)
    elif plot_type == 'raw_shifted':
        plot_raw_shifted(df_shifted['Time'], df_shifted['Value'], 'blue')
    else:
        print("Plotting raw data")
        plot_raw_shifted(df['Time'], df['Value'], 'blue')

    plt.xlabel('Time (ms)')
    plt.ylabel('Value')
    filename_without_extension = os.path.splitext(file_name)[0]
    plt.title(filename_without_extension + f" | [samples per second {samples_per_second}]")
    plt.grid(True)
    plt.show()
