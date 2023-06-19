import argparse
import matplotlib.pyplot as plt
import os
import pandas as pd
import sys

def parse_cli_args():
    parser = argparse.ArgumentParser(description='plot collected data')
    parser.add_argument('file', type=str, help='Serial port which is opened.')
    parser.add_argument('--plot_type', '-t', type=str, choices=['asc', 'ranged', 'trimm', 'both' ], default='', help='How data is plotted.')


    return parser.parse_args(args=None if sys.argv[1:] else ["--help"])

def remove_out_of_range(file_name):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_name)

    # Extract the time and value columns from the DataFrame
    time_ns = df['Time']
    values = df['Value']
    #replace to near or to far with nan
    values[values > 300] = float('NaN')
    # calculate samples per second
    start = time_ns.head(1).values[0]
    stop = time_ns.tail(1).values[0]
    samples_per_second = len(values)/((stop - start) / 1e9)
    # shift start everything relative to start:
    time_ns = time_ns - start

    # Convert time to milliseconds for better visualization
    time_ms = time_ns / 1e6

    return time_ms, values, samples_per_second

def asc(file_name):
        # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_name)

    start = df['Time'].head(1).values[0]
    # shift start everything relative to start:
    df['Time'] = df['Time'] - start

    # Define min and max thresholds
    min_threshold = 6.375
    max_threshold = 8.88

    # Initialize empty DataFrames for A and B
    ascending_df = pd.DataFrame(columns=['Time', 'Value'])
    descending_df = pd.DataFrame(columns=['Time', 'Value'])

    # Initialize condition variable
    ascending = True

    print("starting")
    # Iterate through the DataFrame
    for _, row in df.iterrows():
        value = row['Value']

        if value < min_threshold:
            ascending = True
        elif min_threshold <= value <= max_threshold and ascending:
            ascending_df = pd.concat([ascending_df, row.to_frame().T])
        elif value > max_threshold:
            ascending = False
        elif min_threshold <= value <= max_threshold:
            descending_df = pd.concat([descending_df, row.to_frame().T])

    # ascending values and time
    time_ns_asc = ascending_df['Time']
    values_asc = ascending_df['Value']
    time_ms_asc = time_ns_asc / 1e6
    # descending values and time
    time_ns_desc = descending_df['Time']
    values_desc = descending_df['Value']
    time_ms_desc = time_ns_desc / 1e6
    # Convert time to milliseconds for better visualization


    print(values_asc)

    return time_ms_desc, values_desc, samples_per_second


def trimm(file_name):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_name)

    # Extract the time and value columns from the DataFrame
    time_ns = df['Time']
    values = df['Value']
    #replace to near or to far with nan
    values[values > 8.9] = float('NaN')
    values[values < 6.375] = float('NaN')
    # calculate samples per second
    start = time_ns.head(1).values[0]
    stop = time_ns.tail(1).values[0]
    samples_per_second = len(values)/((stop - start) / 1e9)
    # shift start everything relative to start:
    time_ns = time_ns - start

    # Convert time to milliseconds for better visualization
    time_ms = time_ns / 1e6

    return time_ms, values, samples_per_second

def plot(values, time_ns, color):
    # Create the plot
    plt.plot(time_ms, values)


if __name__=="__main__":
    args = parse_cli_args()
    file_name = args.file
    plot_type = args.plot_type

    samples_per_second = 0.0
    if plot_type == "trimm":
        time_ms, values, samples_per_second = trimm(file_name)
        plt.plot(time_ms, values)
    elif plot_type == "both":
        time_ms, values, samples_per_second = trimm(file_name)
        plt.plot(time_ms, values, color='green', label='trimmed')
        time_ms, values, _ = remove_out_of_range(file_name)
        plt.plot(time_ms, values, color='red', label='out_of_range_removed')
    elif plot_type == "asc":
        time_ms, values, _ = remove_out_of_range(file_name)
        plt.plot(time_ms, values, color='green', label='out_of_range_removed')
        time_ms, values, samples_per_second = asc(file_name)
        plt.scatter(time_ms, values, color='red', label='out_of_range_removed')
    else :
        time_ms, values, samples_per_second = remove_out_of_range(file_name)
        plt.plot(time_ms, values)

    plt.xlabel('Time (ms)')
    plt.ylabel('Value')
    filename_without_extension = os.path.splitext(file_name)[0]
    plt.title(filename_without_extension + f" | [samples per second {samples_per_second}]")
    plt.grid(True)
    plt.show()
