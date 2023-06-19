import argparse
import matplotlib.pyplot as plt
import os
import pandas as pd
import sys

def parse_cli_args():
    parser = argparse.ArgumentParser(description='plot collected data')
    parser.add_argument('file', type=str, help='Serial port which is opened.')

    return parser.parse_args(args=None if sys.argv[1:] else ["--help"])

def plot_data(file_name):
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
    # Convert time to milliseconds for better visualization
    time_ms = time_ns / 1e6


    # Create the plot
    plt.plot(time_ms, values)
    plt.xlabel('Time (ms)')
    plt.ylabel('Value')
    filename_without_extension = os.path.splitext(file_name)[0]
    plt.title(filename_without_extension + f" | [samples per second {samples_per_second}]")

    # Define min and max thresholds
    min_threshold = 6.375
    max_threshold = 8.9

    # Initialize empty DataFrames for A and B
    dataframe_A = pd.DataFrame(columns=['Time', 'Value'])
    dataframe_B = pd.DataFrame(columns=['Time', 'Value'])

    # Initialize condition variable
    cond = False

    print("starting")
    # Iterate through the DataFrame
    for _, row in df.iterrows():
        value = row['Value']

        if value < min_threshold:
            cond = True
            if not dataframe_B.empty:
                dataframe_B = pd.concat([dataframe_B, row.to_frame().T])

        if min_threshold <= value <= max_threshold and cond:
            dataframe_A = pd.concat([dataframe_A, row.to_frame().T])

        if value > max_threshold:
            cond = False
            if not dataframe_A.empty:
                dataframe_A = pd.concat([dataframe_A, row.to_frame().T])

        if min_threshold <= value <= max_threshold:
            dataframe_B = pd.concat([dataframe_B, row.to_frame().T])

    print("plotting")

    # Plotting
    plt.plot(dataframe_A['Time'], dataframe_A['Value'], color='red', label='DataFrame A')
    plt.plot(dataframe_B['Time'], dataframe_B['Value'], color='green', label='DataFrame B')

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

    pos_inclination = []


    # Create the plot
    plt.plot(time_ms, values)

    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

if __name__=="__main__":
    args = parse_cli_args()
    file_name = args.file
    plot_data(file_name)


