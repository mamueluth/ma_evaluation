import argparse
import matplotlib.pyplot as plt
import os
import pandas as pd
import sklearn.linear_model as lm
import sys

out_of_range = 300
max_threshold = 8.85
min_threshold = 6.375

def parse_cli_args():
    parser = argparse.ArgumentParser(description='plot collected data')
    parser.add_argument('file', type=str, help='Serial port which is opened.')
    parser.add_argument('--model', '-m', type=str, choices=['linear', 'ransac', 'linear_ransac'], default='', help='Wich model is fit to the data.')
    parser.add_argument('--debug', '-d', action='store_true', help='If debug is set the ascending and descending data frames are shown.')


    return parser.parse_args(args=None if sys.argv[1:] else ["--help"])

def read_values(file_name):
    # Read the CSV file into a pandas DataFrame
    return pd.read_csv(file_name)

def calculate_samples_per_second(df):
    start = df['Time'].head(1).values[0]
    stop = df['Time'].tail(1).values[0]
    samples_per_second = len(df['Value'])/((stop - start) / 1e9)
    return samples_per_second

def shift_to_zero(df):
    start_time = df['Time'].head(1).values[0]
    # shift start everything relative to start:
    shifted_df = df.copy()
    shifted_df['Time'] = shifted_df['Time'] - start_time
    return shifted_df

def filter_values_out_of_range(df):
    #replace to near or to far with nan
    df['Value'][df['Value'] > out_of_range] = float('NaN')
    return df

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

def plot_base(time, values, color):
    time_ms = convert_to_ms(time)
    plt.plot(time_ms, values, color=color, label='base')

def plot_linear_reg(ascending_frames, descending_frames, color):
    # ascending values and time
    for asc_df in ascending_frames:
        time_ns_asc = asc_df[['Time']]
        values_asc = asc_df[['Value']]
        time_ms_asc = convert_to_ms(time_ns_asc)
        # plt.scatter(time_ms_asc, values_asc, color='green', label='ascending values')
        lin_reg = lm.LinearRegression()
        lin_reg.fit(time_ms_asc, values_asc)
        plt.plot(time_ms_asc, lin_reg.predict(time_ms_asc), color=color, linewidth=1)
        print(f"Asc Score: {lin_reg.score(time_ms_asc, values_asc)}")
        print(f"Asc coef: {lin_reg.coef_}")
    # descending values and time
    for desc_df in descending_frames:
        time_ns_desc = desc_df[['Time']]
        values_desc = desc_df[['Value']]
        time_ms_desc = convert_to_ms(time_ns_desc)
        # plt.scatter(time_ms_desc, values_desc, color='red', label='descending values')
        lin_reg = lm.LinearRegression()
        lin_reg.fit(time_ms_desc, values_desc)
        plt.plot(time_ms_desc, lin_reg.predict(time_ms_desc), color=color, linewidth=1)
        print(f"Des Params: {lin_reg.score(time_ms_desc, values_desc)}")
        print(f"Des coef: {lin_reg.coef_}")

def plot_ransac(ascending_frames, descending_frames, color):
    # ascending values and time
    for asc_df in ascending_frames:
        time_ns_asc = asc_df[['Time']]
        values_asc = asc_df[['Value']]
        time_ms_asc = convert_to_ms(time_ns_asc)
        # plt.scatter(time_ms_asc, values_asc, color='green', label='ascending values')
        ransac = lm.RANSACRegressor()
        ransac.fit(time_ms_asc, values_asc)
        plt.plot(time_ms_asc, ransac.predict(time_ms_asc), color=color, linewidth=1)
        print(f"Asc Score: {ransac.score(time_ms_asc, values_asc)}")
        print(f"Asc coef: {ransac.estimator_.coef_}")
    # descending values and time
    for desc_df in descending_frames:
        time_ns_desc = desc_df[['Time']]
        values_desc = desc_df[['Value']]
        time_ms_desc = convert_to_ms(time_ns_desc)
        # plt.scatter(time_ms_desc, values_desc, color='red', label='descending values')
        ransac = lm.RANSACRegressor()
        ransac.fit(time_ms_desc, values_desc)
        plt.plot(time_ms_desc, ransac.predict(time_ms_desc), color=color, linewidth=1)
        print(f"Des Params: {ransac.score(time_ms_desc, values_desc)}")
        print(f"Des coef: {ransac.estimator_.coef_}")

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

if __name__=="__main__":
    args = parse_cli_args()
    file_name = args.file
    model = args.model
    debug = args.debug

    df = read_values(file_name)
    samples_per_second = calculate_samples_per_second(df)
    df_shifted = shift_to_zero(df)
    df_shifted_filterd = filter_values_out_of_range(df_shifted)
    plot_base(df_shifted_filterd['Time'], df_shifted_filterd['Value'] , 'blue')
    ascending_frames, descending_frames = get_asc_desc_frames(df_shifted_filterd)
    if model == "linear":
        plot_linear_reg(ascending_frames, descending_frames, 'red')
    elif model == "linear_ransac":
        plot_linear_reg(ascending_frames, descending_frames, 'orange')
        plot_ransac(ascending_frames, descending_frames, 'red')
    else :
        plot_ransac(ascending_frames, descending_frames, 'red')

    if(debug):
        plot_asc_desc(ascending_frames, descending_frames, 'green', 'yellow')

    plt.xlabel('Time (ms)')
    plt.ylabel('Value')
    filename_without_extension = os.path.splitext(file_name)[0]
    plt.title(filename_without_extension + f" | [samples per second {samples_per_second}]")
    plt.grid(True)
    plt.show()
