import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

import pandas
import pandas as pd
import sklearn.linear_model as lm
import sys

from dataclasses import dataclass
from functools import reduce
from operator import add
from sklearn.metrics import max_error, mean_absolute_error, mean_squared_error
from sklearn.cluster import DBSCAN

out_of_range = 300
max_threshold = 8.84
min_threshold = 6.36


@dataclass
class PredictionResult:
    preditor: str
    num: int
    score: float
    coeff: float
    frame: pandas.DataFrame


@dataclass
class Results:
    ascending_frame_results: list
    descending_frame_results: list


def parse_cli_args():
    parser = argparse.ArgumentParser(description='plot collected data')
    parser.add_argument('file', type=str, help='Serial port which is opened.')
    parser.add_argument('--model', '-m', type=str, choices=['linear', 'ransac', 'linear_ransac'], default='',
                        help='Which model is fit to the data.')
    parser.add_argument('--debug', '-d', action='store_true',
                        help='If debug is set the ascending and descending data frames are shown.')
    parser.add_argument('--debug_outlier', '-do', action='store_true',
                        help='If debug is set the ascending and descending data frames are shown.')
    parser.add_argument('--asc_desc', '-ad', action='store_true',
                        help='If this flag is set is set the ascending and descending data frames are shown.')
    parser.add_argument('--table', '-t', action='store_true',
                        help='If this flag is set the result is export as a csv table.')

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


def filter_values_out_of_range(df):
    # replace to near or to far with nan
    df['Value'][df['Value'] > out_of_range] = float('NaN')
    return df


def convert_to_ms(time):
    return time / 1e6


def convert_to_s(time):
    return time / 1e9


def db_scan(df, debug):
    df = df.dropna(subset=['Value'])
    df_filtered = df[(df['Value'] >= min_threshold) & (df['Value'] <= max_threshold)]
    time = df_filtered['Time']
    time_ms = convert_to_ms(time)

    df_features = pd.concat([time_ms, df_filtered['Value']], axis=1)

    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=1000, min_samples=100)  # Adjust the parameters as per your requirements
    cluster_labels = dbscan.fit_predict(df_features)

    # Add the cluster labels as a new column in the DataFrame
    df_filtered['Cluster'] = cluster_labels

    if debug:
        # # Plot the clusters
        plt.scatter(time_ms, df_filtered['Value'], c=df_filtered['Cluster'], cmap='viridis')
        plt.colorbar(label='Cluster')

    labels = dbscan.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)

    # Initialize empty DataFrames for ascending and descending frames
    ascending_frames = []
    descending_frames = []
    i = 0
    # Get the unique cluster labels
    unique_clusters = pd.Series(cluster_labels).unique()
    for cluster_label in unique_clusters:
        if i % 2 == 0:
            ascending_frames.append(df_filtered[df_filtered['Cluster'] == cluster_label].drop(columns='Cluster'))
        else:
            descending_frames.append(df_filtered[df_filtered['Cluster'] == cluster_label].drop(columns='Cluster'))
        i = i + 1

    return ascending_frames, descending_frames


def threashold_based_splitting(df, debug):
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


def get_asc_desc_frames(df, cluster_algorithm, debug):
    return cluster_algorithm(df, debug)


def plot_base(time, values, color):
    time_s = convert_to_s(time)
    plt.plot(time_s, values, color=color, label='base', linewidth=3)


def linear_reg(ascending_frames, descending_frames):
    # ascending values and time
    ascending_frame_results = []
    for i, asc_df in enumerate(ascending_frames):
        time_ns_asc = asc_df[['Time']]
        values_asc = asc_df[['Value']]
        time_ms_asc = convert_to_ms(time_ns_asc)
        # plt.scatter(time_ms_asc, values_asc, color='green', label='ascending values')
        lin_reg = lm.LinearRegression()
        lin_reg.fit(time_ms_asc, values_asc)
        asc_df['Prediction'] = lin_reg.predict(time_ms_asc)
        ascending_frame_results.append(
            PredictionResult("linear_reg", i, lin_reg.score(time_ms_asc, values_asc), lin_reg.coef_, asc_df))
        print(f"Linear Regression asc Score: {lin_reg.score(time_ms_asc, values_asc)}")
        print(f"Linear Regression asc coef: {lin_reg.coef_}")
    # descending values and time
    descending_frame_results = []
    for i, desc_df in enumerate(descending_frames):
        time_ns_desc = desc_df[['Time']]
        values_desc = desc_df[['Value']]
        time_ms_desc = convert_to_ms(time_ns_desc)
        # plt.scatter(time_ms_desc, values_desc, color='red', label='descending values')
        lin_reg = lm.LinearRegression()
        lin_reg.fit(time_ms_desc, values_desc)
        desc_df['Prediction'] = lin_reg.predict(time_ms_desc)
        descending_frame_results.append(
            PredictionResult("linear_reg", i, lin_reg.score(time_ms_desc, values_desc), lin_reg.coef_, desc_df))
        print(f"Linear Regression des Params: {lin_reg.score(time_ms_desc, values_desc)}")
        print(f"Linear Regression des coef: {lin_reg.coef_}")

    return Results(ascending_frame_results, descending_frame_results)


def plot_predicted_line(title, df_shifted_filterd, result, color, show_asc_desc_frame):
    plt.figure()
    plt.title(title, fontsize=20, fontweight='bold')
    plt.grid(True)
    plot_base(df_shifted_filterd['Time'], df_shifted_filterd['Value'], 'blue')
    for asc_frame_result in result.ascending_frame_results:
        plt.plot(convert_to_s(asc_frame_result.frame['Time']), asc_frame_result.frame['Prediction'], color=color,
                 linewidth=2)
    for desc_frame_result in result.descending_frame_results:
        plt.plot(convert_to_s(desc_frame_result.frame['Time']), desc_frame_result.frame['Prediction'], color=color,
                 linewidth=2)
    if show_asc_desc_frame:
        plot_asc_desc(ascending_frames, descending_frames, 'green', 'yellow')

    plt.xticks(fontsize=18, fontweight='bold')
    plt.yticks(fontsize=18, fontweight='bold')
    plt.xlabel('Time (s)', fontsize=24, fontweight='bold')
    plt.ylabel('Distance (mm)', fontsize=24, fontweight='bold')
    plt.subplots_adjust(left=0.06, right=0.99, bottom=0.075, top=0.96)


def export_as_table(result):

    for i, asc_frame_result in enumerate(result.ascending_frame_results):
        values = asc_frame_result.frame['Value']
        predictions = asc_frame_result.frame['Prediction']
        print(f"max_error:{max_error(values, predictions)}")
        print(f"mean_absolute_error:{mean_absolute_error(values, predictions)}")
        print(f"mean_squared_error:{mean_squared_error(values, predictions)}")
        deviations = values - predictions
        std_dev = deviations.std()

        asc_boxplot_dict = deviations.boxplot(widths=0.4)
        #
        #
        # # ascending results
        # asc_outliers = asc_boxplot_dict['fliers'][0].get_data()[1]
        #
        # asc_whiskers = [whiskers.get_data()[1] for whiskers in asc_boxplot_dict['whiskers']]
        # asc_min = np.min(asc_whiskers)
        # asc_max = np.max(asc_whiskers)
        #
        # asc_mean = asc_boxplot_dict['means'][0].get_data()[1]

        # descending results



    pass


def plot_result_aio(title, result):
    y_min = -0.3
    y_max = 0.2
    fig, axes = plt.subplots(ncols=len(result.ascending_frame_results) + len(result.descending_frame_results),
                             figsize=(12, 6))
    fig.suptitle(title + " Ascending & Descending", fontsize=18)
    fig.subplots_adjust(left=0.05, right=0.97, wspace=0.55, bottom=0.27)
    print(5 * "#" + "_ASCENDING_" + 5 * "#")
    for i, asc_frame_result in enumerate(result.ascending_frame_results):
        values = asc_frame_result.frame['Value']
        predictions = asc_frame_result.frame['Prediction']
        # print(f"max_error:{max_error(values, predictions)}")
        # print(f"mean_absolute_error:{mean_absolute_error(values, predictions)}")
        # print(f"mean_squared_error:{mean_squared_error(values, predictions)}")
        deviation = values - predictions
        boxplot_dict = axes[i].boxplot(deviation, widths=0.4)
        min_val = boxplot_dict['whiskers'][0].get_ydata()[1]
        max_val = boxplot_dict['whiskers'][1].get_ydata()[1]
        mean = boxplot_dict['medians'][0].get_ydata()[0]
        q1_value = boxplot_dict['boxes'][0].get_ydata()[1]
        q3_value = boxplot_dict['boxes'][0].get_ydata()[2]
        outliers = [flier.get_ydata() for flier in boxplot_dict['fliers']]
        num_outliers = reduce(add, [len(outlier) for outlier in outliers])
        print(f"{i} asc Minimum: {min_val:.5f}")
        print(f"{i} asc Maximum: {max_val:.5f}")
        print(f"{i} asc Q1: {q1_value:.5f}")
        print(f"{i} asc Mean: {mean:.5f}")
        print(f"{i} asc Q3: {q3_value:.5f}")
        print(f"{i} asc Number of Outliers: {num_outliers}")
        print("\n")
        axes[i].set_ylim(y_min, y_max)
        axes[i].grid(True)
        axes[i].tick_params(axis='x', labelsize=16)
        axes[i].tick_params(axis='y', labelsize=16)
        axes[i].text(0.5, -0.10, f'Min.: {round(min_val, 3)}', transform=axes[i].transAxes, ha='center',
                     fontsize=16, fontweight='bold')
        axes[i].text(0.5, -0.14, f'Max.: {round(max_val, 3)}',
                     transform=axes[i].transAxes,
                     ha='center', fontsize=16, fontweight='bold')
        axes[i].text(0.5, -0.18, f'Q1: {round(q1_value, 3)}', transform=axes[i].transAxes, ha='center',
                     fontsize=16, fontweight='bold')
        axes[i].text(0.5, -0.22, f'Mean: {round(mean, 3)}', transform=axes[i].transAxes, ha='center',
                     fontsize=16, fontweight='bold')
        axes[i].text(0.5, -0.26, f'Q3: {round(q3_value, 3)}', transform=axes[i].transAxes, ha='center',
                     fontsize=16, fontweight='bold')
        axes[i].text(0.5, -0.30, f'Score: {round(asc_frame_result.score, 3)}', transform=axes[i].transAxes, ha='center',
                     fontsize=16, fontweight='bold')
        axes[i].text(0.5, -0.34, f'Coef.: {round(asc_frame_result.coeff[0][0], 3)} ',
                     transform=axes[i].transAxes,
                     ha='center', fontsize=16, fontweight='bold')
        mse = mean_squared_error(values, predictions)
        axes[i].text(0.5, -0.38, f'MSE: {round(mse, 7) * 1e3:.3f} x 1e-3', transform=axes[i].transAxes, ha='center',
                     fontsize=16, fontweight='bold')
        axes[i].text(0.5, -0.42, f'Outliers: {num_outliers}', transform=axes[i].transAxes, ha='center',
                     fontsize=16, fontweight='bold')
        axes[i].set_ylabel('Values', fontsize=16)
        axes[i].set_title(f'Ascending Deviation {i + 1}\n', fontsize=16)

    print(5 * "#" + "_DESCENDING_" + 5 * "#")

    for i, desc_frame_result in enumerate(result.descending_frame_results, start=len(result.ascending_frame_results)):
        values = desc_frame_result.frame['Value']
        predictions = desc_frame_result.frame['Prediction']
        # print(f"max_error:{max_error(values, predictions)}")
        # print(f"mean_absolute_error:{mean_absolute_error(values, predictions)}")
        # print(f"mean_squared_error:{mean_squared_error(values, predictions)}")
        deviation = values - predictions
        boxplot_dict = axes[i].boxplot(deviation, widths=0.4)
        min_val = boxplot_dict['whiskers'][0].get_ydata()[1]
        max_val = boxplot_dict['whiskers'][1].get_ydata()[1]
        mean = boxplot_dict['medians'][0].get_ydata()[0]
        q1_value = boxplot_dict['boxes'][0].get_ydata()[1]
        q3_value = boxplot_dict['boxes'][0].get_ydata()[2]
        outliers = [flier.get_ydata() for flier in boxplot_dict['fliers']]
        num_outliers = reduce(add, [len(outlier) for outlier in outliers])
        print(f"{i} asc Minimum: {min_val:.5f}")
        print(f"{i} asc Maximum: {max_val:.5f}")
        print(f"{i} asc Mean: {mean:.5f}")
        print(f"{i} asc Number of Outliers: {num_outliers}")
        print("\n")
        axes[i].set_ylim(y_min, y_max)
        axes[i].grid(True)
        axes[i].tick_params(axis='x', labelsize=16)
        axes[i].tick_params(axis='y', labelsize=16)
        axes[i].text(0.5, -0.10, f'Min.: {round(min_val, 3)}', transform=axes[i].transAxes, ha='center',
                     fontsize=16, fontweight='bold')
        axes[i].text(0.5, -0.14, f'Max.: {round(max_val, 3)}',
                     transform=axes[i].transAxes,
                     ha='center', fontsize=16, fontweight='bold')
        axes[i].text(0.5, -0.18, f'Q1: {round(q1_value, 3)}', transform=axes[i].transAxes, ha='center',
                     fontsize=16, fontweight='bold')
        axes[i].text(0.5, -0.22, f'Mean: {round(mean, 3)}', transform=axes[i].transAxes, ha='center',
                     fontsize=16, fontweight='bold')
        axes[i].text(0.5, -0.26, f'Q3: {round(q3_value, 3)}', transform=axes[i].transAxes, ha='center',
                     fontsize=16, fontweight='bold')
        axes[i].text(0.5, -0.30, f'Score: {round(desc_frame_result.score, 3)}', transform=axes[i].transAxes,
                     ha='center',
                     fontsize=16, fontweight='bold')
        axes[i].text(0.5, -0.34, f'Coef.: {round(desc_frame_result.coeff[0][0], 3)}',
                     transform=axes[i].transAxes,
                     ha='center', fontsize=16, fontweight='bold')
        mse = mean_squared_error(values, predictions)
        axes[i].text(0.5, -0.38, f'MSE: {round(mse, 7) * 1e3:.3f} x 1e-3', transform=axes[i].transAxes, ha='center',
                     fontsize=16, fontweight='bold')
        axes[i].text(0.5, -0.42, f'Outliers: {num_outliers}', transform=axes[i].transAxes, ha='center',
                     fontsize=16, fontweight='bold')
        axes[i].set_ylabel('Values', fontsize=16)
        axes[i].set_title(f'Descending Deviation {i + 1 - len(result.ascending_frame_results)}\n', fontsize=16)


def plot_result(title, result):
    y_min = -0.3
    y_max = 0.2
    fig, axes = plt.subplots(ncols=len(result.ascending_frame_results))
    fig.suptitle(title + " Ascending", fontsize=18)
    fig.subplots_adjust(left=0.092, right=0.92, wspace=0.29, bottom=0.2)
    for i, asc_frame_result in enumerate(result.ascending_frame_results):
        values = asc_frame_result.frame['Value']
        predictions = asc_frame_result.frame['Prediction']
        print(f"max_error:{max_error(values, predictions)}")
        print(f"mean_absolute_error:{mean_absolute_error(values, predictions)}")
        print(f"mean_squared_error:{mean_squared_error(values, predictions)}")
        deviation = values - predictions
        axes[i].boxplot(deviation, widths=0.4)
        axes[i].set_ylim(y_min, y_max)
        axes[i].grid(True)
        axes[i].text(0.5, -0.10, f'Score: {round(asc_frame_result.score, 3)}', transform=axes[i].transAxes, ha='center',
                     fontsize=16)
        axes[i].text(0.5, -0.14, f'Coefficients {round(asc_frame_result.coeff[0][0], 7) * 1e3:.3f} x 1e-3',
                     transform=axes[i].transAxes,
                     ha='center', fontsize=16)
        mse = mean_squared_error(values, predictions)
        axes[i].text(0.5, -0.18, f'MSE {round(mse, 7) * 1e3:.3f} x 1e-3', transform=axes[i].transAxes, ha='center',
                     fontsize=16)
        axes[i].set_ylabel('Values')
        axes[i].set_title(f'Boxplot of Deviation {i + 1}')

    fig, axes = plt.subplots(ncols=len(result.descending_frame_results), figsize=(12, 6))
    fig.suptitle(title + " Descending", fontsize=18)
    fig.subplots_adjust(left=0.092, right=0.92, wspace=0.29, bottom=0.2)
    for i, desc_frame_result in enumerate(result.descending_frame_results):
        values = desc_frame_result.frame['Value']
        predictions = desc_frame_result.frame['Prediction']
        print(f"max_error:{max_error(values, predictions)}")
        print(f"mean_absolute_error:{mean_absolute_error(values, predictions)}")
        print(f"mean_squared_error:{mean_squared_error(values, predictions)}")
        deviation = values - predictions
        axes[i].boxplot(deviation, widths=0.4)
        axes[i].set_ylim(y_min, y_max)
        axes[i].grid(True)
        axes[i].text(0.5, -0.10, f'Score: {round(desc_frame_result.score, 3)}', transform=axes[i].transAxes,
                     ha='center',
                     fontsize=16)
        axes[i].text(0.5, -0.14, f'Coefficients {round(desc_frame_result.coeff[0][0], 7) * 1e3:.3f} x 1e-3',
                     transform=axes[i].transAxes,
                     ha='center', fontsize=16)
        mse = mean_squared_error(values, predictions)
        axes[i].text(0.5, -0.18, f'MSE {round(mse, 7) * 1e3:.3f} x 1e-3', transform=axes[i].transAxes, ha='center',
                     fontsize=16)
        axes[i].set_ylabel('Values')
        axes[i].set_title(f'Boxplot of Deviation {i + 1}')


def ransac(ascending_frames, descending_frames, debug):
    # ascending values and time
    ascending_frame_results = []
    for i, asc_df in enumerate(ascending_frames):
        time_ns_asc = asc_df[['Time']]
        values_asc = asc_df[['Value']]
        time_s_asc = convert_to_s(time_ns_asc)
        ransac = lm.RANSACRegressor()
        ransac.fit(time_s_asc, values_asc)
        asc_df['Prediction'] = ransac.predict(time_s_asc)
        ascending_frame_results.append(
            PredictionResult("ransac", i, ransac.score(time_s_asc, values_asc), ransac.estimator_.coef_, asc_df))
        if (debug):
            print(f"Ransac asc Score: {ransac.score(time_s_asc, values_asc)}")
            print(f"Ransac asc coef: {ransac.estimator_.coef_}")
            # plt.scatter(time_ms_asc, values_asc, color='green', label='ascending values')
            inlier_mask = ransac.inlier_mask_
            outlier_mask = np.logical_not(inlier_mask)
            plt.scatter(time_s_asc[inlier_mask], values_asc[inlier_mask], c='steelblue', marker='o', label='Inliers')
            plt.scatter(time_s_asc[outlier_mask], values_asc[outlier_mask], c='limegreen', marker='s',
                        label='Outliers')
    # descending values and time
    descending_frame_results = []
    for i, desc_df in enumerate(descending_frames):
        time_ns_desc = desc_df[['Time']]
        values_desc = desc_df[['Value']]
        time_s_desc = convert_to_s(time_ns_desc)
        ransac = lm.RANSACRegressor()
        ransac.fit(time_s_desc, values_desc)
        desc_df['Prediction'] = ransac.predict(time_s_desc)
        descending_frame_results.append(
            PredictionResult("ransac", i, ransac.score(time_s_desc, values_desc), ransac.estimator_.coef_, desc_df))
        if (debug):
            print(f"Ransac des Params: {ransac.score(time_s_desc, values_desc)}")
            print(f"Ransac des coef: {ransac.estimator_.coef_}")
            # plt.scatter(time_ms_desc, values_desc, color='red', label='descending values')
            inlier_mask = ransac.inlier_mask_
            outlier_mask = np.logical_not(inlier_mask)
            plt.scatter(time_s_desc[inlier_mask], values_desc[inlier_mask], c='steelblue', marker='o', label='Inliers')
            plt.scatter(time_s_desc[outlier_mask], values_desc[outlier_mask], c='limegreen', marker='s',
                        label='Outliers')

    return Results(ascending_frame_results, descending_frame_results)


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


if __name__ == "__main__":
    args = parse_cli_args()
    file_name = args.file
    model = args.model
    debug = args.debug
    debug_outlier = args.debug_outlier
    show_asc_desc_frame = args.asc_desc

    df = read_values(file_name)
    samples_per_second = calculate_samples_per_second(df)
    filename_without_extension = os.path.splitext(file_name)[0]
    title = filename_without_extension + f" [{round(samples_per_second, 1)} sps]"
    df_shifted = shift_to_zero(df)
    df_shifted_filterd = filter_values_out_of_range(df_shifted)
    ascending_frames, descending_frames = get_asc_desc_frames(df_shifted_filterd, db_scan, debug_outlier)
    # if model == "linear":
    #     result = linear_reg(ascending_frames, descending_frames)
    #     plot_predicted_line(title, df_shifted_filterd, result, 'red', show_asc_desc_frame)
    #     plot_result(title, result)
    # elif model == "linear_ransac":
    #     result = linear_reg(ascending_frames, descending_frames)
    #     plot_predicted_line(title, df_shifted_filterd, result, 'red', show_asc_desc_frame)
    #     plot_result(title, result)
    #     plot_predicted_line(title, df_shifted_filterd, result, 'red', show_asc_desc_frame)
    #     plot_result(title, result)
    # else:
    #     result = ransac(ascending_frames, descending_frames, debug)
    #     plot_predicted_line(title, df_shifted_filterd, result, 'red', show_asc_desc_frame)
    #     plot_result(title, result)

    if model == "ransac":
        result = ransac(ascending_frames, descending_frames, debug)
        export_as_table(result)

    plt.show()
