import argparse
import pandas as pd
import sys
def parse_cli_args():
    parser = argparse.ArgumentParser(description='plot collected data')
    parser.add_argument('file', type=str, help='Serial port which is opened.')
    parser.add_argument('output', type=str, help='Serial port which is opened.')
    return parser.parse_args(args=None if sys.argv[1:] else ["--help"])


if __name__ =="__main__":
    args = parse_cli_args()
    input_file = args.file
    output_file = args.output
    if not output_file:
        output_file = 'output.csv'

    # Read the input CSV file
    df = pd.read_csv(input_file)
    # Transpose the DataFrame
    df_transposed = df.transpose()
    # Write the transposed DataFrame to a new CSV file
    df_transposed.to_csv(output_file, index=True, header=False)
