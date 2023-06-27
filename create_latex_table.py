import argparse
import pandas as pd
import sys
def parse_cli_args():
    parser = argparse.ArgumentParser(description='plot collected data')
    parser.add_argument('file', type=str, help='Serial port which is opened.')
    parser.add_argument('--table_name', '-tn', default='table.txt',
                        help='The file path to save the csv table in.')
    return parser.parse_args(args=None if sys.argv[1:] else ["--help"])


if __name__ =="__main__":
    args = parse_cli_args()
    input_file = args.file
    table_name = args.table_name

    # Read the input CSV file
    df = pd.read_csv(input_file)
    # Transpose the DataFrame
    df_transposed = df.transpose()
    latex_table = df_transposed.to_latex(index=True)
    with open(table_name, 'w') as f:
        f.write(latex_table)

    # # Write the transposed DataFrame to a new CSV file
    # df_transposed.to_csv(output_file, index=True, header=False)
