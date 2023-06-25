import csv
import os
import random

file_path = 'table.csv'
column_values = ['data_name', 'avg score', 'avg coeff', 'avg mse']
random_values = [random.randint(0,100) for _ in range(4)]
# Check if the file exists
file_exists = os.path.isfile(file_path)

if not file_exists:
    # File does not exist, create it and write the header
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(column_values)

# Append a column with values to the existing file
with open(file_path, 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(random_values)
