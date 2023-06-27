import csv
import os
import random

# file_path = 'table.csv'
# column_values = ['data_name', 'avg score', 'avg coeff', 'avg mse']
# random_values = [random.randint(0,100) for _ in range(4)]
# # Check if the file exists
# file_exists = os.path.isfile(file_path)
#
# if not file_exists:
#     # File does not exist, create it and write the header
#     with open(file_path, 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(column_values)
#
# # Append a column with values to the existing file
# with open(file_path, 'a', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(random_values)


# by Yaxin + chatgpt
def save_dict_to_csv(dictionary, file_path):

    if not os.path.isfile(file_path):
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=dictionary.keys())
            writer.writeheader()
            writer.writerow(dictionary)
    else:
        # # append_row_to_csv
        # with open(file_path, 'r', newline='') as csvfile:
        #     reader = csv.reader(csvfile)
        #     fieldnames = next(reader)
        # with open(file_path, 'a', newline='') as csvfile:
        #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        #     writer.writerow(dictionary)

        # Read the existing CSV file
        with open(file_path, 'r') as file:
            reader = csv.DictReader(file)
            rows = list(reader)

        # Add the new columns to the fieldnames
        fieldnames = reader.fieldnames + list(dictionary.keys())

        # Add the column values to each row
        for row in rows:
            row.update(dictionary)

        # Write the modified data back to the CSV file
        with open(file_path, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerows(rows)

