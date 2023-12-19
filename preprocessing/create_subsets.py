import os
import glob
import random

input_directory = "/srv/scratch2/grosjean/Masterarbeit/data"
output_directory = "/srv/scratch2/grosjean/Masterarbeit/data_subsets"

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Get all .txt files in the input directory
file_paths = glob.glob(os.path.join(input_directory, '*.txt'))

lines_per_file = 512

# Calculate the total number of subset files
total_subset_files = sum(len(open(file_path).readlines()) // lines_per_file for file_path in file_paths)

# Generate unique random numbers
random_numbers = set()
while len(random_numbers) < total_subset_files:
    random_numbers.add(random.randint(0, 9999))

# Convert the set to a list and shuffle it
random_numbers = list(random_numbers)
random.shuffle(random_numbers)

random_numbers_iter = iter(random_numbers)

for file_path in file_paths:
    with open(file_path, 'r') as input_file:
        lines = input_file.readlines()

    # Get the base file name without the extension
    base_file_name = os.path.splitext(os.path.basename(file_path))[0]

    # Split the lines into files with 512 lines each
    for i in range(0, len(lines), lines_per_file):
        subset_lines = lines[i:i+lines_per_file]

        # Only write the file if there are exactly 512 lines
        if len(subset_lines) == lines_per_file:
            # Create a new subset file with a unique random number prefix
            random_prefix = next(random_numbers_iter)
            subset_file_path = os.path.join(output_directory, f'{random_prefix}_{base_file_name}.txt')
            with open(subset_file_path, 'w') as subset_file:
                subset_file.writelines(subset_lines)