file_path = 'cat304_.txt'  # Update with the path to your file
output_file_path = 'processed_cat304.csv'  # Update with your desired output file path


# Initialize variables for tracking
current_subject_id = 1
previous_first_col_value = '1'  # Assuming the file starts with 1 as the first value in the first column

processed_lines = []

with open(file_path, 'r') as file:
    for line in file:
        columns = line.strip().split(' ')

        # Check if the first column is '1' and the previous value was not '1', then increment subject ID
        if columns[0] == '1' and previous_first_col_value != '1':
            current_subject_id += 1
        
        # Update the previous first column value
        previous_first_col_value = columns[0]

        # Add the subject ID as the first column and save the modified line
        processed_lines.append(f"{current_subject_id} {line.strip()}")

# Save the processed lines to a new file
with open(output_file_path, 'w') as file:
    file.write('\n'.join(processed_lines))

print(f"Processed file saved as {output_file_path}")
