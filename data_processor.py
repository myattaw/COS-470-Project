from collections import OrderedDict


def remove_duplicates(input_file, output_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()

        # Use OrderedDict to preserve insertion order and remove duplicates
        unique_lines = list(OrderedDict.fromkeys(lines))

        with open(output_file, 'w', encoding='utf-8') as outfile:
            outfile.writelines(unique_lines)

        print(f"Unique lines (preserving order) written to {output_file}")

    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")


# Example usage:
input_file_path = 'chat_message_data.txt'  # Change this to the path of your input file
output_file_path = 'chat_message_data_2.txt'  # Change this to the desired output file path

remove_duplicates(input_file_path, output_file_path)
