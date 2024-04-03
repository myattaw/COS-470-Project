import os


def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')


def count_lines(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return sum(1 for _ in file)
    except FileNotFoundError:
        return 0


def process_messages(input_file_path, output_file_path):
    try:
        lines_to_skip = count_lines(output_file_path)

        with open(input_file_path, 'r', encoding='utf-8') as input_file:
            lines = input_file.readlines()

        output_lines = []

        for i, line in enumerate(lines):
            if i < lines_to_skip:
                continue  # Skip lines that have already been processed

            message = line.strip()
            decision = input(f"'{message}' (0 = bad, 1 = hate, 2 = neutral): ")

            while decision not in ['0', '1', '2']:
                print("Please enter 0 or 1.")
                decision = input(f"{message} (rate: 0 = good, 1 = bad): ")

            output_lines.append(f"{decision} {message}\n")

            # Save file every 5 inputs to make sure we don't lose our data
            if (i + 1) % 5 == 0:
                print(f"\n\033[94mUpdated Text File\033[0m\n")
                with open(output_file_path, 'a', encoding='utf-8') as output_file:
                    clear_console()
                    for output_line in output_lines:
                        output_file.write(output_line)
                output_lines = []

        if output_lines:
            with open(output_file_path, 'a', encoding='utf-8') as output_file:
                for output_line in output_lines:
                    output_file.write(output_line)

    except FileNotFoundError:
        print(f"Error: File '{input_file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


def read_and_print_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for line in lines:
                print(line.strip())  # Stripping to remove newline characters
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    input_file_path = 'extracted_bad_messages.txt'  # Replace with the path to your text file
    output_file_path = 'verified_bad_messages.txt'  # Replace with the desired output file name
    process_messages(input_file_path, output_file_path)
