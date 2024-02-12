def read_and_print_file(file_path):
    try:
        with open(input_file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for line in lines:
                print(line.strip())  # Stripping to remove newline characters
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    input_file_path = 'chat_message_data.txt'  # Replace with the path to your text file
    read_and_print_file(input_file_path)
