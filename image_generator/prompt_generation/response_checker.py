import os
import ast
from datetime import datetime

TODAY = datetime.now().strftime("%Y%m%d")


# Function to check if the content is a valid list of strings
def is_valid_response(content):
    content = content.strip()

    # Check if it starts with '[' and ends with ']'
    if not (content.startswith("[") and content.endswith("]")):
        return False

    try:
        # Try to parse the content as a Python list
        parsed_list = ast.literal_eval(content)
        # Ensure it's a list and all elements are strings
        return isinstance(parsed_list, list) and all(isinstance(item, str) for item in parsed_list)
    except (SyntaxError, ValueError):
        return False  # Not a valid Python list


# Function to attempt fixing an invalid response
def fix_syntax_response(content):
    if not content.startswith("["):
        idx = content.index("[")
        if idx != -1:
            content = content[idx:]

    if not content.endswith("]"):
        if not content.endswith("\n]"):
            if not content.endswith("\"\n]"):
                content = content[:-1] + "\"\n]"
            else:
                content = content[:-1] + "\n]"
        else:
            content = content[:-1] + "]"
    # it surely ends with ]
    elif not content[:-1].endswith("\n"):
        if not content[:-1].endswith("\"\n"):
            content = content[:-2] + "\"\n]"
        else:
            content = content[:-2] + "\n]"
    # it surely ends with \n] 
    elif not content[:-2].endswith("\""):
        content = content[:-3] + "\"\n]"
    else:
        return False
    
    return content.replace("\"\"", "\"")


def check_syntax(input_directory, output_filepath):
    invalid_files = []  # List to store paths of invalid response files
    total = 0

    # Process each file in the directory
    for filename in os.listdir(input_directory):
        if filename.endswith("_response.txt"):
            total += 1
            file_path = os.path.join(input_directory, filename)

            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()

            # Check if the response format is valid
            if not is_valid_response(content):
                new_content = fix_syntax_response(content)
                if not new_content:
                    invalid_files.append(file_path)  # Store path of invalid files
                    # print(f"{file_path}")
                else:
                    with open(file_path, "w", encoding="utf-8") as file:
                        file.write(new_content)

    # Save invalid file paths to a report file
    with open(output_filepath, "w", encoding="utf-8") as file:
        for path in invalid_files:
            file.write(path + "\n")

    print("\n\n")
    print(f"Syntax validation completed. {len(invalid_files)}/{total} invalid files saved to: {output_filepath}")


def has_five_elements(content):
    to_print = False
    try:
        parsed_content = ast.literal_eval(content)
        for x in parsed_content:
            if x == "":
                to_print = True
                break
        
        # Check if it is a list with exactly 5 elements
        if isinstance(parsed_content, list) and len(parsed_content) == 5:
            # Check if all elements are non-empty strings
            return to_print, all(isinstance(item, str) and item.strip() for item in parsed_content)

        return to_print, False  # Not a list with 5 valid strings

    except (SyntaxError, ValueError):
        return to_print, False  # Not a valid Python list


def check_num_of_elements(input_directory, output_filepath):
    invalid_files = []  # List to store paths of invalid response files
    total = 0

    # Process each file in the directory
    for filename in os.listdir(input_directory):
        if filename.endswith("_response.txt"):
            total += 1
            file_path = os.path.join(input_directory, filename)

            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                to_print, has_file_elem = has_five_elements(content)
                if to_print:
                    print(file_path)
                if not has_file_elem:
                    with open(file_path, "w", encoding="utf-8") as file:
                        file.write(content)
    
    # Save invalid file paths to a report file
    with open(output_filepath, "w", encoding="utf-8") as file:
        for path in invalid_files:
            file.write(path + "\n")

    print("\n\n")
    print(f"Num of elements validation completed. {len(invalid_files)}/{total} invalid files saved to: {output_filepath}")

def check_response_syntax(input_directory, output_filepath = "/data/chromei/domestic_image_generator/prompt_generation/invalid_responses.txt"):
    check_syntax(input_directory, output_filepath.replace(".txt", "_syntax.txt"))

    check_num_of_elements(input_directory, output_filepath.replace(".txt", "_elements.txt"))
    
    
def main(input_directory, output_filepath):
    check_syntax(input_directory, output_filepath.replace(".txt", f"_syntax_{TODAY}.txt"))

    check_num_of_elements(input_directory, output_filepath.replace(".txt", f"_elements_{TODAY}.txt"))


if __name__ == "__main__":
    # Set the directory containing the response files
    input_directory = "../data/output_folder"
    output_filepath = "./invalid_responses.txt"
    main(input_directory, output_filepath)

    # Example cases
    # valid_string = '["A", "B", "C", "D", "E"]'
    # invalid_string1 = '["A", "", "C", "D", "E"]'  # Empty string present
    # invalid_string2 = '["A", "B", "C", "D"]'  # Only 4 elements
    # invalid_string3 = '["A", "B", 123, "D", "E"]'  # Non-string element

    # print(has_five_elements(valid_string))      # True
    # print(has_five_elements(invalid_string1))   # False (empty string)
    # print(has_five_elements(invalid_string2))   # False (less than 5 elements)
    # print(has_five_elements(invalid_string3))   # False (non-string element)
    