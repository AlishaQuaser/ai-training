import json
import argparse
from collections import defaultdict
import os


def check_and_remove_common_docs(json_file_1_path, json_file_2_path, output_file_path=None):
    """
    Check for common documents between two JSON files, delete the common ones,
    and save the remaining unique documents to a new file.

    Args:
        json_file_1_path (str): Path to the first JSON file to check
        json_file_2_path (str): Path to the second JSON file to check
        output_file_path (str, optional): Path to save the unique documents to

    Returns:
        tuple: (common_docs, unique_data)
    """
    try:
        with open(json_file_1_path, 'r') as file:
            data1 = json.load(file)

        with open(json_file_2_path, 'r') as file:
            data2 = json.load(file)

        if not isinstance(data1, list) or not isinstance(data2, list):
            print(f"Error: Expected JSON arrays/lists, but got {type(data1).__name__} and {type(data2).__name__}")
            return [], None

        print(f"Loaded {len(data1)} documents from {json_file_1_path}")
        print(f"Loaded {len(data2)} documents from {json_file_2_path}")

        common_docs = [doc for doc in data1 if doc in data2]

        unique_data = [doc for doc in data1 if doc not in common_docs] + [doc for doc in data2 if doc not in common_docs]

        if output_file_path:
            with open(output_file_path, 'w') as file:
                json.dump(unique_data, file, indent=2)
            print(f"Unique documents saved to {output_file_path}")

        return common_docs, unique_data

    except FileNotFoundError:
        print(f"Error: One or both files not found")
        return [], None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file: {e}")
        return [], None
    except Exception as e:
        print(f"Error: {str(e)}")
        return [], None


def print_common_docs_info(common_docs):
    """Print information about common documents found"""
    if not common_docs:
        print("\nNo common documents found.")
        return

    print(f"\nFound {len(common_docs)} common documents:")
    for doc in common_docs:
        print(f"  Document: {doc}")

    print(f"\nTotal common documents: {len(common_docs)}")


def main():
    json_file_1_path = input("Enter path to the first JSON file: ")
    json_file_2_path = input("Enter path to the second JSON file: ")

    output_file_path = input("Enter output file path (or press Enter to use default): ")
    if not output_file_path:
        base_name = os.path.splitext(json_file_1_path)[0]
        output_file_path = f"{base_name}_unique.json"

    print(f"Checking for common documents between: {json_file_1_path} and {json_file_2_path}")
    print(f"Unique data will be saved to: {output_file_path}")

    common_docs, unique_data = check_and_remove_common_docs(json_file_1_path, json_file_2_path, output_file_path)
    print_common_docs_info(common_docs)

    if unique_data:
        original_count_1 = len(json.load(open(json_file_1_path)))
        original_count_2 = len(json.load(open(json_file_2_path)))
        print(f"\nOriginal document count from {json_file_1_path}: {original_count_1}")
        print(f"Original document count from {json_file_2_path}: {original_count_2}")
        print(f"Unique document count: {len(unique_data)}")
        print(f"Removed {original_count_1 + original_count_2 - len(unique_data)} common documents")


if __name__ == "__main__":
    main()
