import json
import argparse
from collections import defaultdict
import os


def check_and_remove_duplicate_names(json_file_path, output_file_path=None):
    """
    Check for duplicate 'name' fields in a JSON file, print duplicates,
    and create a new file with duplicates removed.

    Args:
        json_file_path (str): Path to the JSON file to check
        output_file_path (str, optional): Path to save the deduplicated JSON file

    Returns:
        tuple: (duplicate_counts, deduplicated_data)
    """
    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)

        if not isinstance(data, list):
            print(f"Error: Expected JSON array/list, but got {type(data).__name__}")
            return {}, None

        print(f"Loaded {len(data)} documents from {json_file_path}")

        name_counts = defaultdict(int)
        name_indices = defaultdict(list)

        for i, doc in enumerate(data):
            if not isinstance(doc, dict):
                print(f"Warning: Item at index {i} is not a document/dictionary")
                continue

            if "name" not in doc:
                print(f"Warning: Document at index {i} has no 'name' field")
                continue

            name = doc["name"]
            name_counts[name] += 1
            name_indices[name].append(i)

        duplicates = {name: count for name, count in name_counts.items() if count > 1}

        deduplicated_data = []
        processed_names = set()

        for doc in data:
            if not isinstance(doc, dict) or "name" not in doc:
                deduplicated_data.append(doc)
                continue

            name = doc["name"]
            if name not in processed_names:
                deduplicated_data.append(doc)
                processed_names.add(name)

        if output_file_path:
            with open(output_file_path, 'w') as file:
                json.dump(deduplicated_data, file, indent=2)
            print(f"Deduplicated data saved to {output_file_path}")

        return duplicates, deduplicated_data

    except FileNotFoundError:
        print(f"Error: File not found: {json_file_path}")
        return {}, None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file: {e}")
        return {}, None
    except Exception as e:
        print(f"Error: {str(e)}")
        return {}, None


def print_duplicate_info(duplicates):
    """Print information about duplicate names found"""
    if not duplicates:
        print("\nNo duplicate 'name' fields found.")
        return

    print(f"\nFound {len(duplicates)} duplicate names:")

    sorted_duplicates = sorted(duplicates.items(), key=lambda x: x[1], reverse=True)

    for name, count in sorted_duplicates:
        print(f"  '{name}' appears {count} times")

    total_dupes = sum(duplicates.values()) - len(duplicates)
    print(f"\nTotal duplicate occurrences to be removed: {total_dupes}")


def main():
    parser = argparse.ArgumentParser(description="Check for duplicate 'name' fields in JSON files and remove duplicates.")
    parser.add_argument('file', nargs='?', help="Path to the JSON file")
    parser.add_argument('-o', '--output', help="Output file path for deduplicated JSON")
    args = parser.parse_args()

    json_file_path = args.file
    if not json_file_path:
        json_file_path = input("Enter path to JSON file: ")

    output_file_path = args.output
    if not output_file_path:
        base_name = os.path.splitext(json_file_path)[0]
        output_file_path = f"{base_name}_deduplicated.json"

    print(f"Checking for duplicate 'name' fields in: {json_file_path}")
    print(f"Deduplicated data will be saved to: {output_file_path}")

    duplicates, deduplicated_data = check_and_remove_duplicate_names(json_file_path, output_file_path)
    print_duplicate_info(duplicates)

    if deduplicated_data:
        original_count = 0
        with open(json_file_path, 'r') as file:
            original_data = json.load(file)
            if isinstance(original_data, list):
                original_count = len(original_data)

        print(f"\nOriginal document count: {original_count}")
        print(f"Deduplicated document count: {len(deduplicated_data)}")
        print(f"Removed {original_count - len(deduplicated_data)} duplicate documents")


if __name__ == "__main__":
    main()