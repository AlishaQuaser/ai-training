import json
import argparse
from collections import defaultdict
import datetime
import binascii
import os


def generate_object_id():
    """
    Generate a MongoDB-style ObjectId
    Format: 12 bytes (24 hex chars)
    - 4 bytes: timestamp
    - 3 bytes: machine identifier
    - 2 bytes: process identifier
    - 3 bytes: counter
    """
    # Generate timestamp portion (4 bytes)
    timestamp = int(datetime.datetime.now().timestamp())
    timestamp_hex = format(timestamp, '08x')

    # Generate random parts for the remaining 8 bytes
    remaining_hex = binascii.b2a_hex(os.urandom(8)).decode('ascii')

    return timestamp_hex + remaining_hex


def check_and_fix_duplicate_ids(json_file_path, output_file_path=None):
    """
    Check for duplicate _id fields in a JSON file and replace duplicates with new IDs.

    Args:
        json_file_path (str): Path to the JSON file to check
        output_file_path (str, optional): Path to save the fixed JSON file

    Returns:
        tuple: (duplicates_dict, replacement_map, modified_data)
    """
    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)

        if not isinstance(data, list):
            print(f"Error: Expected JSON array/list, but got {type(data).__name__}")
            return {}, {}, None

        print(f"Loaded {len(data)} documents from {json_file_path}")

        id_map = defaultdict(list)

        for i, doc in enumerate(data):
            if not isinstance(doc, dict):
                print(f"Warning: Item at index {i} is not a document/dictionary")
                continue

            if "_id" not in doc:
                print(f"Warning: Document at index {i} has no _id field")
                continue

            doc_id = doc["_id"]
            id_map[doc_id].append(i)

        duplicates = {id_val: [data[idx] for idx in indices] for id_val, indices in id_map.items() if len(indices) > 1}

        replacement_map = {}
        modified_data = data.copy()

        for id_val, indices in id_map.items():
            if len(indices) > 1:
                for idx in indices[1:]:
                    new_id = generate_object_id()
                    while new_id in id_map or new_id in replacement_map.values():
                        new_id = generate_object_id()

                    replacement_map[modified_data[idx]["_id"]] = new_id

                    modified_data[idx]["_id"] = new_id

        if output_file_path and replacement_map:
            with open(output_file_path, 'w') as file:
                json.dump(modified_data, file, indent=2)
            print(f"Fixed data saved to {output_file_path}")

        return duplicates, replacement_map, modified_data

    except FileNotFoundError:
        print(f"Error: File not found: {json_file_path}")
        return {}, {}, None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file: {e}")
        return {}, {}, None
    except Exception as e:
        print(f"Error: {str(e)}")
        return {}, {}, None


def print_duplicate_info(duplicates, replacement_map=None):
    """Print information about duplicate IDs found and their replacements"""
    if not duplicates:
        print("\nNo duplicate _id fields found.")
        return

    print(f"\nFound {len(duplicates)} duplicate _id values:")

    for id_val, docs in duplicates.items():
        print(f"\n_id: {id_val} (appears {len(docs)} times)")

        for i, doc in enumerate(docs):
            name = doc.get("name", "Unnamed")
            slug = doc.get("slug", "No slug")
            parent_id = doc.get("parentCategoryId", "None")

            print(f"  {i+1}. Name: {name}")
            print(f"     Slug: {slug}")
            print(f"     Parent ID: {parent_id}")

            if replacement_map and i > 0:
                if doc["_id"] in replacement_map:
                    print(f"     NEW _id: {replacement_map[doc['_id']]}")
                else:
                    print(f"     ID NOT REPLACED (first occurrence kept)")


def main():
    parser = argparse.ArgumentParser(description="Check and fix duplicate _id fields in JSON files.")
    parser.add_argument('file', nargs='?', help="Path to the JSON file")
    parser.add_argument('-o', '--output', help="Output file path for fixed JSON")
    args = parser.parse_args()

    json_file_path = args.file
    if not json_file_path:
        json_file_path = input("Enter path to JSON file: ")

    output_file_path = args.output
    if not output_file_path:
        base_name = os.path.splitext(json_file_path)[0]
        output_file_path = f"{base_name}_fixed.json"

    print(f"Checking for duplicate _id fields in: {json_file_path}")
    print(f"Fixed data will be saved to: {output_file_path}")

    duplicates, replacement_map, modified_data = check_and_fix_duplicate_ids(json_file_path, output_file_path)
    print_duplicate_info(duplicates, replacement_map)

    if duplicates:
        print(f"\nTotal unique duplicate IDs found: {len(duplicates)}")
        total_dupes = sum(len(docs) for docs in duplicates.values()) - len(duplicates)
        print(f"Total duplicate documents: {total_dupes}")
        print(f"Total documents fixed: {len(replacement_map)}")


if __name__ == "__main__":
    main()