import json
import sys
from collections import defaultdict


def check_duplicate_ids(json_file_path):
    """
    Check for duplicate _id fields in a JSON file containing category documents.

    Args:
        json_file_path (str): Path to the JSON file to check

    Returns:
        dict: A dictionary with duplicate IDs as keys and lists of documents with those IDs as values
    """
    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)

        if not isinstance(data, list):
            print(f"Error: Expected JSON array/list, but got {type(data).__name__}")
            return {}

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
            id_map[doc_id].append(doc)

        duplicates = {id_val: docs for id_val, docs in id_map.items() if len(docs) > 1}

        return duplicates

    except FileNotFoundError:
        print(f"Error: File not found: {json_file_path}")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file: {e}")
        return {}
    except Exception as e:
        print(f"Error: {str(e)}")
        return {}


def print_duplicate_info(duplicates):
    """Print information about duplicate IDs found"""
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


def main():
    if len(sys.argv) > 1:
        json_file_path = sys.argv[1]
    else:
        json_file_path = input("Enter path to JSON file: ")

    print(f"Checking for duplicate _id fields in: {json_file_path}")

    duplicates = check_duplicate_ids(json_file_path)
    print_duplicate_info(duplicates)

    if duplicates:
        print(f"\nTotal unique duplicate IDs found: {len(duplicates)}")
        total_dupes = sum(len(docs) for docs in duplicates.values()) - len(duplicates)
        print(f"Total duplicate documents: {total_dupes}")


if __name__ == "__main__":
    main()