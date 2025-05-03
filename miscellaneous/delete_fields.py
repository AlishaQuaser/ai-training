import json
import os
import argparse

def clean_json_documents(json_file_path, output_file_path=None):
    """
    Remove '_id' and 'domain' fields from all documents in a JSON file.

    Args:
        json_file_path (str): Path to the JSON file to process
        output_file_path (str, optional): Path to save the cleaned JSON file

    Returns:
        tuple: (processed_count, modified_data)
    """
    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)

        if not isinstance(data, list):
            print(f"Error: Expected JSON array/list, but got {type(data).__name__}")
            return 0, None

        print(f"Loaded {len(data)} documents from {json_file_path}")

        modified_data = []
        processed_count = 0

        for doc in data:
            if not isinstance(doc, dict):
                continue

            cleaned_doc = {key: value for key, value in doc.items()
                           # if key != '_id' and key != 'domain'
                           if key != 'domain'}

            modified_data.append(cleaned_doc)
            processed_count += 1

        if output_file_path:
            with open(output_file_path, 'w') as file:
                json.dump(modified_data, file, indent=2)
            print(f"Cleaned data saved to {output_file_path}")

        return processed_count, modified_data

    except FileNotFoundError:
        print(f"Error: File not found: {json_file_path}")
        return 0, None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file: {e}")
        return 0, None
    except Exception as e:
        print(f"Error: {str(e)}")
        return 0, None

def main():
    parser = argparse.ArgumentParser(description="Remove _id and domain fields from JSON documents.")
    parser.add_argument('file', nargs='?', help="Path to the JSON file")
    parser.add_argument('-o', '--output', help="Output file path for cleaned JSON")
    args = parser.parse_args()

    json_file_path = args.file
    if not json_file_path:
        json_file_path = input("Enter path to JSON file: ")

    output_file_path = args.output
    if not output_file_path:
        base_name = os.path.splitext(json_file_path)[0]
        output_file_path = f"{base_name}_cleaned.json"

    print(f"Processing JSON file: {json_file_path}")
    print(f"Cleaned data will be saved to: {output_file_path}")

    processed_count, _ = clean_json_documents(json_file_path, output_file_path)

    if processed_count > 0:
        print(f"\nTotal documents processed: {processed_count}")
        print(f"Successfully removed 'domain' fields from all documents.")

if __name__ == "__main__":
    main()