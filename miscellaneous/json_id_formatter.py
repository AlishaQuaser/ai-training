import json
import os
import sys
import time

def convert_json_file(input_file, output_file=None):
    """
    Convert JSON file with string _id values to MongoDB extended JSON format with $oid
    Leaves parentCategoryId untouched

    Args:
        input_file: Path to the input JSON file
        output_file: Path to the output JSON file (if None, will create with timestamp)

    Returns:
        Path to the output file
    """
    print(f"Processing file: {input_file}")

    if not output_file:
        timestamp = int(time.time())
        filename, ext = os.path.splitext(input_file)
        output_file = f"{filename}_converted_{timestamp}{ext}"

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"Successfully read JSON with {len(data) if isinstance(data, list) else 1} entries")
    except Exception as e:
        print(f"Error reading input file: {e}")
        return None

    if isinstance(data, list):
        total_converted = 0

        for item in data:
            if "_id" in item and isinstance(item["_id"], str):
                item["_id"] = {"$oid": item["_id"]}
                total_converted += 1

        print(f"Converted {total_converted} _id fields (leaving parentCategoryId untouched)")

    else:
        conversions = 0

        if "_id" in data and isinstance(data["_id"], str):
            data["_id"] = {"$oid": data["_id"]}
            conversions += 1

        print(f"Converted {conversions} fields in single object")

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        print(f"Successfully wrote converted data to {output_file}")
        return output_file
    except Exception as e:
        print(f"Error writing output file: {e}")
        return None

def process_directory(directory_path):
    """
    Process all JSON files in a directory

    Args:
        directory_path: Path to directory containing JSON files

    Returns:
        Number of files successfully processed
    """
    if not os.path.isdir(directory_path):
        print(f"Error: {directory_path} is not a valid directory")
        return 0

    json_files = [f for f in os.listdir(directory_path) if f.endswith('.json')]

    if not json_files:
        print(f"No JSON files found in {directory_path}")
        return 0

    print(f"Found {len(json_files)} JSON files to process")

    success_count = 0
    for json_file in json_files:
        input_path = os.path.join(directory_path, json_file)
        if convert_json_file(input_path):
            success_count += 1

    return success_count

def main():
    """Main function to convert JSON files"""
    print("MongoDB JSON ID Converter")
    print("------------------------")
    print("This script converts string _id values to MongoDB extended JSON format with $oid")
    print("The parentCategoryId field will be left unchanged\n")

    mode = input("Do you want to convert a single file or all files in a directory? (file/dir): ").strip().lower()

    if mode == 'file':
        input_file = input("Enter the path to the JSON file to convert: ").strip()

        if not os.path.isfile(input_file):
            print(f"Error: {input_file} is not a valid file")
            return

        output_file = input("Enter the path for the output file (leave blank for auto-generated): ").strip()
        output_file = output_file if output_file else None

        result = convert_json_file(input_file, output_file)
        if result:
            print(f"\nConversion completed successfully. Output file: {result}")
        else:
            print("\nConversion failed. Check the error messages above.")

    elif mode == 'dir':
        directory = input("Enter the path to the directory containing JSON files: ").strip()

        if not os.path.isdir(directory):
            print(f"Error: {directory} is not a valid directory")
            return

        success_count = process_directory(directory)
        print(f"\nProcessed {success_count} files successfully")

    else:
        print("Invalid option. Please enter 'file' or 'dir'")

if __name__ == "__main__":
    main()