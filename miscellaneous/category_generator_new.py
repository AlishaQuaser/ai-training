from typing_extensions import TypedDict
from pymongo import MongoClient
import os
import json
import random
import uuid
import time
import re
from dotenv import load_dotenv
import math

def get_mongo_client():
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise ValueError("MONGO_URI not set in environment variables.")
    return MongoClient(mongo_uri)

def generate_object_id():
    """Generate a MongoDB-style ObjectId as a 24-character hex string."""
    return ''.join(random.choice('0123456789abcdef') for _ in range(24))

def get_existing_categories():
    """Get all existing categories from the MongoDB collection."""
    client = get_mongo_client()
    db = client["app-test-data"]

    existing_categories = list(db["categories"].find({}, {"name": 1, "slug": 1, "_id": 0}))

    existing_names = {cat.get("name", "").lower() for cat in existing_categories}
    existing_slugs = {cat.get("slug", "").lower() for cat in existing_categories}

    print(f"Found {len(existing_categories)} existing categories in the database")

    return existing_names, existing_slugs

def create_slug(name):
    """Convert a name to a slug format."""
    return name.lower().replace(" ", "-").replace("&", "and").replace("/", "-").replace(".", "").replace(",", "")

def process_parent_categories(parent_file_path, max_categories=None):
    """
    Process parent categories from a text file and create structured category objects.
    Skip categories that already exist in the database.

    Args:
        parent_file_path: Path to the text file containing parent category names
        max_categories: Maximum number of parent categories to process (optional)

    Returns:
        List of parent category objects and a mapping of parent names to their IDs
    """
    existing_names, existing_slugs = get_existing_categories()

    try:
        with open(parent_file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        parent_names = [line.strip() for line in lines if line.strip()]
        print(f"Read {len(parent_names)} parent category names from {parent_file_path}")
    except Exception as e:
        print(f"Error reading parent file: {e}")
        return [], {}

    parent_categories = []
    parents_map = {}
    parent_count = 0

    for name in parent_names:
        if max_categories is not None and len(parent_categories) >= max_categories:
            break

        if name.lower() in existing_names:
            print(f"Skipping existing parent category: {name}")
            continue

        slug = create_slug(name)

        if slug.lower() in existing_slugs:
            print(f"Skipping parent category with existing slug: {name} -> {slug}")
            continue

        parent_id = generate_object_id()
        parent_category = {
            "_id": parent_id,
            "name": name,
            "slug": slug,
            "parentCategoryId": None,
            "title": None,
            "description": f"Main category for {name} solutions and services",
            "marketable": random.choice([True, False]),
            "displayOrder": parent_count + 1,
            "logo": None
        }

        parent_categories.append(parent_category)
        parents_map[name.lower()] = parent_id
        existing_names.add(name.lower())
        existing_slugs.add(slug.lower())
        parent_count += 1

    print(f"Generated {len(parent_categories)} parent categories")
    return parent_categories, parents_map

def process_subcategories(subcategory_file_path, parents_map, existing_names, existing_slugs, max_subcategories=None):
    """
    Process subcategories from a text file and create structured subcategory objects.
    Skip subcategories that already exist in the database.

    Args:
        subcategory_file_path: Path to the text file containing subcategory information
        parents_map: Dictionary mapping parent names to their IDs
        existing_names: Set of existing category names
        existing_slugs: Set of existing category slugs
        max_subcategories: Maximum number of subcategories to process (optional)

    Returns:
        List of subcategory objects
    """
    try:
        with open(subcategory_file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        sections = re.split(r'\n\s*\n', content)

        print(f"Found {len(sections)} potential subcategory sections")
    except Exception as e:
        print(f"Error reading subcategory file: {e}")
        return []

    subcategories = []
    subcat_count = 0

    for section in sections:
        if not section.strip():
            continue

        lines = section.strip().split('\n')

        parent_line = lines[0].strip()
        parent_match = re.match(r'(.*?)\s*subcategories', parent_line, re.IGNORECASE)

        if not parent_match:
            print(f"Could not parse parent category from line: {parent_line}")
            continue

        parent_name = parent_match.group(1).strip()
        parent_id = parents_map.get(parent_name.lower())

        if parent_id is None:
            print(f"Parent category '{parent_name}' not found in generated parents, skipping its subcategories")
            continue

        subcats_list = [line.strip() for line in lines[1:] if line.strip()]
        print(f"Processing {len(subcats_list)} subcategories for parent: {parent_name}")

        for subcat_name in subcats_list:
            if max_subcategories is not None and len(subcategories) >= max_subcategories:
                break

            if subcat_name.lower() in existing_names:
                print(f"Skipping existing subcategory: {subcat_name}")
                continue

            slug = create_slug(subcat_name)

            if slug.lower() in existing_slugs:
                print(f"Skipping subcategory with existing slug: {subcat_name} -> {slug}")
                continue

            subcategory = {
                "_id": generate_object_id(),
                "name": subcat_name,
                "slug": slug,
                "parentCategoryId": parent_id,
                "title": None,
                "description": f"Specialized {subcat_name} subcategory for {parent_name}",
                "marketable": random.choice([True, False]),
                "displayOrder": subcat_count + 1,
                "logo": None
            }

            subcategories.append(subcategory)
            existing_names.add(subcat_name.lower())
            existing_slugs.add(slug.lower())
            subcat_count += 1

    print(f"Generated {len(subcategories)} subcategories total")
    return subcategories

def validate_categories(categories):
    """Validate that generated categories follow the correct structure and are unique."""
    names = set()
    slugs = set()
    valid_categories = []
    validation_issues = []

    for category in categories:
        required_fields = ["_id", "name", "slug", "parentCategoryId", "title",
                           "description", "marketable", "displayOrder", "logo"]

        missing_fields = [field for field in required_fields if field not in category]

        if missing_fields:
            validation_issues.append(f"Missing fields in category: {missing_fields}")
            continue

        name = category.get("name", "").lower()
        slug = category.get("slug", "").lower()

        if name in names:
            validation_issues.append(f"Duplicate name within generated set: {name}")
            continue

        if slug in slugs:
            validation_issues.append(f"Duplicate slug within generated set: {slug}")
            continue

        names.add(name)
        slugs.add(slug)

        valid_categories.append(category)

    print(f"Validation complete: {len(valid_categories)} valid categories out of {len(categories)}")
    if validation_issues:
        print(f"Found {len(validation_issues)} validation issues")

    return valid_categories

def main():
    """Main function to process categories from two separate files and save unique ones."""
    load_dotenv()

    max_parent_categories = None
    max_subcategories = None

    try:
        parent_file_path = input("Enter path to the text file containing parent categories: ")
        subcategory_file_path = input("Enter path to the text file containing subcategory data: ")

        print(f"Starting processing of parent categories from {parent_file_path}...")
        start_time = time.time()

        parent_categories, parents_map = process_parent_categories(parent_file_path, max_parent_categories)

        print(f"\nStarting processing of subcategories from {subcategory_file_path}...")
        existing_names, existing_slugs = get_existing_categories()

        for cat in parent_categories:
            existing_names.add(cat['name'].lower())
            existing_slugs.add(cat['slug'].lower())

        subcategories = process_subcategories(
            subcategory_file_path,
            parents_map,
            existing_names,
            existing_slugs,
            max_subcategories
        )

        print("Validating generated parent categories...")
        valid_parent_categories = validate_categories(parent_categories)

        print("Validating generated subcategories...")
        valid_subcategories = validate_categories(subcategories)

        timestamp = int(time.time())
        parent_filename = f"parent_categories_{timestamp}.json"
        subcategory_filename = f"subcategories_{timestamp}.json"

        with open(parent_filename, "w") as f:
            json.dump(valid_parent_categories, f, indent=2)

        with open(subcategory_filename, "w") as f:
            json.dump(valid_subcategories, f, indent=2)

        end_time = time.time()

        print(f"\nSummary:")
        print(f"Successfully generated and saved {len(valid_parent_categories)} parent categories to {parent_filename}")
        print(f"Successfully generated and saved {len(valid_subcategories)} subcategories to {subcategory_filename}")
        print(f"Processing completed in {end_time - start_time:.2f} seconds")

        parent_cats = valid_parent_categories
        subcats = valid_subcategories
        print(f"Parent categories: {len(parent_cats)}")
        print(f"Subcategories: {len(subcats)}")
        print(f"Total categories: {len(parent_cats) + len(subcats)}")

        show_samples = input("\nDo you want to see some sample categories? (yes/no): ")
        if show_samples.lower() == "yes":
            print(f"\n--- Sample Parent Categories ({min(3, len(parent_cats))}) ---")
            for i, category in enumerate(parent_cats[:3]):
                print(f"\nParent Category {i+1}:")
                print(json.dumps(category, indent=2))

            if subcats:
                print(f"\n\n--- Sample Subcategories ({min(3, len(subcats))}) ---")
                for i, subcat in enumerate(subcats[:3]):
                    parent_id = subcat['parentCategoryId']
                    parent_name = "Unknown"
                    for parent in parent_cats:
                        if parent['_id'] == parent_id:
                            parent_name = parent['name']
                            break

                    print(f"\nSubcategory {i+1} (Parent: {parent_name}):")
                    print(json.dumps(subcat, indent=2))

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()