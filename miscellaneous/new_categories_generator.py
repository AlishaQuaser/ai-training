from typing_extensions import TypedDict
from pymongo import MongoClient
import os
import json
import random
import uuid
import time
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

def process_categories_from_file(file_path, max_categories=1868):
    """
    Process categories from a text file and create structured category objects.
    Skip categories that already exist in the database.

    Args:
        file_path: Path to the text file containing category names
        max_categories: Maximum number of categories to process

    Returns:
        List of category objects
    """
    existing_names, existing_slugs = get_existing_categories()

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        category_names = [line.strip() for line in lines if line.strip()]
        print(f"Read {len(category_names)} category names from {file_path}")
    except Exception as e:
        print(f"Error reading file: {e}")
        return []

    all_categories = []
    parents_map = {}
    parent_count = 0
    subcat_count = 0

    for name in category_names:
        if len(all_categories) >= max_categories:
            break

        is_parent = '/' not in name

        if is_parent:
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

            all_categories.append(parent_category)
            parents_map[name.lower()] = parent_id
            existing_names.add(name.lower())
            existing_slugs.add(slug.lower())
            parent_count += 1

    for name in category_names:
        if len(all_categories) >= max_categories:
            break

        if name.lower() in parents_map:
            continue

        if '/' in name:
            parts = name.split('/', 1)
            parent_name = parts[0].strip()
            subcat_name = parts[1].strip()

            if subcat_name.lower() in existing_names:
                print(f"Skipping existing subcategory: {subcat_name}")
                continue

            slug = create_slug(subcat_name)

            if slug.lower() in existing_slugs:
                print(f"Skipping subcategory with existing slug: {subcat_name} -> {slug}")
                continue

            parent_id = parents_map.get(parent_name.lower())

            if parent_id is None and parent_name.lower() not in existing_names:
                parent_slug = create_slug(parent_name)

                if parent_slug.lower() in existing_slugs:
                    print(f"Cannot create parent for subcategory, slug exists: {parent_name} -> {parent_slug}")
                    continue

                parent_id = generate_object_id()
                parent_category = {
                    "_id": parent_id,
                    "name": parent_name,
                    "slug": parent_slug,
                    "parentCategoryId": None,
                    "title": None,
                    "description": f"Main category for {parent_name} solutions and services",
                    "marketable": random.choice([True, False]),
                    "displayOrder": parent_count + 1,
                    "logo": None
                }

                all_categories.append(parent_category)
                parents_map[parent_name.lower()] = parent_id
                existing_names.add(parent_name.lower())
                existing_slugs.add(parent_slug.lower())
                parent_count += 1

            if parent_id is None:
                print(f"Skipping subcategory, parent not found or conflicting: {name}")
                continue

            subcategory = {
                "_id": generate_object_id(),
                "name": subcat_name,
                "slug": slug,
                "parentCategoryId": parent_id,
                "title": None,
                "description": f"Specialized subcategory for {parent_name}",
                "marketable": random.choice([True, False]),
                "displayOrder": subcat_count + 1,
                "logo": None
            }

            all_categories.append(subcategory)
            existing_names.add(subcat_name.lower())
            existing_slugs.add(slug.lower())
            subcat_count += 1

    final_parents = [c for c in all_categories if c['parentCategoryId'] is None]
    final_subcats = [c for c in all_categories if c['parentCategoryId'] is not None]

    print(f"Generated {len(final_parents)} parent categories and {len(final_subcats)} subcategories")
    print(f"Total: {len(all_categories)} categories")

    return all_categories

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
    """Main function to process categories from file and save unique ones."""
    load_dotenv()

    max_categories = 1868

    try:
        file_path = input("Enter path to the text file containing categories: ")

        print(f"Starting processing of categories from {file_path}...")
        print(f"Will generate up to {max_categories} unique categories...")
        start_time = time.time()

        categories = process_categories_from_file(file_path, max_categories)

        print("Validating generated categories...")
        valid_categories = validate_categories(categories)

        timestamp = int(time.time())
        filename = f"new_unique_categories_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(valid_categories, f, indent=2)

        end_time = time.time()

        print(f"\nSummary:")
        print(f"Successfully generated and saved {len(valid_categories)} unique categories to {filename}")
        print(f"Processing completed in {end_time - start_time:.2f} seconds")

        parent_cats = [c for c in valid_categories if c['parentCategoryId'] is None]
        subcats = [c for c in valid_categories if c['parentCategoryId'] is not None]
        print(f"Parent categories: {len(parent_cats)}")
        print(f"Subcategories: {len(subcats)}")

        if len(valid_categories) < max_categories:
            shortfall = max_categories - len(valid_categories)
            print(f"\nNote: Generated {len(valid_categories)} categories, which is {shortfall} less than the target of {max_categories}.")
            print("This may be due to duplicates being skipped or insufficient unique categories in the input file.")

        show_samples = input("\nDo you want to see some sample categories? (yes/no): ")
        if show_samples.lower() == "yes":
            print(f"\n--- Sample Parent Categories ({min(3, len(parent_cats))}) ---")
            for i, category in enumerate(parent_cats[:3]):
                print(f"\nParent Category {i+1}:")
                print(json.dumps(category, indent=2))

            if parent_cats:
                first_parent = parent_cats[0]
                parent_subcats = [c for c in valid_categories if c.get('parentCategoryId') == first_parent['_id']]
                print(f"\n\n--- Sample Subcategories for {first_parent['name']} ({min(3, len(parent_subcats))}) ---")
                for i, subcat in enumerate(parent_subcats[:3]):
                    print(f"\nSubcategory {i+1}:")
                    print(json.dumps(subcat, indent=2))

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()