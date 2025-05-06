import json
import re
import os
import time

TECH_KEYWORDS = [
    "java", "python", "javascript", "typescript", "html", "css", "react", "angular", "vue",
    "node", "aws", "azure", "cloud", "docker", "kubernetes", "devops", "database", "sql",
    "nosql", "mongodb", "programming", "development", "web", "mobile", "frontend", "backend",
    "fullstack", "data", "ai", "ml", "blockchain", "cyber", "security", "network", "code",
    "coding", "software", "app", "application", "api", "tech", "technology", "computer",
    "computing", "digital", "it", "information technology", "algorithm", "server", "git",
    "framework", "library", "system", "architecture"
]

def determine_vocabulary(category_name):
    """
    Determine if the category is tech-related based on its name.
    Returns "tech" if it's technology-related, otherwise "domain".
    """
    name_lower = category_name.lower()

    for keyword in TECH_KEYWORDS:
        pattern = r'\b' + re.escape(keyword) + r'\b'
        if re.search(pattern, name_lower):
            return "tech"

    return "domain"

def process_json_file(file_path):
    """Process categories from a JSON file and add vocabulary field (tech or domain)."""
    try:
        with open(file_path, 'r') as f:
            categories = json.load(f)

        print(f"Read {len(categories)} categories from {file_path}")

        initial_tech = 0
        initial_domain = 0

        for category in categories:
            category_name = category["name"]

            vocabulary = determine_vocabulary(category_name)

            category["vocabulary"] = vocabulary

            if vocabulary == "tech":
                initial_tech += 1
            else:
                initial_domain += 1

        timestamp = int(time.time())
        output_filename = f"categories_with_vocabulary_{timestamp}.json"
        output_path = os.path.join(os.path.dirname(file_path), output_filename)

        with open(output_path, "w") as f:
            json.dump(categories, f, indent=2)

        print(f"\nAdded vocabulary field to {len(categories)} categories")
        print(f"Categories classified as 'tech': {initial_tech}")
        print(f"Categories classified as 'domain': {initial_domain}")
        print(f"Saved updated categories to {output_filename}")

        sample_size = min(5, len(categories))
        print(f"\n--- Sample Updated Categories ({sample_size}) ---")
        for i, category in enumerate(categories[:sample_size]):
            print(f"\nCategory {i+1}: {category['name']}")
            print(f"Vocabulary: {category['vocabulary']}")

        return categories, output_path

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    """Main function to process categories from a JSON file."""
    print("Simple Category Vocabulary Processor")
    print("------------------------------------")
    print("This script will classify categories as either 'tech' or 'domain' based on their names.")

    while True:
        file_path = input("\nEnter path to the JSON file containing categories: ")

        if os.path.exists(file_path):
            if not file_path.lower().endswith('.json'):
                print("Warning: The file doesn't have a .json extension. Are you sure this is the correct file?")
                confirm = input("Continue anyway? (y/n): ")
                if confirm.lower() != 'y':
                    continue
            break
        else:
            print(f"Error: File not found at '{file_path}'. Please check the path and try again.")

    process_json_file(file_path)

    print("\nProcessing complete!")

if __name__ == "__main__":
    main()