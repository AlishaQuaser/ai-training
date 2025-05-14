from typing_extensions import TypedDict
import os
import json
import random
import time
import re
from dotenv import load_dotenv

class AgencyGenerator:
    def __init__(self, agency_json_path=None):
        self.agency_json_path = agency_json_path
        self.loaded_agencies = self.load_agencies_from_json() if agency_json_path else []

        self.tech_keywords = [
            'Machine Learning', 'Cloud Computing', 'Blockchain',
            'Cybersecurity', 'Web Development', 'Mobile App',
            'Data Analytics', 'AI Solutions', 'Enterprise Software'
        ]

        self.industry_domains = [
            'Financial Services', 'Healthcare', 'E-commerce',
            'Education Technology', 'Logistics', 'Media',
            'Telecommunications', 'Manufacturing', 'Gaming',
            'Retail Technology', 'Travel Tech', 'Real Estate Tech',
            'Agricultural Technology', 'Energy Sector'
        ]

        self.programming_languages = [
            'JavaScript', 'Python', 'Java', 'TypeScript', 'Go',
            'Rust', 'Kotlin', 'Swift', 'Scala', 'Dart', 'C#',
            'Ruby', 'PHP', 'Lua', 'R', 'Elixir', 'Haskell',
            'Clojure', 'Erlang', 'Crystal', 'Nim'
        ]

        self.frameworks_libraries = [
            'React', 'Node.js', 'Django', 'Spring Boot', 'Flutter',
            'Angular', 'Vue.js', 'TensorFlow', 'PyTorch', 'Kubernetes',
            'Express.js', 'Laravel', 'Ruby on Rails', 'FastAPI',
            'Phoenix', 'Svelte', 'Next.js', 'Nuxt.js', 'Gatsby',
            'Electron', 'React Native', 'Xamarin', 'Qt', 'PrimeNG'
        ]

        self.cloud_platforms = [
            'AWS', 'Google Cloud', 'Azure', 'DigitalOcean', 'Heroku',
            'IBM Cloud', 'Oracle Cloud', 'Alibaba Cloud', 'Linode',
            'Vultr', 'FoundationDB', 'OpenStack'
        ]

        self.databases = [
            'MongoDB', 'PostgreSQL', 'MySQL', 'Redis', 'Cassandra',
            'MariaDB', 'SQLite', 'Oracle', 'Microsoft SQL Server',
            'Couchbase', 'Neo4j', 'DynamoDB', 'Firebase'
        ]

    def generate_agency_name(self):
        """Generate a realistic agency name."""
        prefixes = [
            'Digital', 'Tech', 'Innovative', 'Global', 'Creative',
            'Advanced', 'Smart', 'Future', 'Cutting Edge', 'Elite'
        ]
        suffixes = [
            'Solutions', 'Technologies', 'Innovations', 'Systems',
            'Ventures', 'Group', 'Labs', 'Studio', 'Collective', 'Agency'
        ]
        tech_keywords = [
            'Tech', 'Software', 'Digital', 'Cloud', 'Data',
            'AI', 'Web', 'Mobile', 'Enterprise', 'Network'
        ]

        name_patterns = [
            lambda: f"{random.choice(prefixes)} {random.choice(tech_keywords)} {random.choice(suffixes)}",
            lambda: f"{random.choice(tech_keywords)} {random.choice(suffixes)}",
            lambda: f"{random.choice(prefixes)} {random.choice(tech_keywords)}"
        ]

        return random.choice(name_patterns)()

    def load_agencies_from_json(self):
        """Load agency details from the provided JSON file."""
        try:
            with open(self.agency_json_path, 'r', encoding='utf-8') as f:
                agencies = json.load(f)
            print(f"Loaded {len(agencies)} agencies from {self.agency_json_path}")
            return agencies
        except Exception as e:
            print(f"Error loading agencies from JSON: {e}")
            return []

    def generate_object_id(self):
        """Generate a MongoDB-style ObjectId as a 24-character hex string."""
        return ''.join(random.choice('0123456789abcdef') for _ in range(24))

    def generate_freelancer_name(self):
        """Generate a realistic freelancer name."""
        first_names = [
            'Alex', 'Sam', 'Jordan', 'Taylor', 'Casey', 'Riley',
            'Morgan', 'Charlie', 'Avery', 'Quinn', 'Blake', 'Jamie',
            'Cameron', 'Skylar', 'Parker', 'Harper', 'Drew', 'Reese'
        ]
        last_names = [
            'Smith', 'Johnson', 'Williams', 'Brown', 'Jones',
            'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez',
            'Lee', 'Kim', 'Chen', 'Patel', 'Singh', 'Ahmed', 'Wong'
        ]
        return f"{random.choice(first_names)} {random.choice(last_names)}"

    def generate_tech_stack(self, is_agency=True):
        """Generate a realistic tech stack."""
        if is_agency:
            tech_count = random.randint(3, 8)
            all_techs = (
                    self.programming_languages +
                    self.frameworks_libraries +
                    self.cloud_platforms +
                    self.databases
            )
            tech_count = min(tech_count, len(all_techs))
            tech_stack = random.sample(all_techs, tech_count)
        else:
            tech_count = random.randint(3, 6)
            all_techs = (
                    self.programming_languages +
                    self.frameworks_libraries +
                    self.cloud_platforms +
                    self.databases
            )
            tech_count = min(tech_count, len(all_techs))
            tech_stack = random.sample(all_techs, tech_count)

        return tech_stack

    def generate_project_domain(self):
        """Generate a project domain."""
        return random.choice(self.industry_domains)

    def generate_project_technologies(self):
        """Generate technologies used for a project."""
        all_techs = (
                self.programming_languages +
                self.frameworks_libraries +
                self.cloud_platforms +
                self.databases
        )

        tech_count = random.randint(2, 5)
        tech_count = min(tech_count, len(all_techs))
        project_techs = random.sample(all_techs, tech_count)

        return project_techs

    def generate_representative_text(self, is_agency=True, agency_details=None):
        """Generate representative text following the specified format."""
        if agency_details and 'name' in agency_details:
            name = agency_details.get('name', 'Unknown Agency')
        elif is_agency:
            name = self.generate_agency_name()
        else:
            name = self.generate_freelancer_name()

        entity_type = "agency" if is_agency else "freelancer"

        tech_stack = self.generate_tech_stack(is_agency)

        num_domains = random.randint(7, 10) if is_agency else random.randint(3, 5)
        project_domains = []

        available_domains = self.industry_domains.copy()
        for _ in range(min(num_domains, len(available_domains))):
            if not available_domains:
                break

            domain = random.choice(available_domains)
            available_domains.remove(domain)
            project_techs = self.generate_project_technologies()
            project_domains.append((domain, project_techs))

        rep_text = f"{name} is an {entity_type} with following skill set {', '.join(tech_stack)}"

        if project_domains:
            first_domain, first_techs = project_domains[0]
            domain_text = f" have done projects on {first_domain} using {', '.join(first_techs)}"

            for domain, techs in project_domains[1:]:
                domain_text += f" ; {domain} using {', '.join(techs)}"

            rep_text += domain_text

        return rep_text

    def generate_agencies_and_freelancers(self, num_agencies=500, num_freelancers=50):
        """Generate a list of agency and freelancer documents."""
        documents = []

        for i in range(num_agencies):
            agency_details = self.loaded_agencies[i] if i < len(self.loaded_agencies) else None

            rep_text = self.generate_representative_text(is_agency=True, agency_details=agency_details)

            name = rep_text.split(" is an agency")[0] if " is an agency" in rep_text else agency_details.get('name', self.generate_agency_name())

            doc = {
                "_id": {
                    "$oid": agency_details.get('_id', {}).get('$oid', self.generate_object_id()) if agency_details else self.generate_object_id()
                },
                "name": name,
                "hourlyRate": agency_details.get('hourlyRate', {
                    "min": random.randint(20, 50),
                    "max": random.randint(70, 150),
                    "currency": "USD"
                }),
                "representativeText": rep_text,
                "type": "agency",
                "founded": agency_details.get('founded', random.randint(2010, 2023))
            }
            documents.append(doc)

        for _ in range(num_freelancers):
            rep_text = self.generate_representative_text(is_agency=False)

            name = rep_text.split(" is a freelancer")[0] if " is a freelancer" in rep_text else self.generate_freelancer_name()

            doc = {
                "_id": {
                    "$oid": self.generate_object_id()
                },
                "name": name,
                "hourlyRate": {
                    "min": random.randint(50, 80),
                    "max": random.randint(100, 200),
                    "currency": "USD"
                },
                "representativeText": rep_text,
                "type": "freelancer",
                "founded": random.randint(2010, 2023)
            }
            documents.append(doc)

        return documents

class DocumentProcessor:
    def __init__(self, input_file):
        self.input_file = input_file

        self.generator = AgencyGenerator()

    def load_documents(self):
        """Load documents from the input file."""
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading documents from {self.input_file}: {e}")
            return []

    def process_documents(self):
        """Process documents to update the representativeText field only."""
        documents = self.load_documents()
        processed_docs = []

        for doc in documents:
            processed_doc = doc.copy()

            is_agency = doc.get("type", "agency") == "agency"
            rep_text = self.generator.generate_representative_text(is_agency=is_agency, agency_details=doc)
            processed_doc["representativeText"] = rep_text

            processed_docs.append(processed_doc)

        return processed_docs

    def save_processed_documents(self, processed_docs, output_file=None):
        """Save processed documents to a JSON file."""
        if not output_file:
            timestamp = int(time.time())
            output_file = f"processed_documents_{timestamp}.json"

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_docs, f, indent=2, ensure_ascii=False)
            print(f"Successfully saved {len(processed_docs)} documents to {output_file}")
            return output_file
        except Exception as e:
            print(f"Error saving processed documents: {e}")
            return None

def main():
    """Main function to handle both generation and processing of documents."""
    load_dotenv()

    try:
        print("Choose an option:")
        print("1. Generate new agency and freelancer documents")
        print("2. Process existing documents to update representativeText")
        choice = input("Enter your choice (1 or 2): ").strip()

        if choice == "1":
            agency_json_path = input("Enter path to the JSON file with agency details (or press Enter to skip): ").strip()

            print("Starting generation of agency and freelancer documents...")
            start_time = time.time()

            generator = AgencyGenerator(agency_json_path if agency_json_path else None)

            documents = generator.generate_agencies_and_freelancers(
                num_agencies=500,
                num_freelancers=50
            )

            timestamp = int(time.time())
            filename = f"agency_freelancer_docs_{timestamp}.json"

            with open(filename, "w", encoding='utf-8') as f:
                json.dump(documents, f, indent=2, ensure_ascii=False)

            end_time = time.time()

            print(f"\nSummary:")
            print(f"Successfully generated and saved {len(documents)} documents to {filename}")
            print(f"Processing completed in {end_time - start_time:.2f} seconds")

            print("\n--- Sample Entries ---")
            for i, doc in enumerate(documents[:3], 1):
                print(f"\nEntry {i}:")
                print(json.dumps(doc, indent=2))

        elif choice == "2":
            input_file = input("Enter path to the JSON file with documents to process: ").strip()

            if not input_file:
                print("No input file provided. Exiting.")
                return

            print(f"Starting to process documents from {input_file}...")
            start_time = time.time()

            processor = DocumentProcessor(input_file)
            processed_docs = processor.process_documents()

            if processed_docs:
                output_file = processor.save_processed_documents(processed_docs)

                end_time = time.time()

                print(f"\nSummary:")
                print(f"Successfully processed {len(processed_docs)} documents")
                print(f"Processing completed in {end_time - start_time:.2f} seconds")

                print("\n--- Sample Processed Entry ---")
                if processed_docs:
                    print(json.dumps(processed_docs[0], indent=2))
            else:
                print("No documents were processed.")
        else:
            print("Invalid choice. Please run the script again and select option 1 or 2.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()