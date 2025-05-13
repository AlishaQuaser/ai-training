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
            'Electron', 'React Native', 'Xamarin', 'Qt'
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
            all_techs = (
                    self.programming_languages +
                    self.frameworks_libraries +
                    self.cloud_platforms +
                    self.databases +
                    [
                        'GraphQL', 'gRPC', 'WebAssembly', 'Terraform',
                        'Ansible', 'Docker', 'Nginx', 'Apache',
                        'Kafka', 'RabbitMQ', 'Redis Cluster', 'Elasticsearch',
                        'Prometheus', 'Grafana', 'Jenkins', 'GitLab CI/CD',
                        'Kubernetes', 'OpenShift', 'ArgoCD', 'Istio',
                        'TensorFlow', 'PyTorch', 'Keras', 'Scikit-learn',
                        'CUDA', 'OpenCL', 'Metal', 'DirectX',
                        'OpenGL', 'Vulkan', 'WebRTC', 'WASM',
                        'CoreML', 'ONNX', 'MXNet', 'Caffe',
                        'Xamarin', 'Ionic', 'PhoneGap', 'Unity',
                        'Unreal Engine', 'Godot', 'Blender', 'Maya'
                    ]
            )

            if len(all_techs) < 50:
                raise ValueError("Not enough unique technologies to generate 50 skills")

            tech_stack = random.sample(all_techs, min(50, len(all_techs)))
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

    def generate_project_details(self):
        """Generate project details with domain, technologies, and case study style."""
        detailed_domains = [
            'Financial Sector Digital Transformation',
            'Healthcare Technology Innovation',
            'E-commerce Platform Development',
            'Educational Technology Solutions',
            'Logistics and Supply Chain Optimization',
            'Media and Content Streaming Platforms',
            'Telecommunications Infrastructure',
            'Manufacturing Process Automation',
            'Gaming and Interactive Entertainment',
            'Retail Technology Modernization',
            'Travel and Hospitality Tech',
            'Real Estate Technology',
            'Agricultural Technology Solutions',
            'Renewable Energy Management Systems',
            'Cybersecurity and Threat Detection',
            'Smart City Infrastructure',
            'IoT and Embedded Systems'
        ]

        domain = random.choice(detailed_domains)

        all_techs = (
                self.programming_languages +
                self.frameworks_libraries +
                self.cloud_platforms +
                self.databases
        )

        project_name = f"Case Study: {domain}"

        tech_count = random.randint(2, 6)
        tech_count = min(tech_count, len(all_techs))
        project_techs = random.sample(all_techs, tech_count)

        return project_name, domain, project_techs

    def generate_representative_text(self, is_agency=True, agency_details=None):
        """Generate comprehensive representative text."""
        if is_agency:
            if agency_details:
                name = agency_details.get('name', 'Unknown Agency')
                team_size = random.randint(15, 250)
                founded_year = agency_details.get('founded', random.randint(2010, 2023))
            else:
                name = self.generate_agency_name()
                team_size = random.randint(15, 250)
                founded_year = random.randint(2010, 2023)
        else:
            name = self.generate_freelancer_name()
            team_size = 1
            founded_year = random.randint(2010, 2023)

        tech_stack = self.generate_tech_stack(is_agency)

        num_projects = random.randint(7, 10) if is_agency else random.randint(1, 3)
        projects = []

        for _ in range(num_projects):
            project_name, domain, project_techs = self.generate_project_details()
            projects.append({
                'name': project_name,
                'domain': domain,
                'technologies': project_techs
            })

        if is_agency:
            rep_text = f"{name} is an agency with the following skill sets: {', '.join(tech_stack)}. "

            domain_projects = {}
            for proj in projects:
                if proj['domain'] not in domain_projects:
                    domain_projects[proj['domain']] = []
                domain_projects[proj['domain']].append(proj)

            project_descriptions = []
            for domain, domain_projs in domain_projects.items():
                domain_techs = set()
                for proj in domain_projs:
                    domain_techs.update(proj['technologies'])

                domain_proj_names = [proj['name'] for proj in domain_projs]
                domain_desc = f"have done projects on {domain} using {', '.join(domain_techs)}"
                project_descriptions.append(domain_desc)

            rep_text += " ".join(project_descriptions) + "."
        else:
            rep_text = f"{name} is a freelancer with the following skill set: {', '.join(tech_stack)}. "

            project_descriptions = []
            for proj in projects:
                proj_desc = f"Worked on {proj['name']} in {proj['domain']} "
                proj_desc += f"using {', '.join(proj['technologies'])}"
                project_descriptions.append(proj_desc)

            rep_text += "Worked on following projects: " + ". ".join(project_descriptions) + "."

        return {
            'name': name,
            'is_agency': is_agency,
            'team_size': team_size,
            'tech_stack': tech_stack,
            'projects': projects,
            'operational_year': founded_year
        }, rep_text

    def generate_agencies_and_freelancers(self, num_agencies=500, num_freelancers=50):
        """Generate a list of agency and freelancer documents."""
        documents = []

        for i in range(num_agencies):
            agency_details = self.loaded_agencies[i] if i < len(self.loaded_agencies) else None

            details, rep_text = self.generate_representative_text(is_agency=True, agency_details=agency_details)

            if agency_details and 'hourlyRate' in agency_details:
                hourly_rate = agency_details['hourlyRate']
            else:
                hourly_rate = {
                    "min": random.randint(20, 50),
                    "max": random.randint(70, 150),
                    "currency": "USD"
                }

            doc = {
                "_id": {
                    "$oid": agency_details.get('_id', {}).get('$oid', self.generate_object_id()) if agency_details else self.generate_object_id()
                },
                "name": details["name"],
                "hourlyRate": hourly_rate,
                "representativeText": rep_text,
                "teamSize": details["team_size"],
                "type": "agency",
                "founded": details["operational_year"]
            }
            documents.append(doc)

        for _ in range(num_freelancers):
            details, rep_text = self.generate_representative_text(is_agency=False)

            doc = {
                "_id": {
                    "$oid": self.generate_object_id()
                },
                "name": details["name"],
                "hourlyRate": {
                    "min": random.randint(50, 80),
                    "max": random.randint(100, 200),
                    "currency": "USD"
                },
                "representativeText": rep_text,
                "teamSize": 1,
                "type": "freelancer",
                "founded": details["operational_year"]
            }
            documents.append(doc)

        return documents

class DocumentProcessor:
    def __init__(self, input_file):
        self.input_file = input_file

    def load_documents(self):
        """Load documents from the input file."""
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading documents from {self.input_file}: {e}")
            return []

    def process_documents(self):
        """Process documents to add new fields and remove info from representativeText."""
        documents = self.load_documents()
        processed_docs = []

        for doc in documents:
            rep_text = doc.get("representativeText", "")

            is_agency = "is an agency" in rep_text.lower()
            doc_type = "agency" if is_agency else "freelancer"


            team_size = 1
            if is_agency:
                team_size_match = re.search(r"has (\d+) team members", rep_text)
                if team_size_match:
                    team_size = int(team_size_match.group(1))


            founded_year = None
            year_match = re.search(r"operational since (\d{4})", rep_text)
            if year_match:
                founded_year = int(year_match.group(1))

            new_rep_text = re.sub(r"has \d+ team members", "", rep_text)
            new_rep_text = re.sub(r"operational since \d{4}", "", new_rep_text)
            new_rep_text = re.sub(r"\s+\.", ".", new_rep_text)
            new_rep_text = re.sub(r"\s{2,}", " ", new_rep_text)
            new_rep_text = new_rep_text.strip()

            if is_agency:
                new_rep_text = re.sub(r"is an agency that", "is an agency", new_rep_text)
            else:
                new_rep_text = re.sub(r"is a freelancer who", "is a freelancer", new_rep_text)

            doc["teamSize"] = team_size
            doc["type"] = doc_type

            if founded_year:
                doc["founded"] = founded_year

            doc["representativeText"] = new_rep_text
            processed_docs.append(doc)

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
        print("2. Process existing documents to add fields and update representativeText")
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