from typing_extensions import TypedDict
import os
import json
import random
import time
from dotenv import load_dotenv
import re

class AgencyGenerator:
    def __init__(self, agency_json_path=None):
        # Expanded and comprehensive lists to guide generation
        self.agency_json_path = agency_json_path
        self.loaded_agencies = self.load_agencies_from_json() if agency_json_path else []

        # Expanded and comprehensive lists to guide generation
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

        # Greatly expanded and comprehensive tech lists
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

        # Randomly choose between different naming patterns
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
    # For agencies, generate at least 50 unique skills
        if is_agency:
            # Combine all possible tech categories and expand the list
            all_techs = (
                    self.programming_languages +
                    self.frameworks_libraries +
                    self.cloud_platforms +
                    self.databases +
                    [
                        # Add more specialized and niche technologies
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

            # Ensure at least 50 unique skills
            if len(all_techs) < 50:
                raise ValueError("Not enough unique technologies to generate 50 skills")

            tech_stack = random.sample(all_techs, min(50, len(all_techs)))
        else:
            # Existing freelancer logic
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
    # Expand domain list for more variety
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

        # Select a random detailed domain
        domain = random.choice(detailed_domains)

        # Combine all tech categories for project technologies
        all_techs = (
                self.programming_languages +
                self.frameworks_libraries +
                self.cloud_platforms +
                self.databases
        )

        # Generate project name with more professional tone
        project_name = f"Case Study: {domain}"

        # Select 2-6 unique technologies for the project
        tech_count = random.randint(2, 6)
        tech_count = min(tech_count, len(all_techs))
        project_techs = random.sample(all_techs, tech_count)

        return project_name, domain, project_techs

    def generate_representative_text(self, is_agency=True, agency_details=None):
        """Generate comprehensive representative text."""
        # Generate basic details
        if is_agency:
            # Use details from the loaded agencies if available
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

        # Generate tech stack
        tech_stack = self.generate_tech_stack(is_agency)

        # Generate multiple projects
        num_projects = random.randint(7, 10) if is_agency else random.randint(1, 3)
        projects = []

        for _ in range(num_projects):
            project_name, domain, project_techs = self.generate_project_details()
            projects.append({
                'name': project_name,
                'domain': domain,
                'technologies': project_techs
            })

        # Construct representative text
        if is_agency:
            # Format for agency representation
            rep_text = f"{name} is an agency that has {team_size} team members with the following skill sets: {', '.join(tech_stack)}. "

            # Grouping projects by domain if possible
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

            rep_text += " ".join(project_descriptions) + ". "
            rep_text += f"This agency is operational since {founded_year}."

        else:
            # Format for freelancer representation
            rep_text = f"{name} is a freelancer who has the following skill set: {', '.join(tech_stack)}. "

            project_descriptions = []
            for proj in projects:
                proj_desc = f"Worked on {proj['name']} in {proj['domain']} "
                proj_desc += f"using {', '.join(proj['technologies'])}"
                project_descriptions.append(proj_desc)

            rep_text += "Worked on following projects: " + ". ".join(project_descriptions) + ". "
            rep_text += f"This freelancer is operational since {founded_year}."

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

        # Generate agencies using loaded data or random generation
        for i in range(num_agencies):
            # Use agency details from loaded JSON if available
            agency_details = self.loaded_agencies[i] if i < len(self.loaded_agencies) else None

            details, rep_text = self.generate_representative_text(is_agency=True, agency_details=agency_details)

            # Use agency details from JSON for hourly rate if available
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
                "representativeText": rep_text
            }
            documents.append(doc)

        # Generate freelancers
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
                "representativeText": rep_text
            }
            documents.append(doc)

        return documents

def main():
    """Main function to generate and save agency/freelancer documents."""
    load_dotenv()

    try:
        # Prompt for input JSON file
        agency_json_path = input("Enter path to the JSON file with agency details (or press Enter to skip): ").strip()

        print("Starting generation of agency and freelancer documents...")
        start_time = time.time()

        # Initialize generator with optional JSON path
        generator = AgencyGenerator(agency_json_path if agency_json_path else None)

        # Generate documents
        documents = generator.generate_agencies_and_freelancers(
            num_agencies=500,
            num_freelancers=50
        )

        # Save to JSON file with timestamp
        timestamp = int(time.time())
        filename = f"agency_freelancer_docs_{timestamp}.json"

        with open(filename, "w", encoding='utf-8') as f:
            json.dump(documents, f, indent=2, ensure_ascii=False)

        end_time = time.time()

        print(f"\nSummary:")
        print(f"Successfully generated and saved {len(documents)} documents to {filename}")
        print(f"Processing completed in {end_time - start_time:.2f} seconds")

        # Show a few sample entries
        print("\n--- Sample Entries ---")
        for i, doc in enumerate(documents[:3], 1):
            print(f"\nEntry {i}:")
            print(json.dumps(doc, indent=2))

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()