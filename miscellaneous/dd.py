# import os
# from mistralai import Mistral
# import datetime
# from dotenv import load_dotenv
#
# # Load environment variables from .env file
# load_dotenv()
#
#
# class TextEnhancer:
#     def __init__(self, api_key):
#         """Initialize the TextEnhancer with Mistral client."""
#         self.api_key = api_key
#         self.mistral_client = Mistral(api_key=api_key)
#         self.model = "mistral-large-latest"
#
#     def enhance_text(self, raw_text):
#         """
#         Use Mistral client to enhance the text about an agency or freelancer.
#
#         Args:
#             raw_text (str): The raw text about the agency or freelancer
#
#         Returns:
#             str: Enhanced, more natural-sounding text
#         """
#         # Create a prompt that instructs Mistral how to improve the text
#         prompt = f"""
#         You are a professional copywriter specializing in enhancing professional profiles.
#
#         Below is a raw text description of a freelancer or agency from a talent platform.
#         Your task is to improve this text to make it:
#         - More engaging and natural-sounding
#         - Well-structured with good flow between ideas
#         - Professional and confident in tone
#         - Fix any grammatical errors or redundant phrases
#
#         IMPORTANT REQUIREMENTS:
#         - DO NOT add any information that is not present in the original text
#         - DO NOT use marketing buzzwords or hype language
#         - DO NOT change any factual information, including names, locations, rates, etc.
#         - DO remove redundant or nonsensical phrases like repetitions of words
#         - DO fix grammatical errors and improve sentence structure
#         - DO PRESERVE all placeholder text (like "sdsdsd", "dsad") as they may be template placeholders
#         - KEEP all meaningful content but present it in a more coherent way
#
#         Here is the text to enhance:
#
#         {raw_text}
#
#         Respond with ONLY the enhanced text, with no additional comments or explanations.
#         """
#
#         messages = [
#             {"role": "user", "content": prompt}
#         ]
#
#         try:
#             response = self.mistral_client.chat(
#                 model=self.model,
#                 messages=messages
#             )
#
#             enhanced_text = response.choices[0].message.content.strip()
#             return enhanced_text
#         except Exception as e:
#             print(f"Error calling Mistral API: {e}")
#             # In case of failure, return the original text
#             return raw_text
#
#
# def save_text_to_file(text, filename):
#     """
#     Save text to a file in the current directory
#
#     Args:
#         text (str): Text to save
#         filename (str): Name of the file
#     """
#     with open(filename, 'w', encoding='utf-8') as file:
#         file.write(text)
#     print(f"Text saved to {filename}")
#
#
# def get_raw_text():
#     """Get raw text from user input."""
#     print("\nEnter or paste the raw text to enhance (press Enter twice when done):")
#     lines = []
#     while True:
#         line = input()
#         if not line and lines and not lines[-1]:
#             break
#         lines.append(line)
#     return '\n'.join(lines)
#
#
# def main():
#     # Check for API key
#     api_key = os.getenv('MISTRAL_API_KEY')
#     if api_key is None:
#         print('You need to set your MISTRAL_API_KEY environment variable in the .env file')
#         exit(1)
#
#     # Get raw text from user
#     raw_text = get_raw_text()
#
#     # Exit if no text was provided
#     if not raw_text.strip():
#         print("No text was provided. Exiting.")
#         return
#
#     # Create output filename with timestamp
#     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#     output_filename = f"enhanced_text_{timestamp}.txt"
#
#     # Save the raw text
#     raw_filename = f"raw_text_{timestamp}.txt"
#     save_text_to_file(raw_text, raw_filename)
#
#     print("\nEnhancing text using Mistral API...")
#
#     # Enhance the text
#     try:
#         enhancer = TextEnhancer(api_key)
#         enhanced_text = enhancer.enhance_text(raw_text)
#
#         # Save the enhanced text
#         save_text_to_file(enhanced_text, output_filename)
#
#         print("\n=== ENHANCED TEXT ===")
#         print(enhanced_text)
#         print(f"\nEnhanced text saved to {output_filename}")
#
#     except Exception as e:
#         print(f"Error enhancing text: {e}")
#
#
# if __name__ == "__main__":
#     main()
import os
from pymongo import MongoClient, UpdateOne
from collections import Counter, defaultdict
from datetime import datetime
from dotenv import load_dotenv
from bson import ObjectId

load_dotenv()

def connect_to_mongodb():
    """Connect to MongoDB using environment variables."""
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        print("You need to set MONGO_URI in your environment variables.")
        exit(1)

    client = MongoClient(mongo_uri)
    db = client["hiretalentt"]
    return db

def get_agency_type(agency, profiles):
    """Determine if agency is a business or freelancer based on team size."""
    if len(profiles) > 1:
        return 'business'
    return 'freelancer'

def extract_agency_basic_info(agency, agency_type):
    """Extract basic information from agency document."""
    name = agency.get('name', 'Unnamed Agency')
    description = agency.get('description', '')
    tagline = agency.get('tagline', '')

    hourly_rate = agency.get('hourlyRate', {})
    if isinstance(hourly_rate, dict):
        min_rate = hourly_rate.get('min', 'N/A')
        max_rate = hourly_rate.get('max', 'N/A')
        currency = hourly_rate.get('currency', 'USD')
        rate_str = f"{currency} {min_rate}-{max_rate}" if min_rate != 'N/A' and max_rate != 'N/A' else "Rate not specified"
    else:
        rate_str = "Rate not specified"

    locations = agency.get('locations', [])
    location_strs = []
    for loc in locations:
        if isinstance(loc, dict):
            city = loc.get('city', '')
            state = loc.get('state', '')
            country = loc.get('country', {})
            country_name = country.get('countryName', '') if isinstance(country, dict) else ''

            loc_parts = []
            if city:
                loc_parts.append(city)
            if state:
                loc_parts.append(state)
            if country_name:
                loc_parts.append(country_name)

            loc_str = ", ".join(loc_parts)
            if loc_str:
                location_strs.append(loc_str)

    location_text = ", ".join(location_strs) if location_strs else "Location not specified"

    services = agency.get('services', [])
    service_names = []
    for service in services:
        if isinstance(service, dict) and service.get('name'):
            service_names.append(service.get('name'))

    services_text = ""
    if service_names:
        services_text = f" They specialize in {', '.join(service_names)}."

    website = agency.get('website', '')

    approval_status = ""
    if agency.get('approvedByHiretalentt') is True:
        approval_status = " This agency is approved by HireTalentt."

    # Basic info starts with name and type
    basic_info = f"{name} is a {agency_type}"

    # Add description if available
    if description:
        basic_info += f". {description}"

    # Add tagline if available
    if tagline:
        basic_info += f" {tagline}"

    # Add services
    basic_info += services_text

    # Add location if available
    if location_text != "Location not specified":
        basic_info += f" Based in {location_text}."

    # Add website if available
    if website:
        basic_info += f" Website: {website}."

    # Add rate if available
    if rate_str != "Rate not specified":
        basic_info += f" Typical hourly rate: {rate_str}."

    # Add approval status
    basic_info += approval_status

    return basic_info.strip()

def extract_profiles_for_agency(db, agency_id):
    """Extract all profiles related to a specific agency."""
    profiles_collection = db["profiles"]
    profiles = list(profiles_collection.find({"businessId": agency_id}))
    return profiles

def extract_case_studies_for_agency(db, agency_id):
    """Extract all case studies related to a specific agency, handling both ObjectId and string IDs."""
    case_studies_collection = db["case_studies"]

    agency_id_str = str(agency_id)

    case_studies = list(case_studies_collection.find({"caseStudyBusinessId": agency_id_str}))

    return case_studies

def collect_all_skills(profiles, case_studies):
    """Collect all unique skills from profiles and case studies."""
    unique_skills = set()

    for profile in profiles:
        highlighted_skills = profile.get('highlightedSkills', [])
        for skill in highlighted_skills:
            if isinstance(skill, dict):
                skill_name = skill.get('name', '')
                if skill_name:
                    unique_skills.add(skill_name)
            elif isinstance(skill, str):
                unique_skills.add(skill)

        additional_skills = profile.get('additionalSkill', [])
        for skill in additional_skills:
            if isinstance(skill, dict):
                skill_name = skill.get('name', '')
                if skill_name:
                    unique_skills.add(skill_name)
            elif isinstance(skill, str):
                unique_skills.add(skill)

        experience_entries = profile.get('experience', [])
        for exp in experience_entries:
            if isinstance(exp, dict):
                practiced_skills = exp.get('practicedSkills', [])
                for skill in practiced_skills:
                    if isinstance(skill, dict):
                        skill_name = skill.get('name', '')
                        if skill_name:
                            unique_skills.add(skill_name)
                    elif isinstance(skill, str):
                        unique_skills.add(skill)

    for case_study in case_studies:
        tech_stack = case_study.get('techStack', [])
        for tech in tech_stack:
            if isinstance(tech, dict):
                tech_name = tech.get('name', '')
                if tech_name:
                    unique_skills.add(tech_name)
            elif isinstance(tech, str):
                unique_skills.add(tech)

        teams = case_study.get('teams', [])
        for team_member in teams:
            if isinstance(team_member, dict):
                member_skills = team_member.get('highlightedSkills', [])
                for skill in member_skills:
                    if isinstance(skill, dict):
                        skill_name = skill.get('name', '')
                        if skill_name:
                            unique_skills.add(skill_name)
                    elif isinstance(skill, str):
                        unique_skills.add(skill)

    return list(unique_skills)

def collect_education_and_certifications(profiles):
    """Collect education and certification information from profiles."""
    education_data = []
    certification_data = []
    experience_data = []

    name_exclusions = [
        "Uzair Uddin", "John", "Smith", "Johnson", "Williams", "Brown", "Jones", "Miller",
        "Davis", "Garcia", "Rodriguez", "Wilson", "Martinez", "Anderson", "Taylor", "Thomas",
        "Hernandez", "Moore", "Martin", "Jackson", "Thompson", "White", "Lopez", "Lee", "Gonzalez"
    ]

    name_exclusions_lower = {name.lower() for name in name_exclusions}

    for profile in profiles:
        education_entries = profile.get('education', [])
        for edu in education_entries:
            if isinstance(edu, dict):
                institution = edu.get('institution', '')
                degree = edu.get('degree', '')
                if institution and degree:
                    education_data.append(f"{degree} from {institution}")
                elif institution:
                    education_data.append(institution)

        cert_entries = profile.get('certifications', [])
        for cert in cert_entries:
            cert_name = ""
            if isinstance(cert, dict):
                cert_name = cert.get('name', '')
            elif isinstance(cert, str):
                cert_name = cert

            if cert_name and cert_name.lower() not in name_exclusions_lower:
                if not (len(cert_name.split()) >= 2 and cert_name.split()[0][0].isupper() and cert_name.split()[1][0].isupper()):
                    certification_data.append(cert_name)

        career_summary = profile.get('carrierSummary', '')
        if career_summary:
            experience_data.append(career_summary)

        experience_entries = profile.get('experience', [])
        for exp in experience_entries:
            if isinstance(exp, dict):
                position = exp.get('position', '')
                company = exp.get('company', '')
                description = exp.get('responsibilityDescription', '')

                if position and company:
                    experience_data.append(f"{position} at {company}")

                if description:
                    experience_data.append(description)

    education_data = list(set(education_data))
    certification_data = list(set(certification_data))
    experience_data = list(set(experience_data))

    return education_data, certification_data, experience_data

def collect_projects_information(case_studies):
    """Collect project information including industries, clients, and features."""
    projects_by_industry = defaultdict(list)
    clients = set()
    all_features = set()
    all_project_types = set()

    for case_study in case_studies:
        industry = case_study.get('industry', 'Other')
        if industry == 'none' or not industry:
            industry = 'Other'

        tech_stack = []
        for tech in case_study.get('techStack', []):
            if isinstance(tech, dict):
                tech_name = tech.get('name', '')
                if tech_name:
                    tech_stack.append(tech_name)
            elif isinstance(tech, str):
                tech_stack.append(tech)

        title = case_study.get('title', 'Untitled Project')
        client = case_study.get('client', '')
        # duration = case_study.get('duration', '')
        project_type = case_study.get('type', '')
        overview = case_study.get('overview', '')
        feature_statement = case_study.get('featureStatement', '')

        features = case_study.get('features', [])
        for feature in features:
            if isinstance(feature, dict):
                feature_title = feature.get('title', '')
                if feature_title:
                    all_features.add(feature_title)

        if client:
            clients.add(client)

        if project_type:
            all_project_types.add(project_type)

        projects_by_industry[industry].append({
            'title': title,
            'technologies': tech_stack,
            'client': client,
            # 'duration': duration,
            'type': project_type,
            'overview': overview,
            'feature_statement': feature_statement
        })

    return projects_by_industry, list(clients), list(all_features), list(all_project_types)

def generate_agency_text(agency, profiles, case_studies):
    """Generate comprehensive representative text for an agency."""
    agency_type = get_agency_type(agency, profiles)

    # Get basic info
    agency_info = extract_agency_basic_info(agency, agency_type)

    all_skills = collect_all_skills(profiles, case_studies)

    education_data, certification_data, experience_data = collect_education_and_certifications(profiles)

    projects_by_industry, clients, features, project_types = collect_projects_information(case_studies)

    # Check if we have any substantial data
    has_data = bool(all_skills or education_data or certification_data or
                    experience_data or projects_by_industry or clients or
                    features or project_types)

    # If we have no data, just return the basic info
    if not has_data:
        return agency_info

    # Build skills section
    skills_text = ""
    if all_skills:
        # FIX: Change wording based on agency type to avoid redundancy
        if agency_type == 'freelancer':
            skills_text = f" has expertise in {', '.join(all_skills)}"
        else:
            skills_text = f" with expertise in {', '.join(all_skills)}"

    # Build education section
    education_text = ""
    if education_data:
        if agency_type == 'freelancer':
            education_text = f" They have educational background from {', '.join(education_data)}."
        else:
            education_text = f" Team members have educational backgrounds from {', '.join(education_data)}."

    # Build certification section
    certification_text = ""
    if certification_data:
        certification_text = f" Certifications held include {', '.join(certification_data)}."

    # Build experience section
    experience_text = ""
    if experience_data:
        key_experiences = experience_data[:5]
        if len(experience_data) > 5:
            if agency_type == 'freelancer':
                experience_text = f" They have extensive experience across various roles and responsibilities."
            else:
                experience_text = f" The team has extensive experience across various roles and responsibilities."
        else:
            if agency_type == 'freelancer':
                experience_text = f" They have experience in {', '.join(key_experiences)}."
            else:
                experience_text = f" The team has experience in {', '.join(key_experiences)}."

    # Build project types section
    project_types_text = ""
    if project_types:
        project_types_text = f" They specialize in {', '.join(project_types)} projects."

    # Build industry section
    industry_text = ""
    if projects_by_industry:
        industry_parts = []

        for industry, projects in projects_by_industry.items():
            if industry == 'Other' and len(projects_by_industry) > 1:
                continue

            all_tech = []
            for project in projects:
                all_tech.extend(project['technologies'])

            tech_set = set(all_tech)

            if tech_set:
                industry_parts.append(f"{industry} using {', '.join(tech_set)}")
            else:
                industry_parts.append(industry)

        if industry_parts:
            industry_text = f" They work across industries including {', '.join(industry_parts)}."

    # Build clients section
    clients_text = ""
    if clients:
        clients_text = f" They have worked with clients such as {', '.join(clients)}."

    # Build features section
    features_text = ""
    if features:
        features_text = f" Their projects feature {', '.join(features)}."

    # Put it all together
    # FIX: Check each section before adding it to avoid empty sections
    # First add skills to the basic info
    if skills_text:
        agency_text = f"{agency_info}{skills_text}."
    else:
        agency_text = agency_info

    # Now add additional sections only if they have content
    additional_sections = []
    if education_text.strip():
        additional_sections.append(education_text)
    if certification_text.strip():
        additional_sections.append(certification_text)
    if experience_text.strip():
        additional_sections.append(experience_text)
    if project_types_text.strip():
        additional_sections.append(project_types_text)
    if industry_text.strip():
        additional_sections.append(industry_text)
    if clients_text.strip():
        additional_sections.append(clients_text)
    if features_text.strip():
        additional_sections.append(features_text)

    # Add the additional sections to the agency text
    if additional_sections:
        agency_text += " " + " ".join(additional_sections)

    # Clean up any double spaces or double periods
    agency_text = agency_text.replace("  ", " ")
    agency_text = agency_text.replace("..", ".")

    return agency_text.strip()

def generate_and_store_agency_texts():
    """Generate representative texts for all agencies and store them in MongoDB."""
    db = connect_to_mongodb()
    agencies_collection = db["agencies2"]

    agencies = list(agencies_collection.find())
    print(f"Found {len(agencies)} agencies")

    operations = []

    for i, agency in enumerate(agencies):
        agency_id = agency.get('_id')

        profiles = extract_profiles_for_agency(db, agency_id)
        case_studies = extract_case_studies_for_agency(db, agency_id)

        print(f"Processing agency {i+1}/{len(agencies)}: {agency.get('name', 'Unnamed')} - {len(profiles)} profiles, {len(case_studies)} case studies)")

        agency_text = generate_agency_text(agency, profiles, case_studies)

        operations.append(
            UpdateOne(
                {"_id": agency_id},
                {"$set": {"representativeText": agency_text, "textGeneratedAt": datetime.now()}}
            )
        )

        print(f"Generated text for {agency.get('name', 'Unnamed')}: {agency_text[:100]}...")

        if len(operations) >= 50:
            result = agencies_collection.bulk_write(operations)
            print(f"Bulk updated {result.modified_count} agencies")
            operations = []

    if operations:
        result = agencies_collection.bulk_write(operations)
        print(f"Bulk updated {result.modified_count} agencies")

def main():
    """Main function to run the agency text generation process."""
    print("Starting agency text generation process...")
    generate_and_store_agency_texts()
    print("Agency text generation process completed.")

if __name__ == "__main__":
    main()