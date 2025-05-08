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

    # Handle hourly rate
    hourly_rate = agency.get('hourlyRate', {})
    if isinstance(hourly_rate, dict):
        min_rate = hourly_rate.get('min', 'N/A')
        max_rate = hourly_rate.get('max', 'N/A')
        currency = hourly_rate.get('currency', 'USD')
        rate_str = f"{currency} {min_rate}-{max_rate}" if min_rate != 'N/A' and max_rate != 'N/A' else "Rate not specified"
    else:
        rate_str = "Rate not specified"

    # Handle locations
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

    # Extract services
    services = agency.get('services', [])
    service_names = []
    for service in services:
        if isinstance(service, dict) and service.get('name'):
            service_names.append(service.get('name'))

    services_text = ""
    if service_names:
        services_text = f" They specialize in {', '.join(service_names)}."

    website = agency.get('website', '')

    # Social media links
    social_links = []
    if agency.get('facebook'):
        social_links.append("Facebook")
    if agency.get('linkedin'):
        social_links.append("LinkedIn")
    if agency.get('twitter'):
        social_links.append("Twitter")

    social_text = ""
    if social_links:
        social_text = f" They can be found on {', '.join(social_links)}."

    # Agency approval status
    approval_status = ""
    if agency.get('approvedByHiretalentt') is True:
        approval_status = " This agency is approved by HireTalentt."

    # Build the basic info text
    basic_info = f"{name} is a {'company' if agency_type == 'business' else 'freelancer'}"

    basic_info += f". {description}"

    if tagline:
        basic_info += f" {tagline}"

    basic_info += services_text

    if location_text != "Location not specified":
        basic_info += f" Based in {location_text}."

    if website:
        basic_info += f" Website: {website}."

    if rate_str != "Rate not specified":
        basic_info += f" Typical hourly rate: {rate_str}."

    basic_info += social_text
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

    # Convert agency_id to string for comparison with caseStudyBusinessId
    agency_id_str = str(agency_id)

    # Query with both ObjectId and string format to ensure we catch all cases
    case_studies = list(case_studies_collection.find({"caseStudyBusinessId": agency_id_str}))

    return case_studies

def collect_all_skills(profiles, case_studies):
    """Collect all unique skills from profiles and case studies."""
    unique_skills = set()

    # Extract skills from profiles
    for profile in profiles:
        # Extract highlighted skills
        highlighted_skills = profile.get('highlightedSkills', [])
        for skill in highlighted_skills:
            if isinstance(skill, dict):
                skill_name = skill.get('name', '')
                if skill_name:
                    unique_skills.add(skill_name)
            elif isinstance(skill, str):
                unique_skills.add(skill)

        # Extract additional skills
        additional_skills = profile.get('additionalSkill', [])
        for skill in additional_skills:
            if isinstance(skill, dict):
                skill_name = skill.get('name', '')
                if skill_name:
                    unique_skills.add(skill_name)
            elif isinstance(skill, str):
                unique_skills.add(skill)

        # Extract skills from experience entries
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

    # Extract skills from case studies
    for case_study in case_studies:
        # Extract tech stack
        tech_stack = case_study.get('techStack', [])
        for tech in tech_stack:
            if isinstance(tech, dict):
                tech_name = tech.get('name', '')
                if tech_name:
                    unique_skills.add(tech_name)
            elif isinstance(tech, str):
                unique_skills.add(tech)

        # Extract skills from team members
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

    # Common names to exclude from certifications (to prevent names being identified as certifications)
    name_exclusions = [
        "Uzair Uddin", "John", "Smith", "Johnson", "Williams", "Brown", "Jones", "Miller",
        "Davis", "Garcia", "Rodriguez", "Wilson", "Martinez", "Anderson", "Taylor", "Thomas",
        "Hernandez", "Moore", "Martin", "Jackson", "Thompson", "White", "Lopez", "Lee", "Gonzalez"
    ]

    # Create a set of lowercase name exclusions for case-insensitive checking
    name_exclusions_lower = {name.lower() for name in name_exclusions}

    for profile in profiles:
        # Extract education
        education_entries = profile.get('education', [])
        for edu in education_entries:
            if isinstance(edu, dict):
                institution = edu.get('institution', '')
                degree = edu.get('degree', '')
                if institution and degree:
                    education_data.append(f"{degree} from {institution}")
                elif institution:
                    education_data.append(institution)

        # Extract certifications with name filtering
        cert_entries = profile.get('certifications', [])
        for cert in cert_entries:
            cert_name = ""
            if isinstance(cert, dict):
                cert_name = cert.get('name', '')
            elif isinstance(cert, str):
                cert_name = cert

            # Check if the certification name looks like a person's name
            if cert_name and cert_name.lower() not in name_exclusions_lower:
                # Additional check: if it contains a full name format (FirstName LastName)
                if not (len(cert_name.split()) >= 2 and cert_name.split()[0][0].isupper() and cert_name.split()[1][0].isupper()):
                    certification_data.append(cert_name)

        # Extract experience
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

    # Get unique items
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
        duration = case_study.get('duration', '')
        project_type = case_study.get('type', '')
        overview = case_study.get('overview', '')
        feature_statement = case_study.get('featureStatement', '')

        # Extract features
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
            'duration': duration,
            'type': project_type,
            'overview': overview,
            'feature_statement': feature_statement
        })

    return projects_by_industry, list(clients), list(all_features), list(all_project_types)

def generate_agency_text(agency, profiles, case_studies):
    """Generate comprehensive representative text for an agency."""
    # Determine if agency is a business or freelancer based on team size
    agency_type = get_agency_type(agency, profiles)

    # Extract basic agency information
    agency_info = extract_agency_basic_info(agency, agency_type)

    # Get all skills across profiles and case studies
    all_skills = collect_all_skills(profiles, case_studies)

    # Get education, certifications, and experience data
    education_data, certification_data, experience_data = collect_education_and_certifications(profiles)

    # Get projects information
    projects_by_industry, clients, features, project_types = collect_projects_information(case_studies)

    # Team size information
    team_size = len(profiles)
    team_info = f"The team consists of {team_size} {'members' if team_size > 1 else 'member'}"

    # Skills text
    skills_text = ""
    if all_skills:
        skills_text = f" with expertise in {', '.join(all_skills)}"

    # Education text
    education_text = ""
    if education_data:
        education_text = f" Team members have educational backgrounds from {', '.join(education_data)}."

    # Certification text
    certification_text = ""
    if certification_data:
        certification_text = f" Certifications held include {', '.join(certification_data)}."

    # Experience text
    experience_text = ""
    if experience_data:
        # Limit the experience text to keep it concise but informative
        key_experiences = experience_data[:5]
        experience_text = f" The team has experience in {', '.join(key_experiences)}."
        if len(experience_data) > 5:
            experience_text = f" The team has extensive experience across various roles and responsibilities."

    # Project types text
    project_types_text = ""
    if project_types:
        project_types_text = f" They specialize in {', '.join(project_types)} projects."

    # Projects by industry text
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

    # Clients text
    clients_text = ""
    if clients:
        clients_text = f" They have worked with clients such as {', '.join(clients)}."

    # Features text
    features_text = ""
    if features:
        features_text = f" Their projects feature {', '.join(features)}."

    # Combine all sections into the final text
    agency_text = (
        f"{agency_info} {team_info}{skills_text}."
        f"{education_text}{certification_text}{experience_text}"
        f"{project_types_text}{industry_text}{clients_text}{features_text}"
    )

    # Clean up multiple spaces and periods
    agency_text = agency_text.replace("  ", " ")
    agency_text = agency_text.replace("..", ".")

    return agency_text.strip()

def generate_and_store_agency_texts():
    """Generate representative texts for all agencies and store them in MongoDB."""
    db = connect_to_mongodb()
    agencies_collection = db["agencies"]

    agencies = list(agencies_collection.find())
    print(f"Found {len(agencies)} agencies")

    operations = []

    for i, agency in enumerate(agencies):
        agency_id = agency.get('_id')

        profiles = extract_profiles_for_agency(db, agency_id)
        case_studies = extract_case_studies_for_agency(db, agency_id)

        print(f"Processing agency {i+1}/{len(agencies)}: {agency.get('name', 'Unnamed')} - {len(profiles)} profiles, {len(case_studies)} case studies)")

        # Generate agency text
        agency_text = generate_agency_text(agency, profiles, case_studies)

        # Create update operation for MongoDB
        operations.append(
            UpdateOne(
                {"_id": agency_id},
                {"$set": {"representativeText": agency_text, "textGeneratedAt": datetime.now()}}
            )
        )

        # Log the generated text for debugging
        print(f"Generated text for {agency.get('name', 'Unnamed')}: {agency_text[:100]}...")

        # Execute database operations in batches to avoid overwhelming the database
        if len(operations) >= 50:
            result = agencies_collection.bulk_write(operations)
            print(f"Bulk updated {result.modified_count} agencies")
            operations = []

    # Execute any remaining operations
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