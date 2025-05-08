import os
from pymongo import MongoClient, UpdateOne
from collections import Counter, defaultdict
from datetime import datetime
from dotenv import load_dotenv

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

def get_agency_type(agency):
    """Determine if agency is a business or freelancer."""
    if agency.get('teamSize', 0) > 1 or agency.get('businessType') == 'company':
        return 'business'
    return 'freelancer'

def extract_agency_basic_info(agency):
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

    founded = agency.get('founded', 'N/A')

    locations = agency.get('locations', [])
    location_strs = []
    for loc in locations:
        if isinstance(loc, dict):
            city = loc.get('city', '')
            state = loc.get('state', '')
            country = loc.get('country', {})
            country_name = country.get('countryName', '') if isinstance(country, dict) else ''
            phone = loc.get('phoneNumber', '')

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

    website = agency.get('website', '')

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

    approval_status = ""
    if agency.get('approvedByHiretalentt') is True:
        approval_status = " This agency is approved by HireTalentt."

    basic_info = f"{name} is a {'company' if get_agency_type(agency) == 'business' else 'freelancer'}"
    if founded != 'N/A':
        basic_info += f" operational since {founded}"

    basic_info += f". {description}"

    if tagline:
        basic_info += f" {tagline}"

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
    """Extract all case studies related to a specific agency."""
    case_studies_collection = db["casestudies"]
    case_studies = list(case_studies_collection.find({"caseStudyBusinessId": agency_id}))
    return case_studies

def collect_skills_from_profiles(profiles):
    """Collect and count skills from profiles."""
    skill_counter = Counter()

    for profile in profiles:
        highlighted_skills = profile.get('highlightedSkills', [])
        for skill in highlighted_skills:
            if isinstance(skill, dict):
                skill_name = skill.get('name', '')
                if skill_name:
                    skill_counter[skill_name] += 1
            elif isinstance(skill, str):
                skill_counter[skill] += 1

        additional_skills = profile.get('additionalSkill', [])
        for skill in additional_skills:
            if isinstance(skill, dict):
                skill_name = skill.get('name', '')
                if skill_name:
                    skill_counter[skill_name] += 1
            elif isinstance(skill, str):
                skill_counter[skill] += 1

        experience_entries = profile.get('experience', [])
        for exp in experience_entries:
            if isinstance(exp, dict):
                practiced_skills = exp.get('practicedSkills', [])
                for skill in practiced_skills:
                    if isinstance(skill, dict):
                        skill_name = skill.get('name', '')
                        if skill_name:
                            skill_counter[skill_name] += 1
                    elif isinstance(skill, str):
                        skill_counter[skill] += 1

    return skill_counter

def collect_education_and_certifications(profiles):
    """Collect education and certification information from profiles."""
    education_data = []
    certification_data = []

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
            if isinstance(cert, dict):
                cert_name = cert.get('name', '')
                if cert_name:
                    certification_data.append(cert_name)
            elif isinstance(cert, str):
                certification_data.append(cert)

    education_data = list(set(education_data))
    certification_data = list(set(certification_data))

    return education_data, certification_data

def collect_skills_from_case_studies(case_studies):
    """Collect and count skills from case studies."""
    skill_counter = Counter()

    for case_study in case_studies:
        tech_stack = case_study.get('techStack', [])
        for tech in tech_stack:
            if isinstance(tech, dict):
                tech_name = tech.get('name', '')
                if tech_name:
                    skill_counter[tech_name] += 1
            elif isinstance(tech, str):
                skill_counter[tech] += 1

        teams = case_study.get('teams', [])
        for team_member in teams:
            if isinstance(team_member, dict):
                member_skills = team_member.get('highlightedSkills', [])
                for skill in member_skills:
                    if isinstance(skill, dict):
                        skill_name = skill.get('name', '')
                        if skill_name:
                            skill_counter[skill_name] += 1
                    elif isinstance(skill, str):
                        skill_counter[skill] += 1

    return skill_counter

def collect_projects_by_industry(case_studies):
    """Organize projects by industry and collect used technologies and clients."""
    projects_by_industry = defaultdict(list)
    clients = set()

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

        if client:
            clients.add(client)

        projects_by_industry[industry].append({
            'title': title,
            'technologies': tech_stack,
            'client': client,
            'duration': duration,
            'type': project_type
        })

    return projects_by_industry, list(clients)

def generate_agency_text(agency, profiles, case_studies):
    """Generate comprehensive representative text for an agency."""
    agency_info = extract_agency_basic_info(agency)

    profile_skills = collect_skills_from_profiles(profiles)
    case_study_skills = collect_skills_from_case_studies(case_studies)

    all_skills = profile_skills + case_study_skills
    top_skills = [skill for skill, count in all_skills.most_common(20)]

    education_data, certification_data = collect_education_and_certifications(profiles)
    projects_by_industry, clients = collect_projects_by_industry(case_studies)

    team_size = len(profiles)
    team_info = f"The team consists of {team_size} {'members' if team_size > 1 else 'member'}"

    skills_text = ""
    if top_skills:
        skills_text = f" with expertise in {', '.join(top_skills[:10])}"
        if len(top_skills) > 10:
            skills_text += f" and other technologies"

    education_text = ""
    if education_data:
        top_education = education_data[:5]
        education_text = f" Team members have educational backgrounds from institutions including {', '.join(top_education)}"
        if len(education_data) > 5:
            education_text += " and others"

    certification_text = ""
    if certification_data:
        top_certifications = certification_data[:5]
        certification_text = f" Certifications held include {', '.join(top_certifications)}"
        if len(certification_data) > 5:
            certification_text += " and more"

    projects_text = ""
    if projects_by_industry:
        industry_parts = []

        for industry, projects in projects_by_industry.items():
            if industry == 'Other' and len(projects_by_industry) > 1:
                continue

            all_tech = []
            project_types = set()
            durations = []

            for project in projects:
                all_tech.extend(project['technologies'])
                if project['type']:
                    project_types.add(project['type'])
                if project['duration']:
                    durations.append(project['duration'])

            tech_counter = Counter(all_tech)
            top_tech = [tech for tech, count in tech_counter.most_common(5)]

            industry_text = f"{industry}"

            if project_types:
                industry_text += f" ({', '.join(list(project_types)[:3])})"

            if top_tech:
                industry_text += f" using {', '.join(top_tech)}"

            if durations:
                avg_duration = max(set(durations), key=durations.count) if durations else ""
                if avg_duration:
                    industry_text += f" with typical project duration of {avg_duration}"

            industry_parts.append(industry_text)

        if industry_parts:
            projects_text = f" They specialize in projects across {' and '.join(industry_parts)}."

    clients_text = ""
    if clients:
        top_clients = clients[:5]
        clients_text = f" They have worked with clients such as {', '.join(top_clients)}"
        if len(clients) > 5:
            clients_text += " and others"

    agency_text = (
        f"{agency_info} {team_info}{skills_text}.{education_text}.{certification_text}."
        f"{projects_text}{clients_text}."
    )

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

        print(f"Processing agency {i+1}/{len(agencies)}: {agency.get('name', 'Unnamed')} - {len(profiles)} profiles, {len(case_studies)} case studies")

        representative_text = generate_agency_text(agency, profiles, case_studies)

        operations.append(
            UpdateOne(
                {"_id": agency_id},
                {"$set": {
                    "representativeText": representative_text,
                    "textGeneratedAt": datetime.now()
                }}
            )
        )

        if len(operations) >= 50:
            result = agencies_collection.bulk_write(operations)
            print(f"Updated {result.modified_count} agencies with representative text")
            operations = []

    if operations:
        result = agencies_collection.bulk_write(operations)
        print(f"Updated {result.modified_count} agencies with representative text")

def main():
    """Main function to run the agency text generation process."""
    print("Starting agency text generation process...")
    generate_and_store_agency_texts()
    print("Agency text generation process completed.")

if __name__ == "__main__":
    main()