import os
from pymongo import MongoClient, UpdateOne
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()


def connect_to_mongodb():
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        print("You need to set MONGO_URI in your environment variables.")
        exit(1)

    client = MongoClient(mongo_uri)
    db = client["hiretalentt"]
    return db


def get_agency_type(profiles):
    if len(profiles) > 1:
        return 'agency'
    return 'freelancer'


def extract_profiles_for_agency(db, agency_id):
    profiles_collection = db["profiles"]
    agency_id_str = str(agency_id)
    all_profiles = list(profiles_collection.find())
    matching_profiles = []

    for profile in all_profiles:
        business_id = profile.get('businessId')
        if business_id:
            business_id_str = str(business_id)
            if business_id_str == agency_id_str:
                matching_profiles.append(profile)

    return matching_profiles


def extract_case_studies_for_agency(db, agency_id):
    case_studies_collection = db["case_studies"]
    agency_id_str = str(agency_id)
    case_studies = list(case_studies_collection.find({"caseStudyBusinessId": agency_id_str}))
    return case_studies


def collect_all_skills(profiles, case_studies):
    skill_case_map = {}

    for profile in profiles:
        additional_skills = profile.get('additionalSkill', [])
        for skill in additional_skills:
            if isinstance(skill, dict):
                skill_name = skill.get('name', '')
                if skill_name:
                    skill_name_lower = skill_name.lower()
                    skill_case_map[skill_name_lower] = skill_name
            elif isinstance(skill, str):
                skill_name_lower = skill.lower()
                skill_case_map[skill_name_lower] = skill

        experience_entries = profile.get('experience', [])
        for exp in experience_entries:
            if isinstance(exp, dict):
                practiced_skills = exp.get('practicedSkills', [])
                for skill in practiced_skills:
                    if isinstance(skill, dict):
                        skill_name = skill.get('name', '')
                        if skill_name:
                            skill_name_lower = skill_name.lower()
                            skill_case_map[skill_name_lower] = skill_name
                    elif isinstance(skill, str):
                        skill_name_lower = skill.lower()
                        skill_case_map[skill_name_lower] = skill

    for case_study in case_studies:
        tech_stack = case_study.get('techStack', [])
        for tech in tech_stack:
            if isinstance(tech, dict):
                tech_name = tech.get('name', '')
                if tech_name:
                    tech_name_lower = tech_name.lower()
                    skill_case_map[tech_name_lower] = tech_name
            elif isinstance(tech, str):
                tech_name_lower = tech.lower()
                skill_case_map[tech_name_lower] = tech

    return list(skill_case_map.values())


def collect_projects_information(case_studies):
    projects_info = []
    tech_case_map = {}

    for case_study in case_studies:
        tech_stack = []
        for tech in case_study.get('techStack', []):
            if isinstance(tech, dict):
                tech_name = tech.get('name', '')
                if tech_name:
                    tech_name_lower = tech_name.lower()
                    tech_case_map[tech_name_lower] = tech_name
                    tech_stack.append(tech_name_lower)
            elif isinstance(tech, str):
                tech_name_lower = tech.lower()
                tech_case_map[tech_name_lower] = tech
                tech_stack.append(tech_name_lower)

        project_type = case_study.get('type', '')
        if project_type:
            projects_info.append({
                'type': project_type,
                'tech_stack': tech_stack,
                'tech_case_map': tech_case_map
            })

    return projects_info


def generate_agency_text(agency, profiles, case_studies):
    if not profiles:
        return ""

    agency_type = get_agency_type(profiles)
    name = agency.get('name', 'Unnamed Agency')

    if agency_type == 'freelancer' and profiles:
        name = profiles[0].get('firstName', 'Unnamed Professional')

    member_count = len(profiles)
    member_text = f" has {member_count} team members" if agency_type == 'agency' else ""

    founded_date = agency.get('founded')
    founded_text = ""
    if founded_date:
        try:
            founded_text = f" This {agency_type} is operational since {founded_date}."
        except:
            pass

    all_skills = collect_all_skills(profiles, case_studies)
    skills_text = ""
    if all_skills:
        skills_text = f" with the following skill sets: {', '.join(all_skills)}"

    projects_info = collect_projects_information(case_studies)
    projects_by_type = defaultdict(list)
    tech_case_map = {}

    for project in projects_info:
        project_type = project['type']
        tech_stack = project['tech_stack']
        project_tech_case_map = project.get('tech_case_map', {})
        tech_case_map.update(project_tech_case_map)

        if tech_stack:
            projects_by_type[project_type].extend(tech_stack)

    projects_text = ""
    if projects_by_type:
        projects_parts = []
        for project_type, tech_stack in projects_by_type.items():
            unique_tech_lower = set(tech_stack)
            unique_tech = [tech_case_map.get(tech.lower(), tech) for tech in unique_tech_lower]

            if unique_tech:
                projects_parts.append(f"{project_type} using {', '.join(unique_tech)}")
            else:
                projects_parts.append(project_type)

        if projects_parts:
            projects_text = f" and have done projects on {' and '.join(projects_parts)}."

    agency_text = f"{name} is a {agency_type}{member_text}{skills_text}{projects_text}{founded_text}"

    return agency_text.strip()


def generate_and_store_agency_texts():
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

        agency_text = generate_agency_text(agency, profiles, case_studies)

        update_data = {
            "representativeText": agency_text,
        }

        operations.append(
            UpdateOne(
                {"_id": agency_id},
                {"$set": update_data}
            )
        )

        if len(operations) >= 50:
            result = agencies_collection.bulk_write(operations)
            print(f"Bulk updated {result.modified_count} agencies")
            operations = []

    if operations:
        result = agencies_collection.bulk_write(operations)
        print(f"Bulk updated {result.modified_count} agencies")


def main():
    print("Starting agency text generation process...")
    generate_and_store_agency_texts()
    print("Agency text generation process completed.")


if __name__ == "__main__":
    main()