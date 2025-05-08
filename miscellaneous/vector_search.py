import os
import numpy as np
from langchain_mistralai import MistralAIEmbeddings
from pymongo import MongoClient, UpdateOne
from dotenv import load_dotenv

load_dotenv()

def setup_embeddings():
    api_key = os.getenv('MISTRAL_API_KEY')
    if api_key is None:
        print('You need to set your environment variable MISTRAL_API_KEY')
        exit(1)

    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        mistral_api_key=api_key
    )
    return embeddings

def extract_documents_from_mongodb(db_name, collection_name):
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        print("You need to set MONGO_URI in your environment variables.")
        exit(1)

    client = MongoClient(mongo_uri)
    db = client["hiretalentt"]
    collection = db["profiles"]

    docs = list(collection.find())
    if not docs:
        print(f"No documents found in collection '{collection_name}'.")
        exit(1)

    print(f"Extracted {len(docs)} documents from MongoDB collection '{collection_name}'")
    return docs, collection

def prepare_profile_text(profile):
    """Create a comprehensive text representation of the profile with all fields"""
    first_name = profile.get('firstName', '')
    last_name = profile.get('lastName', '')
    full_name = f"{first_name} {last_name}".strip()

    if not full_name:
        full_name = f"Profile {profile.get('_id', 'Unknown')}"

    area_of_expertise = profile.get('areaOfExpertise', 'Not specified')

    location = profile.get('currentLocation', {}) or {}
    if not isinstance(location, dict):
        location = {}
    location_str = f"{location.get('city', '')}, {location.get('state', '')}, {location.get('country', '')}".strip()
    if not location_str or location_str == ", , ":
        location_str = "Location not specified"

    summary = profile.get('carrierSummary', '') or profile.get('summary', '') or profile.get('bio', '') or 'No summary provided.'

    education_entries = profile.get('education', []) or []
    education_str = "\n".join([
        f"- {edu.get('degree', 'Degree not specified')} at {edu.get('institute', 'Institute not specified')} ({edu.get('startDate', 'Date not specified')})"
        for edu in education_entries if isinstance(edu, dict)
    ])

    experience_entries = profile.get('experience', []) or []
    experience_str_parts = []

    for exp in experience_entries:
        if not isinstance(exp, dict):
            continue

        position = exp.get('position', 'Position not specified')
        company = exp.get('company', 'Company not specified')
        start_date = exp.get('startDate', 'Start date not specified')
        experience_entry = f"- {position} at {company} ({start_date})"
        experience_str_parts.append(experience_entry)

    experience_str = "\n".join(experience_str_parts)

    skills = profile.get('highlightedSkills', []) or []
    skill_names = []

    for skill in skills:
        if isinstance(skill, dict):
            skill_name = skill.get('name', '')
            if skill_name:
                skill_names.append(skill_name)
        elif isinstance(skill, str):
            skill_names.append(skill)

    skill_str = ", ".join(skill_names) if skill_names else "No skills listed"

    profile_text = f"""
    Name: {full_name}
    Expertise: {area_of_expertise}
    Location: {location_str}
    Summary: {summary}
    
    Education:
    {education_str if education_str else 'No education data'}
    
    Experience:
    {experience_str if experience_str else 'No experience data'}
    
    Highlighted Skills:
    {skill_str}
    """

    return profile_text

def generate_and_store_embeddings(db_name, collection_name):
    embeddings_model = setup_embeddings()
    profiles, collection = extract_documents_from_mongodb(db_name, collection_name)

    operations = []

    print(f"Generating embeddings for {len(profiles)} profiles")

    for i, profile in enumerate(profiles):
        profile_text = prepare_profile_text(profile)

        embedding = embeddings_model.embed_query(profile_text)

        operations.append(
            UpdateOne(
                {"_id": profile["_id"]},
                {"$set": {
                    "embedding": embedding
                }}
            )
        )

        if (i+1) % 10 == 0:
            print(f"Generated embeddings for {i+1}/{len(profiles)} profiles")

    if operations:
        result = collection.bulk_write(operations)
        updated_count = result.modified_count
        print(f"Updated {updated_count} profiles with embeddings")
    else:
        print("No profiles to update")

def main():
    db_name = "hiretalentt"
    collection_name = "profiles"
    generate_and_store_embeddings(db_name, collection_name)

if __name__ == "__main__":
    main()