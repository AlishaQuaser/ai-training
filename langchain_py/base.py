import os
from pymongo import MongoClient
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

load_dotenv()  # Load env variables from .env file


def get_mongo_client():
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise ValueError("MONGO_URI not set in environment variables.")
    return MongoClient(mongo_uri)


def load_profiles_from_db():
    client = get_mongo_client()
    db = client["app-dev"]
    collection = db["profiles"]

    profiles = list(collection.find())
    if not profiles:
        return "No profiles found."

    all_profiles_string = ""
    for profile in profiles:
        full_name = f"{profile.get('firstName', '')} {profile.get('lastName', '')}".strip()
        area = profile.get('areaOfExpertise', 'Not specified')
        location = profile.get('currentLocation', {})
        location_str = f"{location.get('city', '')}, {location.get('state', '')}, {location.get('country', '')}"
        summary = profile.get('carrierSummary', 'No summary provided.')

        education_entries = profile.get('education', [])
        education_str = "\n".join([
            f"- {edu.get('degree')} at {edu.get('institute')} ({edu.get('startDate')})"
            for edu in education_entries
        ])

        experience_entries = profile.get('experience', [])
        experience_str = "\n".join([
            f"- {exp.get('position')} at {exp.get('company')} ({exp.get('startDate')})"
            for exp in experience_entries
        ])

        skills = profile.get('highlightedSkills', [])
        skill_str = ", ".join([s.get('name') for s in skills])

        profile_string = f"""
        Name: {full_name}
        Expertise: {area}
        Location: {location_str}
        Summary: {summary}

        Education:
        {education_str if education_str else 'No education data'}

        Experience:
        {experience_str if experience_str else 'No experience data'}

        Highlighted Skills:
        {skill_str if skill_str else 'No skills listed'}
        """

        all_profiles_string += profile_string.strip() + "\n\n"

    return f"You are provided with multiple user profiles: \n\n{all_profiles_string.strip()}"


def load_pdf_documents(file_path: str):
    """Loads a PDF file into a list of Document objects using PyPDFLoader."""
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs

