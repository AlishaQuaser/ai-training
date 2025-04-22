import os
import json
import time
from pymongo import MongoClient
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.core import Document
import random

load_dotenv()


def setup_mistral_embedding():
    return MistralAIEmbedding(model_name="mistral-embed")


Settings.embed_model = setup_mistral_embedding()


def get_random_profiles_from_db(limit=50):
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        print("You need to set MONGO_URI in your environment variables.")
        exit(1)

    client = MongoClient(mongo_uri)
    db = client["app-dev"]
    collection = db["profiles"]

    total_profiles = list(collection.find())
    if not total_profiles:
        print("No profiles found in the database.")
        exit(1)

    # Get a random sample of 50 profiles
    random_profiles = random.sample(total_profiles, min(limit, len(total_profiles)))
    return random_profiles


def prepare_documents_from_profiles(profiles):
    documents = []

    for profile in profiles:
        full_name = f"{profile.get('firstName', '')} {profile.get('lastName', '')}".strip()
        area_of_expertise = profile.get('areaOfExpertise', 'Not specified')
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
        Expertise: {area_of_expertise}
        Location: {location_str}
        Summary: {summary}

        Education:
        {education_str if education_str else 'No education data'}

        Experience:
        {experience_str if experience_str else 'No experience data'}

        Highlighted Skills:
        {skill_str if skill_str else 'No skills listed'}
        """

        documents.append(Document(text=profile_string.strip()))
        time.sleep(1.5)  # Sleep to avoid rate limiting

    return documents


if __name__ == "__main__":
    profiles = get_random_profiles_from_db(limit=20)
    documents = prepare_documents_from_profiles(profiles)

    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=MistralAIEmbedding(model_name="mistral-embed")
    )

    persist_dir = "profile_index"
    index.storage_context.persist(persist_dir=persist_dir)
