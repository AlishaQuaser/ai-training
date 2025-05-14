import os
from pymongo import MongoClient, UpdateOne
from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings

from langchain_openai import OpenAIEmbeddings

load_dotenv()

def connect_to_mongodb():
    """Connect to MongoDB and return the database"""
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        print("You need to set MONGO_URI in your environment variables.")
        exit(1)

    client = MongoClient(mongo_uri)
    db = client["hiretalentt"]
    return db

def setup_embeddings():
    """Set up OpenAI embeddings"""
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        print('You need to set your environment variable OPENAI_API_KEY')
        exit(1)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=api_key,
        dimensions=2048
    )
    return embeddings

def generate_embeddings_for_agencies():
    """Generate and store embeddings for agency representative texts"""
    db = connect_to_mongodb()

    agencies_collection = db["agencies_"]

    embeddings_model = setup_embeddings()

    agencies = list(agencies_collection.find({"representativeText": {"$exists": True, "$ne": ""}}))
    print(f"Found {len(agencies)} agencies with representative text")

    operations = []

    for i, agency in enumerate(agencies):
        try:
            representative_text = agency['representativeText']
            embedding = embeddings_model.embed_query(representative_text)

            operations.append(
                UpdateOne(
                    {"_id": agency['_id']},
                    {"$set": {"embedding": embedding}}
                )
            )

            print(f"Generated embedding for agency {i+1}/{len(agencies)}: {agency.get('name', 'Unnamed')}")

        except Exception as e:
            print(f"Error generating embedding for agency {agency.get('name', 'Unnamed')}: {e}")

        if len(operations) >= 50:
            result = agencies_collection.bulk_write(operations)
            print(f"Bulk updated {result.modified_count} agencies")
            operations = []

    if operations:
        result = agencies_collection.bulk_write(operations)
        print(f"Bulk updated {result.modified_count} agencies")

def main():
    print("Starting agency embedding generation process...")
    generate_embeddings_for_agencies()
    print("Agency embedding generation process completed.")

if __name__ == "__main__":
    main()