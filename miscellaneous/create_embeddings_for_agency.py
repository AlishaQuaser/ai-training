import os
from pymongo import MongoClient
from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings

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
    """Set up Mistral embeddings"""
    api_key = os.getenv('MISTRAL_API_KEY')
    if api_key is None:
        print('You need to set your environment variable MISTRAL_API_KEY')
        exit(1)

    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        mistral_api_key=api_key
    )
    return embeddings

def generate_embedding_for_agency(agency_name, force_regenerate=False):
    """
    Generate or retrieve embedding for a specific agency's representative text

    :param agency_name: Name of the agency to generate embedding for
    :param force_regenerate: If True, regenerate embedding even if it exists
    :return: Embedding vector or None if agency not found or text is empty
    """
    db = connect_to_mongodb()

    agencies_collection = db["test_data"]

    embeddings_model = setup_embeddings()

    agency = agencies_collection.find_one({"name": {"$regex": f"^{agency_name}$", "$options": "i"}})

    if not agency:
        print(f"No agency found with the name: {agency_name}")
        return None

    representative_text = agency.get('representativeText', '')

    if not representative_text:
        print(f"No representative text found for agency: {agency_name}")
        return None

    existing_embedding = agency.get('embedding')

    if existing_embedding and not force_regenerate:
        print(f"Existing embedding found for {agency_name}. Use force_regenerate=True to overwrite.")
        return existing_embedding

    try:
        embedding = embeddings_model.embed_query(representative_text)

        result = agencies_collection.update_one(
            {"_id": agency['_id']},
            {"$set": {"embedding": embedding}}
        )

        print(f"Successfully {'regenerated' if force_regenerate else 'generated'} embedding for {agency_name}")
        return embedding

    except Exception as e:
        print(f"Error generating embedding for {agency_name}: {e}")
        return None

def main():
    agency_name = input("Enter the name of the agency to generate embedding for: ").strip()

    force_regen = input("Force regenerate embedding? (y/n): ").strip().lower() == 'y'

    embedding = generate_embedding_for_agency(agency_name, force_regenerate=force_regen)

    if embedding:
        print("Embedding details:")
        print(f"Embedding dimensions: {len(embedding)}")
        print("First few values:", embedding[:5])

def regenerate_all_embeddings(force=False):
    """
    Regenerate embeddings for all agencies

    :param force: If True, regenerate all embeddings even if they exist
    """
    db = connect_to_mongodb()

    agencies_collection = db["agencies_test_data"]

    embeddings_model = setup_embeddings()

    agencies = list(agencies_collection.find({"representativeText": {"$exists": True, "$ne": ""}}))

    print(f"Found {len(agencies)} agencies with representative text")

    total_processed = 0
    regenerated = 0
    skipped = 0

    for agency in agencies:
        try:
            existing_embedding = agency.get('embedding')

            if existing_embedding and not force:
                skipped += 1
                continue

            representative_text = agency['representativeText']
            embedding = embeddings_model.embed_query(representative_text)

            agencies_collection.update_one(
                {"_id": agency['_id']},
                {"$set": {"embedding": embedding}}
            )

            total_processed += 1
            if existing_embedding:
                regenerated += 1

            print(f"Processed {total_processed}: {agency.get('name', 'Unnamed Agency')}")

        except Exception as e:
            print(f"Error processing {agency.get('name', 'Unnamed Agency')}: {e}")

    print("\nEmbedding Generation Summary:")
    print(f"Total processed: {total_processed}")
    print(f"Regenerated: {regenerated}")
    print(f"Skipped (existing): {skipped}")

if __name__ == "__main__":
    regenerate_all_embeddings(force=False)