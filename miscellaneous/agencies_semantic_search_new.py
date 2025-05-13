import os
from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.retrievers.hybrid_search import MongoDBAtlasHybridSearchRetriever
from dotenv import load_dotenv
import re

load_dotenv()


def connect_to_mongodb():
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        print("Please set environment variable MONGO_URI")
        exit(1)

    client = MongoClient(mongo_uri)
    db = client["hiretalentt"]
    return db, client


def setup_embeddings():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set environment variable OPENAI_API_KEY")
        exit(1)

    return OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=api_key,
        dimensions=2048
    )


def parse_query(query):
    """Parse complex queries to extract special conditions like team size requirements."""
    conditions = {
        "text_query": query,
        "min_team_size": None,
        "skills": []
    }

    team_size_match = re.search(r'(\d+)\s+team\s+members', query, re.IGNORECASE)
    if team_size_match:
        conditions["min_team_size"] = int(team_size_match.group(1))

    skills_match = re.findall(r'skills?\s+in\s+([A-Za-z0-9\.\+]+)', query, re.IGNORECASE)
    if skills_match:
        conditions["skills"] = skills_match

    return conditions


def filter_results(results, conditions):
    """Filter results based on extracted conditions."""
    filtered_results = []

    for doc in results:
        meets_criteria = True
        text_content = doc.page_content.lower() + " " + doc.metadata.get('text', '').lower()

        if conditions["min_team_size"]:
            team_size_match = re.search(r'(\d+)\s+team\s+members', text_content)
            if not team_size_match or int(team_size_match.group(1)) < conditions["min_team_size"]:
                meets_criteria = False

        for skill in conditions["skills"]:
            skill_pattern = r'\b' + re.escape(skill) + r'\b'
            if not re.search(skill_pattern, text_content, re.IGNORECASE):
                meets_criteria = False

        if meets_criteria:
            filtered_results.append(doc)

    return filtered_results


def search_cli():
    db, client = connect_to_mongodb()
    embeddings = setup_embeddings()

    collection = db["agencies_new"]

    metadata_field_names = ["_id", "name", "text", "hourlyRate", "teamSize", "skills"]

    vector_store = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name="agencies_new_search_index",
        text_key="representativeText",
        embedding_key="embedding",
        metadata_field_names=metadata_field_names
    )


    retriever = MongoDBAtlasHybridSearchRetriever(
        vectorstore=vector_store,
        search_index_name="agencies_new_search_index",
        top_k=100,
        fulltext_penalty=30,
        vector_penalty=40
    )

    print("Welcome to Advanced Agency Hybrid Search")
    print("----------------------------------------")

    while True:
        query = input("\nEnter your search query (or 'q' to quit): ").strip()
        if query.lower() == 'q':
            break
        if not query:
            print("Empty query. Please try again.")
            continue

        print(f"\nSearching for: '{query}'...\n")

        conditions = parse_query(query)

        initial_results = retriever.invoke(conditions["text_query"])

        results = filter_results(initial_results, conditions)

        if results:
            print(f"Found {len(results)} matching agencies:")
            for i, doc in enumerate(results):
                doc_id = doc.metadata.get('_id', 'No ID')
                print(f"{i+1}. ID: {doc_id}")
                print(f"   Name: {doc.metadata.get('name', 'Unknown Name')}")
                print(f"   Hourly Rate: {doc.metadata.get('hourlyRate', 'NA')}")

                team_size_match = re.search(r'(\d+)\s+team\s+members', doc.page_content)
                team_size = team_size_match.group(1) if team_size_match else "Unknown"
                print(f"   Team Size: {team_size}")

                skills_text = re.search(r'skill sets?: (.*?)\.', doc.page_content)
                skills = skills_text.group(1) if skills_text else "Not specified"
                print(f"   Skills: {skills}")

                print(f"   Full-text score: {doc.metadata.get('fulltext_score', 0):.2f}")
                print(f"   Vector score: {doc.metadata.get('vector_score', 0):.2f}")
                print(f"   Total score: {doc.metadata.get('score', 0):.2f}")
                print("\n" + "-" * 70)
        else:
            print("No matching agencies found.")

    client.close()
    print("\nThank you for using Advanced Agency Hybrid Search. Goodbye!")


if __name__ == "__main__":
    search_cli()