import os
from pymongo import MongoClient

from langchain_openai import OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.retrievers.hybrid_search import MongoDBAtlasHybridSearchRetriever
from dotenv import load_dotenv

load_dotenv()


def connect_to_mongodb():
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        exit(1)

    client = MongoClient(mongo_uri)
    db = client["hiretalentt"]
    return db, client


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


def search_cli():
    db, client = connect_to_mongodb()
    embeddings = setup_embeddings()

    collection = db["agencies_new"]

    metadata_field_names = ["_id", "name", "text", "hourlyRate"]

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
        top_k=10,
        fulltext_penalty=50,
        vector_penalty=50
    )

    print("Welcome to Agency Hybrid Search")
    print("-------------------------------")

    while True:
        query = input("\nEnter your search query (or 'q' to quit): ").strip()
        if query.lower() == 'q':
            break
        if not query:
            print("Empty query. Please try again.")
            continue

        print(f"\nSearching for: '{query}'...\n")
        results = retriever.invoke(query)

        if results:
            print(results[0])
            for i, doc in enumerate(results):
                doc_id = doc.metadata.get('_id', 'No ID')
                print(f"{i+1}. ID: {doc_id}")
                print(f"   Name: {doc.metadata.get('name', 'Unknown Name')}")
                print(f"   Hourly Rates: {doc.metadata.get('hourlyRate', 'NA')}")
                print(f"   Content: {doc.page_content}")
                print(f"   Full-text score: {doc.metadata.get('fulltext_score', 0):.2f}")
                print(f"   Vector score: {doc.metadata.get('vector_score', 0):.2f}")
                print(f"   Total score: {doc.metadata.get('score', 0):.2f}")
                print("\n" + "-" * 60)
        else:
            print("No matching agencies found.")

    client.close()
    print("\nThank you for using Agency Hybrid Search. Goodbye!")


if __name__ == "__main__":
    search_cli()

