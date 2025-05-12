import os
from pymongo import MongoClient
from langchain_mistralai import MistralAIEmbeddings
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
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        print("Please set environment variable MISTRAL_API_KEY")
        exit(1)

    return MistralAIEmbeddings(model="mistral-embed", mistral_api_key=api_key)


def search_cli():
    db, client = connect_to_mongodb()
    embeddings = setup_embeddings()

    collection = db["agencies_test_data"]

    vector_store = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name="default",
        text_key="representativeText",
        embedding_key="embedding"
    )

    retriever = MongoDBAtlasHybridSearchRetriever(
        vectorstore=vector_store,
        search_index_name="default",
        top_k=5,
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
            for i, doc in enumerate(results):
                print(f"{i+1}. {doc.metadata.get('name', 'Unknown Name')}")
                print(f"   Content: {doc.page_content[:200]}...")
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

