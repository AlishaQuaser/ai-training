from dotenv import load_dotenv

load_dotenv()
import os
from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch


class MongoDBSimilaritySearch:
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        if self.api_key is None:
            print('You need to set your environment variable OPENAI_API_KEY')
            exit(1)

        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=self.api_key,
            dimensions=2048
        )

        # Connect to MongoDB
        self.mongo_uri = os.getenv("MONGO_URI")
        if not self.mongo_uri:
            print("You need to set MONGO_URI in your environment variables.")
            exit(1)

        self.client = MongoClient(self.mongo_uri)
        self.db = self.client["hiretalentt"]  # Use your actual database name
        self.collection = self.db["agencies_new"]  # Use your actual collection name

        # Set up vector store
        self.vector_store = self._setup_vector_store()

    def _setup_vector_store(self):
        """Connect to the vector store"""
        # Define metadata fields to include in results
        metadata_field_names = ["_id", "name", "text", "hourlyRate"]

        # Create vector store connection
        vector_store = MongoDBAtlasVectorSearch(
            collection=self.collection,
            embedding=self.embeddings,  # Required for the interface but used only for query encoding
            index_name="agencies_new_search_index",
            text_key="representativeText",
            embedding_key="embedding",
            metadata_field_names=metadata_field_names
        )

        return vector_store

    def perform_similarity_search(self, query, k=5):
        """Perform similarity search using the vector store"""
        if not self.vector_store:
            print("Vector store not initialized.")
            return None

        # LangChain will handle embedding generation internally
        results = self.vector_store.similarity_search_with_score(query, k=k)
        return results

    def format_search_results(self, results):
        """Format search results for display"""
        formatted_results = []

        for i, (doc, score) in enumerate(results):
            formatted_results.append(f"\n--- Result {i+1} (Similarity Score: {score:.4f}) ---")
            formatted_results.append(f"ID: {doc.metadata.get('_id', 'Unknown ID')}")
            formatted_results.append(f"Name: {doc.metadata.get('name', 'Unknown Name')}")
            formatted_results.append(f"Hourly Rate: {doc.metadata.get('hourlyRate', 'NA')}")
            formatted_results.append(f"Content: {doc.page_content}")

        return "\n".join(formatted_results)

    def close(self):
        """Close the MongoDB connection"""
        if self.client:
            self.client.close()


def main():
    search_engine = MongoDBSimilaritySearch()

    print("\n=== MongoDB Similarity Search ===")
    print("Type 'quit' to exit")

    try:
        while True:
            query = input("\nEnter your search query: ")
            if query.lower() == 'quit':
                break

            results = search_engine.perform_similarity_search(query, k=5)
            if results:
                formatted_results = search_engine.format_search_results(results)
                print(formatted_results)
            else:
                print("No results found.")
    finally:
        search_engine.close()
        print("\nThank you for using MongoDB Similarity Search. Goodbye!")


if __name__ == "__main__":
    main()