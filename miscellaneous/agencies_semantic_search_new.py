from dotenv import load_dotenv
import json
import os
from typing import Dict, List, Any, Tuple
import re
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient

load_dotenv()

class QueryParser:
    """Parse natural language queries into structured filters and semantic search components"""

    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        if self.api_key is None:
            raise ValueError('You need to set your environment variable OPENAI_API_KEY')

        self.llm = ChatOpenAI(
            model="gpt-4",
            openai_api_key=self.api_key,
            temperature=0
        )

    def parse_query(self, query: str) -> Tuple[Dict[str, Any], str]:
        """
        Parse a natural language query into structured filters and semantic search query

        Args:
            query: Natural language query string

        Returns:
            Tuple of (structured_filters, semantic_query)
        """
        system_prompt = """
        You are a query parser that extracts structured filters and semantic intent from natural language search queries.
        You will extract the following structured fields if mentioned, in order of priority:
        
        1. type: HIGHEST PRIORITY - Must be strictly enforced
           - For "agency" type: Match when user mentions "agencies", "businesses", "companies", "firms", etc.
           - For "freelancer" type: Match when user mentions "freelancers", "individuals", "contractors", etc.
           - If type is specified, it's critical this filter is correctly applied with NO EXCEPTIONS
        
        2. teamSize: HIGH PRIORITY
           - Parse team size requirements, including operators like >, <, >=, <= or ranges
           - For agencies, this refers to number of team members
        
        3. hourlyRate: MEDIUM PRIORITY
           - Parse hourly rate requirements, including currency, operators, and ranges
        
        4. founded: LOWER PRIORITY
           - Parse founding year requirements if mentioned
        
        Return a JSON object with:
        1. "filters": Object containing structured filters for MongoDB
        2. "semantic_query": String containing the remaining semantic search intent
        
        Make sure to handle phrases like:
        - If text contains any variant of "agencies" or "companies": type = "agency" (NEVER match freelancers)
        - If text contains any variant of "freelancers" or "individuals": type = "freelancer" (NEVER match agencies)
        - "200+ members" → teamSize: {"$gte": 200}
        - "at least 50 people" → teamSize: {"$gte": 50}
        - "charging $200-300/hour" → hourlyRate: {"min": {"$lte": 300}, "max": {"$gte": 200}}
        - "less than $150 per hour" → hourlyRate: {"min": {"$lt": 150}}
        - "founded after 2015" → founded: {"$gt": 2015}
        
        Your response should be a valid JSON object with no additional text.
        """

        user_message = f"Extract structured filters and semantic intent from this search query: '{query}'"

        response = self.llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        )

        try:
            json_match = re.search(r'({[\s\S]*})', response.content)
            if json_match:
                json_str = json_match.group(1)
                parsed_data = json.loads(json_str)
            else:
                parsed_data = json.loads(response.content)

            filters = parsed_data.get("filters", {})
            semantic_query = parsed_data.get("semantic_query", query)

            return filters, semantic_query

        except json.JSONDecodeError as e:
            print(f"Failed to parse LLM response as JSON: {e}")
            print(f"Raw response: {response.content}")
            return {}, query


class HybridSearchSystem:
    """Hybrid search system combining structured filters and semantic search"""

    def __init__(self):
        self.mongo_uri = os.getenv("MONGO_URI")
        if not self.mongo_uri:
            raise ValueError("You need to set MONGO_URI in your environment variables.")

        self.client = MongoClient(self.mongo_uri)
        self.db = self.client["hiretalentt"]
        self.collection = self.db["agencies_"]

        self.api_key = os.getenv('OPENAI_API_KEY')
        if self.api_key is None:
            raise ValueError('You need to set your environment variable OPENAI_API_KEY')

        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=self.api_key,
            dimensions=2048
        )

        self.query_parser = QueryParser()

        self.metadata_field_names = ["_id", "name", "hourlyRate", "teamSize", "type", "founded", "representativeText"]


    def _setup_vector_store(self, pre_filter):
        """
        Set up vector store with pre-filter for MongoDB Atlas Vector Search

        Args:
            pre_filter: MongoDB query filter to apply before semantic search

        Returns:
            Configured vector store
        """
        vector_store = MongoDBAtlasVectorSearch(
            collection=self.collection,
            embedding=self.embeddings,
            index_name="vector_index",
            text_key="representativeText",
            embedding_key="embedding",
            metadata_field_names=self.metadata_field_names,
            pre_filter=pre_filter
        )

        return vector_store

    def _transform_filters_for_mongodb(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform parsed filters into MongoDB query format

        Args:
            filters: The filters dictionary from the query parser

        Returns:
            MongoDB-compatible query filter
        """
        mongodb_filters = {}

        if "type" in filters:
            mongodb_filters["type"] = filters["type"]
            print(f"Strictly filtering by type: {filters['type']}")

        if "teamSize" in filters:
            mongodb_filters["teamSize"] = filters["teamSize"]

        if "hourlyRate" in filters:
            hourly_rate_filter = filters["hourlyRate"]

            if "min" in hourly_rate_filter:
                for op, value in hourly_rate_filter["min"].items():
                    mongodb_filters["hourlyRate.min"] = {op: value}

            if "max" in hourly_rate_filter:
                for op, value in hourly_rate_filter["max"].items():
                    mongodb_filters["hourlyRate.max"] = {op: value}

        if "founded" in filters:
            mongodb_filters["founded"] = filters["founded"]

        return mongodb_filters

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform hybrid search on the collection

        Args:
            query: Natural language query string
            k: Number of results to return

        Returns:
            List of search results with scores
        """
        filters, semantic_query = self.query_parser.parse_query(query)
        print(f"Extracted filters: {json.dumps(filters, indent=2)}")
        print(f"Semantic query: {semantic_query}")

        mongodb_filters = self._transform_filters_for_mongodb(filters)
        print(f"MongoDB filters: {json.dumps(mongodb_filters, indent=2)}")

        try:
            vector_store = self._setup_vector_store(mongodb_filters)

            results = vector_store.similarity_search_with_score(semantic_query, k=k)

            formatted_results = []
            for doc, score in results:
                metadata = doc.metadata

                hourly_rate = metadata.get('hourlyRate', {})
                if isinstance(hourly_rate, dict):
                    hourly_rate_str = f"{hourly_rate.get('min', 'N/A')}-{hourly_rate.get('max', 'N/A')} {hourly_rate.get('currency', 'USD')}"
                else:
                    hourly_rate_str = str(hourly_rate)

                result = {
                    "id": str(metadata.get('_id', 'Unknown ID')),
                    "name": metadata.get('name', 'Unknown Name'),
                    "hourly_rate": hourly_rate_str,
                    "team_size": metadata.get('teamSize', 'N/A'),
                    "type": metadata.get('type', 'N/A'),
                    "founded": metadata.get('founded', 'N/A'),
                    "content": doc.page_content,
                    "similarity_score": score
                }
                formatted_results.append(result)

            return formatted_results

        except Exception as e:
            print(f"Error during search: {e}")
            print("Falling back to direct MongoDB query with structured filters...")

            try:
                raw_results = list(self.collection.find(mongodb_filters).limit(k))

                formatted_results = []
                for doc in raw_results:
                    hourly_rate = doc.get('hourlyRate', {})
                    if isinstance(hourly_rate, dict):
                        hourly_rate_str = f"{hourly_rate.get('min', 'N/A')}-{hourly_rate.get('max', 'N/A')} {hourly_rate.get('currency', 'USD')}"
                    else:
                        hourly_rate_str = str(hourly_rate)

                    result = {
                        "id": str(doc.get('_id', 'Unknown ID')),
                        "name": doc.get('name', 'Unknown Name'),
                        "hourly_rate": hourly_rate_str,
                        "team_size": doc.get('teamSize', 'N/A'),
                        "type": doc.get('type', 'N/A'),
                        "founded": doc.get('founded', 'N/A'),
                        "content": doc.get('representativeText', ''),
                        "similarity_score": 0.0
                    }
                    formatted_results.append(result)

                return formatted_results

            except Exception as e2:
                print(f"Fallback also failed: {e2}")
                return []

    def close(self):
        """Close the MongoDB connection"""
        if hasattr(self, 'client') and self.client:
            self.client.close()


def main():
    search_engine = HybridSearchSystem()

    print("\n=== Hybrid Search System (LLM + MongoDB + Vector Search) ===")
    print("Type 'quit' to exit")
    print("\nIMPORTANT: Make sure your MongoDB Atlas Vector Search index includes these fields:")
    print("- Vector field: 'embedding'")
    print("- Filter fields: 'type', 'teamSize', 'hourlyRate.min', 'hourlyRate.max', 'founded'")
    print("\nRecommended index definition:")
    print("""{
  "fields": [
    {
      "numDimensions": 2048,
      "path": "embedding",
      "similarity": "cosine",
      "type": "vector"
    },
    {
      "path": "type",
      "type": "filter"
    },
    {
      "path": "teamSize",
      "type": "filter" 
    },
    {
      "path": "founded",
      "type": "filter"
    },
    {
      "path": "hourlyRate.min",
      "type": "filter"
    },
    {
      "path": "hourlyRate.max",
      "type": "filter"
    }
  ]
}""")

    try:
        while True:
            query = input("\nEnter your search query: ")
            if query.lower() == 'quit':
                break

            print("\nProcessing query...")
            results = search_engine.search(query, k=5)

            if results:
                print("\n=== Search Results ===")
                for i, result in enumerate(results):
                    print(f"\n--- Result {i+1} (Similarity Score: {result['similarity_score']:.4f}) ---")
                    print(f"ID: {result['id']}")
                    print(f"Name: {result['name']}")
                    print(f"Type: {result['type']}")
                    print(f"Team Size: {result['team_size']}")
                    print(f"Hourly Rate: {result['hourly_rate']}")
                    print(f"Founded: {result['founded']}")
                    print(f"Content: {result['content'][:200]}...")
            else:
                print("No results found.")
    finally:
        search_engine.close()
        print("\nThank you for using Hybrid Search System. Goodbye!")


if __name__ == "__main__":
    main()