from dotenv import load_dotenv
import json
import os
from typing import Dict, List, Any, Tuple
import re
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pymongo import MongoClient
import numpy as np

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
           - For "freelancer" type: Match when user mentions "freelancers", "individuals", "contractors", "lone developer", etc.
           - If type is specified, it's critical this filter is correctly applied with NO EXCEPTIONS
        
        2. teamSize: HIGH PRIORITY
           - Parse team size requirements, including operators like >, <, >=, <= or ranges
           - For agencies, this refers to number of team members
           - Be precise about inequalities: "200+" means greater than 200 (NOT greater than or equal)
           - "atleast 200" or "minimum 200" means greater than or equal to 200
        
        3. hourlyRate: MEDIUM PRIORITY
           - Parse hourly rate requirements, including currency, operators, and ranges
           - Include currency specification if mentioned (USD, EUR, GBP, etc.)
           - IMPORTANT: For single threshold values, prioritize filtering on min values:
             - "under $100" should be interpreted as hourlyRate: {"min": {"$lt": 100}}
             - "less than $150" should be interpreted as hourlyRate: {"min": {"$lt": 150}}
             - "maximum $200" should be interpreted as hourlyRate: {"min": {"$lte": 200}}
           - For ranges, use both min and max:
             - "$100-200" should be interpreted as hourlyRate: {"min": {"$gte": 100}, "max": {"$lte": 200}}
        
        4. founded: LOWER PRIORITY
           - Parse founding year requirements if mentioned
        
        Return a JSON object with:
        1. "filters": Object containing structured filters for MongoDB
        2. "semantic_query": String containing the remaining semantic search intent
        
        Make sure to handle phrases like:
        - If text contains any variant of "agencies", "companies", "businesses", "firms": type = "agency" (NEVER match freelancers)
        - If text contains any variant of "freelancers", "individuals", "contractors", "lone developer": type = "freelancer" (NEVER match agencies)
        - "200+ members" → teamSize: {"$gt": 200}  (STRICTLY greater than, not greater than or equal)
        - "at least 50 people" → teamSize: {"$gte": 50}
        - "charging $200-300/hour" → hourlyRate: {"min": {"$gte": 200}, "max": {"$lte": 300}}
        - "less than $150 per hour" → hourlyRate: {"min": {"$lt": 150}}
        - "under $100/hour" → hourlyRate: {"min": {"$lt": 100}}
        - "maximum $200 hourly" → hourlyRate: {"min": {"$lte": 200}}
        - "founded after 2015" → founded: {"$gt": 2015}
        - "hourly rate in EUR" → hourlyRate: {"currency": "EUR"}
        
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


class DirectMongoSemanticSearch:
    """
    Direct MongoDB query with semantic search applied to filtered results
    - First applies MongoDB query filters
    - Then applies semantic search to the filtered documents
    """

    def __init__(self):
        self.mongo_uri = os.getenv("MONGO_URI")
        if not self.mongo_uri:
            raise ValueError("You need to set MONGO_URI in your environment variables.")

        self.client = MongoClient(self.mongo_uri)
        self.db = self.client["hiretalentt"]
        self.collection = self.db["agencies_new"]

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
            if "$gt" in filters["teamSize"]:
                print(f"Filtering by team size > {filters['teamSize']['$gt']}")
            elif "$gte" in filters["teamSize"]:
                print(f"Filtering by team size >= {filters['teamSize']['$gte']}")
            elif "$lt" in filters["teamSize"]:
                print(f"Filtering by team size < {filters['teamSize']['$lt']}")
            elif "$lte" in filters["teamSize"]:
                print(f"Filtering by team size <= {filters['teamSize']['$lte']}")

        if "hourlyRate" in filters:
            hourly_rate_filter = filters["hourlyRate"]

            if "min" in hourly_rate_filter:
                for op, value in hourly_rate_filter["min"].items():
                    if op in ["$lt", "$lte"]:
                        mongodb_filters["hourlyRate.min"] = {op: value}
                        print(f"Filtering by hourly rate minimum {op} {value}")
                    else:
                        mongodb_filters["hourlyRate.min"] = {op: value}
                        print(f"Filtering by hourly rate minimum {op} {value}")

            if "max" in hourly_rate_filter:
                for op, value in hourly_rate_filter["max"].items():
                    if op in ["$gt", "$gte"]:
                        mongodb_filters["hourlyRate.max"] = {op: value}
                        print(f"Filtering by hourly rate maximum {op} {value}")
                    else:
                        mongodb_filters["hourlyRate.max"] = {op: value}
                        print(f"Filtering by hourly rate maximum {op} {value}")

            if "currency" in hourly_rate_filter:
                mongodb_filters["hourlyRate.currency"] = hourly_rate_filter["currency"]
                print(f"Filtering by hourly rate currency: {hourly_rate_filter['currency']}")

        if "founded" in filters:
            mongodb_filters["founded"] = filters["founded"]
            if "$gt" in filters["founded"]:
                print(f"Filtering by founded > {filters['founded']['$gt']}")
            elif "$gte" in filters["founded"]:
                print(f"Filtering by founded >= {filters['founded']['$gte']}")
            elif "$lt" in filters["founded"]:
                print(f"Filtering by founded < {filters['founded']['$lt']}")
            elif "$lte" in filters["founded"]:
                print(f"Filtering by founded <= {filters['founded']['$lte']}")

        return mongodb_filters


    def _calculate_semantic_similarity(self, query_embedding, document_embedding):
        """
        Calculate cosine similarity between query embedding and document embedding

        Args:
            query_embedding: The embedding vector of the query
            document_embedding: The embedding vector of the document

        Returns:
            Cosine similarity score (higher means more similar)
        """
        if isinstance(query_embedding, list):
            query_embedding = np.array(query_embedding)
        if isinstance(document_embedding, list):
            document_embedding = np.array(document_embedding)

        dot_product = np.dot(query_embedding, document_embedding)
        query_norm = np.linalg.norm(query_embedding)
        doc_norm = np.linalg.norm(document_embedding)

        if query_norm == 0 or doc_norm == 0:
            return 0.0

        similarity = dot_product / (query_norm * doc_norm)
        return float(similarity)

    def search(self, query: str, k: int = 5, max_filter_results: int = 100) -> List[Dict[str, Any]]:
        """
        Perform direct MongoDB query followed by semantic search

        Args:
            query: Natural language query string
            k: Number of final results to return
            max_filter_results: Maximum number of documents to retrieve from MongoDB

        Returns:
            List of search results with scores
        """
        filters, semantic_query = self.query_parser.parse_query(query)
        print(f"Extracted filters: {json.dumps(filters, indent=2)}")
        print(f"Semantic query: {semantic_query}")

        mongodb_filters = self._transform_filters_for_mongodb(filters)
        print(f"MongoDB filters: {json.dumps(mongodb_filters, indent=2)}")

        try:
            print(f"Executing direct MongoDB query with filters...")
            cursor = self.collection.find(mongodb_filters).limit(max_filter_results)
            filtered_docs = list(cursor)
            filtered_count = len(filtered_docs)
            print(f"Retrieved {filtered_count} documents from MongoDB")

            if not filtered_docs:
                print("No documents matched the filters")
                return []

            if filtered_count <= k:
                print(f"Warning: Only {filtered_count} documents matched filters (less than requested {k} results)")
            else:
                print(f"Performing semantic search on {filtered_count} filtered documents")

            print(f"Generating embedding for semantic query: '{semantic_query}'")
            query_embedding = self.embeddings.embed_query(semantic_query)

            results_with_scores = []
            for doc in filtered_docs:
                similarity_score = self._calculate_semantic_similarity(query_embedding, doc['embedding'])

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
                    "similarity_score": similarity_score
                }

                results_with_scores.append(result)

            results_with_scores.sort(key=lambda x: x['similarity_score'], reverse=True)

            final_result_count = min(k, len(results_with_scores))
            print(f"Returning top {final_result_count} results")
            return results_with_scores[:final_result_count]

        except Exception as e:
            print(f"Error during search: {e}")
            return []

        except Exception as e:
            print(f"Error during search: {e}")
            return []

    def close(self):
        """Close the MongoDB connection"""
        if hasattr(self, 'client') and self.client:
            self.client.close()


def main():
    search_engine = DirectMongoSemanticSearch()

    print("\n=== Direct MongoDB Query with Semantic Search ===")
    print("Type 'quit' to exit")

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
                    print(f"Content: {result['content']}...")
            else:
                print("No results found.")
    finally:
        search_engine.close()


if __name__ == "__main__":
    main()