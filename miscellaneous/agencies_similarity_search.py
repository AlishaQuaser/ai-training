from dotenv import load_dotenv
import json
import os
from typing import Dict, List, Any, Tuple
import re
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
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
            temperature=0,
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
        
        1. teamSize: HIGH PRIORITY
           - Parse team size requirements, including operators like >, <, >=, <= or ranges
           - For agencies, this refers to number of team members
           - Be precise about inequalities: "200+" means greater than 200 (NOT greater than or equal)
           - "atleast 200" or "minimum 200" means greater than or equal to 200
        
        2. hourlyRate: MEDIUM PRIORITY
           - Parse hourly rate requirements, including currency, operators, and ranges
           - Include currency specification if mentioned (USD, EUR, GBP, etc.)
           - IMPORTANT: For single threshold values, prioritize filtering on min values:
             - "under $100" should be interpreted as hourlyRate: {"min": {"$lt": 100}}
             - "less than $150" should be interpreted as hourlyRate: {"min": {"$lt": 150}}
             - "maximum $200" should be interpreted as hourlyRate: {"min": {"$lte": 200}}
           - For ranges, use both min and max:
             - "$100-200" should be interpreted as hourlyRate: {"min": {"$gte": 100}, "max": {"$lte": 200}}
        
        3. founded: LOWER PRIORITY
           - Parse founding year requirements if mentioned
        
        Return a JSON object with:
        1. "filters": Object containing structured filters for MongoDB
        2. "semantic_query": String containing the remaining semantic search intent
        
        Make sure to handle phrases like:
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


class AtlasVectorSearch:
    """
    MongoDB Atlas Vector Search implementation
    - First applies MongoDB query filters
    - Then applies Atlas Vector Search to the filtered documents
    """

    def __init__(self):
        self.mongo_uri = os.getenv("MONGO_URI")
        if not self.mongo_uri:
            raise ValueError("You need to set MONGO_URI in your environment variables.")

        self.client = MongoClient(self.mongo_uri)
        self.db = self.client["hiretalentt"]
        self.collection = self.db["agencies"]

        self.index_name = "agencies_vector_index"
        self.embedding_field = "embedding"

        self.api_key = os.getenv('OPENAI_API_KEY')
        if self.api_key is None:
            raise ValueError('You need to set your environment variable OPENAI_API_KEY')

        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=self.api_key,
            dimensions=2048
        )

        self.query_parser = QueryParser()
        self.metadata_field_names = ["_id", "name", "hourlyRate", "teamSize", "founded", "representativeText"]

    def _transform_filters_for_mongodb(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform parsed filters into MongoDB query format

        Args:
            filters: The filters dictionary from the query parser

        Returns:
            MongoDB-compatible query filter
        """
        mongodb_filters = {}

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

    def search(self, query: str, k: int = 5, max_vector_results: int = 616) -> List[Dict[str, Any]]:
        """
        Perform Atlas Vector Search first, then apply filters to the results

        Args:
            query: Natural language query string
            k: Number of final results to return after filtering
            max_vector_results: Maximum number of documents to retrieve from initial vector search

        Returns:
            List of search results with scores
        """
        filters, semantic_query = self.query_parser.parse_query(query)
        print(f"Extracted filters: {json.dumps(filters, indent=2)}")
        print(f"Semantic query: {semantic_query}")

        mongodb_filters = self._transform_filters_for_mongodb(filters)
        print(f"MongoDB filters: {json.dumps(mongodb_filters, indent=2)}")

        try:
            print(f"Generating embedding for semantic query: '{semantic_query}'")
            query_embedding = self.embeddings.embed_query(semantic_query)

            print(f"Performing Atlas Vector Search on collection")

            pipeline = [
                {"$vectorSearch": {
                    "index": self.index_name,
                    "path": self.embedding_field,
                    "queryVector": query_embedding,
                    "numCandidates": 616,
                    "limit": max_vector_results
                }},

                {"$project": {
                    "_id": 1,
                    "name": 1,
                    "hourlyRate": 1,
                    "teamSize": 1,
                    "founded": 1,
                    "representativeText": 1,
                    "similarity_score": {"$meta": "vectorSearchScore"}
                }}
            ]

            vector_results = list(self.collection.aggregate(pipeline))
            print(f"Retrieved {len(vector_results)} documents from Atlas Vector Search")

            if not vector_results:
                print("No results found from vector search")
                return []

            print(f"Applying filters to vector search results")
            filtered_results = []

            for doc in vector_results:
                matches_all_filters = True

                if "teamSize" in mongodb_filters:
                    team_size = doc.get("teamSize")
                    if team_size is not None:
                        for op, value in mongodb_filters["teamSize"].items():
                            if op == "$gt" and not (team_size > value):
                                matches_all_filters = False
                            elif op == "$gte" and not (team_size >= value):
                                matches_all_filters = False
                            elif op == "$lt" and not (team_size < value):
                                matches_all_filters = False
                            elif op == "$lte" and not (team_size <= value):
                                matches_all_filters = False

                if "hourlyRate.min" in mongodb_filters:
                    hourly_rate = doc.get("hourlyRate", {})
                    if isinstance(hourly_rate, dict):
                        min_rate = hourly_rate.get("min")
                        if min_rate is not None:
                            for op, value in mongodb_filters["hourlyRate.min"].items():
                                if op == "$gt" and not (min_rate > value):
                                    matches_all_filters = False
                                elif op == "$gte" and not (min_rate >= value):
                                    matches_all_filters = False
                                elif op == "$lt" and not (min_rate < value):
                                    matches_all_filters = False
                                elif op == "$lte" and not (min_rate <= value):
                                    matches_all_filters = False

                if "hourlyRate.max" in mongodb_filters:
                    hourly_rate = doc.get("hourlyRate", {})
                    if isinstance(hourly_rate, dict):
                        max_rate = hourly_rate.get("max")
                        if max_rate is not None:
                            for op, value in mongodb_filters["hourlyRate.max"].items():
                                if op == "$gt" and not (max_rate > value):
                                    matches_all_filters = False
                                elif op == "$gte" and not (max_rate >= value):
                                    matches_all_filters = False
                                elif op == "$lt" and not (max_rate < value):
                                    matches_all_filters = False
                                elif op == "$lte" and not (max_rate <= value):
                                    matches_all_filters = False

                if "hourlyRate.currency" in mongodb_filters:
                    hourly_rate = doc.get("hourlyRate", {})
                    if isinstance(hourly_rate, dict):
                        currency = hourly_rate.get("currency")
                        if currency != mongodb_filters["hourlyRate.currency"]:
                            matches_all_filters = False

                if "founded" in mongodb_filters:
                    founded = doc.get("founded")
                    if founded is not None:
                        for op, value in mongodb_filters["founded"].items():
                            if op == "$gt" and not (founded > value):
                                matches_all_filters = False
                            elif op == "$gte" and not (founded >= value):
                                matches_all_filters = False
                            elif op == "$lt" and not (founded < value):
                                matches_all_filters = False
                            elif op == "$lte" and not (founded <= value):
                                matches_all_filters = False

                if matches_all_filters:
                    filtered_results.append(doc)

            print(f"After filtering: {len(filtered_results)} documents remain")

            formatted_results = []
            for doc in filtered_results[:k]:
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
                    "founded": doc.get('founded', 'N/A'),
                    "content": doc.get('representativeText', ''),
                    "similarity_score": doc.get('similarity_score', 0.0)
                }

                formatted_results.append(result)

            return formatted_results

        except Exception as e:
            print(f"Error during search: {e}")
            return []

    def close(self):
        """Close the MongoDB connection"""
        if hasattr(self, 'client') and self.client:
            self.client.close()


def main():
    search_engine = AtlasVectorSearch()

    print("\n=== MongoDB Atlas Vector Search ===")
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
                    print(f"\n--- Result {i+1} (Similarity Score: {result['similarity_score']: .4f}) ---")
                    print(f"ID: {result['id']}")
                    print(f"Name: {result['name']}")
                    print(f"Team Size: {result['team_size']}")
                    print(f"Hourly Rate: {result['hourly_rate']}")
                    print(f"Founded: {result['founded']}")
                    print(f"Content: {result['content']}")
            else:
                print("No results found.")
    finally:
        search_engine.close()


if __name__ == "__main__":
    main()