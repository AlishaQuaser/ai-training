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


class MongoDBVectorSearch:
    """
    MongoDB vector search with filters
    - Uses MongoDB Atlas vector search for combining semantic search and filtering
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
        self.vector_search_index = "agencies_search_index"

    def _transform_filters_for_atlas_search(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Transform parsed filters into MongoDB Atlas Search filter format

        Args:
            filters: The filters dictionary from the query parser

        Returns:
            List of Atlas Search filter operators
        """
        search_filters = []

        if "type" in filters:
            search_filters.append({
                "equals": {
                    "path": "type",
                    "value": filters["type"]
                }
            })
            print(f"Strictly filtering by type: {filters['type']}")

        if "teamSize" in filters:
            team_size_filter = filters["teamSize"]
            range_filter = {"path": "teamSize", "gt": None, "gte": None, "lt": None, "lte": None}

            if "$gt" in team_size_filter:
                range_filter["gt"] = team_size_filter["$gt"]
                print(f"Filtering by team size > {team_size_filter['$gt']}")
            if "$gte" in team_size_filter:
                range_filter["gte"] = team_size_filter["$gte"]
                print(f"Filtering by team size >= {team_size_filter['$gte']}")
            if "$lt" in team_size_filter:
                range_filter["lt"] = team_size_filter["$lt"]
                print(f"Filtering by team size < {team_size_filter['$lt']}")
            if "$lte" in team_size_filter:
                range_filter["lte"] = team_size_filter["$lte"]
                print(f"Filtering by team size <= {team_size_filter['$lte']}")

            # Remove None values
            range_filter = {k: v for k, v in range_filter.items() if v is not None}
            if len(range_filter) > 1:  # More than just the path
                search_filters.append({"range": range_filter})

        if "hourlyRate" in filters:
            hourly_rate_filter = filters["hourlyRate"]

            if "min" in hourly_rate_filter:
                for op, value in hourly_rate_filter["min"].items():
                    range_filter = {"path": "hourlyRate.min"}
                    if op == "$lt":
                        range_filter["lt"] = value
                        print(f"Filtering by hourly rate minimum < {value}")
                    elif op == "$lte":
                        range_filter["lte"] = value
                        print(f"Filtering by hourly rate minimum <= {value}")
                    elif op == "$gt":
                        range_filter["gt"] = value
                        print(f"Filtering by hourly rate minimum > {value}")
                    elif op == "$gte":
                        range_filter["gte"] = value
                        print(f"Filtering by hourly rate minimum >= {value}")
                    search_filters.append({"range": range_filter})

            if "max" in hourly_rate_filter:
                for op, value in hourly_rate_filter["max"].items():
                    range_filter = {"path": "hourlyRate.max"}
                    if op == "$lt":
                        range_filter["lt"] = value
                        print(f"Filtering by hourly rate maximum < {value}")
                    elif op == "$lte":
                        range_filter["lte"] = value
                        print(f"Filtering by hourly rate maximum <= {value}")
                    elif op == "$gt":
                        range_filter["gt"] = value
                        print(f"Filtering by hourly rate maximum > {value}")
                    elif op == "$gte":
                        range_filter["gte"] = value
                        print(f"Filtering by hourly rate maximum >= {value}")
                    search_filters.append({"range": range_filter})

            if "currency" in hourly_rate_filter:
                search_filters.append({
                    "equals": {
                        "path": "hourlyRate.currency",
                        "value": hourly_rate_filter["currency"]
                    }
                })
                print(f"Filtering by hourly rate currency: {hourly_rate_filter['currency']}")

        if "founded" in filters:
            founded_filter = filters["founded"]
            range_filter = {"path": "founded", "gt": None, "gte": None, "lt": None, "lte": None}

            if "$gt" in founded_filter:
                range_filter["gt"] = founded_filter["$gt"]
                print(f"Filtering by founded > {founded_filter['$gt']}")
            if "$gte" in founded_filter:
                range_filter["gte"] = founded_filter["$gte"]
                print(f"Filtering by founded >= {founded_filter['$gte']}")
            if "$lt" in founded_filter:
                range_filter["lt"] = founded_filter["$lt"]
                print(f"Filtering by founded < {founded_filter['$lt']}")
            if "$lte" in founded_filter:
                range_filter["lte"] = founded_filter["$lte"]
                print(f"Filtering by founded <= {founded_filter['$lte']}")

            # Remove None values
            range_filter = {k: v for k, v in range_filter.items() if v is not None}
            if len(range_filter) > 1:  # More than just the path
                search_filters.append({"range": range_filter})

        return search_filters

    def search(self, query: str, k: int = None) -> List[Dict[str, Any]]:
        """
        Perform MongoDB vector search with filters

        Args:
            query: Natural language query string
            k: Number of results to return (None means return all)

        Returns:
            List of search results with scores
        """
        filters, semantic_query = self.query_parser.parse_query(query)
        print(f"Extracted filters: {json.dumps(filters, indent=2)}")
        print(f"Semantic query: {semantic_query}")

        search_filters = self._transform_filters_for_atlas_search(filters)
        print(f"Atlas Search filters: {json.dumps(search_filters, indent=2)}")

        try:
            print(f"Generating embedding for semantic query: '{semantic_query}'")
            query_embedding = self.embeddings.embed_query(semantic_query)

            limit = k if k is not None else 10000

            # Create the search stage
            search_stage = {
                "$search": {
                    "index": self.vector_search_index,
                }
            }

            # If we have filters, use the knnBeta directly and apply filters at the aggregation level
            if search_filters:
                # Use knnBeta for vector search
                search_stage["$search"]["knnBeta"] = {
                    "vector": query_embedding,
                    "path": "embedding",
                    "k": limit
                }
            else:
                # No filters, just use knnBeta directly
                search_stage["$search"]["knnBeta"] = {
                    "vector": query_embedding,
                    "path": "embedding",
                    "k": limit
                }

            pipeline = [
                search_stage
            ]

            # If we have filters, apply them after the vector search using standard MongoDB query operators
            if search_filters:
                # Convert Atlas Search filters back to MongoDB filters for post-search filtering
                mongo_filters = {}

                for filter_item in search_filters:
                    if "equals" in filter_item:
                        path = filter_item["equals"]["path"]
                        value = filter_item["equals"]["value"]
                        mongo_filters[path] = value

                    elif "range" in filter_item:
                        range_filter = filter_item["range"]
                        path = range_filter["path"]
                        mongo_path_filter = {}

                        if "gt" in range_filter:
                            mongo_path_filter["$gt"] = range_filter["gt"]
                        if "gte" in range_filter:
                            mongo_path_filter["$gte"] = range_filter["gte"]
                        if "lt" in range_filter:
                            mongo_path_filter["$lt"] = range_filter["lt"]
                        if "lte" in range_filter:
                            mongo_path_filter["$lte"] = range_filter["lte"]

                        mongo_filters[path] = mongo_path_filter

                # Add a match stage to filter results after vector search
                if mongo_filters:
                    pipeline.append({"$match": mongo_filters})

            # Add projection stage
            pipeline.append({
                "$project": {
                    "_id": 1,
                    "name": 1,
                    "hourlyRate": 1,
                    "teamSize": 1,
                    "type": 1,
                    "founded": 1,
                    "representativeText": 1,
                    "score": {"$meta": "searchScore"}
                }
            })

            print(f"Executing MongoDB vector search pipeline...")
            cursor = self.collection.aggregate(pipeline)
            results = list(cursor)
            results_count = len(results)
            print(f"Retrieved {results_count} documents from MongoDB vector search")

            if not results:
                print("No documents matched the query")
                return []

            formatted_results = []
            for doc in results:
                hourly_rate = doc.get('hourlyRate', {})
                if isinstance(hourly_rate, dict):
                    hourly_rate_str = f"{hourly_rate.get('min', 'N/A')}-{hourly_rate.get('max', 'N/A')} {hourly_rate.get('currency', 'USD')}"
                else:
                    hourly_rate_str = str(hourly_rate)

                formatted_result = {
                    "id": str(doc.get('_id', 'Unknown ID')),
                    "name": doc.get('name', 'Unknown Name'),
                    "hourly_rate": hourly_rate_str,
                    "team_size": doc.get('teamSize', 'N/A'),
                    "type": doc.get('type', 'N/A'),
                    "founded": doc.get('founded', 'N/A'),
                    "content": doc.get('representativeText', ''),
                    "similarity_score": doc.get('score', 0.0)
                }

                formatted_results.append(formatted_result)

            print(f"Returning {len(formatted_results)} results")
            return formatted_results

        except Exception as e:
            print(f"Error during search: {e}")
            return []

    def close(self):
        """Close the MongoDB connection"""
        if hasattr(self, 'client') and self.client:
            self.client.close()


def main():
    search_engine = MongoDBVectorSearch()

    print("\n=== MongoDB Vector Search with Filters ===")
    print("Type 'quit' to exit")

    try:
        while True:
            query = input("\nEnter your search query: ")
            if query.lower() == 'quit':
                break

            print("\nProcessing query...")
            results = search_engine.search(query)

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