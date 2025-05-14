from dotenv import load_dotenv
import json
import os
from typing import Dict, List, Any, Tuple
import re
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pymongo import MongoClient
from pymongo_paginate import PyMongoPaginate

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


class MongoDBAtlasVectorSearch:
    """
    MongoDB Atlas Vector Search implementation using an existing search index
    - Uses Atlas Vector Search with $vectorSearch operator
    """

    def __init__(self):
        self.mongo_uri = os.getenv("MONGO_URI")
        if not self.mongo_uri:
            raise ValueError("You need to set MONGO_URI in your environment variables.")

        self.client = MongoClient(self.mongo_uri)
        self.db = self.client["hiretalentt"]
        self.collection = self.db["agencies_new"]

        self.vector_index_name = "agencies_search_index"

        self.api_key = os.getenv('OPENAI_API_KEY')
        if self.api_key is None:
            raise ValueError('You need to set your environment variable OPENAI_API_KEY')

        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=self.api_key,
            dimensions=2048
        )

        self.query_parser = QueryParser()

    def _transform_filters_for_vectorsearch(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform parsed filters into MongoDB $vectorSearch filter format

        Args:
            filters: The filters dictionary from the query parser

        Returns:
            MongoDB $vectorSearch compatible filter
        """
        vectorsearch_filters = {}

        if "type" in filters:
            vectorsearch_filters["type"] = filters["type"]
            print(f"Strictly filtering by type: {filters['type']}")

        if "teamSize" in filters:
            vectorsearch_filters["teamSize"] = filters["teamSize"]
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
                    vectorsearch_filters["hourlyRate.min"] = {op: value}
                    print(f"Filtering by hourly rate minimum {op} {value}")

            if "max" in hourly_rate_filter:
                for op, value in hourly_rate_filter["max"].items():
                    vectorsearch_filters["hourlyRate.max"] = {op: value}
                    print(f"Filtering by hourly rate maximum {op} {value}")

        if "founded" in filters:
            vectorsearch_filters["founded"] = filters["founded"]
            if "$gt" in filters["founded"]:
                print(f"Filtering by founded > {filters['founded']['$gt']}")
            elif "$gte" in filters["founded"]:
                print(f"Filtering by founded >= {filters['founded']['$gte']}")
            elif "$lt" in filters["founded"]:
                print(f"Filtering by founded < {filters['founded']['$lt']}")
            elif "$lte" in filters["founded"]:
                print(f"Filtering by founded <= {filters['founded']['$lte']}")

        return vectorsearch_filters

    def search(
            self,
            query: str,
            page: int = 1,
            page_size: int = 10,
            num_candidates: int = 10000,
            max_results: int = 10000
    ) -> Dict[str, Any]:
        """
        Perform vector search using MongoDB Atlas $vectorSearch operator

        Args:
            query: Natural language query string
            page: Current page number (1-indexed)
            page_size: Number of results per page
            num_candidates: Number of candidates to consider for scoring
            max_results: Maximum number of results to retrieve (MongoDB requires a limit)

        Returns:
            Dictionary containing search results, pagination info, and metadata
        """
        filters, semantic_query = self.query_parser.parse_query(query)
        print(f"Extracted filters: {json.dumps(filters, indent=2)}")
        print(f"Semantic query: {semantic_query}")

        vectorsearch_filters = self._transform_filters_for_vectorsearch(filters)
        print(f"Vector search filters: {json.dumps(vectorsearch_filters, indent=2)}")

        try:
            print(f"Generating embedding for semantic query: '{semantic_query}'")
            query_embedding = self.embeddings.embed_query(semantic_query)

            pipeline = [
                {
                    "$vectorSearch": {
                        "index": self.vector_index_name,
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": num_candidates,
                        "limit": max_results
                    }
                }
            ]

            if vectorsearch_filters:
                pipeline[0]["$vectorSearch"]["filter"] = vectorsearch_filters

            pipeline.append({
                "$project": {
                    "_id": 1,
                    "name": 1,
                    "hourlyRate": 1,
                    "teamSize": 1,
                    "type": 1,
                    "founded": 1,
                    "representativeText": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            })

            total_count = len(list(self.collection.aggregate(pipeline)))
            print(f"Documents matching both filters and semantic search: {total_count}")

            count_pipeline = pipeline.copy()
            count_pipeline.append({"$count": "total"})
            count_result = list(self.collection.aggregate(count_pipeline))
            total_count = count_result[0]["total"] if count_result else 0
            print(f"Documents matching both filters and semantic search: {total_count}")

            pagination_pipeline = pipeline.copy()
            pagination_pipeline.append({"$skip": (page - 1) * page_size})
            pagination_pipeline.append({"$limit": page_size})

            search_results = list(self.collection.aggregate(pagination_pipeline))

            if not search_results:
                print("No results found matching the query")
                return {
                    "results": [],
                    "pagination": {
                        "total": 0,
                        "pages": 0,
                        "current_page": page,
                        "page_size": page_size
                    },
                    "filters_applied": bool(vectorsearch_filters),
                    "semantic_query": semantic_query
                }

            results_with_scores = []
            for doc in search_results:
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
                    "content": doc.get('representativeText', '')[:200] + "..." if doc.get('representativeText') else "",
                    "similarity_score": doc.get('score', 0)
                }
                results_with_scores.append(result)

            print(f"Retrieved {len(results_with_scores)} results")

            return {
                "results": results_with_scores,
                "pagination": {
                    "total": total_count,
                    "pages": (total_count + page_size - 1) // page_size,
                    "current_page": page,
                    "page_size": page_size
                },
                "filters_applied": bool(vectorsearch_filters),
                "semantic_query": semantic_query
            }

        except Exception as e:
            print(f"Error during search: {e}")
            return {
                "results": [],
                "pagination": {
                    "total": 0,
                    "pages": 0,
                    "current_page": page,
                    "page_size": page_size
                },
                "error": str(e)
            }

    def ensure_embeddings(self, max_docs=100, batch_size=10):
        """
        Ensure all documents have embeddings by generating them for documents that don't

        Args:
            max_docs: Maximum number of documents to process
            batch_size: Number of documents to embed in a single batch
        """
        try:
            query = {"embedding": {"$exists": False}}
            docs_without_embeddings = list(self.collection.find(query).limit(max_docs))

            if not docs_without_embeddings:
                print("All documents already have embeddings")
                return

            total_docs = len(docs_without_embeddings)
            print(f"Found {total_docs} documents without embeddings")

            for i in range(0, total_docs, batch_size):
                batch = docs_without_embeddings[i:i+batch_size]
                batch_updates = []

                for doc in batch:
                    if 'representativeText' in doc and doc['representativeText']:
                        print(f"Generating embedding for document: {doc.get('name', 'Unknown')}")
                        embedding = self.embeddings.embed_query(doc['representativeText'])

                        batch_updates.append({
                            "filter": {"_id": doc["_id"]},
                            "update": {"$set": {"embedding": embedding}}
                        })
                    else:
                        print(f"Skipping document with no text to embed: {doc.get('name', 'Unknown')}")

                if batch_updates:
                    for update in batch_updates:
                        self.collection.update_one(
                            update["filter"],
                            update["update"]
                        )

                print(f"Processed batch {i//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size}")

            print("Finished generating embeddings")

        except Exception as e:
            print(f"Error ensuring embeddings: {e}")

    def close(self):
        """Close the MongoDB connection"""
        if hasattr(self, 'client') and self.client:
            self.client.close()


def main():
    search_engine = MongoDBAtlasVectorSearch()

    print("\n=== Ensuring all documents have embeddings ===")
    search_engine.ensure_embeddings()

    print("\n=== MongoDB Atlas Vector Search ===")
    print("Type 'quit' to exit")

    try:
        while True:
            query = input("\nEnter your search query: ")
            if query.lower() == 'quit':
                break

            print("\nProcessing query...")
            page = 1
            page_size = 10

            search_response = search_engine.search(
                query=query,
                page=page,
                page_size=page_size,
                num_candidates=10000,
                max_results=10000
            )

            results = search_response.get("results", [])
            pagination = search_response.get("pagination", {})

            if results:
                print(f"\n=== Search Results (Page {page} of {pagination.get('pages', 1)}) ===")
                print(f"Total results: {pagination.get('total', 0)}")

                if search_response.get("filters_applied"):
                    print(f"Applied filters with semantic query: '{search_response.get('semantic_query', '')}'")

                for i, result in enumerate(results):
                    print(f"\n--- Result {i+1} (Similarity Score: {result['similarity_score']:.4f}) ---")
                    print(f"ID: {result['id']}")
                    print(f"Name: {result['name']}")
                    print(f"Type: {result['type']}")
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