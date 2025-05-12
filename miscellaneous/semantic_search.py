import os
from pymongo import MongoClient
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv
import numpy as np

load_dotenv()


def connect_to_mongodb():
    """Connect to MongoDB using environment variables."""
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        print("You need to set MONGO_URI in your environment variables.")
        exit(1)

    client = MongoClient(mongo_uri)
    db = client["hiretalentt"]
    return db, client


def setup_embeddings():
    """Set up Mistral AI embeddings."""
    api_key = os.getenv('MISTRAL_API_KEY')
    if api_key is None:
        print('You need to set your environment variable MISTRAL_API_KEY')
        exit(1)

    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        mistral_api_key=api_key
    )
    return embeddings


def hybrid_semantic_search(query, db, embeddings_model, top_n=5, approach="hybrid"):
    """
    Perform semantic search using the hybrid approach.

    Args:
        query: Natural language query string
        db: MongoDB database connection
        embeddings_model: The embedding model to use
        top_n: Number of results to return
        approach: Which search approach to use
                 - "fast": Use only aggregated embeddings
                 - "precise": Use only chunk embeddings
                 - "hybrid": Use aggregated for initial search, rerank with chunks

    Returns:
        List of matched agencies with their relevance scores and details
    """
    agencies_collection = db["agencies"]

    query_embedding = embeddings_model.embed_query(query)

    if approach == "fast" or approach == "hybrid":
        pipeline = [
            {
                "$search": {
                    "index": "agencies_search_index",
                    "knnBeta": {
                        "vector": query_embedding,
                        "path": "aggregatedEmbedding",
                        "k": top_n if approach == "fast" else top_n * 2
                    }
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "name": 1,
                    "textChunks": 1,
                    "representativeText": 1,
                    "score": { "$meta": "searchScore" }
                }
            }
        ]

        results = list(agencies_collection.aggregate(pipeline))

        if approach == "hybrid" and results:
            reranked_results = []

            for agency in results:
                chunks = agency.get("textChunks", [])
                best_chunk = None
                best_score = -1

                for chunk in chunks:
                    chunk_embedding = chunk.get("embedding")
                    if chunk_embedding:
                        similarity = cosine_similarity(query_embedding, chunk_embedding)

                        if similarity > best_score:
                            best_score = similarity
                            best_chunk = chunk

                if best_chunk:
                    reranked_results.append({
                        "agency_id": agency["_id"],
                        "name": agency.get("name"),
                        "original_score": agency.get("score"),
                        "reranked_score": best_score,
                        "text": agency.get("representativeText"),
                        "best_matching_chunk": best_chunk.get("text")
                    })

            reranked_results = sorted(reranked_results, key=lambda x: x["reranked_score"], reverse=True)[:top_n]
            return reranked_results

        return [
            {
                "agency_id": result["_id"],
                "name": result.get("name"),
                "score": result.get("score"),
                "text": result.get("representativeText")
            } for result in results[:top_n]
        ]

    elif approach == "precise":
        pipeline = [
            {
                "$search": {
                    "index": "agencies_search_index",
                    "knnBeta": {
                        "vector": query_embedding,
                        "path": "textChunks.embedding",
                        "k": top_n * 5
                    }
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "name": 1,
                    "representativeText": 1,
                    "matchedChunk": {
                        "$filter": {
                            "input": "$textChunks",
                            "as": "chunk",
                            "cond": {
                                "$eq": [
                                    "$$chunk.embedding",
                                    { "$meta": "searchVectorHighlight" }
                                ]
                            }
                        }
                    },
                    "score": { "$meta": "searchScore" }
                }
            },
            { "$sort": { "score": -1 } },
            { "$group": {
                "_id": "$_id",
                "name": { "$first": "$name" },
                "representativeText": { "$first": "$representativeText" },
                "matchedChunk": { "$first": "$matchedChunk" },
                "score": { "$first": "$score" }
            }},
            { "$sort": { "score": -1 } },
            { "$limit": top_n }
        ]

        try:
            results = list(agencies_collection.aggregate(pipeline))

            return [
                {
                    "agency_id": result["_id"],
                    "name": result.get("name"),
                    "score": result.get("score"),
                    "text": result.get("representativeText"),
                    "best_matching_chunk": result.get("matchedChunk")[0].get("text") if result.get("matchedChunk") else None
                } for result in results
            ]
        except Exception as e:
            print(f"Error with precise search: {e}")
            return fallback_precise_search(query, db, embeddings_model, top_n)

    else:
        raise ValueError("Invalid approach. Use 'fast', 'precise', or 'hybrid'.")


def fallback_precise_search(query, db, embeddings_model, top_n=5):
    """Fallback implementation for precise search if the aggregation pipeline fails."""
    agencies_collection = db["agencies"]
    query_embedding = embeddings_model.embed_query(query)

    agencies = list(agencies_collection.find(
        {"textChunks": {"$exists": True}},
        {"_id": 1, "name": 1, "textChunks": 1, "representativeText": 1}
    ))

    all_chunks = []

    for agency in agencies:
        for chunk in agency.get("textChunks", []):
            chunk_embedding = chunk.get("embedding")
            if chunk_embedding:
                all_chunks.append({
                    "agency_id": agency["_id"],
                    "name": agency.get("name"),
                    "agency_text": agency.get("representativeText"),
                    "chunk_text": chunk.get("text"),
                    "embedding": chunk_embedding
                })

    for chunk in all_chunks:
        chunk["score"] = cosine_similarity(query_embedding, chunk["embedding"])

    all_chunks.sort(key=lambda x: x["score"], reverse=True)

    seen_agencies = set()
    results = []

    for chunk in all_chunks:
        agency_id = chunk["agency_id"]
        if agency_id not in seen_agencies and len(results) < top_n:
            seen_agencies.add(agency_id)
            results.append({
                "agency_id": agency_id,
                "name": chunk["name"],
                "score": chunk["score"],
                "text": chunk["agency_text"],
                "best_matching_chunk": chunk["chunk_text"]
            })

    return results


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    v1 = np.array(vec1)
    v2 = np.array(vec2)

    dot_product = np.dot(v1, v2)
    norm_a = np.linalg.norm(v1)
    norm_b = np.linalg.norm(v2)

    if norm_a == 0 or norm_b == 0:
        return 0

    return dot_product / (norm_a * norm_b)


def search_cli():
    """Command-line interface for semantic search."""
    db, _ = connect_to_mongodb()
    embeddings_model = setup_embeddings()

    print("Welcome to Agency Semantic Search")
    print("--------------------------------")

    while True:
        print("\nSearch approaches:")
        print("1. Fast search (using aggregated embeddings)")
        print("2. Precise search (using chunk embeddings)")
        print("3. Hybrid search (fast + reranking)")
        print("q. Quit")

        choice = input("\nSelect search approach (1/2/3/q): ").strip().lower()

        if choice == 'q':
            break

        approach = ""
        if choice == '1':
            approach = "fast"
        elif choice == '2':
            approach = "precise"
        elif choice == '3':
            approach = "hybrid"
        else:
            print("Invalid choice. Please try again.")
            continue

        top_n = 5
        try:
            top_n_input = input(f"Number of results to show (default: {top_n}): ").strip()
            if top_n_input:
                top_n = int(top_n_input)
        except ValueError:
            print(f"Invalid number. Using default: {top_n}")

        query = input("\nEnter your search query: ").strip()

        if not query:
            print("Empty query. Please try again.")
            continue

        print(f"\nSearching for: '{query}' using {approach} approach...\n")

        results = hybrid_semantic_search(query, db, embeddings_model, top_n, approach)

        if results:
            print(f"Found {len(results)} matching agencies:\n")

            for i, result in enumerate(results):
                print(f"{i+1}. {result['name']}")

                if approach == "hybrid":
                    print(f"   Original Score: {result.get('original_score', 0):.4f}")
                    print(f"   Reranked Score: {result.get('reranked_score', 0):.4f}")
                else:
                    print(f"   Score: {result.get('score', 0):.4f}")

                if "best_matching_chunk" in result and result["best_matching_chunk"]:
                    print("\n   Best matching content snippet:")
                    print(f"   \"{result['best_matching_chunk'][:200]}...\"")

                show_full = input("\n   Show full agency description? (y/n): ").strip().lower()
                if show_full == 'y':
                    print("\n   Full Description:")
                    print(result["text"])
                print("\n" + "-" * 40)
        else:
            print("No matching agencies found.")


if __name__ == "__main__":
    search_cli()

