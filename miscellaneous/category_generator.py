from typing_extensions import TypedDict
from pymongo import MongoClient
import os
import json
import requests
from bs4 import BeautifulSoup
import random
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from typing_extensions import Annotated
from langgraph import graph as langgraph
import uuid


class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str
    analysis: str
    feedback: str
    generated_categories: list
    validation_result: str


class QueryOutput(TypedDict):
    """Generated MongoDB query."""
    query: Annotated[str, ..., "Valid MongoDB query in Python syntax."]


class CategoryOutput(TypedDict):
    """Generated new categories."""
    categories: Annotated[list, ..., "List of valid new category objects following DB schema."]


def get_mongo_client():
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise ValueError("MONGO_URI not set in environment variables.")
    return MongoClient(mongo_uri)


def get_categories_collection_info():
    client = get_mongo_client()
    db = client["app-dev"]

    collection = db["categories"]
    sample_doc = collection.find_one()

    if sample_doc:
        return {"categories": list(sample_doc.keys())}
    else:
        return {"categories": []}


def write_query(state: State):
    """Generate MongoDB query to fetch information from categories collection."""
    collection_info = get_categories_collection_info()

    prompt = f"""
    Write a MongoDB query in Python syntax to answer the question: {state["question"]}
    
    Available collection and its fields:
    {collection_info}
    
    Return only the Python code for executing the MongoDB query.
    The code should use the 'db["categories"]' collection which is already connected to the database.
    Only query the 'categories' collection.
    """

    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    return {"query": result["query"]}


def execute_query(state: State):
    """Execute MongoDB query."""
    client = get_mongo_client()
    db = client["app-dev"]

    result = eval(state["query"])

    if hasattr(result, 'to_list'):
        result = list(result)

    return {"result": str(result)}


def generate_answer(state: State):
    """Answer question using retrieved information as context."""
    prompt = (
        "Given the following user question, corresponding MongoDB query, "
        "and query result, answer the user question.\n\n"
        f'Question: {state["question"]}\n'
        f'MongoDB Query: {state["query"]}\n'
        f'Query Result: {state["result"]}'
    )
    response = llm.invoke(prompt)
    return {"answer": response.content}


def analyze_data(state: State):
    """Analyze categories collection data patterns."""
    client = get_mongo_client()
    db = client["app-dev"]

    sample_docs = list(db["categories"].find().limit(5))
    count = db["categories"].count_documents({})

    fields = {}
    if sample_docs:
        for doc in sample_docs:
            for field, value in doc.items():
                if field not in fields:
                    fields[field] = {"type": type(value).__name__, "examples": []}
                if len(fields[field]["examples"]) < 3 and value not in fields[field]["examples"]:
                    fields[field]["examples"].append(value)

    analysis_prompt = f"""
    Analyze the following MongoDB collection data and identify patterns, data types, relationships, and potential insights:
    
    Collection: categories
    Total documents: {count}
    
    Fields and their types:
    {fields}
    
    Sample documents:
    {sample_docs[:3]}
    
    Provide a comprehensive analysis that includes:
    1. Data structure and organization
    2. Types of data stored and their formats
    3. Potential relationships between fields
    4. Common patterns or anomalies
    5. Suggestions for how this data might be used
    """

    response = llm.invoke(analysis_prompt)
    return {"analysis": response.content}


def get_analysis_feedback(state: State):
    """Process feedback on the data analysis."""
    prompt = f"""
    Review the feedback provided on your data analysis and determine if it indicates improvement is needed.
    
    Original Analysis:
    {state["analysis"]}
    
    Feedback:
    {state["feedback"]}
    
    Based on this feedback, should the analysis be revised? If so, what specific improvements are needed?
    """

    response = llm.invoke(prompt)
    return {"answer": response.content}


def generate_new_categories(state: State):
    """Generate new categories based on specified structure with 5 parent categories and 40 subcategories."""
    print("\nGenerating 5 parent categories and 40 subcategories...\n")

    domain_topic = ""
    if "domain_topic" in state:
        domain_topic = state["domain_topic"]
    else:
        domain_topic = input("Enter the domain for the categories (e.g., Development Tools, Marketing, Finance): ")

    print(f"\nGenerating categories in the '{domain_topic}' domain...")

    parent_categories_prompt = f"""
    Create 5 unique parent categories for the '{domain_topic}' domain following this exact structure:
    {{
      "_id": "generate a valid MongoDB ObjectId as string without the $oid wrapper",
      "name": "Category Name",
      "slug": "category-name-in-lowercase-with-hyphens",
      "parentCategoryId": null,
      "title": null,
      "description": "One line description of the category",
      "marketable": boolean (randomly true or false),
      "displayOrder": integer between 1-5,
      "logo": null
    }}
    
    Requirements:
    1. Each category must have a unique name and slug
    2. parentCategoryId must be null for all parent categories
    3. Each name should be a real, commonly used category in the '{domain_topic}' industry
    4. The description should be a single informative sentence
    5. Generate a valid MongoDB ObjectId as a string for _id (24 character hex string)
    6. displayOrder should be a number from 1 to 5
    7. title and logo should be null
    8. marketable should be randomly set to true or false
    
    Return exactly 5 parent category objects.
    """

    parent_categories_result = llm.invoke(parent_categories_prompt)

    try:
        import re
        parent_json_match = re.search(r'\[\s*{\s*"_id".*}\s*\]', parent_categories_result.content, re.DOTALL)
        if parent_json_match:
            parent_categories = json.loads(parent_json_match.group(0))
        else:
            objects_pattern = r'{\s*"_id"[^}]*}'
            objects_matches = re.findall(objects_pattern, parent_categories_result.content, re.DOTALL)
            parent_categories = [json.loads(obj.replace("'", '"')) for obj in objects_matches][:5]
    except Exception as e:
        print(f"Error parsing parent categories: {e}")
        parent_categories = []
        for i in range(5):
            oid = ''.join(random.choice('0123456789abcdef') for _ in range(24))
            parent_categories.append({
                "_id": oid,
                "name": f"{domain_topic} Category {i+1}",
                "slug": f"{domain_topic.lower().replace(' ', '-')}-category-{i+1}",
                "parentCategoryId": None,
                "title": None,
                "description": f"This is a parent category for {domain_topic} #{i+1}",
                "marketable": random.choice([True, False]),
                "displayOrder": i+1,
                "logo": None
            })

    while len(parent_categories) < 5:
        oid = ''.join(random.choice('0123456789abcdef') for _ in range(24))
        parent_categories.append({
            "_id": oid,
            "name": f"{domain_topic} Category {len(parent_categories)+1}",
            "slug": f"{domain_topic.lower().replace(' ', '-')}-category-{len(parent_categories)+1}",
            "parentCategoryId": None,
            "title": None,
            "description": f"This is a parent category for {domain_topic} #{len(parent_categories)+1}",
            "marketable": random.choice([True, False]),
            "displayOrder": len(parent_categories)+1,
            "logo": None
        })

    subcategories = []
    parent_ids = [p["_id"] for p in parent_categories]

    subcats_per_parent = [8, 8, 8, 8, 8]

    for parent_idx, parent_id in enumerate(parent_ids):
        num_subcats = subcats_per_parent[parent_idx]
        parent_name = parent_categories[parent_idx]["name"]

        subcat_prompt = f"""
        Create {num_subcats} unique subcategories for the parent category '{parent_name}' in the '{domain_topic}' domain.
        
        Each subcategory must follow this exact structure:
        {{
          "_id": "generate a valid MongoDB ObjectId as string without the $oid wrapper",
          "name": "Subcategory Name",
          "slug": "subcategory-name-in-lowercase-with-hyphens",
          "parentCategoryId": "{parent_id}",
          "title": null,
          "description": "One line description of the subcategory",
          "marketable": boolean (randomly true or false),
          "displayOrder": integer between 1-{num_subcats},
          "logo": null
        }}
        
        Requirements:
        1. Each subcategory must have a unique name and slug
        2. parentCategoryId must be set to "{parent_id}" for all subcategories
        3. Each name should be a real, commonly used subcategory related to {parent_name} in the '{domain_topic}' industry
        4. The description should be a single informative sentence
        5. Generate a valid MongoDB ObjectId as a string for _id (24 character hex string)
        6. displayOrder should be a number from 1 to {num_subcats} (each subcategory should have a unique displayOrder)
        7. title and logo should be null
        8. marketable should be randomly set to true or false
        
        Return exactly {num_subcats} subcategory objects.
        """

        subcat_result = llm.invoke(subcat_prompt)

        try:
            import re
            subcat_json_match = re.search(r'\[\s*{\s*"_id".*}\s*\]', subcat_result.content, re.DOTALL)
            if subcat_json_match:
                parent_subcats = json.loads(subcat_json_match.group(0))
            else:
                objects_pattern = r'{\s*"_id"[^}]*}'
                objects_matches = re.findall(objects_pattern, subcat_result.content, re.DOTALL)
                parent_subcats = [json.loads(obj.replace("'", '"')) for obj in objects_matches][:num_subcats]
        except Exception as e:
            print(f"Error parsing subcategories for parent {parent_idx+1}: {e}")
            parent_subcats = []
            for i in range(num_subcats):
                oid = ''.join(random.choice('0123456789abcdef') for _ in range(24))
                parent_subcats.append({
                    "_id": oid,
                    "name": f"{parent_name} Subcategory {i+1}",
                    "slug": f"{parent_name.lower().replace(' ', '-')}-subcat-{i+1}",
                    "parentCategoryId": parent_id,
                    "title": None,
                    "description": f"This is a subcategory for {parent_name} #{i+1}",
                    "marketable": random.choice([True, False]),
                    "displayOrder": i+1,
                    "logo": None
                })

        while len(parent_subcats) < num_subcats:
            oid = ''.join(random.choice('0123456789abcdef') for _ in range(24))
            parent_subcats.append({
                "_id": oid,
                "name": f"{parent_name} Subcategory {len(parent_subcats)+1}",
                "slug": f"{parent_name.lower().replace(' ', '-')}-subcat-{len(parent_subcats)+1}",
                "parentCategoryId": parent_id,
                "title": None,
                "description": f"This is a subcategory for {parent_name} #{len(parent_subcats)+1}",
                "marketable": random.choice([True, False]),
                "displayOrder": len(parent_subcats)+1,
                "logo": None
            })

        subcategories.extend(parent_subcats)

    all_categories = parent_categories + subcategories

    validated_categories = []
    for category in all_categories:
        if all(key in category for key in ["_id", "name", "slug", "parentCategoryId", "title",
                                           "description", "marketable", "displayOrder", "logo"]):
            validated_categories.append(category)

    print(f"Generated {len(validated_categories)} total categories ({len(parent_categories)} parents, {len(subcategories)} subcategories)")

    return {"generated_categories": validated_categories, "domain_topic": domain_topic}


def validate_new_categories(state: State):
    """Validate that generated categories follow the correct structure and are unique."""
    client = get_mongo_client()
    db = client["app-dev"]

    existing_categories = list(db["categories"].find({}, {"_id": 0}))

    generated_categories = state["generated_categories"]
    validation_issues = []
    valid_categories = []

    for category in generated_categories:
        required_fields = ["_id", "name", "slug", "parentCategoryId", "title",
                           "description", "marketable", "displayOrder", "logo"]

        missing_fields = [field for field in required_fields if field not in category]

        if missing_fields:
            validation_issues.append(f"Missing fields in category: {missing_fields}")
            continue

        is_duplicate = False
        for existing in existing_categories:
            if (category.get("name") == existing.get("name") or
                    category.get("slug") == existing.get("slug")):
                is_duplicate = True
                validation_issues.append(f"Duplicate category found: {category}")
                break

        if not is_duplicate:
            valid_categories.append(category)

    if valid_categories:
        timestamp = int(time.time())
        filename = f"new_categories_{timestamp}.json"
        with open(filename, "w") as f:
            json.dump(valid_categories, f, indent=2)

        result_message = f"Successfully validated {len(valid_categories)} categories. Saved to {filename}."
        if validation_issues:
            result_message += f" Issues found: {len(validation_issues)}"
    else:
        result_message = f"No valid categories generated. Issues: {validation_issues}"

    return {"validation_result": result_message}


def chat_session():
    """Run an interactive chat session for the categories collection."""
    print("MongoDB Chatbot for Categories Collection started. Type 'exit' to end the conversation.\n")
    print("Type 'analyze' to run a data analysis on the categories collection.\n")
    print("Type 'generate-categories' to create new categories based on the existing structure.\n")

    session_id = "chat_session_" + str(os.urandom(4).hex())
    config = {"configurable": {"thread_id": session_id}}

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        if user_input.lower() == "analyze":
            try:
                print("\nAnalyzing categories collection data...\n")

                client = get_mongo_client()
                db = client["app-dev"]

                sample_docs = list(db["categories"].find().limit(5))
                count = db["categories"].count_documents({})

                fields = {}
                if sample_docs:
                    for doc in sample_docs:
                        for field, value in doc.items():
                            if field not in fields:
                                fields[field] = {"type": type(value).__name__, "examples": []}
                            if len(fields[field]["examples"]) < 3 and value not in fields[field]["examples"]:
                                fields[field]["examples"].append(value)

                analysis_prompt = f"""
                Analyze the following MongoDB collection data and identify patterns, data types, relationships, and potential insights:
                
                Collection: categories
                Total documents: {count}
                
                Fields and their types:
                {fields}
                
                Sample documents:
                {sample_docs[:3]}
                
                Provide a comprehensive analysis that includes:
                1. Data structure and organization
                2. Types of data stored and their formats
                3. Potential relationships between fields
                4. Common patterns or anomalies
                5. Suggestions for how this data might be used
                """

                analysis = llm.invoke(analysis_prompt)
                print(f"\nData Analysis:\n{analysis.content}\n")

                user_feedback = input("\nProvide feedback on this analysis (or type 'skip' to continue): ")
                if user_feedback.lower() != "skip":
                    feedback_prompt = f"""
                    Review the feedback provided on your data analysis and determine if it indicates improvement is needed.
                    
                    Original Analysis:
                    {analysis.content}
                    
                    Feedback:
                    {user_feedback}
                    
                    Based on this feedback, what specific improvements would you make to the analysis?
                    """

                    improved_analysis = llm.invoke(feedback_prompt)
                    print(f"\nRevised Analysis:\n{improved_analysis.content}\n")

                continue
            except Exception as e:
                print(f"An error occurred during analysis: {str(e)}\n")
                continue

        elif user_input.lower() == "generate-categories":
            try:
                print("\nGenerating categories with 5 parents and 40 subcategories...\n")

                initial_state = {}

                print("Step 1: Generating new categories...")
                generation_result = generate_new_categories(initial_state)

                initial_state.update(generation_result)

                print("Step 2: Validating generated categories...")
                validation_result = validate_new_categories(initial_state)

                initial_state.update(validation_result)

                print(f"\nGeneration complete. {len(initial_state['generated_categories'])} categories generated.")
                print(f"Validation result: {initial_state['validation_result']}")

                show_categories = input("\nDo you want to see the generated categories? (yes/no): ")
                if show_categories.lower() == "yes":
                    parent_cats = [c for c in initial_state['generated_categories'] if c['parentCategoryId'] is None]
                    print(f"\n--- Parent Categories ({len(parent_cats)}) ---")
                    for i, category in enumerate(parent_cats):
                        print(f"\nParent Category {i+1}:")
                        print(json.dumps(category, indent=2))

                    for parent in parent_cats:
                        subcats = [c for c in initial_state['generated_categories']
                                   if c['parentCategoryId'] == parent['_id']]
                        print(f"\n\n--- Subcategories for {parent['name']} ({len(subcats)}) ---")
                        for i, subcat in enumerate(subcats):
                            print(f"\nSubcategory {i+1}:")
                            print(json.dumps(subcat, indent=2))

                continue
            except Exception as e:
                print(f"An error occurred during category generation: {str(e)}\n")
                continue

        try:
            initial_state = {"question": user_input}
            initial_result = graph.invoke(initial_state, config)

            print(f"\nGenerated MongoDB query: {initial_result['query']}")

            user_approval = input("Do you want to execute this query? (yes/no): ")
            if user_approval.lower() != "yes":
                print("Query execution cancelled.\n")
                continue

            client = get_mongo_client()
            db = client["app-dev"]

            query_code = initial_result['query']
            query_result = eval(query_code)

            if hasattr(query_result, 'to_list'):
                query_result = list(query_result)

            print(f"\nQuery result: {query_result}")

            answer_prompt = (
                "Given the following user question, corresponding MongoDB query, "
                "and query result, answer the user question.\n\n"
                f'Question: {user_input}\n'
                f'MongoDB Query: {query_code}\n'
                f'Query Result: {query_result}'
            )
            answer = llm.invoke(answer_prompt)
            print(f"\nAnswer: {answer.content}\n")

        except Exception as e:
            print(f"An error occurred: {str(e)}\n")


if __name__ == "__main__":
    load_dotenv()
    import time

    api_key = os.getenv("MISTRAL_API_KEY")
    if api_key is None:
        print("You need to set your MISTRAL_API_KEY environment variable")
        exit(1)

    llm = init_chat_model("mistral-large-latest", model_provider="mistralai")

    client = get_mongo_client()
    db = client["app-dev"]
    print(f"Connected to MongoDB database: app-dev")
    print(f"Using collection: categories")
    sample = db["categories"].find_one()
    if sample:
        print(f"Sample category fields: {list(sample.keys())}")
    else:
        print("The categories collection is empty.")

    builder = langgraph.StateGraph(State)
    builder.add_node("write_query", write_query)
    builder.add_node("execute_query", execute_query)
    builder.add_node("generate_answer", generate_answer)
    builder.add_node("analyze_data", analyze_data)
    builder.add_node("get_analysis_feedback", get_analysis_feedback)
    builder.add_node("generate_new_categories", generate_new_categories)
    builder.add_node("validate_new_categories", validate_new_categories)

    builder.add_edge("write_query", "execute_query")
    builder.add_edge("execute_query", "generate_answer")

    builder.set_entry_point("write_query")
    graph = builder.compile()

    chat_session()