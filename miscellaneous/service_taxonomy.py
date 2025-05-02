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
    """Generate new categories based on specified structure with 70-80 parent categories
    and approximately 5000 total categories for a hiring platform."""
    print("\nGenerating 70-80 parent categories and approximately 5000 total categories for a hiring platform...\n")

    domain_topics = []
    if "domain_topics" in state:
        domain_topics = state["domain_topics"]
    else:
        domain_input = input("Enter comma-separated domains for the categories (e.g., Recruitment, Technology, Finance): ")
        domain_topics = [topic.strip() for topic in domain_input.split(',')]

    print(f"\nGenerating categories for the following domains: {', '.join(domain_topics)}")

    # Determine the number of parent categories (random between 70-80)
    num_parent_categories = random.randint(70, 80)
    print(f"Will generate {num_parent_categories} parent categories")

    # Calculate how many subcategories we need for approximately 5000 total
    total_desired = 5000
    num_subcategories_needed = total_desired - num_parent_categories

    # Distribute parent categories across domains
    domains_count = len(domain_topics)
    parents_per_domain = {}
    remaining_parents = num_parent_categories

    # Distribute parent categories evenly across domains with some randomness
    for i, domain in enumerate(domain_topics):
        if i == domains_count - 1:  # Last domain gets all remaining parents
            parents_per_domain[domain] = remaining_parents
        else:
            # Allocate parents with some randomness
            domain_parents = max(1, int(num_parent_categories / domains_count) +
                                 random.randint(-2, 2))
            domain_parents = min(domain_parents, remaining_parents - (domains_count - i - 1))
            parents_per_domain[domain] = domain_parents
            remaining_parents -= domain_parents

    print(f"Parent categories distribution across domains: {parents_per_domain}")

    # Generate parent categories by domain
    parent_categories = []
    for domain, domain_parent_count in parents_per_domain.items():
        print(f"\nGenerating {domain_parent_count} parent categories for domain: {domain}")

        # Generate parent categories in batches of 10 to avoid token limits
        for batch in range(0, domain_parent_count, 10):
            batch_size = min(10, domain_parent_count - batch)
            overall_display_order = len(parent_categories) + 1  # For global ordering

            parent_categories_prompt = f"""
            Create {batch_size} unique parent categories for a hiring platform in the '{domain}' domain following this exact structure:
            {{
              "_id": "generate a valid MongoDB ObjectId as string without the $oid wrapper",
              "name": "Category Name",
              "slug": "category-name-in-lowercase-with-hyphens",
              "parentCategoryId": null,
              "title": null,
              "description": "One line description of the category for a hiring platform",
              "marketable": boolean (randomly true or false),
              "displayOrder": integer between {overall_display_order}-{overall_display_order+batch_size-1},
              "logo": null,
              "domain": "{domain}"
            }}
            
            Requirements:
            1. Each category must have a unique name and slug
            2. parentCategoryId must be null for all parent categories
            3. Each name should be a real, commonly used job category or industry sector in the '{domain}' field for a hiring platform
            4. The description should be a single informative sentence explaining this job category
            5. Generate a valid MongoDB ObjectId as a string for _id (24 character hex string)
            6. displayOrder should be a number from {overall_display_order} to {overall_display_order+batch_size-1}
            7. title and logo should be null
            8. marketable should be randomly set to true or false
            9. Include "domain": "{domain}" in each object
            
            Return exactly {batch_size} parent category objects for a hiring platform.
            """

            parent_categories_result = llm.invoke(parent_categories_prompt)

            try:
                import re
                parent_json_match = re.search(r'\[\s*{\s*"_id".*}\s*\]', parent_categories_result.content, re.DOTALL)
                if parent_json_match:
                    batch_parent_categories = json.loads(parent_json_match.group(0))
                else:
                    objects_pattern = r'{\s*"_id"[^}]*}'
                    objects_matches = re.findall(objects_pattern, parent_categories_result.content, re.DOTALL)
                    batch_parent_categories = [json.loads(obj.replace("'", '"')) for obj in objects_matches][:batch_size]
            except Exception as e:
                print(f"Error parsing parent categories batch for domain {domain}: {e}")
                batch_parent_categories = []
                for i in range(batch_size):
                    oid = ''.join(random.choice('0123456789abcdef') for _ in range(24))
                    batch_parent_categories.append({
                        "_id": oid,
                        "name": f"{domain} Category {batch+i+1}",
                        "slug": f"{domain.lower().replace(' ', '-')}-category-{batch+i+1}",
                        "parentCategoryId": None,
                        "title": None,
                        "description": f"This is a parent job category for {domain} #{batch+i+1}",
                        "marketable": random.choice([True, False]),
                        "displayOrder": overall_display_order + i,
                        "logo": None,
                        "domain": domain
                    })

            while len(batch_parent_categories) < batch_size:
                oid = ''.join(random.choice('0123456789abcdef') for _ in range(24))
                batch_parent_categories.append({
                    "_id": oid,
                    "name": f"{domain} Category {batch+len(batch_parent_categories)+1}",
                    "slug": f"{domain.lower().replace(' ', '-')}-category-{batch+len(batch_parent_categories)+1}",
                    "parentCategoryId": None,
                    "title": None,
                    "description": f"This is a parent job category for {domain} #{batch+len(batch_parent_categories)+1}",
                    "marketable": random.choice([True, False]),
                    "displayOrder": overall_display_order + len(batch_parent_categories),
                    "logo": None,
                    "domain": domain
                })

            for category in batch_parent_categories:
                # Ensure domain field is present
                if "domain" not in category:
                    category["domain"] = domain

            parent_categories.extend(batch_parent_categories)
            print(f"Generated batch of {len(batch_parent_categories)} parent categories for {domain}. Total parents so far: {len(parent_categories)}")

    # Calculate average number of subcategories per parent
    avg_subcats_per_parent = num_subcategories_needed / num_parent_categories

    # Distribute subcategories among parents with significant variation
    # Since we're creating so many subcategories, we can have more variance
    subcats_distribution = []
    remaining = num_subcategories_needed

    # Create a hierarchy - some parents will have many subcategories and some of those will have sub-subcategories
    # This better matches the example JSON structure
    for i in range(num_parent_categories - 1):  # Distribute for all except the last parent
        if remaining <= 0:
            subcats_distribution.append(0)
        else:
            # Hiring platforms often have varied category depths with some having deep hierarchies
            # Use a longer-tailed distribution where some categories have many more subcategories
            if random.random() < 0.15:  # 15% chance of being a "major" category with many subcategories
                subcats = int(max(30, avg_subcats_per_parent * 2 + random.uniform(-20, 40)))
            elif random.random() < 0.3:  # 30% chance of being a "medium" category
                subcats = int(max(15, avg_subcats_per_parent + random.uniform(-10, 20)))
            else:  # 55% chance of being a "minor" category with fewer subcategories
                subcats = int(max(5, avg_subcats_per_parent * 0.5 + random.uniform(-5, 10)))

            subcats = min(subcats, remaining)
            subcats_distribution.append(subcats)
            remaining -= subcats

    # Assign remaining subcategories to the last parent
    subcats_distribution.append(max(0, remaining))

    print(f"Distributing {sum(subcats_distribution)} subcategories across {num_parent_categories} parent categories")

    subcategories = []
    parent_ids = [p["_id"] for p in parent_categories]

    for parent_idx, parent_id in enumerate(parent_ids):
        num_subcats = subcats_distribution[parent_idx]
        if num_subcats == 0:
            continue

        parent_name = parent_categories[parent_idx]["name"]

        # Generate subcategories in batches of 25 to avoid token limits
        # Using larger batches since we have so many more subcategories to generate
        for sub_batch in range(0, num_subcats, 25):
            sub_batch_size = min(25, num_subcats - sub_batch)

            subcat_prompt = f"""
            Create {sub_batch_size} unique subcategories for the parent category '{parent_name}' in the '{parent_categories[parent_idx]["domain"]}' domain 
            for a hiring platform. Follow this exact structure:
            {{
              "_id": "generate a valid MongoDB ObjectId as string without the $oid wrapper",
              "name": "Subcategory Name",
              "slug": "subcategory-name-in-lowercase-with-hyphens",
              "parentCategoryId": "{parent_id}",
              "title": null,
              "description": "One line description of the job subcategory",
              "marketable": boolean (randomly true or false),
              "displayOrder": integer between {sub_batch+1}-{sub_batch+sub_batch_size},
              "logo": null
            }}
            
            Requirements:
            1. Each subcategory must have a unique name and slug
            2. parentCategoryId must be set to "{parent_id}" for all subcategories
            3. Each name should be a real, commonly used job subcategory related to {parent_name} for hiring platforms
            4. The description should be a single informative sentence about this job subcategory
            5. Generate a valid MongoDB ObjectId as a string for _id (24 character hex string)
            6. displayOrder should be a number from {sub_batch+1} to {sub_batch+sub_batch_size} (each subcategory should have a unique displayOrder)
            7. title and logo should be null
            8. marketable should be randomly set to true or false
            9. For about 20% of the subcategories, make them detailed specializations (like "Senior React Developer" instead of just "React Developer")
            
            Return exactly {sub_batch_size} subcategory objects.
            """

            subcat_result = llm.invoke(subcat_prompt)

            try:
                import re
                subcat_json_match = re.search(r'\[\s*{\s*"_id".*}\s*\]', subcat_result.content, re.DOTALL)
                if subcat_json_match:
                    batch_subcats = json.loads(subcat_json_match.group(0))
                else:
                    objects_pattern = r'{\s*"_id"[^}]*}'
                    objects_matches = re.findall(objects_pattern, subcat_result.content, re.DOTALL)
                    batch_subcats = [json.loads(obj.replace("'", '"')) for obj in objects_matches][:sub_batch_size]
            except Exception as e:
                print(f"Error parsing subcategories for parent {parent_idx+1}, batch {sub_batch//10+1}: {e}")
                batch_subcats = []
                for i in range(sub_batch_size):
                    oid = ''.join(random.choice('0123456789abcdef') for _ in range(24))
                    batch_subcats.append({
                        "_id": oid,
                        "name": f"{parent_name} Subcategory {sub_batch+i+1}",
                        "slug": f"{parent_name.lower().replace(' ', '-')}-subcat-{sub_batch+i+1}",
                        "parentCategoryId": parent_id,
                        "title": None,
                        "description": f"This is a job subcategory for {parent_name} #{sub_batch+i+1}",
                        "marketable": random.choice([True, False]),
                        "displayOrder": sub_batch+i+1,
                        "logo": None
                    })

            while len(batch_subcats) < sub_batch_size:
                oid = ''.join(random.choice('0123456789abcdef') for _ in range(24))
                batch_subcats.append({
                    "_id": oid,
                    "name": f"{parent_name} Subcategory {sub_batch+len(batch_subcats)+1}",
                    "slug": f"{parent_name.lower().replace(' ', '-')}-subcat-{sub_batch+len(batch_subcats)+1}",
                    "parentCategoryId": parent_id,
                    "title": None,
                    "description": f"This is a job subcategory for {parent_name} #{sub_batch+len(batch_subcats)+1}",
                    "marketable": random.choice([True, False]),
                    "displayOrder": sub_batch+len(batch_subcats)+1,
                    "logo": None
                })

            # For approximately 15% of subcategory batches, create sub-subcategories (third level)
            create_third_level = random.random() < 0.15 and len(batch_subcats) > 0

            if create_third_level:
                # Select a random subcategory from this batch to have sub-subcategories
                sub_subcat_parent_idx = random.randint(0, len(batch_subcats) - 1)
                sub_subcat_parent = batch_subcats[sub_subcat_parent_idx]

                # Determine number of sub-subcategories (3-8)
                num_sub_subcats = random.randint(3, 8)

                sub_subcat_prompt = f"""
                Create {num_sub_subcats} unique sub-subcategories (third level) for the subcategory '{sub_subcat_parent["name"]}' 
                in the hiring platform. Follow this exact structure:
                {{
                  "_id": "generate a valid MongoDB ObjectId as string without the $oid wrapper",
                  "name": "Sub-Subcategory Name",
                  "slug": "sub-subcategory-name-in-lowercase-with-hyphens",
                  "parentCategoryId": "{sub_subcat_parent["_id"]}",
                  "title": null,
                  "description": "One line description of this specialized job category",
                  "marketable": boolean (randomly true or false),
                  "displayOrder": integer between 1-{num_sub_subcats},
                  "logo": null
                }}
                
                Requirements:
                1. Each sub-subcategory must have a unique name and slug
                2. parentCategoryId must be set to "{sub_subcat_parent["_id"]}" for all sub-subcategories
                3. Each name should be a specific job specialization related to {sub_subcat_parent["name"]}
                4. Make them very specific job titles like "Senior JavaScript Developer with React" or "Junior UI Designer with Figma Experience"
                5. The description should be a single informative sentence about this specialized job role
                6. Generate a valid MongoDB ObjectId as a string for _id (24 character hex string)
                7. displayOrder should be a number from 1 to {num_sub_subcats}
                8. title and logo should be null
                9. marketable should be randomly set to true or false
                
                Return exactly {num_sub_subcats} sub-subcategory objects.
                """

                sub_subcat_result = llm.invoke(sub_subcat_prompt)

                try:
                    import re
                    sub_subcat_json_match = re.search(r'\[\s*{\s*"_id".*}\s*\]', sub_subcat_result.content, re.DOTALL)
                    if sub_subcat_json_match:
                        sub_subcats = json.loads(sub_subcat_json_match.group(0))
                    else:
                        objects_pattern = r'{\s*"_id"[^}]*}'
                        objects_matches = re.findall(objects_pattern, sub_subcat_result.content, re.DOTALL)
                        sub_subcats = [json.loads(obj.replace("'", '"')) for obj in objects_matches][:num_sub_subcats]
                except Exception as e:
                    print(f"Error parsing sub-subcategories: {e}")
                    sub_subcats = []
                    for i in range(num_sub_subcats):
                        oid = ''.join(random.choice('0123456789abcdef') for _ in range(24))
                        sub_subcats.append({
                            "_id": oid,
                            "name": f"{sub_subcat_parent['name']} Specialization {i+1}",
                            "slug": f"{sub_subcat_parent['name'].lower().replace(' ', '-')}-spec-{i+1}",
                            "parentCategoryId": sub_subcat_parent["_id"],
                            "title": None,
                            "description": f"This is a specialized job category for {sub_subcat_parent['name']} #{i+1}",
                            "marketable": random.choice([True, False]),
                            "displayOrder": i+1,
                            "logo": None
                        })

                while len(sub_subcats) < num_sub_subcats:
                    oid = ''.join(random.choice('0123456789abcdef') for _ in range(24))
                    sub_subcats.append({
                        "_id": oid,
                        "name": f"{sub_subcat_parent['name']} Specialization {len(sub_subcats)+1}",
                        "slug": f"{sub_subcat_parent['name'].lower().replace(' ', '-')}-spec-{len(sub_subcats)+1}",
                        "parentCategoryId": sub_subcat_parent["_id"],
                        "title": None,
                        "description": f"This is a specialized job category for {sub_subcat_parent['name']} #{len(sub_subcats)+1}",
                        "marketable": random.choice([True, False]),
                        "displayOrder": len(sub_subcats)+1,
                        "logo": None
                    })

                subcategories.extend(sub_subcats)
                print(f"Generated {len(sub_subcats)} sub-subcategories (third level) for '{sub_subcat_parent['name']}'")

            subcategories.extend(batch_subcats)
            print(f"Generated {len(batch_subcats)} subcategories for parent #{parent_idx+1}. Total subcategories so far: {len(subcategories)}")

            # Add a progress indicator since this will take a while with 5000 categories
            if len(subcategories) % 500 == 0:
                print(f"===== Milestone: Generated {len(subcategories)} subcategories ({len(subcategories) + len(parent_categories)} total categories) =====")

            # If we've already reached our total goal, we can break early
            if len(subcategories) + len(parent_categories) >= total_desired:
                print(f"Reached target of {total_desired} total categories early. Breaking generation loop.")
                break

        # If we've already reached our total goal, we can break early
        if len(subcategories) + len(parent_categories) >= total_desired:
            break

    all_categories = parent_categories + subcategories

    validated_categories = []
    for category in all_categories:
        if all(key in category for key in ["_id", "name", "slug", "parentCategoryId", "title",
                                           "description", "marketable", "displayOrder", "logo"]):
            validated_categories.append(category)

    # Calculate some statistics about the distribution
    if subcategories:
        subcats_per_parent = {}
        for subcat in subcategories:
            parent_id = subcat["parentCategoryId"]
            if parent_id in subcats_per_parent:
                subcats_per_parent[parent_id] += 1
            else:
                subcats_per_parent[parent_id] = 1

        max_subcats = max(subcats_per_parent.values()) if subcats_per_parent else 0
        min_subcats = min(subcats_per_parent.values()) if subcats_per_parent else 0
        avg_subcats = sum(subcats_per_parent.values()) / len(subcats_per_parent) if subcats_per_parent else 0

        print(f"\nSubcategory distribution:")
        print(f"- Average subcategories per parent: {avg_subcats:.2f}")
        print(f"- Maximum subcategories for a parent: {max_subcats}")
        print(f"- Minimum subcategories for a parent: {min_subcats}")
        print(f"- Number of parents with subcategories: {len(subcats_per_parent)} out of {len(parent_categories)}")

    # Create nested structure like the example JSON
    def create_nested_structure(categories):
        # Create a dictionary to store the nested structure
        nested_structure = {}

        # First pass: identify all parent categories
        for category in categories:
            if category["parentCategoryId"] is None:
                nested_structure[category["name"]] = {}

        # Second pass: add subcategories to their parents
        for category in categories:
            if category["parentCategoryId"] is not None:
                # Find the parent category
                parent = next((c for c in categories if c["_id"] == category["parentCategoryId"]), None)
                if parent:
                    if parent["parentCategoryId"] is None:
                        # This is a second-level subcategory
                        if nested_structure.get(parent["name"]) is None:
                            nested_structure[parent["name"]] = {}
                        nested_structure[parent["name"]][category["name"]] = {}
                    else:
                        # This is a third-level subcategory
                        grandparent = next((c for c in categories if c["_id"] == parent["parentCategoryId"]), None)
                        if grandparent and grandparent["parentCategoryId"] is None:
                            if nested_structure.get(grandparent["name"]) is None:
                                nested_structure[grandparent["name"]] = {}
                            if nested_structure[grandparent["name"]].get(parent["name"]) is None:
                                nested_structure[grandparent["name"]][parent["name"]] = {}
                            nested_structure[grandparent["name"]][parent["name"]][category["name"]] = {}

        return nested_structure

    # Create the nested structure similar to example JSON
    nested_categories = create_nested_structure(validated_categories)

    print(f"\nGenerated hierarchical category structure with {len(nested_categories)} top-level categories")

    # Dump a preview of the nested structure (top 5 categories with their immediate children)
    preview_structure = {}
    for i, (category, subcats) in enumerate(nested_categories.items()):
        if i >= 5:  # Only show 5 top categories in preview
            break
        preview_structure[category] = {}
        for j, (subcategory, subsubcats) in enumerate(subcats.items()):
            if j >= 3:  # Only show 3 subcategories per parent in preview
                preview_structure[category][subcategory + " ... and more"] = {}
                break
            preview_structure[category][subcategory] = {}
            if subsubcats:
                for k, subsubcat in enumerate(subsubcats.keys()):
                    if k >= 2:  # Only show 2 sub-subcategories per subcategory in preview
                        preview_structure[category][subcategory][subsubcat + " ... and more"] = {}
                        break
                    preview_structure[category][subcategory][subsubcat] = {}

    print("\nPreview of hierarchical structure (truncated):")
    print(json.dumps(preview_structure, indent=2))

    # Return both flat and nested formats
    return {
        "generated_categories": validated_categories,
        "domain_topics": domain_topics,
        "nested_categories": nested_categories
    }

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