from typing_extensions import TypedDict
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from typing_extensions import Annotated
from langgraph import graph as langgraph


class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str


class QueryOutput(TypedDict):
    """Generated MongoDB query."""
    query: Annotated[str, ..., "Valid MongoDB query in Python syntax."]


def get_mongo_client():
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise ValueError("MONGO_URI not set in environment variables.")
    return MongoClient(mongo_uri)


def get_collection_info():
    client = get_mongo_client()
    db = client["app-dev"]

    collection_names = db.list_collection_names()
    collection_info = {}

    for collection_name in collection_names:
        collection = db[collection_name]
        sample_doc = collection.find_one()
        if sample_doc:
            collection_info[collection_name] = list(sample_doc.keys())

    return collection_info


def write_query(state: State):
    """Generate MongoDB query to fetch information."""
    collection_info = get_collection_info()

    prompt = f"""
    Write a MongoDB query in Python syntax to answer the question: {state["question"]}
    
    Available collections and their fields:
    {collection_info}
    
    Return only the Python code for executing the MongoDB query.
    The code should use the 'db' variable which is already connected to the database.
    """

    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    return {"query": result["query"]}


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


def execute_query(state: State):
    """Execute MongoDB query."""
    client = get_mongo_client()
    db = client["app-dev"]

    result = eval(state["query"])

    if hasattr(result, 'to_list'):
        result = list(result)

    return {"result": str(result)}


def chat_session():
    """Run an interactive chat session."""
    print("MongoDB Chatbot started. Type 'exit' to end the conversation.\n")

    session_id = "chat_session_" + str(os.urandom(4).hex())
    config = {"configurable": {"thread_id": session_id}}

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

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

    api_key = os.getenv("MISTRAL_API_KEY")
    if api_key is None:
        print("You need to set your MISTRAL_API_KEY environment variable")
        exit(1)

    llm = init_chat_model("mistral-large-latest", model_provider="mistralai")

    client = get_mongo_client()
    db = client["app-dev"]
    print(f"Connected to MongoDB database: app-dev")
    print(f"Available collections: {db.list_collection_names()}")

    builder = langgraph.StateGraph(State)
    builder.add_node("write_query", write_query)
    builder.add_node("execute_query", execute_query)
    builder.add_node("generate_answer", generate_answer)

    builder.add_edge("write_query", "execute_query")
    builder.add_edge("execute_query", "generate_answer")

    builder.set_entry_point("write_query")
    graph = builder.compile()

    chat_session()