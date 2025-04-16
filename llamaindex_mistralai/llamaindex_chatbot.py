from llamaindex_mistralai.index_manager import load_or_create_index

DOCUMENTS_PATH = r"C:/Users/user/Desktop/Alisha/google genai 5 days workshop/trash"

if __name__ == "__main__":
    index = load_or_create_index(DOCUMENTS_PATH)
    query_engine = index.as_query_engine()

    print("\n=== Document Chatbot ===")
    print("Type 'exit' to quit")

    while True:
        user_input = input("\nYour question: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            break

        response = query_engine.query(user_input)
        print("\nResponse:", response.response)


