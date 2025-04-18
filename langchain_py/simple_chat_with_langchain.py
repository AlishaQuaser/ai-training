import getpass  # To securely prompt the user for their API key if not set in env
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model  # For initializing a Mistral-based chat model
from langchain_core.messages import HumanMessage, SystemMessage  # Message types for the chat


load_dotenv()

if __name__ == '__main__':
    # Prompt for API key if it's not already set in environment variables
    if not os.environ.get("MISTRAL_API_KEY"):
        os.environ["MISTRAL_API_KEY"] = getpass.getpass("Enter API key for Mistral AI: ")

    # Initialize the chat model with the Mistral provider
    model = init_chat_model("mistral-large-latest", model_provider="mistralai")
    # Define the list of messages for the conversation
    messages = [
        SystemMessage("Translate the following into Spanish"),
        HumanMessage("My name is Alisha and I stay in Bhopal!"),
    ]
    # response = model.invoke(messages)
    # print(response.content)

    # Stream the response from the model token-by-token and print it
    for token in model.stream(messages):
        print(token.content, end="")
