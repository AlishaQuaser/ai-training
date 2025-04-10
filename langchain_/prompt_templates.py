import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate  # To define and use a prompt template
load_dotenv()

if __name__ == "__main__":
    if not os.environ.get("MISTRAL_API_KEY"):
        os.environ["MISTRAL_API_KEY"] = os.getenv("Enter API key for Mistral AI: ")

    model = init_chat_model("mistral-large-latest", model_provider="mistralai")

    # Define the system template with a placeholder for language
    system_template = "Translate the following from English into {language}"

    # Create a prompt template that takes both system and user messages with placeholders
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_template), ("user", "{text}")]
    )

    # Inject actual values into the prompt template
    prompt = prompt_template.invoke({"language": "Urdu", "text": "My name is Alisha!"})

    # Print the constructed prompt and the message format (for debugging or understanding)
    print(prompt)
    print(prompt.to_messages())

    # Get the model's response to the constructed prompt
    response = model.invoke(prompt)

    # Print the final translated content
    print(response.content)
