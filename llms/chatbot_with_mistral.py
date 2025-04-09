# Used to interact with operating system, like reading environment variables
import os

# Import the Mistral class from the mistralai library (used to talk to Mistral AI)
from mistralai import Mistral

# Loads environment variables from .env file into the program
from dotenv import load_dotenv

# Actually loads the environment variables, so we can use them later
load_dotenv()

# This line makes sure the code inside only runs when this file is run directly (not when imported)
if __name__ == '__main__':
    api_key = os.getenv('MISTRAL_API_KEY')    # Reads the API key named 'MISTRAL_API_KEY' from the .env file
    if api_key is None:                       # If the API key is missing, show a message and stop the program
        print('You need to set your MISTRAL_API_KEY environment variable')
        exit(1)

    # print("api key is : {}".format(api_key))

    model = "mistral-large-latest"             # Set which model you want to use (like choosing a version of the chatbot)
    client = Mistral(api_key=api_key)          # Create a client that talks to Mistral AI using your API key
    chat_response = client.chat.complete(      # Send a chat request to Mistral AI with a conversation made of messages
        model=model,
        messages=[
            {
                "role": "system",              # Sets context or background info for the chatbot
                "content": "My name is Alisha Quaser"
            },
            {
                "role": "user",                # This is what the user is asking
                "content": "What is the capital of Madhya Pradesh?",
            },
            {
                "role": "user",
                "content": "What is my name ?",  # Asking the bot something it should remember
            }

        ]
    )
    print(chat_response.choices[0].message.content)  # Print just the chatbot's reply (first one it gives)

