from dotenv import load_dotenv
import os
from groq import Groq
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("Error: GROQ_API_KEY not found in environment variables")
    exit(1)

client = Groq(api_key=api_key)


def main():
    messages = []

    print("Groq Chatbot (type 'exit' to quit)")
    print("-" * 50)
    print("Using model: llama3-70b-8192")

    while True:
        # Get user input
        user_input = input("\nYou: ")

        # Check if user wants to exit
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("\nGoodbye!")
            break

        messages.append({"role": "user", "content": user_input})

        # Get the response
        print("\nBot: ", end="", flush=True)

        stream = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=messages,
            stream=True
        )

        response_content = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                response_content += content

        print()
        messages.append({"role": "assistant", "content": response_content})


if __name__ == "__main__":
    main()
