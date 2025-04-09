import os
from mistralai import Mistral
from dotenv import load_dotenv
load_dotenv()

if __name__ == '__main__':
    api_key = os.getenv('MISTRAL_API_KEY')
    model = "mistral-large-latest"

    client = Mistral(api_key=api_key)

    # stream() sends messages and receives a response in real-time chunks (instead of waiting for the whole reply)
    stream_response = client.chat.stream(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "What is the best French cheese?",
            },
        ]
    )

    # Instead of getting one full reply, we loop through the streamed chunks
    for chunk in stream_response:
        # Print each piece of the response as it comes
        # 'delta' contains the current partial output being generated
        # 'print(..., end="", flush=True)' makes it appear like live typing
        # The end="" parameter tells print() not to add a newline after each chunk, so all the tokens are printed on the same line like a flowing sentence.
        print(chunk.data.choices[0].delta.content, end="")
