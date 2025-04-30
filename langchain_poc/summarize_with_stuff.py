from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
docs = loader.load()

if __name__ == "__main__":
    load_dotenv()

    api_key = os.getenv("MISTRAL_API_KEY")
    if api_key is None:
        print("You need to set your MISTRAL_API_KEY environment variable")
        exit(1)

    model_name = "mistral-large-latest"
    model_provider = "mistralai"
    llm = init_chat_model(model_name, model_provider=model_provider)

    prompt = ChatPromptTemplate.from_messages(
        [("system", "Write a concise summary of the following:\\n\\n{context}")]
    )

    chain = create_stuff_documents_chain(llm, prompt)

    result = chain.invoke({"context": docs})
    print(result)

    for token in chain.stream({"context": docs}):
        print(token, end="|")
