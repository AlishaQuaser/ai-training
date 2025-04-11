import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from base import load_pdf_documents

load_dotenv()


class ChatBot:
    def __init__(self, model_name, document_retriever, model_provider="mistralai"):
        self.model = init_chat_model(model_name, model_provider=model_provider)
        self.document_retriever = document_retriever

        self.system_template = (
            "You are a helpful assistant. Use the following context to answer the user's question:\n\n{context}"
        )
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_template),
            ("user", "{question}")
        ])
        self.conversation_history = []

    def ask(self, question):
        self.conversation_history.append(HumanMessage(content=question))

        retrieved_docs = self.document_retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        prompt = self.prompt_template.invoke({
            "context": context,
            "question": question
        })

        print("\nAnswer:")

        for token in self.model.stream(prompt):
            print(token.content, end="")
        print("\n")


if __name__ == "__main__":
    api_key = os.getenv("MISTRAL_API_KEY")
    if api_key is None:
        print("Please set environment variable MISTRAL_API_KEY")
        exit(1)

    file_path = "C:/Users/user/Desktop/Alisha/google genai 5 days workshop/nke-10k-2023.pdf"
    documents = load_pdf_documents(file_path)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)

    embeddings = MistralAIEmbeddings(model="mistral-embed")
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(chunks)

    # kwargs stands for "keyword arguments"
    # It's a Python concept that lets you pass a dictionary of named arguments to a function.
    # The as_retriever() method allows you to customize how the retriever fetches documents.
    # search_kwargs is a dictionary of parameters that control how the retriever searches the vector store.
    # means:
    # "When retrieving documents, return the top 4 most relevant documents."
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # This would mean:
    # Retrieve up to 4 results
    # But only return documents where similarity score is above 0.8
    # Not all retrievers support all options — k is common and usually supported.
    # retriever = vector_store.as_retriever(search_kwargs={"k": 4, "score_threshold": 0.8})

    bot = ChatBot("mistral-large-latest", document_retriever=retriever)

    print("Ask your questions (type 'exit' to stop):\n")
    while True:
        user_question = input("You: ")
        if user_question.strip().lower() == "exit":
            print("Goodbye!")
            break
        bot.ask(user_question)
