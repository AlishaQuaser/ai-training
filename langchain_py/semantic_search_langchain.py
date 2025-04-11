import os
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from base import load_pdf_documents

load_dotenv()


if __name__ == "__main__":
    api_key = os.getenv("MISTRAL_API_KEY")
    if api_key is None:
        print("Please set environment variable MISTRAL_API_KEY")
        exit(1)

    file_path = "C:/Users/user/Desktop/Alisha/google genai 5 days workshop/nke-10k-2023.pdf"
    docs = load_pdf_documents(file_path)

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)
    print(f"Total splits created: {len(all_splits)}")

    # Create embeddings
    embeddings = MistralAIEmbeddings(model="mistral-embed")

    # Store embeddings in memory
    vector_store = InMemoryVectorStore(embeddings)
    ids = vector_store.add_documents(documents=all_splits)

    # Perform a basic similarity search
    query_1 = "How many distribution centers does Nike have in the US?"
    results = vector_store.similarity_search(query_1)
    print("\nTop result for text query 1:")
    print(results[0].page_content)

    # Perform similarity search with score
    query_2 = "What was Nike's revenue in 2023?"
    results_with_score = vector_store.similarity_search_with_score(query_2)
    doc, score = results_with_score[0]
    print(f"\nTop result for text query 2 with score:\nScore: {score}")
    print(doc.page_content)

    # Perform similarity search using embedding vector
    query_3 = "How were Nike's margins impacted in 2023?"
    query_vector = embeddings.embed_query(query_3)
    vector_results = vector_store.similarity_search_by_vector(query_vector)
    print("\nTop result for vector-based similarity search:")
    print(vector_results[0].page_content)



