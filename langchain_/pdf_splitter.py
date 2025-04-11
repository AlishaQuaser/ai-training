from langchain_text_splitters import RecursiveCharacterTextSplitter
from base import load_pdf_documents

file_path = "C:/Users/user/Desktop/Alisha/google genai 5 days workshop/whitepaper_Foundational Large Language models & text generation.pdf"
docs = load_pdf_documents(file_path)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)

all_splits = text_splitter.split_documents(docs)

if __name__ == '__main__':
    print(f"Total number of chunks: {len(all_splits)}")
    page_chunks = [doc for doc in all_splits if doc.metadata["page"] == 7]
    if len(page_chunks) >= 2:
        print("--- Second chunk from page 8 ---\n")
        print(page_chunks[1].page_content)
        print("\nMetadata:", page_chunks[1].metadata)
    else:
        print("Less than 2 chunks on page 8.")

