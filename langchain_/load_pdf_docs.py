from langchain_community.document_loaders import PyPDFLoader

file_path = "C:/Users/user/Desktop/Alisha/google genai 5 days workshop/whitepaper_Foundational Large Language models & text generation.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

print(len(docs))
print("\n")
print("first page content:")
print(docs[0].page_content[:200])  # shows first 200 characters of text from page 1
print("\n")
print("Metadata:")
print(docs[0].metadata)

