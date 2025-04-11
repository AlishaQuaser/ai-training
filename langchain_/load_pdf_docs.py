# pdf_loader_demo.py

from base import load_pdf_documents

docs = load_pdf_documents("C:/Users/user/Desktop/Alisha/google genai 5 days workshop/whitepaper_Foundational Large Language models & text generation.pdf")

if __name__ == '__main__':
    print(len(docs))
    print("\n")
    print("First page content:")
    print(docs[0].page_content[:200])
    print("\n")
    print("Metadata:")
    print(docs[0].metadata)

