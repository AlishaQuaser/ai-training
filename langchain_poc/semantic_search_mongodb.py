import os
import numpy as np
import faiss
from pymongo import MongoClient
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

load_dotenv()


class MongoFAISSSearchEngine:
    def __init__(self):
        self.api_key = os.getenv('MISTRAL_API_KEY')
        if self.api_key is None:
            print('You need to set your environment variable MISTRAL_API_KEY')
            exit(1)

        self.embeddings = MistralAIEmbeddings(
            model="mistral-embed",
            mistral_api_key=self.api_key
        )

        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )

    def extract_documents_from_mongodb(self, db_name, collection_name):
        mongo_uri = os.getenv("MONGO_URI")
        if not mongo_uri:
            print("You need to set MONGO_URI in your environment variables.")
            exit(1)

        client = MongoClient(mongo_uri)
        db = client[db_name]
        collection = db[collection_name]

        docs = list(collection.find())
        if not docs:
            print(f"No documents found in collection '{collection_name}'.")
            exit(1)

        print(f"Extracted {len(docs)} documents from MongoDB collection '{collection_name}'")
        return docs

    def process_documents(self, docs):
        processed_docs = []
        for profile in docs:
            first_name = profile.get('firstName', '')
            last_name = profile.get('lastName', '')
            full_name = f"{first_name} {last_name}".strip()

            if not full_name:
                full_name = f"Profile {profile.get('_id', 'Unknown')}"

            area_of_expertise = profile.get('areaOfExpertise', 'Not specified')

            location = profile.get('currentLocation', {}) or {}
            if not isinstance(location, dict):
                location = {}
            location_str = f"{location.get('city', '')}, {location.get('state', '')}, {location.get('country', '')}".strip()
            if not location_str or location_str == ", , ":
                location_str = "Location not specified"

            summary = profile.get('carrierSummary', '') or profile.get('summary', '') or profile.get('bio', '') or 'No summary provided.'

            education_entries = profile.get('education', []) or []
            education_str = "\n".join([
                f"- {edu.get('degree', 'Degree not specified')} at {edu.get('institute', 'Institute not specified')} ({edu.get('startDate', 'Date not specified')})"
                for edu in education_entries if isinstance(edu, dict)
            ])

            experience_entries = profile.get('experience', []) or []
            experience_str_parts = []

            for exp in experience_entries:
                if not isinstance(exp, dict):
                    continue

                position = exp.get('position', 'Position not specified')
                company = exp.get('company', 'Company not specified')
                start_date = exp.get('startDate', 'Start date not specified')
                experience_entry = f"- {position} at {company} ({start_date})"
                experience_str_parts.append(experience_entry)

            experience_str = "\n".join(experience_str_parts)

            skills = profile.get('highlightedSkills', []) or []
            skill_names = []

            for skill in skills:
                if isinstance(skill, dict):
                    skill_name = skill.get('name', '')
                    if skill_name:
                        skill_names.append(skill_name)
                elif isinstance(skill, str):
                    skill_names.append(skill)

            skill_str = ", ".join(skill_names) if skill_names else "No skills listed"

            profile_text = f"""
            Name: {full_name}
            Expertise: {area_of_expertise}
            Location: {location_str}
            Summary: {summary}
    
            Education:
            {education_str if education_str else 'No education data'}
    
            Experience:
            {experience_str if experience_str else 'No experience data'}
    
            Highlighted Skills:
            {skill_str}
            """

            processed_docs.append({
                "page_content": profile_text,
                "metadata": {"id": str(profile.get('_id'))}
            })

        return processed_docs

    def generate_embeddings_and_create_index(self, documents):
        from langchain_core.documents import Document

        embeddings_list = []
        langchain_docs = []

        print(f"Generating embeddings for {len(documents)} documents")

        for i, doc in enumerate(documents):
            content = doc["page_content"]
            metadata = doc["metadata"]

            embedding = self.embeddings.embed_query(content)
            embeddings_list.append(embedding)

            langchain_doc = Document(
                page_content=content,
                metadata=metadata
            )
            langchain_docs.append(langchain_doc)

            if (i+1) % 10 == 0:
                print(f"Generated embeddings for {i+1}/{len(documents)} documents")

        embedding_dim = len(embeddings_list[0])
        embeddings_array = np.array(embeddings_list).astype('float32')

        index = faiss.IndexFlatL2(embedding_dim)
        index.add(embeddings_array)

        self.vector_store = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=InMemoryDocstore({i: doc for i, doc in enumerate(langchain_docs)}),
            index_to_docstore_id={i: i for i in range(len(langchain_docs))}
        )

        print(f"Created FAISS index with dimension {embedding_dim}")
        return self.vector_store

    def save_faiss_index(self, path="./faiss_index"):
        if not self.vector_store:
            print("No vector store to save.")
            return False

        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.vector_store.save_local(path)
        print(f"FAISS index saved to {path}")
        return True

    def load_faiss_index(self, path="./faiss_index"):
        if os.path.exists(path):
            self.vector_store = FAISS.load_local(
                path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"FAISS index loaded from {path}")
            return self.vector_store
        else:
            print(f"No FAISS index found at {path}")
            return None

    def perform_similarity_search(self, query, k=5):
        if not self.vector_store:
            print("Vector store not initialized. Please create or load a vector store first.")
            return None

        results = self.vector_store.similarity_search_with_score(query, k=k)
        return results

    def format_search_results(self, results):
        formatted_results = []

        for i, (doc, score) in enumerate(results):
            formatted_results.append(f"\n--- Result {i+1} (Similarity Score: {score:.4f}) ---")
            formatted_results.append(doc.page_content)

        return "\n".join(formatted_results)


def main():
    search_engine = MongoFAISSSearchEngine()

    faiss_index_path = "./faiss_index"
    loaded = search_engine.load_faiss_index(faiss_index_path)

    if not loaded:
        print("Creating new index from MongoDB data...")

        docs = search_engine.extract_documents_from_mongodb(db_name="app-dev", collection_name="profiles")

        processed_docs = search_engine.process_documents(docs)

        search_engine.generate_embeddings_and_create_index(processed_docs)

        search_engine.save_faiss_index(faiss_index_path)

    print("\n=== Profile Semantic Search ===")
    print("Type 'quit' to exit")
    while True:
        query = input("\nEnter your search query: ")
        if query.lower() == 'quit':
            break

        results = search_engine.perform_similarity_search(query, k=3)
        if results:
            formatted_results = search_engine.format_search_results(results)
            print(formatted_results)
        else:
            print("No results found.")


if __name__ == "__main__":
    main()