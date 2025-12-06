from rag_engine import load_vector_store
import os
from dotenv import load_dotenv

load_dotenv()

def debug_search():
    print("Loading vector store...")
    vector_store = load_vector_store()
    
    if not vector_store:
        print("Error: Could not load vector store from 'chroma_db_new'.")
        return

    query = "suggest me best college for engineering for my rank of 26000 in cse stream"
    print(f"\nQuery: {query}")
    
    print("\n--- Performing Similarity Search ---")
    # Retrieve top 5 docs to see what's being found
    docs = vector_store.similarity_search(query, k=5)
    
    for i, doc in enumerate(docs):
        print(f"\n[Document {i+1}]")
        # Print a snippet of the content to verify relevance
        content_snippet = doc.page_content[:500].replace('\n', ' ')
        print(f"Content: {content_snippet}...")
        print("-" * 50)

if __name__ == "__main__":
    debug_search()
