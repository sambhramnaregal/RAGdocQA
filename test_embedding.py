import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import time

load_dotenv()

def test():
    api_key = os.getenv("GOOGLE_API_KEY")
    print(f"API Key found: {bool(api_key)}")
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
    
    try:
        print("Attempting to embed 'Hello World'...")
        res = embeddings.embed_query("Hello World")
        print(f"Success! Embedding length: {len(res)}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test()
