import os
from rag_engine import get_pdf_text, get_text_chunks, get_vector_store
from dotenv import load_dotenv

load_dotenv()

def build_database():
    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"Directory '{data_dir}' not found.")
        return

    pdf_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pdf")]
    
    if not pdf_files:
        print("No PDF files found in 'data/' directory.")
        return

    print(f"Found {len(pdf_files)} PDF(s). Starting processing...")
    
    # Process files
    raw_text = get_pdf_text(pdf_files)
    print("Text extracted. Splitting into chunks...")
    
    text_chunks = get_text_chunks(raw_text)
    print(f"Created {len(text_chunks)} text chunks. Generating embeddings (this may take a while)...")
    
    # Create and persist vector store
    get_vector_store(text_chunks, persist_directory="chroma_db_final")
    print("Success! Vector database built and saved to 'chroma_db_final/'.")

if __name__ == "__main__":
    build_database()
