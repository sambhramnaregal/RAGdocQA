import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

import time

def get_vector_store(text_chunks):
    api_key = os.getenv("GOOGLE_API_KEY")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    
    # Initialize Chroma with persistence
    vector_store = Chroma(embedding_function=embeddings, persist_directory="chroma_db")
    
    # Add texts in batches to avoid hitting rate limits
    batch_size = 5  # Process 5 chunks at a time
    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i + batch_size]
        try:
            vector_store.add_texts(batch)
            print(f"Processed batch {i//batch_size + 1}/{(len(text_chunks) + batch_size - 1)//batch_size}")
            time.sleep(2)  # Sleep for 2 seconds between batches
        except Exception as e:
            print(f"Error processing batch: {e}")
            # Optional: Wait longer and retry once
            time.sleep(10)
            try:
                vector_store.add_texts(batch)
            except Exception as retry_e:
                print(f"Retry failed: {retry_e}")

    return vector_store

def load_vector_store():
    api_key = os.getenv("GOOGLE_API_KEY")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    # Load from 'chroma_db' directory
    if os.path.exists("chroma_db"):
        vector_store = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
        return vector_store
    return None

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, vector_store):
    api_key = os.getenv("GOOGLE_API_KEY")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    # In a real app with persistent Chroma, we would load it here. 
    # For this simple implementation, we pass the vector_store object or assume it's in memory/session state.
    # However, Chroma.from_texts returns a vector store. 
    # If we want to search, we use that instance.
    
    docs = vector_store.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question}
        , return_only_outputs=True)
    return response["output_text"]
