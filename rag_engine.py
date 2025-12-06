import os
import time
from pypdf import PdfReader
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

load_dotenv()


# ------------------------------
# Extract text from PDF files
# ------------------------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    return text


# ------------------------------
# Split extracted text into chunks
# ------------------------------
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    return chunks


# ------------------------------
# Create vector DB and store embeddings
# ------------------------------
def get_vector_store(text_chunks, persist_directory=None):
    api_key = os.getenv("GOOGLE_API_KEY")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key
    )

    if persist_directory:
        # Permanent Store
        vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
    else:
        # Ephemeral Store (In-memory)
        vector_store = Chroma(
            embedding_function=embeddings
        )

    batch_size = 10
    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i + batch_size]
        try:
            vector_store.add_texts(batch)
            print(f"Processed {i+1}/{len(text_chunks)} chunks")
            time.sleep(1)
        except Exception as e:
            print(f"Error: {e}. Retrying...")
            time.sleep(30)
            try:
                vector_store.add_texts(batch)
            except Exception as retry_e:
                print(f"Retry failed: {retry_e}")

    return vector_store


# ------------------------------
# Load existing Chroma DB
# ------------------------------
def load_vector_store():
    api_key = os.getenv("GOOGLE_API_KEY")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key
    )

    if os.path.exists("chroma_db_final"):
        return Chroma(
            persist_directory="chroma_db_final",
            embedding_function=embeddings
        )
    return None


# ------------------------------
# LLM + Prompt + QA Chain
# ------------------------------
def get_conversational_chain():
    prompt_template = """
    You are an intelligent assistant. Your task is to answer the question using the provided context.
    
    CRITICAL INSTRUCTION: The context might contain two types of data:
    1. **Messy Tables** (e.g., Engineering Cutoff Ranks with dashes like "-- --"):
       - You MUST format this into a CLEAN Markdown Table.
       - Ignore garbage characters.
       - Extract relevant columns (e.g., College, Branch, Rank).
    2. **Standard Text** (e.g., Project Reports, Paragraphs):
       - Answer normal questions based on this text.
       - Do NOT try to force this into a table if it doesn't make sense.
    
    If the answer is not found in the context, reply: "answer is not available in the context".

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    api_key = os.getenv("GOOGLE_API_KEY")

    # Updated Model Here 👇👇👇
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",  # or "gemini-2.0-pro"
        temperature=0.3,
        google_api_key=api_key,
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    qa_chain = load_qa_chain(
        llm,
        chain_type="stuff",
        prompt=prompt
    )

    return qa_chain


# ------------------------------
# Final user query processing
# ------------------------------
def user_input(user_question, vector_store):
    print(f"DEBUG: Querying '{user_question}'")
    
    # Retry mechanism for API timeouts (504 errors)
    max_retries = 3
    retry_delay = 2

    docs = []
    for attempt in range(max_retries):
        try:
            docs = vector_store.similarity_search(user_question)
            break
        except Exception as e:
            print(f"DEBUG: similarity_search attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
            else:
                # If all retries fail, re-raise the last exception
                raise e

    print(f"DEBUG: Found {len(docs)} documents")
    if docs:
        print(f"DEBUG: Top Doc: {docs[0].page_content[:200]}...")
    
    chain = get_conversational_chain()

    # Retry mechanism for LLM generation
    response = None
    for attempt in range(max_retries):
        try:
            response = chain(
                {"input_documents": docs, "question": user_question},
                return_only_outputs=True
            )
            break
        except Exception as e:
            print(f"DEBUG: chain invocation attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
            else:
                raise e

    return response["output_text"], docs
