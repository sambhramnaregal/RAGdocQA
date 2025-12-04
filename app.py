import streamlit as st
from rag_engine import get_pdf_text, get_text_chunks, get_vector_store, user_input, load_vector_store
from dotenv import load_dotenv
import os

load_dotenv()

def main():
    st.set_page_config("RAGdocQA")
    st.header("RAGdocQA 💁")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Try to load existing vector store if not already loaded
    if "vector_store" not in st.session_state:
        persisted_store = load_vector_store()
        if persisted_store:
            st.session_state.vector_store = persisted_store
            # st.success("Loaded existing knowledge base from disk.")
        else:
            # If no persisted store, check for data/ files and process them automatically
            data_dir = "data"
            static_files = []
            if os.path.exists(data_dir):
                for file in os.listdir(data_dir):
                    if file.endswith(".pdf"):
                        static_files.append(os.path.join(data_dir, file))
            
            if static_files:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(static_files)
                    text_chunks = get_text_chunks(raw_text)
                    vector_store = get_vector_store(text_chunks)
                    st.session_state.vector_store = vector_store
                    # st.success("Knowledge base initialized from static files.")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        if "vector_store" in st.session_state:
            response = user_input(user_question, st.session_state.vector_store)
            st.write("Reply: ", response)
            st.session_state.chat_history.append(("User", user_question))
            st.session_state.chat_history.append(("Bot", response))
        else:
            st.error("Knowledge base is empty. Please upload a PDF or add files to the 'data/' folder.")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        
        # Check for static files in data/ directory
        data_dir = "data"
        static_files = []
        if os.path.exists(data_dir):
            for file in os.listdir(data_dir):
                if file.endswith(".pdf"):
                    static_files.append(os.path.join(data_dir, file))
        
        # if static_files:
        #     st.info(f"Found {len(static_files)} static PDF(s) in 'data/' folder.")

        if st.button("Submit & Process"):
            all_files = []
            if pdf_docs:
                all_files.extend(pdf_docs)
            if static_files:
                all_files.extend(static_files)

            if all_files:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(all_files)
                    text_chunks = get_text_chunks(raw_text)
                    vector_store = get_vector_store(text_chunks)
                    st.session_state.vector_store = vector_store
                    st.success("Done")
            else:
                st.warning("Please upload at least one PDF file or add files to the 'data/' folder.")

if __name__ == "__main__":
    main()
