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
    
    # Cache the loading of the vector store to improve performance
    @st.cache_resource
    def load_cached_vector_store():
        return load_vector_store()

    # Try to load existing vector store if not already loaded
    if "vector_store" not in st.session_state:
        persisted_store = load_cached_vector_store()
        if persisted_store:
            st.session_state.vector_store = persisted_store
            # st.success("Loaded existing knowledge base from disk.")
        else:
            st.error("Vector Store not found. Please run `python build_db.py` to build the database first.")


    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        if "vector_store" in st.session_state:
            response, docs = user_input(user_question, st.session_state.vector_store)
            st.markdown("### Reply:")
            st.markdown(response)
            
            with st.expander("View Source Documents"):
                for i, doc in enumerate(docs):
                    st.markdown(f"**Source {i+1}**")
                    st.write(doc.page_content)
                    st.divider()
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
                with st.spinner("Processing & Saving Permanently..."):
                    # 1. Save uploaded files to disk
                    if not os.path.exists("data"):
                        os.makedirs("data")
                    
                    saved_files = []
                    # Handle static files (already path strings)
                    for f in static_files:
                        saved_files.append(f)

                    # Handle uploaded files (Streamlit UploadedFile objects)
                    if pdf_docs:
                        for uploaded_file in pdf_docs:
                            save_path = os.path.join("data", uploaded_file.name)
                            with open(save_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            saved_files.append(save_path)

                    # 2. Process all files (both old and new)
                    raw_text = get_pdf_text(saved_files)
                    text_chunks = get_text_chunks(raw_text)
                    
                    # 3. Save to Permanent DB
                    vector_store = get_vector_store(text_chunks, persist_directory="chroma_db_final")
                    
                    st.session_state.vector_store = vector_store
                    st.success("Done! Files saved to 'data/' and added to Permanent Database.")
            else:
                st.warning("Please upload at least one PDF file or add files to the 'data/' folder.")

if __name__ == "__main__":
    main()
