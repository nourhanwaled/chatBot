import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
import time

# Import utility classes
from modules.utils.doc_utils import DocUtils
from modules.vector_database.faiss_vb import FAISSVectorDB
from modules.vector_database.qdrant_vb import QdrantVectorDB
from modules.embedding_model.embedding_model import EmbeddingModel
from modules.cache.semantic_cache import SemanticCache
from modules.cost_calculator import CostCalculator
from credentials import API_KEY
genai.configure(api_key=API_KEY)


vdb_utils = QdrantVectorDB(collection_name="doc_vb", host="192.168.1.110")
semantic_cache = SemanticCache(EmbeddingModel().get_embeddings(), threshold=0.75)
vector_store = vdb_utils.load_vector_store(EmbeddingModel().get_embeddings())
# Function to handle user input and generate answers
def user_input(user_question, embeddings):
    """Handles the user's input, decides whether to use cached or fresh response."""
    start_time = time.time()
    response = semantic_cache.ask(user_question)
    if response:
        st.write(f"Cached answer: {response}")
    else:
        
        # Get relevant context using VectorDatabaseUtils
        context = vdb_utils.get_relevant_context(user_question, vector_store)
    
        # print("Context: ", context)
        # If context is not empty, pass it into the prompt
        if context and context != []:
            prompt_template = """You are a helpful and knowledgeable AI assistant. Your task is to provide accurate and comprehensive answers to questions while distinguishing whether the response is based on the provided context or your general knowledge.

                Important Instructions:
                1. If the answer is based on the provided context, clearly indicate this by starting with:
                - In English: "Based on the provided context..."
                - In Arabic: "على حسب المصادر ..."
                2. If the answer is not grounded in the provided context and relies on general knowledge, start by saying:
                - In English: "This answer is based on general knowledge..."
                - In Arabic: "الإجابة دي جاية من معلومات عامة..."
                3. Always respond in the same language as the question:
                - If the question is in English, answer entirely in English.
                - If the question is in Arabic, answer entirely in Egyptian Arabic.
                4. Be thorough and ensure clarity in your response.


            Context: {context}
            Question: {user_question}
            
            Answer: """
        else:
            prompt_template = """You are a helpful and knowledgeable AI assistant. Your task is to provide accurate and comprehensive answers to questions while distinguishing whether the response is based on the provided context or your general knowledge.

                Important Instructions:
                1. If the answer is based on the provided context, clearly indicate this by starting with:
                - In English: "Based on the provided context..."
                - In Arabic: "على حسب المصادر ..."
                2. If the answer is not grounded in the provided context and relies on general knowledge, start by saying:
                - In English: "This answer is based on general knowledge..."
                - In Arabic: "الإجابة دي جاية من معلومات عامة..."
                3. Always respond in the same language as the question:
                - If the question is in English, answer entirely in English.
                - If the question is in Arabic, answer entirely in Egyptian Arabic.
                4. Be thorough and ensure clarity in your response.


            Question: {user_question}
            
            Answer: """
        
        model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=API_KEY)
        response = model.predict(prompt_template.format(
            context=context,
            user_question=user_question
        ))
        # print("Response: ", response)
        semantic_cache.add_to_cache(user_question, response)
        st.write("Reply (from Gemini Pro): ", response)

    end_time = time.time()
    elapsed_time = end_time - start_time
    cost, total_tokens = CostCalculator.calculate_cost("gemini-pro", user_question, response)

    st.write(f"Time taken: {elapsed_time:.2f} seconds")
    st.write(f"Total tokens: {total_tokens}")
    st.write(f"Estimated cost: ${cost:.4f}")


# Main Streamlit app
def main():
    st.set_page_config(page_title="Chat with PDF")
    st.header("Chat with PDF using Gemini 💬")

    # Initialize session state
    if "vector_store_created" not in st.session_state:
        st.session_state["vector_store_created"] = False

    # File uploader for PDF files
    with st.sidebar:
        st.title("Upload PDF")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)

        if st.button("Process PDFs"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    # Use utility classes for processing
                    raw_text = DocUtils.get_pdf_text(pdf_docs)
                    text_chunks = DocUtils.get_text_chunks(raw_text)
                    embeddings = EmbeddingModel().get_embeddings()
                    vdb_utils.create_vector_store(text_chunks, embeddings)
                    st.session_state["vector_store_created"] = True
                    st.success("PDFs processed and vector store created!")
            else:
                st.warning("Please upload PDF files before processing.")

    # User question input
    user_question = st.text_input("Ask a question about the uploaded PDFs")

    # Handle user questions
    if user_question and st.session_state["vector_store_created"]:
        embeddings = EmbeddingModel().get_embeddings()
        user_input(user_question, embeddings)
    elif user_question:
        st.warning("Please process the PDFs first before asking questions.")


if __name__ == "__main__":
    main()
