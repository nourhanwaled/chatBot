import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
import time


COST_PER_1000_TOKENS_GEMINI_BASE = 0.01
COST_PER_1000_TOKENS_GEMINI_PRO = 0.05

genai.configure(api_key="GOOGLE_API_KEY")


# Extract text from PDF
def get_pdf_text(pdf_docs):
    """Extracts text from the uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Split text into manageable chunks
def get_text_chunks(text):
    """Splits extracted text into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a FAISS vector store
def create_vector_store(text_chunks, embeddings):
    """Creates a FAISS vector store from text chunks."""
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# Function to load the vector store (with deserialization safety)
def load_vector_store(embeddings):
    """Loads the existing FAISS vector store."""
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return vector_store

def get_relevant_context(query, vector_store):
    """Retrieve the most relevant context chunk for the query."""
    results = vector_store.similarity_search(query, k=3) 
    context = "\n".join([result.page_content for result in results])
    return context




# Test performance
def estimate_tokens(text):
    # A rough estimate: 1 token is roughly equivalent to 4 characters (for English text)
    return len(text) // 4


def calculate_cost(model, input_text, output_text):
    input_tokens = estimate_tokens(input_text)
    output_tokens = estimate_tokens(output_text)
    total_tokens = input_tokens + output_tokens
    
    if model == "gemini-base":
        cost = (total_tokens / 1000) * COST_PER_1000_TOKENS_GEMINI_BASE
    elif model == "gemini-pro":
        cost = (total_tokens / 1000) * COST_PER_1000_TOKENS_GEMINI_PRO
    else:
        cost = 0
    
    return cost, total_tokens


# Function to handle user input and generate answers
def user_input(user_question, embeddings):
    """Handles the user's input, decides whether to use cached or fresh response."""
    start_time = time.time()

    # Load the vector store
    vector_store = load_vector_store(embeddings)
    
    # Retrieve the relevant context for the query
    context = get_relevant_context(user_question, vector_store)
    
    # If context is not empty, pass it into the prompt
    if context:
        prompt_template = f"""
        Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in
        the provided context just say, "answer is not available in the context". Don't provide a wrong answer.

        Context: {context}
        Question: {user_question}

        Answer:
        """
        model = ChatGoogleGenerativeAI(model="gemini-pro")
        response = model.predict(prompt_template)
    else:
        response = "Answer is not available in the context."

    # Cache the response in session state
    if 'question_cache' not in st.session_state:
        st.session_state['question_cache'] = []
    
    # Check if the current question already exists in cache
    question_lower = user_question.lower()
    cached_answer = None
    for cached in st.session_state['question_cache']:
        if cached['question'].lower() == question_lower:
            cached_answer = cached['answer']
            break
    
    if cached_answer:
        # If a cached answer is found, show it instead of making a new API call
        st.write(f"Cached answer: {cached_answer}")
    else:
        # If no cached answer, save the new question-answer pair in the cache
        st.session_state['question_cache'].append({"question": user_question, "answer": response})
        st.write("Reply (from Gemini Pro): ", response)
    
    # Print the time and response
    end_time = time.time()
    elapsed_time = end_time - start_time
    cost, total_tokens = calculate_cost("gemini-pro", user_question, response)

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
                    # Extract text and create vector store
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                    create_vector_store(text_chunks, embeddings)
                    st.session_state["vector_store_created"] = True
                    st.success("PDFs processed and vector store created!")
            else:
                st.warning("Please upload PDF files before processing.")

    # User question input
    user_question = st.text_input("Ask a question about the uploaded PDFs")

    # Handle user questions
    if user_question and st.session_state["vector_store_created"]:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        user_input(user_question, embeddings)
    elif user_question:
        st.warning("Please process the PDFs first before asking questions.")


if __name__ == "__main__":
    main()
