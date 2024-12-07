import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from sklearn.metrics.pairwise import cosine_similarity
import os
import time


COST_PER_1000_TOKENS_GEMINI_BASE = 0.01
COST_PER_1000_TOKENS_GEMINI_PRO = 0.05

genai.configure(api_key="AIzaSyCosxAPjrz73sWCDQKSbKvqITMwqcezYhQ")


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

def get_relevant_context(query, vector_store, k=5):
    """Retrieve the most relevant context chunks for the query."""
    results = vector_store.similarity_search(query, k=k) 
    context = "\n".join([result.page_content for result in results])
    return context

def estimate_tokens(text):
    """Estimate the number of tokens in text."""
    return len(text) // 4

def calculate_cost(model, input_text, output_text):
    """Calculate the cost of using the model based on tokens."""
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

def is_semantically_similar(new_question, cached_questions, embeddings, threshold=0.9):
    """Check if a new question is semantically similar to cached ones."""
    new_embedding = embeddings.embed_query(new_question)
    for cached in cached_questions:
        cached_embedding = embeddings.embed_query(cached['question'])
        similarity = cosine_similarity([new_embedding], [cached_embedding])[0][0]
        if similarity > threshold:
            return cached['answer']
    return None

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
        Answer the question based on the following context. If the information is not directly available, try to infer logically. If the answer is still unavailable, respond with "The answer is not available in the context."

        Context: {context}
        Question: {user_question}

        Answer:
        """
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        response = model.predict(prompt_template)
    else:
        response = "The answer is not available in the context."

    # Cache the response in session state
    if 'question_cache' not in st.session_state:
        st.session_state['question_cache'] = []
    
    # Check if the current question already exists in cache (semantic similarity)
    cached_answer = is_semantically_similar(user_question, st.session_state['question_cache'], embeddings)
    
    if cached_answer:
        # If a cached answer is found, show it instead of making a new API call
        st.write(f"Cached answer: {cached_answer}")
    else:
        # If no cached answer, save the new question-answer pair in the cache
        st.session_state['question_cache'].append({"question": user_question, "answer": response})
        st.write("Reply (from gemini-1.5-flash): ", response)
    
    # Print the time and response
    end_time = time.time()
    elapsed_time = end_time - start_time
    cost, total_tokens = calculate_cost("gemini-1.5-flash", user_question, response)

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
