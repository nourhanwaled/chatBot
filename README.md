# ChatBot Project

This project is a chatbot application that utilizes the Google Gemini API with advanced features for efficient question answering. Key components include:

- Semantic Caching: The application implements semantic caching to store and retrieve previous question-answer pairs, reducing API calls and improving response times for similar questions.
- Qdrant Vector Database: Uses Qdrant as a vector database for storing and retrieving document embeddings, enabling efficient similarity search and context retrieval.
- Google Gemini API: Powers the core language model capabilities for generating accurate and contextual responses.
- Docker Containerization: The entire application is containerized using Docker, making it easy to deploy and run on any system with Docker installed. Includes separate containers for the main application and Qdrant database.

## Prerequisites

- Docker: Ensure you have Docker installed on your system. You can download it from [Docker's official website](https://www.docker.com/get-started).
- Docker Compose: This is typically included with Docker Desktop, but you can also install it separately if needed.

## Setup and Launch

Follow these steps to set up and launch the chatbot application:

1. **Configure the Google Gemini API:**
   - Add your Google Gemini API credentials to the `docker-compose.yml` file. This will allow the application to authenticate and interact with the API.

2. **Build and Run the Application:**
   - Open a terminal and navigate to the project directory.
   - Run the following command to build and start the application:
     ```bash
     docker-compose up --build
     ```

3. **Access the Application:**
   - Once the application is running, open your web browser and go to `http://localhost:8501` to interact with the chatbot.



