import os
import openai
from llama_index import (
    ServiceContext, 
    VectorStoreIndex, 
    Document, 
    SimpleDirectoryReader
)
from langchain.agents import initialize_agent, AgentType
from langchain.agents import tool
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
import requests

# Load your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# 1. File Ingestion using LlamaIndex

# Function to ingest files into LlamaIndex
def ingest_files(directory: str):
    # Reading the files from a directory
    documents = SimpleDirectoryReader(directory).load_data()
    
    # Creating a service context for querying the documents
    service_context = ServiceContext.from_defaults()
    
    # Creating the index (vector store) for the documents
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    
    return index

# 2. Retrieval-Augmented Generation (RAG) using LlamaIndex

# Function to query the index with RAG
def retrieve_and_generate(query: str, index):
    # Use LlamaIndex's query method for RAG
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    return response

# 3. Arithmetic Operations with Function Calling

# Create a simple tool for basic arithmetic operations
@tool
def add(a: float, b: float) -> float:
    return a + b

@tool
def subtract(a: float, b: float) -> float:
    return a - b

@tool
def multiply(a: float, b: float) -> float:
    return a * b

@tool
def divide(a: float, b: float) -> float:
    if b == 0:
        return "Error: Division by zero."
    return a / b

# 4. Function Calling to Generate Notes or Summaries from Content

# Function to generate summary
def generate_summary(content: str) -> str:
    prompt = "Summarize the following text:\n" + content
    llm = OpenAI(temperature=0.5)
    chain = LLMChain(llm=llm, prompt=PromptTemplate(template=prompt, input_variables=[]))
    summary = chain.run("")
    return summary

# Function to generate notes
def generate_notes(content: str) -> str:
    prompt = "Generate notes from the following text:\n" + content
    llm = OpenAI(temperature=0.5)
    chain = LLMChain(llm=llm, prompt=PromptTemplate(template=prompt, input_variables=[]))
    notes = chain.run("")
    return notes

# 5. Web Search Integration

# Function to search the web and augment responses
def search_web(query: str) -> str:
    # This can be implemented using a web search tool like SerpAPI, GoogleSearch, or custom API
    # Placeholder response for demonstration
    api_key = "your_serpapi_key"  # Replace with your API key
    params = {
        "q": query,
        "api_key": api_key
    }
    response = requests.get("https://serpapi.com/search", params=params)
    search_results = response.json()
    
    # Process the search results and return relevant data
    return search_results.get("organic_results", [{}])[0].get("snippet", "No relevant information found.")

# 6. Putting Everything Together

def process_input(query: str, directory: str):
    # Ingest files
    index = ingest_files(directory)

    # RAG response
    rag_response = retrieve_and_generate(query, index)

    # Generate a summary of the query (just an example)
    summary = generate_summary(rag_response)

    # Return both RAG response and summary
    return {
        "rag_response": rag_response,
        "summary": summary
    }
