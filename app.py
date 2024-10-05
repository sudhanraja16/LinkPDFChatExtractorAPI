from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
import os
import uuid
import fitz  # PyMuPDF
import json
from pydantic import BaseModel
from fake_useragent import UserAgent
import requests
from bs4 import BeautifulSoup

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.preprocessing import normalize  # To normalize vectors
import numpy as np
import faiss

# Initialize the FastAPI application
app = FastAPI()

# Model configuration for embedding generation
model_name = "./bge-small-en"  # Path to the pre-trained model
model_kwargs = {"device": "cpu"}  # Specify the device (CPU or GPU)
encode_kwargs = {"normalize_embeddings": True}  # Enable normalization of embeddings

# Load the Hugging Face model for embeddings
hf = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

def remove_unwanted_char(text_string):
    """
    Cleans up the given text string by removing unwanted characters.

    Args:
    text_string (str): The string to be cleaned.

    Returns:
    str: The cleaned string with unwanted characters removed.
    """
    # Replace unwanted characters with spaces or nothing
    return text_string.replace('\n', ' ').replace('\u200c', '').replace('    ', '').replace('\t', '').replace('\xa0', '').strip()

# Path for the JSON file to store chat data
file_path = "./testing.json"

def append_to_json(new_data):
    """
    Appends new data to a JSON file.

    Args:
    new_data (dict): The new data to append.
    """
    # Try to read existing data from the JSON file; initialize empty dict if not found
    try:
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}

    # Append the new data to the existing data
    data.update(new_data)

    # Write the updated data back to the JSON file
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def pdf_extractor(pdf_file_path):
    """
    Extracts text from a PDF file and stores it in JSON format.

    Args:
    pdf_file_path (str): The path of the PDF file to extract text from.

    Returns:
    dict: A dictionary containing the chat ID and extracted text.
    """
    storage = dict()  # Dictionary to store extracted data
    chat_id = str(uuid.uuid4())  # Generate a unique chat ID
    text_content = ""

    # Open the PDF and extract text from each page
    with fitz.open(pdf_file_path) as pdf_doc:
        for page in pdf_doc:
            text_content += page.get_text()
            print(text_content)  # Print extracted text for debugging

    # Clean and store the extracted text
    storage[chat_id] = remove_unwanted_char(text_content)
    append_to_json(storage)  # Append the new data to the JSON file
    
    return storage  # Return the storage for feedback

def url_extractor(url):
    """
    Extracts text from a given URL and stores it in JSON format.

    Args:
    url (str): The URL to extract text from.

    Returns:
    dict: A dictionary containing the chat ID and extracted text.
    """
    user_agent = UserAgent()  # Create a user agent for requests
    random_agent = user_agent.random  # Get a random user agent
    headers = {
        'User-Agent': random_agent  # Set headers for the request
    }

    cleaned_text = ""
    chat_id = str(uuid.uuid4())  # Generate a unique chat ID
    response = requests.get(url, headers=headers)  # Fetch the URL content

    # Check if the request was successful
    if response.status_code == 200:
        content = BeautifulSoup(response.content, 'html.parser')  # Parse the HTML content
        cleaned_text = remove_unwanted_char(content.get_text())  # Clean the extracted text
    else:
        # Raise an HTTP exception if the request failed
        raise HTTPException(status_code=response.status_code, detail=f"Failed to fetch URL content: {response.reason}")

    storage = {chat_id: cleaned_text}  # Store the cleaned text
    append_to_json(storage)  # Append the new data to the JSON file
    return storage  # Return the storage for feedback

def create_faiss_index(cleaned_text, chunk_size=100, chunk_overlap=100):
    """
    Creates a FAISS index for the cleaned text to enable efficient similarity search.

    Args:
    cleaned_text (str): The text to index.
    chunk_size (int): The size of text chunks to create.
    chunk_overlap (int): The overlap between chunks.

    Returns:
    tuple: A tuple containing the FAISS index and the list of text chunks.
    """
    # Split the cleaned text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(cleaned_text)  # Split the text

    # Generate embeddings for the chunks
    document_embeddings = hf.embed_documents(chunks)

    # Normalize the embeddings to unit length
    document_embeddings = normalize(document_embeddings, axis=1, norm='l2')

    # Convert embeddings to a NumPy array
    document_embeddings = np.array(document_embeddings)

    # Create a FAISS index for efficient inner product search
    dimension = document_embeddings.shape[1]  # Get the number of dimensions
    index = faiss.IndexFlatIP(dimension)  # Create the FAISS index

    # Add the normalized document embeddings to the index
    index.add(document_embeddings)
    
    return index, chunks  # Return the index and chunks

def get_input_query(text, query, top_k=2):
    """
    Searches for the most relevant chunks in the indexed text for the given query.

    Args:
    text (str): The text to search within.
    query (str): The search query.
    top_k (int): The number of top results to return.

    Returns:
    list: A list of the most relevant chunks and their similarity scores.
    """
    index, chunks = create_faiss_index(cleaned_text=text)  # Create FAISS index from the text
    # Generate and normalize the embedding for the query
    query_embedding = hf.embed_query(query)
    query_embedding = normalize(np.array(query_embedding).reshape(1, -1), axis=1, norm='l2')

    # Perform similarity search using the FAISS index
    distances, indices = index.search(query_embedding, top_k)

    # Collect the most similar documents and their details
    results = []
    for i in range(top_k):
        results.append({
            "chunk_index": indices[0][i],  # Index of the chunk
            "similarity_score": distances[0][i],  # Similarity score
            "chunk_text": chunks[indices[0][i]]  # The text chunk
        })
    
    return results  # Return the results

def chat(chat_id, query):
    """
    Handles the chat functionality by retrieving relevant text for a given chat ID and query.

    Args:
    chat_id (str): The ID of the chat to retrieve data for.
    query (str): The user query to search within the chat data.

    Returns:
    str: The most relevant chunk of text for the given query.
    """
    try:
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)  # Load existing chat data
        
        if chat_id in data:
            actual_text = data[chat_id]  # Retrieve the actual text for the chat ID
            results = get_input_query(actual_text, query)  # Get relevant results
            return results[0]['chunk_text']  # Return the top result
        else:
            raise KeyError("Chat ID not found.")  # Raise error if chat ID is not found
            
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))  # Handle not found error
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")  # Handle general errors

# Define request model for the chat endpoint
class ChatRequest(BaseModel):
    chat_id: str  # Chat ID to retrieve data for
    query: str  # User query to search within the chat data

# Endpoint to handle chat interaction
@app.post('/chat')
async def chat_interaction(request: ChatRequest):
    """
    Endpoint for processing chat interactions.

    Args:
    request (ChatRequest): The request containing chat ID and query.

    Returns:
    dict: The response containing the relevant text chunk.
    """
    try:
        result = chat(request.chat_id, request.query)  # Call the chat function
        return {
            "response": result  # Return the response
        }
    except HTTPException as e:
        raise e  # Raise known HTTP exceptions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))  # Handle unexpected errors

# Define request model for the URL processing endpoint
class URLRequest(BaseModel):
    url: str  # URL to be processed

# Endpoint to extract text from a URL
@app.post('/process_url')
async def url_scrape(request: URLRequest):
    """
    Endpoint for processing text extraction from a URL.

    Args:
    request (URLRequest): The request containing the URL.

    Returns:
    dict: The response containing chat ID and success message.
    """
    try:
        result = url_extractor(request.url)  # Extract text from the URL
        chat_id = list(result.keys())[0]  # Extract chat ID from the returned dictionary
        return {
            "chat_id": chat_id,  # Return the chat ID
            "message": "URL content processed and stored successfully."  # Success message
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))  # Handle unexpected errors

# Endpoint to upload and process a PDF file
@app.post('/process_pdf')
async def upload_file(file: UploadFile = File(...)):
    """
    Endpoint for uploading a PDF file and extracting its text.

    Args:
    file (UploadFile): The PDF file to be uploaded.

    Returns:
    dict: The response containing chat ID and success message.
    """
    try:
        if file is None:
            raise HTTPException(status_code=400, detail="No file uploaded")  # Handle no file error

        pdf_path = f"uploaded_pdfs/{file.filename}"  # Define the path to save the uploaded PDF
        os.makedirs(os.path.dirname(pdf_path), exist_ok=True)  # Create directory if it doesn't exist

        # Write the uploaded PDF to the file system
        with open(pdf_path, "wb") as pdf_file:
            pdf_file.write(await file.read())

        result = pdf_extractor(pdf_path)  # Extract text from the uploaded PDF
        os.remove(pdf_path)  # Remove the PDF after processing
        chat_id = list(result.keys())[0]  # Extract chat ID from the returned dictionary
        return {
            "chat_id": chat_id,  # Return the chat ID
            "message": "PDF content processed and stored successfully."  # Success message
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))  # Handle unexpected errors

# Run the FastAPI application
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)  # Start the server
