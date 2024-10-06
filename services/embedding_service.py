# services/embedding_service.py

import os
import llama_index
from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader, Document
from config import config as cf

def embed_frame_texts(frames_folder_path, index_save_path):
    """
    Embeds the text content from frame text files and creates a vector store index.

    Parameters:
    frames_folder_path (str): Path to the folder containing frame text files.
    index_save_path (str): Path where the vector store index will be saved.

    Returns:
    VectorStoreIndex: The created vector store index.
    """
    # Initialize LLM and Embedding Model from config
    llm = MistralAI(api_key=cf.API_KEY, model=cf.TEXT_MODEL)  # Adjust model if necessary
    embed_model = MistralAIEmbedding(model_name="mistral-embed", api_key=cf.API_KEY)
    
    Settings.llm = llm
    Settings.embed_model = embed_model

    documents = []

    # Iterate over all text files in the frames folder
    for filename in sorted(os.listdir(frames_folder_path)):
        if filename.endswith('.txt') and filename.startswith('frame_'):
            file_path = os.path.join(frames_folder_path, filename)
            with open(file_path, 'r') as file:
                text_content = file.read().strip()
            
            # Extract timestamp from filename (e.g., frame_12.34.txt -> 12.34)
            timestamp = filename.replace('frame_', '').replace('.txt', '')
            
            # Create a Document with metadata
            doc = Document(
                text=text_content,
                metadata={"timestamp": timestamp}
            )
            documents.append(doc)

    # Create vector store index from documents
    index = VectorStoreIndex.from_documents(documents)

    # Save the index to disk
    index.save_to_disk(index_save_path)

    print(f"Vector store index created and saved to {index_save_path}")

    return index

def query_index(index_path, query, similarity_top_k=2):
    """
    Queries the vector store index with the given query.

    Parameters:
    index_path (str): Path to the saved vector store index.
    query (str): The query string.
    similarity_top_k (int): Number of top similar documents to retrieve.

    Returns:
    str: The response from the query.
    """
    # Load the index from disk
    index = VectorStoreIndex.load_from_disk(index_path)

    # Create a query engine
    query_engine = index.as_query_engine(similarity_top_k=similarity_top_k)

    # Execute the query
    response = query_engine.query(query)

    return str(response)
