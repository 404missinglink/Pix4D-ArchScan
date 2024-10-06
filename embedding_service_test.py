import os
from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.core import Settings, VectorStoreIndex, Document, StorageContext, load_index_from_storage

from config import config as cf  # Ensure this config has your API keys

def embed_pixtral_texts(pixtral_folder_path, persist_dir):
    """
    Embeds the text content from the pixtral_response text files and creates a vector store index.
    """
    # Initialize LLM and Embedding Model from config
    llm = MistralAI(api_key=cf.API_KEY, model=cf.TEXT_MODEL)  # Adjust model if necessary
    embed_model = MistralAIEmbedding(model_name="mistral-embed", api_key=cf.API_KEY)
    
    Settings.llm = llm
    Settings.embed_model = embed_model

    documents = []

    # Iterate over all text files in the pixtral_response folder
    for filename in sorted(os.listdir(pixtral_folder_path)):
        if filename.endswith('.txt') and filename.startswith('frame_'):
            file_path = os.path.join(pixtral_folder_path, filename)
            with open(file_path, 'r') as file:
                text_content = file.read().strip()
            
            # Extract timestamp from filename (e.g., frame_1.00.txt -> 1.00)
            timestamp = filename.replace('frame_', '').replace('.txt', '')
            
            # Create a Document with metadata
            doc = Document(
                text=text_content,
                metadata={"timestamp": timestamp}
            )
            documents.append(doc)

    # Create vector store index from documents
    index = VectorStoreIndex.from_documents(documents)
    
    # Create a storage context and persist the index to disk
    storage_context = index.storage_context
    storage_context.persist(persist_dir=persist_dir)

    print(f"Vector store index created and persisted to {persist_dir}")

    return index


def query_pixtral_index(persist_dir, query, similarity_top_k=2):
    """
    Queries the vector store index with the given query.
    """
    # Rebuild the storage context from the persisted directory
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    
    # Load the index from the storage context
    index = load_index_from_storage(storage_context)

    # Create a query engine
    query_engine = index.as_query_engine(similarity_top_k=similarity_top_k)

    # Execute the query
    response = query_engine.query(query)

    return str(response)


if __name__ == "__main__":
    # Define paths
    pixtral_folder = "data/rkfFCSbWDyY/pixtral_response"  # Adjust this to your correct folder path
    persist_dir = "data/index"  # This should be a folder where the index is saved
    
    # Step 1: Embed the pixtral text files and persist the index
    print("Embedding text files...")
    embed_pixtral_texts(pixtral_folder, persist_dir)
    
    # Step 2: Query the index
    query = "How many frames are there along with the timestamp."  # Replace this with the query you want to test
    print(f"Querying with: '{query}'")
    response = query_pixtral_index(persist_dir, query)
    
    print(f"Query response: {response}")

