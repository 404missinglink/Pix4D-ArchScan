import os
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
from unstructured.partition.text import partition_text
import faiss
import numpy as np
from flask import Flask, render_template, request, jsonify

# Check if GPU is available
device = 0 if torch.cuda.is_available() else -1  # Use GPU if available, otherwise CPU

# Load a pre-trained BERT model for embeddings
bert_model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
bert_model = AutoModel.from_pretrained(bert_model_name).to(device)

# Load a summarization pipeline and use GPU if available
summarizer = pipeline("summarization", device=device)

# Flask app initialization
app = Flask(__name__)

# Function to generate BERT embeddings from text
def embed_text_with_bert(text, max_length=512):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to GPU if available
    with torch.no_grad():
        outputs = bert_model(**inputs)
        # Use the CLS token embedding as the sentence embedding
        embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token is at index 0
    return embeddings.cpu().numpy()  # Move embeddings back to CPU if necessary

# Function to process .txt files in a folder and return embeddings and file names
def process_txt_files_and_embed(folder_path):
    embeddings_list = []
    file_names = []

    if not os.path.exists(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return embeddings_list, file_names

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                elements = partition_text(filename=file_path)
                text = " ".join(str(element) for element in elements)

                if text.strip():  # Ensure text is not empty
                    embeddings = embed_text_with_bert(text)
                    embeddings_list.append(embeddings)
                    file_names.append(file_path)

    return embeddings_list, file_names

# Function to create a FAISS index from embeddings
def create_faiss_index(embeddings):
    if not embeddings:
        return None

    d = embeddings[0].shape[1]  # Dimension of the embeddings
    index = faiss.IndexFlatL2(d)
    embeddings_np = np.vstack(embeddings)
    index.add(embeddings_np)

    return index

# Function to query FAISS index and retrieve the most relevant files
def query_faiss_index(query, index, file_names, k=5):
    if index is None:
        return "Error: FAISS index is not initialized."
    
    query_embedding = embed_text_with_bert(query)
    D, I = index.search(query_embedding, k)

    relevant_files = [file_names[i] for i in I[0]]
    return relevant_files

# Function to chunk text if it's too long for the model
def chunk_text(text, max_length=1000):
    """Splits the input text into smaller chunks"""
    words = text.split()
    for i in range(0, len(words), max_length):
        yield " ".join(words[i:i + max_length])

# Function to summarize or return content of files
def process_file_content(file_paths, summarize=False):
    all_text = ""
    
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            content = file.read()
            all_text += content + "\n"

    if all_text.strip():
        if summarize:
            # Chunk the text to avoid hitting model limits
            summaries = []
            for chunk in chunk_text(all_text, max_length=500):  # Adjust max_length as needed
                summary = summarizer(chunk, max_length=200, min_length=50, do_sample=False)
                summaries.append(summary[0]['summary_text'])
            return " ".join(summaries)
        else:
            return all_text  # Return raw content

    return "No valid content to display."

# Preprocessing step to build the FAISS index from .txt files
import os
cwd = os.getcwd()
folder_path = os.path.join(cwd, "data", "7Q6DU-AuZyI", "pixtral_response")

embeddings_list, file_names = process_txt_files_and_embed(folder_path)

faiss_index = None
if embeddings_list:
    faiss_index = create_faiss_index(embeddings_list)

# Flask route to render the chat interface
@app.route("/")
def home():
    return render_template("chat.html")

# Flask route to handle chat queries and return relevant responses
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")

    if user_input:
        if "summary" in user_input.lower():  # If the user asks for a summary
            relevant_files = query_faiss_index(user_input, faiss_index, file_names, k=5)
            if isinstance(relevant_files, str):  # If there's an error in querying FAISS
                response = relevant_files
            else:
                summary = process_file_content(relevant_files, summarize=True)
                response = f"Summary: {summary}"
        else:
            relevant_files = query_faiss_index(user_input, faiss_index, file_names, k=5)
            if isinstance(relevant_files, str):
                response = relevant_files
            else:
                content = process_file_content(relevant_files, summarize=False)
                response = f"File Content:\n{content}"
                
        return jsonify({"response": response})
    
    return jsonify({"response": "Please enter a query."})

if __name__ == "__main__":
    app.run(debug=True)
