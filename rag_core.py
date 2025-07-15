import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import sys
import os
import fitz  # PyMuPDF
import numpy as np

# --- This file contains the core RAG and model logic. ---
# --- It is designed to be imported by the GUI. ---

def load_and_chunk_pdf(file_path):
    """
    Loads a PDF, extracts text, and splits it into smaller chunks.
    """
    if not file_path or not os.path.exists(file_path):
        return []
        
    doc = fitz.open(file_path)
    chunks = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        page_chunks = text.split('\n\n')
        chunks.extend([chunk.strip() for chunk in page_chunks if len(chunk.strip()) > 100])
    
    return chunks

def create_vector_store(chunks, model):
    """
    Creates embeddings for each text chunk.
    """
    if not chunks:
        return None
    embeddings = model.encode(chunks, convert_to_tensor=True, show_progress_bar=False)
    return embeddings

def retrieve_context(query, model, chunks, embeddings, top_k=3):
    """
    Retrieves the most relevant text chunks for a given query.
    """
    if embeddings is None or not chunks:
        return ""
    query_embedding = model.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, embeddings)[0]
    top_results = torch.topk(cos_scores, k=min(top_k, len(chunks)))
    context = "\n---\n".join([chunks[i] for i in top_results.indices])
    return context

class ChatbotModel:
    def __init__(self):
        """
        Initializes and loads all the required models.
        """
        self.pipe = None
        self.retriever_model = None
        
        # Load Retriever Model
        print("Loading sentence transformer model for RAG...")
        self.retriever_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Retriever model loaded.")

        # Load Generator Model (Llama 3.2)
        print("\n--- Llama 3.2 Chatbot Initializing ---")
        model_kwargs = {"torch_dtype": torch.bfloat16}
        
        if not torch.cuda.is_available():
            print("\nWARNING: CUDA not available. Using CPU.")
            num_cores = os.cpu_count()
            torch.set_num_threads(num_cores)
            model_kwargs = {}

        try:
            model_id = "MetaAI/Llama-3.2-3B-Instruct"
            self.pipe = pipeline(
                "text-generation",
                model=model_id,
                model_kwargs=model_kwargs,
                device_map="auto",
            )
            print("--- Llama 3.2 Model loaded successfully! ---")
        except Exception as e:
            print(f"Error loading Llama 3.2 model: {e}")
            self.pipe = None

    def generate_response(self, query, context):
        """
        Generates a response using the Llama 3.2 model with the given context.
        """
        if not self.pipe:
            return "The generator model is not available."

        rag_prompt = f"""
        System: You are an expert Q&A assistant. Use the following context from a document to answer the user's question.
        If the answer is not found in the context, state that you cannot find the answer in the provided document.
        Do not use any outside knowledge.

        Context:
        ---
        {context}
        ---

        User's Question: {query}
        """
        messages = [{"role": "system", "content": rag_prompt}]

        try:
            outputs = self.pipe(
                messages,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )
            assistant_response = outputs[0]["generated_text"][-1]
            if assistant_response.get("role") == "assistant":
                return assistant_response.get("content", "").strip()
            return "Could not generate a valid response."
        except Exception as e:
            return f"Error during generation: {e}"


