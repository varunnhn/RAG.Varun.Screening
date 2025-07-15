import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import sys
import os
import fitz  # PyMuPDF
import numpy as np

def load_and_chunk_pdf(file_path):
    """
    Loads a PDF, extracts text, and splits it into smaller chunks.
    
    Args:
        file_path (str): The path to the PDF file.

    Returns:
        list: A list of text chunks (strings).
    """
    print(f"Loading and processing '{file_path}'...")
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        sys.exit(1)
        
    doc = fitz.open(file_path)
    chunks = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        # Simple chunking strategy: split by paragraphs
        page_chunks = text.split('\n\n')
        # Filter out empty or very short chunks
        chunks.extend([chunk.strip() for chunk in page_chunks if len(chunk.strip()) > 100])
    
    if not chunks:
        print("Warning: No text chunks could be extracted from the PDF. The chatbot will not have any context.")
        
    print(f"PDF processed into {len(chunks)} chunks.")
    return chunks

def create_vector_store(chunks, model):
    """
    Creates embeddings for each text chunk.

    Args:
        chunks (list): A list of text chunks.
        model (SentenceTransformer): The sentence transformer model.

    Returns:
        torch.Tensor: A tensor containing the embeddings for all chunks.
    """
    print("Creating vector store from text chunks...")
    embeddings = model.encode(chunks, convert_to_tensor=True, show_progress_bar=True)
    print("Vector store created successfully.")
    return embeddings

def retrieve_context(query, model, chunks, embeddings, top_k=3):
    """
    Retrieves the most relevant text chunks for a given query.

    Args:
        query (str): The user's question.
        model (SentenceTransformer): The sentence transformer model.
        chunks (list): The list of text chunks.
        embeddings (torch.Tensor): The embeddings for the text chunks.
        top_k (int): The number of top chunks to retrieve.

    Returns:
        str: A single string containing the concatenated relevant context.
    """
    query_embedding = model.encode(query, convert_to_tensor=True)
    # Calculate cosine similarity between query and all chunks
    cos_scores = util.cos_sim(query_embedding, embeddings)[0]
    
    # Get the top_k most similar chunks
    top_results = torch.topk(cos_scores, k=min(top_k, len(chunks)))
    
    # Concatenate the relevant chunks into a single context string
    context = "\n---\n".join([chunks[i] for i in top_results.indices])
    return context

def main():
    """
    Main function to run the command-line RAG chatbot.
    """
    pdf_path = "document.pdf"
    
    # --- Step 1: Load PDF and create vector store ---
    # Load a sentence transformer model for creating embeddings
    print("Loading sentence transformer model for RAG...")
    retriever_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    text_chunks = load_and_chunk_pdf(pdf_path)
    chunk_embeddings = create_vector_store(text_chunks, retriever_model)

    # --- Step 2: Initialize the Llama 3.2 Generator Model ---
    print("\n--- Llama 3.2 Chatbot Initializing ---")
    model_kwargs = {"torch_dtype": torch.bfloat16}
    device_map = "auto"

    if not torch.cuda.is_available():
        print("\nWARNING: CUDA not available. The model will run on the CPU, which will be slower.")
        num_cores = os.cpu_count()
        print(f"Configuring PyTorch to use all {num_cores} available CPU cores.")
        torch.set_num_threads(num_cores)
        model_kwargs = {}
        print("For faster performance, please run this on a machine with a CUDA-enabled GPU.\n")

    try:
        model_id = "MetaAI/Llama-3.2-3B-Instruct"
        pipe = pipeline(
            "text-generation",
            model=model_id,
            model_kwargs=model_kwargs,
            device_map=device_map,
        )
        print("--- Llama 3.2 Model loaded successfully! ---")
    except Exception as e:
        print(f"Error: Failed to load the model pipeline: {e}")
        print("\nPlease ensure you have 'transformers', 'torch', and 'accelerate' installed.")
        print("You may also need to install: pip install sentence-transformers PyMuPDF")
        sys.exit(1)

    # --- Step 3: Start the Interactive Chat Loop ---
    print("\nChatbot is ready. Ask questions about your document.")
    print("Type 'exit' or 'quit' to end the conversation.")
    print("-" * 60)

    while True:
        user_input = input("\nYou: ")
        if user_input.lower().strip() in ["exit", "quit"]:
            print("\nLlama: Goodbye!")
            break

        if not text_chunks:
            print("\nLlama: I can't answer questions without context from the PDF.")
            continue
            
        # --- RAG Step: Retrieve context before generating ---
        print("Searching for relevant context in the document...")
        context = retrieve_context(user_input, retriever_model, text_chunks, chunk_embeddings)

        # --- Prompt Engineering: Create a prompt with the retrieved context ---
        rag_prompt = f"""
        System: You are an expert Q&A assistant. Use the following context from a document to answer the user's question.
        If the answer is not found in the context, state that you cannot find the answer in the provided document.
        Do not use any outside knowledge.

        Context:
        ---
        {context}
        ---

        User's Question: {user_input}
        """

        # We use a simple, stateless message list for each Q&A turn
        messages = [{"role": "system", "content": rag_prompt}]

        try:
            # Generate a response using the pipeline
            outputs = pipe(
                messages,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )
            
            assistant_response = outputs[0]["generated_text"][-1]
            if assistant_response.get("role") == "assistant":
                response_text = assistant_response.get("content", "").strip()
                print(f"\nLlama: {response_text}")
            else:
                print("\nLlama: I'm sorry, I couldn't generate a proper response.")

        except Exception as e:
            print(f"\nAn error occurred during text generation: {e}")
            continue

if __name__ == "__main__":
    main()
