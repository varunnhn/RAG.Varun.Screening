import os
import requests # New import for making HTTP requests
from bs4 import BeautifulSoup # New import for parsing HTML
# Updated import for HuggingFaceEmbeddings as per deprecation warning
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
# from langchain_community.document_loaders import PyPDFLoader # No longer needed for web scraping
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
import torch
import transformers
from transformers import AutoTokenizer

# Import for custom prompt template
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.documents import Document # Import Document for creating Langchain documents from scraped text


# --- Configuration ---
# Define the URL of the website you want to scrape
WEBSITE_URL = "https://fossee.in/" # <--- IMPORTANT: Replace with the actual URL you want to scrape
# For demonstration, you might want to pick a simpler page or a specific article.
# Example: "https://en.wikipedia.org/wiki/Artificial_intelligence"

# Configuration for Generative LLM (Llama 3.2-1B via Hugging Face Transformers)
# IMPORTANT: Ensure you have accepted the Llama 3.2 license on Hugging Face and run `huggingface-cli login`
# If you have the model downloaded locally in Hugging Face format, replace with your local path:
LLAMA_HF_MODEL_NAME = r'C:\Users\pg401\.llama\checkpoints\Llama3.2-1B' # Path to your local Llama 3.2-1B model

# Configuration for Embedding Model (for Retrieval)
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

# Define a path to save your FAISS index for persistence
# Using a full path
FAISS_INDEX_PATH = r'C:\Users\pg401\Varun\Programs\note-taking-app\RAG\faiss_index_website_content' # Changed path for website data

# Retrieval Parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_K = 4


# --- Helper Function for Web Scraping ---
def scrape_website_content(url):
    """
    Scrapes the text content from a given URL using requests and BeautifulSoup.
    Returns the cleaned text content.
    """
    try:
        print(f"Attempting to scrape content from: {url}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)

        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script and style elements
        for script_or_style in soup(['script', 'style']):
            script_or_style.extract()

        # Get text, strip whitespace, and join lines
        text = soup.get_text(separator='\n', strip=True)

        # Basic cleaning: remove multiple newlines
        cleaned_text = os.linesep.join([s for s in text.splitlines() if s.strip()])
        print(f"Successfully scraped content from {url}. Length: {len(cleaned_text)} characters.")
        return cleaned_text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return None
    except Exception as e:
        print(f"Error parsing HTML from {url}: {e}")
        return None


# --- Main Execution ---
if __name__ == "__main__":
    print("Starting Website RAG System with Llama 3.2-1B via Hugging Face Transformers (Langchain-based)...")
    print(f"Attempting to scrape content from: '{WEBSITE_URL}'")
    print("\n*** IMPORTANT: Please ensure the following for Llama 3.2-1B model loading: ***")
    print("   1. You have accepted the Llama 3.2 license on Hugging Face (meta-llama/Llama-3.2-1B page).")
    print("   2. You have logged in to Hugging Face from your terminal using `huggingface-cli login`.")
    print("   (If loading from a local path, ensure that path contains all HF model files including `tokenizer.model`.)")
    print("   3. You have sufficient RAM/VRAM for the Llama model (1.3B parameters requires several GBs).")

    # --- Step 1: Web Scraping and Splitting ---
    documents = []
    # We will re-scrape and re-process if the FAISS index doesn't exist or if you want to force an update.
    # For a robust system, you might want to add a mechanism to check if the content has changed.
    if not os.path.exists(FAISS_INDEX_PATH) or True: # Set to True to always re-scrape for demonstration
        print("\nScraping website content (required for new FAISS index or forced refresh)...")
        scraped_text = scrape_website_content(WEBSITE_URL)

        if scraped_text:
            # Create a Langchain Document object from the scraped text
            # This is crucial for Langchain's text splitter and vector store to work correctly.
            # You can add metadata if available, e.g., source=WEBSITE_URL
            web_document = Document(page_content=scraped_text, metadata={"source": WEBSITE_URL})

            # Use CharacterTextSplitter for chunking
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len,
                is_separator_regex=False,
            )
            documents = text_splitter.split_documents([web_document])
            print(f"Scraped and split website content into {len(documents)} text chunks (chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}).")
        else:
            print("Error: Could not scrape website content. Cannot proceed with RAG.")
            exit()
    else:
        print(f"\nFAISS index found at '{FAISS_INDEX_PATH}'. Skipping web scraping/splitting.")


    # --- Step 2: Create Embeddings and Vector Store (with persistence logic) ---
    print("\nCreating embeddings and building/loading FAISS vector store...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

        os.makedirs(FAISS_INDEX_PATH, exist_ok=True)

        if os.path.exists(os.path.join(FAISS_INDEX_PATH, "index.faiss")) and \
           os.path.exists(os.path.join(FAISS_INDEX_PATH, "index.pkl")) and not documents:
            # If documents list is empty but FAISS exists, it means we are loading existing.
            print(f"Loading existing FAISS index from '{FAISS_INDEX_PATH}'...")
            db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            print("FAISS index loaded successfully.")
        else:
            print("No existing FAISS index found or incomplete, or new content scraped. Creating a new one...")
            if not documents:
                print("Error: No documents available from scraping to create FAISS index.")
                exit()
            db = FAISS.from_documents(documents, embeddings)
            db.save_local(FAISS_INDEX_PATH)
            print(f"New FAISS vector store created and saved to '{FAISS_INDEX_PATH}'.")

        retriever = db.as_retriever(search_kwargs={"k": RETRIEVAL_K})
        print(f"Vector store (or retriever) ready with k={RETRIEVAL_K}.")
    except Exception as e:
        print(f"Error creating/loading embeddings or vector store: {e}")
        print("Please ensure you have an active internet connection for embedding model download, or it's cached.")
        print("Also check file permissions for the FAISS_INDEX_PATH.")
        exit()

    # --- Step 3: Load the Generative LLM (Llama 3.2-1B) and set up Langchain pipeline ---
    print(f"\nLoading Generative LLM: {LLAMA_HF_MODEL_NAME} and setting up Langchain pipeline...")
    try:
        llama_tokenizer = AutoTokenizer.from_pretrained(LLAMA_HF_MODEL_NAME, trust_remote_code=True, legacy=False)
        if llama_tokenizer.pad_token is None:
            llama_tokenizer.pad_token = llama_tokenizer.eos_token

        llm_pipeline = transformers.pipeline(
            "text-generation",
            model=LLAMA_HF_MODEL_NAME,
            tokenizer=llama_tokenizer,
            trust_remote_code=True,
            model_kwargs={
                "torch_dtype": torch.bfloat16,
                "device_map": "auto",
            },
        )

        stop_sequences = [llama_tokenizer.eos_token, "<|eot_id|>"]

        llm = HuggingFacePipeline(
            pipeline=llm_pipeline,
            model_kwargs={
                "temperature": 0.1,
                "max_new_tokens": 500,
                "do_sample": True,
                "top_p": 0.9,
                "eos_token_id": llama_tokenizer.eos_token_id,
                "stop_sequences": stop_sequences
            }
        )
        print("Llama 3.2-1B generative model and Langchain pipeline loaded successfully.")
    except Exception as e:
        print(f"Error loading Llama 3.2-1B model or setting up pipeline: {e}")
        print("Please review the 'IMPORTANT' notes above regarding Hugging Face access and local paths.")
        print("Also check your system's RAM/VRAM availability.")
        exit()

    # --- Step 4: Set up the RetrievalQA chain with a custom prompt ---
    print("\nSetting up RetrievalQA chain with a custom prompt for strict context adherence...")

    system_template = (
        "You are a helpful assistant. Your task is to answer the user's question based ONLY on the provided context. "
        "If the answer cannot be found in the given context, please state that you cannot answer based on the provided information. "
        "Do not use any outside knowledge.\n\n"
        "Context:\n{context}"
    )
    human_template = "{question}"

    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template),
    ])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": chat_prompt}
    )
    print("RetrievalQA chain ready with strict context adherence.")

    # --- Step 5: Interactive Query Loop ---
    while True:
        query = input("\nEnter your question about the website content (type 'exit' to quit): ")
        if query.lower() == 'exit':
            break

        print(f"\nProcessing query: '{query}'...")
        try:
            result = qa_chain({"query": query})
            raw_answer = result["result"]
            answer = raw_answer.strip()

            print("\n--- Answer ---")
            if not answer:
                print("No answer was generated by the model (empty response).")
                print(f"Raw model output (before strip): '{raw_answer}'")
            else:
                print(answer)

            print("-" * 30)

        except Exception as e:
            print(f"An error occurred during query processing: {e}")
            print("Please check the model loading and pipeline setup.")

    print("\nExiting RAG system.")