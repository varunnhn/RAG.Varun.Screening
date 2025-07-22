import os
import requests
from bs4 import BeautifulSoup
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
import torch
import transformers
from transformers import AutoTokenizer

from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.documents import Document


# --- Configuration ---
WEBSITE_URL = "https://fossee.in/" # Target URL

LLAMA_HF_MODEL_NAME = r'C:\Users\pg401\.llama\checkpoints\Llama3.2-1B'
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
FAISS_INDEX_PATH = r'C:\Users\pg401\Varun\Programs\note-taking-app\RAG\faiss_index_website_content_refined' # Changed index path

CHUNK_SIZE = 700 # Slightly reduced chunk size, might fit more context in Llama's window
CHUNK_OVERLAP = 100 # Reduced overlap slightly
RETRIEVAL_K = 3 # Adjusted retrieval k, less noise, more focused


# --- Helper Function for Web Scraping (Enhanced) ---
def scrape_website_content(url):
    """
    Scrapes the text content from a given URL using requests and BeautifulSoup,
    focusing on main content areas.
    Returns the cleaned text content.
    """
    try:
        print(f"Attempting to scrape content from: {url}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15) # Increased timeout
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove irrelevant elements that are not part of the main content
        for script_or_style in soup(['script', 'style', 'header', 'footer', 'nav', 'aside', 'form', '.sidebar', '.menu', '.ads']):
            script_or_style.extract()

        # Try to find main content areas. This is site-specific!
        # Inspect fossee.in's HTML to find appropriate tags/classes.
        # Common candidates: <article>, <main>, <div id="content">, <div class="main-body">, <div class="post-content">
        # For fossee.in, looking at its structure, it's quite diverse.
        # We'll try to get text from common text-holding tags or the body if specific sections aren't clear.
        main_content_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'li', 'td', 'div']) # Include common text tags

        # Filter out very short or clearly non-content divs if necessary
        filtered_text_parts = []
        for element in main_content_elements:
            text = element.get_text(strip=True)
            if text and len(text) > 50: # Only consider elements with substantial text
                 # Check if the parent is also some navigation or header (heuristic)
                is_nav_or_header_parent = False
                current_parent = element.parent
                while current_parent:
                    if current_parent.name in ['nav', 'header', 'footer', 'aside'] or \
                       ('class' in current_parent.attrs and any(c in current_parent['class'] for c in ['sidebar', 'menu'])):
                        is_nav_or_header_parent = True
                        break
                    current_parent = current_parent.parent
                if not is_nav_or_header_parent:
                    filtered_text_parts.append(text)

        if not filtered_text_parts:
            # Fallback to getting all text if no specific content elements are found
            text = soup.get_text(separator='\n', strip=True)
            print("Warning: No specific main content elements found, falling back to full page text.")
        else:
            text = '\n\n'.join(filtered_text_parts) # Join with double newlines for better paragraph separation

        # Basic cleaning: remove multiple newlines
        cleaned_text = os.linesep.join([s for s in text.splitlines() if s.strip()])
        # Further cleaning for multiple spaces
        cleaned_text = ' '.join(cleaned_text.split())

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
    # Force re-scrape for demonstration of changes, remove `or True` for persistence
    if not os.path.exists(FAISS_INDEX_PATH) or True:
        print("\nScraping website content (required for new FAISS index or forced refresh)...")
        scraped_text = scrape_website_content(WEBSITE_URL)

        if scraped_text:
            web_document = Document(page_content=scraped_text, metadata={"source": WEBSITE_URL})

            # Use RecursiveCharacterTextSplitter for more robust chunking
            text_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", " ", ""], # Try to split by larger units first
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len,
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
                "temperature": 0.05,
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
        "If the answer cannot be found in the given context, please state that you cannot answer based on the provided information, "
        "and suggest asking a more specific question related to the provided context. "
        "Do not use any outside knowledge or provide information not explicitly stated in the context.\n\n"
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