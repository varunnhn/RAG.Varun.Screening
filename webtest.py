import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque
import time
import re
import xml.etree.ElementTree as ET # For parsing XML sitemap

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
import torch
import transformers
from transformers import AutoTokenizer
import csv

from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.documents import Document

# --- Configuration ---
WEBSITE_URL = "https://fossee.in"
SITEMAP_URL = "https://fossee.in/sitemap.xml" # New: Sitemap URL

LLAMA_HF_MODEL_NAME = r'C:\Users\pg401\.llama\checkpoints\Llama3.2-1B'
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
FAISS_INDEX_PATH = r'C:\Users\pg401\Varun\Programs\note-taking-app\RAG\faiss_index_website_content_sitemap' # Updated for sitemap crawling
SCRAPED_DATA_CSV = r'C:\Users\pg401\Varun\Programs\note-taking-app\RAG\scraped_fossee_content_sitemap.csv' # Updated for sitemap crawling

CHUNK_SIZE = 700 # Updated as per your request
CHUNK_OVERLAP = 100
RETRIEVAL_K = 3 # Updated as per your request

# --- Crawler Configuration ---
MAX_CRAWL_DEPTH = 2 # How many links deep to follow from the sitemap/start URL
MAX_PAGES_TO_CRAWL = 100 # Maximum number of unique pages to scrape from sitemap or general crawl
CRAWL_DELAY_SECONDS = 0.5 # Delay between page requests


# --- Helper Function for Web Scraping (raw content) ---
def scrape_website_content(url):
    """
    Scrapes the text content from a given URL with minimal filtering,
    retaining essentially all text from the body after removing script/style tags.
    Returns the cleaned text content.
    """
    try:
        print(f"  Scraping: {url}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script and style elements
        for script_or_style in soup(['script', 'style']):
            script_or_style.extract()

        # Get all text from the body
        page_text = soup.get_text(separator=' ', strip=True)

        # Normalize whitespace
        cleaned_text = re.sub(r'\s+', ' ', page_text).strip()

        if len(cleaned_text) < 100:
             print(f"  Warning: Scraped content from {url} is very short ({len(cleaned_text)} chars), might be incomplete.")
             return None

        print(f"  Successfully scraped content from {url}. Length: {len(cleaned_text)} characters.")
        return cleaned_text
    except requests.exceptions.RequestException as e:
        print(f"  Error fetching URL {url}: {e}")
        return None
    except Exception as e:
        print(f"  Error parsing HTML from {url}: {e}")
        return None


# --- Function to Parse Sitemap ---
def parse_sitemap(sitemap_url):
    """
    Fetches and parses a sitemap.xml file to extract all URLs.
    Returns a set of URLs.
    """
    urls = set()
    print(f"\nAttempting to fetch sitemap from: {sitemap_url}")
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(sitemap_url, headers=headers, timeout=20)
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)

        root = ET.fromstring(response.content)

        # Namespace for sitemap. For sitemap.xml, it's usually http://www.sitemaps.org/schemas/sitemap/0.9
        # Check if root tag has a namespace
        ns_match = re.match(r'\{.*\}', root.tag)
        ns = ns_match.group(0) if ns_match else ''

        # Find all <loc> tags within <url> or <sitemap> tags
        for url_element in root.findall(f'{ns}url'):
            loc_element = url_element.find(f'{ns}loc')
            if loc_element is not None:
                urls.add(loc_element.text)
        
        # Handle sitemap index files (sitemap contains links to other sitemaps)
        for sitemap_element in root.findall(f'{ns}sitemap'):
            loc_element = sitemap_element.find(f'{ns}loc')
            if loc_element is not None:
                print(f"  Found nested sitemap: {loc_element.text}. Attempting to parse it...")
                urls.update(parse_sitemap(loc_element.text)) # Recursively parse nested sitemaps

        print(f"  Found {len(urls)} URLs in sitemap(s).")
        return urls
    except requests.exceptions.RequestException as e:
        print(f"  Error fetching sitemap {sitemap_url}: {e}")
        return set()
    except ET.ParseError as e:
        print(f"  Error parsing sitemap XML from {sitemap_url}: {e}")
        return set()
    except Exception as e:
        print(f"  An unexpected error occurred while processing sitemap {sitemap_url}: {e}")
        return set()


# --- Web Crawler Function (Now can start from sitemap URLs) ---
def crawl_website(start_urls, max_depth=2, max_pages=50, delay=0.5):
    """
    Crawls a website to collect content from multiple pages.
    Performs a breadth-first search.
    Can start from multiple URLs (e.g., from a sitemap).
    Returns a list of Document objects, each representing a scraped page.
    """
    if not start_urls:
        print("No start URLs provided for crawling.")
        return []

    # Get the base domain from the first start URL (assuming all are from the same domain)
    base_domain = urlparse(list(start_urls)[0]).netloc
    visited_urls = set()
    urls_to_visit = deque([(url, 0) for url in start_urls]) # (url, depth)
    scraped_documents = []

    print(f"\nStarting web crawl from {len(start_urls)} initial URL(s).")
    print(f"Max depth: {max_depth}, Max pages: {max_pages}, Delay: {delay}s")

    page_count = 0
    while urls_to_visit and page_count < max_pages:
        current_url, depth = urls_to_visit.popleft()

        if current_url in visited_urls:
            continue

        if depth > max_depth:
            continue

        # Ensure we stay within the same domain
        if urlparse(current_url).netloc != base_domain:
            print(f"  Skipping external link: {current_url}")
            continue

        visited_urls.add(current_url)
        print(f"Crawling URL (Depth {depth}/{max_depth}, Scraped: {page_count}/{max_pages}): {current_url}")

        # Scrape content from the current page
        content = scrape_website_content(current_url)
        if content: # Only add if content was successfully scraped
            scraped_documents.append(Document(page_content=content, metadata={"source": current_url}))
            page_count += 1 # Increment page count only for successfully scraped pages

        # Find new links to add to the queue (only if within max_pages and depth)
        if page_count < max_pages and depth < max_depth:
            try:
                response = requests.get(current_url, timeout=10)
                soup = BeautifulSoup(response.text, 'html.parser')
                for link_tag in soup.find_all('a', href=True):
                    href = link_tag['href']
                    absolute_url = urljoin(current_url, href).split('#')[0] # Remove fragments

                    if absolute_url.startswith(('http://', 'https://')) and \
                       urlparse(absolute_url).netloc == base_domain and \
                       absolute_url not in visited_urls:
                        # Filter out common file types or unwanted paths
                        if not any(absolute_url.lower().endswith(ext) for ext in ['.pdf', '.zip', '.tar.gz', '.docx', '.pptx', '.xlsx', '.jpg', '.png', '.gif', '.mp4', '.avi', '.mov']):
                            if not any(path_segment in absolute_url.lower() for path_segment in ['/wp-content/', '/uploads/', '/feed/', '/comment-page-', '/tag/', '/category/', '/author/']):
                                urls_to_visit.append((absolute_url, depth + 1))
            except requests.exceptions.RequestException as e:
                print(f"  Error getting links from {current_url}: {e}")
            except Exception as e:
                print(f"  Error parsing links from {current_url}: {e}")

        time.sleep(delay) # Be polite

    print(f"\nFinished crawling. Scraped {len(scraped_documents)} pages.")
    return scraped_documents


# --- Function: Save Documents to CSV ---
def save_documents_to_csv(documents, output_csv_path):
    """
    Saves a list of Document objects to a CSV file.
    Each row contains 'content' and 'source'.
    Returns True on success, False on failure.
    """
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    print(f"\nSaving {len(documents)} scraped documents to CSV: {output_csv_path}")
    if not documents:
        print("No documents to save to CSV.")
        return False

    try:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['content', 'source']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for doc in documents:
                writer.writerow({'content': doc.page_content, 'source': doc.metadata.get('source', 'Unknown Source')})
        print(f"Successfully saved all scraped content to '{output_csv_path}'.")
        return True
    except IOError as e:
        print(f"Error writing to CSV file {output_csv_path}: {e}")
        return False


# --- Function: Load from CSV ---
def load_documents_from_csv(csv_path):
    """
    Loads documents from a CSV file.
    Each row is expected to have 'content' and 'source' columns.
    Returns a list of Document objects.
    """
    documents = []
    if not os.path.exists(csv_path):
        print(f"CSV file not found at '{csv_path}'.")
        return []

    print(f"Loading documents from CSV: '{csv_path}'...")
    try:
        with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                content = row.get('content')
                source = row.get('source', 'Unknown Source')
                if content:
                    documents.append(Document(page_content=content, metadata={"source": source}))
        print(f"Loaded {len(documents)} document(s) from '{csv_path}'.")
    except Exception as e:
        print(f"Error reading CSV file '{csv_path}': {e}")
    return documents


# --- Main Execution ---
if __name__ == "__main__":
    print("Starting Website RAG System with Llama 3.2-1B via Hugging Face Transformers (Langchain-based)...")
    print(f"Target Base URL: '{WEBSITE_URL}'")
    print(f"Sitemap URL: '{SITEMAP_URL}'")
    print("\n*** IMPORTANT: Please ensure the following for Llama 3.2-1B model loading: ***")
    print("    1. You have accepted the Llama 3.2 license on Hugging Face (meta-llama/Llama-3.2-1B page).")
    print("    2. You have logged in to Hugging Face from your terminal using `huggingface-cli login`.")
    print("    (If loading from a local path, ensure that that path contains all HF model files including `tokenizer.model`.)")
    print("    3. You have sufficient RAM/VRAM for the Llama model (1.3B parameters requires several GBs).")

    # --- Step 1: Web Crawling to CSV or Loading from CSV ---
    documents = []
    # Set FORCE_REBUILD to True to ensure scraping happens every time you run
    # For production, set to False to only rebuild if files are missing.
    FORCE_REBUILD = True # Set to True for development to force re-scrape

    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(SCRAPED_DATA_CSV) or FORCE_REBUILD:
        print("\nInitiating web crawling based on sitemap or default URL (required for new data or forced refresh)...")
        
        sitemap_urls = parse_sitemap(SITEMAP_URL)
        
        # If sitemap parsing fails or returns no URLs, fallback to scraping the main WEBSITE_URL
        if not sitemap_urls:
            print(f"Sitemap parsing failed or returned no URLs. Falling back to single-page scrape: {WEBSITE_URL}")
            start_urls_for_crawl = {WEBSITE_URL} # Use a set to be consistent with crawl_website input
        else:
            print(f"Successfully parsed sitemap. Found {len(sitemap_urls)} URLs. Starting crawl.")
            start_urls_for_crawl = sitemap_urls

        scraped_docs = crawl_website(
            start_urls=start_urls_for_crawl,
            max_depth=MAX_CRAWL_DEPTH,
            max_pages=MAX_PAGES_TO_CRAWL,
            delay=CRAWL_DELAY_SECONDS
        )

        if scraped_docs:
            if save_documents_to_csv(scraped_docs, SCRAPED_DATA_CSV):
                documents = load_documents_from_csv(SCRAPED_DATA_CSV)
            else:
                print("Error: Could not save scraped content to CSV. Cannot proceed with RAG.")
                exit()
        else:
            print("Error: No content scraped from the website(s). Cannot proceed with RAG.")
            exit()
    else:
        print(f"\nCSV file found at '{SCRAPED_DATA_CSV}'. Loading content from CSV...")
        documents = load_documents_from_csv(SCRAPED_DATA_CSV)
        if not documents:
            print("Error: No documents loaded from CSV. Cannot proceed with RAG.")
            exit()

    # --- Step 1.5: Split documents (always split the loaded/scraped documents) ---
    if documents:
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Split content into {len(chunks)} text chunks (chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}).")
        documents = chunks # Update 'documents' to refer to the chunks for further processing

        # --- DEBUG: Print Sample Scraped Documents (Chunks) ---
        print("\n--- Printing Sample Scraped Documents (Chunks) ---")
        for i, doc in enumerate(documents[:5]): # Print only first 5 for brevity
            print(f"\n--- Document {i+1} ---")
            print(f"Source: {doc.metadata.get('source', 'N/A')}")
            print(f"Length: {len(doc.page_content)} characters")
            print("Content (first 200 chars):")
            print(doc.page_content[:200])
            print("--------------------")
        if len(documents) > 5:
            print(f"... and {len(documents) - 5} more documents.")
        print("\n--- End of Sample Documents ---")
        # --- END DEBUG ---
    else:
        print("No documents available after loading/splitting. Exiting.")
        exit()


    # --- Step 2: Create Embeddings and Vector Store (with persistence logic) ---
    print("\nCreating embeddings and building/loading FAISS vector store...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

        os.makedirs(FAISS_INDEX_PATH, exist_ok=True)

        if os.path.exists(os.path.join(FAISS_INDEX_PATH, "index.faiss")) and \
           os.path.exists(os.path.join(FAISS_INDEX_PATH, "index.pkl")) and \
           not FORCE_REBUILD:
            print(f"Loading existing FAISS index from '{FAISS_INDEX_PATH}'...")
            db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            print("FAISS index loaded successfully.")
        else:
            print("No existing FAISS index found or incomplete, or new content scraped. Creating a new one...")
            if not documents:
                print("Error: No documents available from scraping/loading to create FAISS index.")
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

        stop_sequences = [llama_tokenizer.eos_token, "<|eot_id|>", "Human:", "Context:"]

        llm = HuggingFacePipeline(
            pipeline=llm_pipeline,
            model_kwargs={
                "temperature": 0.01,
                "max_new_tokens": 200,
                "do_sample": True,
                "top_p": 0.8,
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
    "You are a helpful and knowledgeable AI assistant. Your task is to answer the user's question "
    "based *solely* on the provided context. "
    "If the answer is not present or cannot be inferred from the context, state that you cannot answer. "
    "Do not use external knowledge or make up information. Be concise and to the point."

    "\n\nContext:\n{context}\n\n"
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
        return_source_documents=True,
        chain_type_kwargs={"prompt": chat_prompt}
    )
    print("RetrievalQA chain ready with strict context adherence.")

    # --- Step 5: Interactive Query Loop ---
    while True:
        query = input("\nEnter your question about the website content (type 'exit' to quit): ")
        if query.lower() == 'exit':
            break

        clean_query = query.split('(')[0].strip()

        print(f"\nProcessing query: '{clean_query}'...")
        try:
            result = qa_chain({"query": clean_query})
            raw_answer = result["result"]
            answer = raw_answer.strip()
            source_documents = result.get("source_documents", [])

            print("\n--- Answer ---")
            if not answer or "cannot answer this question based on the provided information" in answer.lower():
                print(answer if answer else "No answer was generated by the model (empty response or explicit refusal).")
            else:
                print(answer)

            # Print sources for verification
            if source_documents:
                print("\n--- Sources Used (Top Retrieved Documents) ---")
                for i, doc in enumerate(source_documents):
                    print(f"Doc {i+1} from: {doc.metadata.get('source', 'N/A')} (Length: {len(doc.page_content)})")
                    print(f"  Content snippet: {doc.page_content[:200]}...")
                print("---------------------------------------------")

            print("-" * 30)

        except Exception as e:
            print(f"An error occurred during query processing: {e}")
            print("Please check the model loading and pipeline setup.")

    print("\nExiting RAG system.")
