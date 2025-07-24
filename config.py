# config.py
import os

# --- Define your actual, absolute paths here ---
# IMPORTANT: Replace these with the correct paths on your system!
LLAMA_MODEL_PATH = r"C:\Users\pg401\.llama\checkpoints\Llama3.2-1B" # Your local Llama 3.2-1B model path
FAISS_INDEX_DIRECTORY = os.path.join(os.getcwd(), "faiss_index_data") # Directory to store FAISS index
SCRAPED_DATA_FILE = os.path.join(os.getcwd(), "scraped_website_content.csv") # CSV file for scraped content

# Default URLs for the GUI's input fields
DEFAULT_WEBSITE_URL = "https://fossee.in"
DEFAULT_SITEMAP_URL = "https://fossee.in/sitemap.xml"


def load_config_to_env():
    """
    Loads configuration variables from this file into os.environ.
    This allows other parts of the application to read them using os.getenv().
    In a production environment, these environment variables would ideally
    be set externally (e.g., via shell scripts, Docker environment variables,
    or deployment platform settings) rather than being loaded from a file
    in the application itself.
    """
    print("Loading paths from config.py into environment variables...")
    os.environ['LLAMA_HF_MODEL_NAME'] = LLAMA_MODEL_PATH
    os.environ['FAISS_INDEX_PATH'] = FAISS_INDEX_DIRECTORY
    os.environ['SCRAPED_DATA_CSV'] = SCRAPED_DATA_FILE
    os.environ['DEFAULT_WEBSITE_URL'] = DEFAULT_WEBSITE_URL
    os.environ['DEFAULT_SITEMAP_URL'] = DEFAULT_SITEMAP_URL
    print("Environment variables updated from config.py.")

# Example usage if you run config.py directly (optional)
if __name__ == "__main__":
    load_config_to_env()
    print("\nVerify some loaded variables:")
    print(f"LLAMA_HF_MODEL_NAME: {os.getenv('LLAMA_HF_MODEL_NAME')}")
    print(f"FAISS_INDEX_PATH: {os.getenv('FAISS_INDEX_PATH')}")