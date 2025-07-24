import tkinter as tk
from tkinter import scrolledtext, messagebox, END
import os
import threading
import sys

# Ensure config.py and rag_backend.py are discoverable
# This assumes they are in the same directory as this script.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- NEW: Import and load configuration from config.py ---
import config
config.load_config_to_env() # This will set the necessary environment variables

import rag_backend

# Load environment variables from .env file (if using, e.g., for API keys)
# Note: config.py explicitly sets the paths, so .env for paths might be redundant here.
# But keep it if you use .env for other things.
from dotenv import load_dotenv
load_dotenv()


class RAGChatbotApp:
    def __init__(self, master):
        self.master = master
        master.title("Website RAG Chatbot (Tkinter)")
        master.geometry("800x700") # Set initial window size

        self.vector_store = None
        self.rag_chain = None
        self.llm_instance = None
        self.rag_model_ready = False
        
        self.current_rag_task_thread = None # To keep track of running tasks

        # --- Configuration / Input Frame ---
        config_frame = tk.LabelFrame(master, text="Configuration", padx=10, pady=10)
        config_frame.pack(pady=10, padx=10, fill="x")

        tk.Label(config_frame, text="Website URL:").grid(row=0, column=0, sticky="w", pady=2)
        self.website_url_entry = tk.Entry(config_frame, width=50)
        # --- NEW: Retrieve default from environment variable ---
        self.website_url_entry.insert(0, os.getenv("DEFAULT_WEBSITE_URL", "https://fossee.in"))
        self.website_url_entry.grid(row=0, column=1, padx=5, pady=2, sticky="ew")

        tk.Label(config_frame, text="Sitemap URL:").grid(row=1, column=0, sticky="w", pady=2)
        self.sitemap_url_entry = tk.Entry(config_frame, width=50)
        # --- NEW: Retrieve default from environment variable ---
        self.sitemap_url_entry.insert(0, os.getenv("DEFAULT_SITEMAP_URL", "https://fossee.in/sitemap.xml"))
        self.sitemap_url_entry.grid(row=1, column=1, padx=5, pady=2, sticky="ew")

        self.force_rebuild_var = tk.BooleanVar(value=rag_backend.FORCE_REBUILD_ON_INIT)
        tk.Checkbutton(config_frame, text="Force Rebuild RAG Model", variable=self.force_rebuild_var) \
                       .grid(row=2, column=0, columnspan=2, sticky="w", pady=5)

        self.build_button = tk.Button(config_frame, text="Build/Rebuild RAG Model", command=self.start_build_rag_model)
        self.build_button.grid(row=3, column=0, columnspan=2, pady=10)

        self.status_label = tk.Label(config_frame, text="Status: Ready to build RAG model.", fg="blue")
        self.status_label.grid(row=4, column=0, columnspan=2, sticky="w", pady=5)

        config_frame.grid_columnconfigure(1, weight=1) # Allow website URL entry to expand

        # --- Chat Display Frame ---
        chat_frame = tk.LabelFrame(master, text="Chat", padx=10, pady=10)
        chat_frame.pack(pady=10, padx=10, fill="both", expand=True)

        self.chat_display = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, state='disabled', font=("Arial", 10))
        self.chat_display.pack(pady=5, fill="both", expand=True)

        # --- User Input Frame ---
        input_frame = tk.Frame(master, padx=10, pady=10)
        input_frame.pack(pady=5, padx=10, fill="x")

        self.user_input_entry = tk.Entry(input_frame, width=70, font=("Arial", 10))
        self.user_input_entry.pack(side=tk.LEFT, fill="x", expand=True, padx=(0, 5))
        self.user_input_entry.bind("<Return>", self.send_query_event) # Bind Enter key

        self.send_button = tk.Button(input_frame, text="Send", command=self.send_query)
        self.send_button.pack(side=tk.RIGHT)

        # Initial message
        self.append_message("Bot", "Welcome! Please configure the website and build the RAG model to start chatting.")

        # Try to load RAG model on startup if files exist
        self.master.after(100, self.start_initial_load) # Call after 100ms to allow UI to render

    def update_status(self, message, color="blue"):
        self.status_label.config(text=f"Status: {message}", fg=color)
        self.master.update_idletasks() # Force UI update

    def append_message(self, sender, message):
        self.chat_display.config(state='normal')
        self.chat_display.insert(END, f"{sender}: {message}\n\n")
        self.chat_display.config(state='disabled')
        self.chat_display.yview(END) # Scroll to bottom

    def enable_ui(self):
        self.build_button.config(state="normal")
        self.user_input_entry.config(state="normal")
        self.send_button.config(state="normal")

    def disable_ui(self):
        self.build_button.config(state="disabled")
        self.user_input_entry.config(state="disabled")
        self.send_button.config(state="disabled")

    def start_initial_load(self):
        # Initial load attempt on app startup
        self.update_status("Attempting to load existing RAG model or will build if not found...", "blue")
        self.disable_ui()
        # Start the build/load process in a new thread
        self.current_rag_task_thread = threading.Thread(target=self._initial_load_rag_model_task)
        self.current_rag_task_thread.start()

    def start_build_rag_model(self):
        # Ensure only one build process runs at a time
        if self.current_rag_task_thread and self.current_rag_task_thread.is_alive():
            messagebox.showwarning("Busy", "A RAG model build/load process is already running. Please wait.")
            return

        self.update_status("Building/Rebuilding RAG model... This may take a few minutes.", "blue")
        self.disable_ui()
        # Start the build process in a new thread
        self.current_rag_task_thread = threading.Thread(target=self._build_rag_model_task)
        self.current_rag_task_thread.start()

    def _initial_load_rag_model_task(self):
        try:
            website_url = self.website_url_entry.get()
            sitemap_url = self.sitemap_url_entry.get()
            # For initial load, we don't force rebuild unless the checkbox is explicitly set
            force_rebuild = self.force_rebuild_var.get()

            self.vector_store = rag_backend.process_website_content(website_url, sitemap_url, force_rebuild)

            if self.vector_store:
                self.update_status("RAG model ready! Initializing LLM...", "green")
                self.llm_instance = rag_backend.initialize_llm()
                if self.llm_instance:
                    self.rag_chain = rag_backend.get_rag_chain(self.vector_store, self.llm_instance)
                    if self.rag_chain:
                        self.rag_model_ready = True
                        self.update_status("Chatbot engine initialized and ready to chat!", "green")
                        self.append_message("Bot", "RAG model and chatbot engine are ready! Ask me anything about the website.")
                    else:
                        self.update_status("Failed to initialize chatbot engine (RAG chain).", "red")
                        self.rag_model_ready = False
                else:
                    self.update_status("Failed to initialize LLM. Check model path/access.", "red")
                    self.rag_model_ready = False
            else:
                self.update_status("Failed to build RAG model. Please check console for details.", "red")
                self.rag_model_ready = False
        except Exception as e:
            self.update_status(f"An error occurred during initial load: {e}", "red")
            messagebox.showerror("Error", f"An error occurred during initial load: {e}")
            self.rag_model_ready = False
        finally:
            self.enable_ui()

    def _build_rag_model_task(self):
        try:
            website_url = self.website_url_entry.get()
            sitemap_url = self.sitemap_url_entry.get()
            force_rebuild = self.force_rebuild_var.get()

            self.vector_store = rag_backend.process_website_content(website_url, sitemap_url, force_rebuild)

            if self.vector_store:
                self.update_status("RAG model built successfully! Initializing LLM...", "green")
                self.llm_instance = rag_backend.initialize_llm()
                if self.llm_instance:
                    self.rag_chain = rag_backend.get_rag_chain(self.vector_store, self.llm_instance)
                    if self.rag_chain:
                        self.rag_model_ready = True
                        self.update_status("Chatbot engine initialized and ready to chat!", "green")
                        self.append_message("Bot", "RAG model and chatbot engine are ready! Ask me anything about the website.")
                    else:
                        self.update_status("Failed to initialize chatbot engine (RAG chain).", "red")
                        self.rag_model_ready = False
                else:
                    self.update_status("Failed to initialize LLM. Check model path/access.", "red")
                    self.rag_model_ready = False
            else:
                self.update_status("Failed to build RAG model. Check console for details.", "red")
                self.rag_model_ready = False
        except Exception as e:
            self.update_status(f"An error occurred during build: {e}", "red")
            messagebox.showerror("Error", f"An error occurred during build: {e}")
            self.rag_model_ready = False
        finally:
            self.enable_ui()

    def send_query_event(self, event=None): # Added event parameter for bind
        self.send_query()

    def send_query(self):
        user_query = self.user_input_entry.get()
        if not user_query.strip():
            return

        self.append_message("You", user_query)
        self.user_input_entry.delete(0, END) # Clear input field

        if not self.rag_model_ready or self.rag_chain is None:
            self.append_message("Bot", "RAG model is not ready. Please build/rebuild it first.")
            return

        self.disable_ui() # Disable input while processing
        self.update_status("Generating response...", "blue")
        # Run query in a separate thread
        query_thread = threading.Thread(target=self._process_query_task, args=(user_query,))
        query_thread.start()

    def _process_query_task(self, query):
        try:
            result = self.rag_chain({"query": query})
            response = result["result"].strip()
            source_documents = result.get("source_documents", [])

            if not response or "cannot answer this question based on the provided information" in response.lower():
                final_response = response if response else "I cannot answer this question based on the provided information."
            else:
                final_response = response

            # Append sources to the response for transparency
            if source_documents:
                final_response += "\n\n**Sources:**\n"
                unique_sources = set()
                for doc in source_documents:
                    if 'source' in doc.metadata:
                        unique_sources.add(doc.metadata['source'])
                
                for i, source_url in enumerate(list(unique_sources)):
                    final_response += f"- [Source {i+1}]({source_url})\n"
            
            self.append_message("Bot", final_response)
            self.update_status("Ready.", "green")

        except Exception as e:
            self.append_message("Bot", f"An error occurred while generating a response: {e}")
            self.update_status("Error during query processing.", "red")
            messagebox.showerror("Error", f"An error occurred during query processing: {e}")
        finally:
            self.enable_ui()


if __name__ == "__main__":
    root = tk.Tk()
    app = RAGChatbotApp(root)
    root.mainloop()