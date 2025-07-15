import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import threading
import queue

# Import the backend logic from the other file
import rag_core

class ChatbotGUI:
    def __init__(self, root):
        """
        Initializes the GUI components.
        """
        self.root = root
        self.root.title("RAG Chatbot")
        self.root.geometry("800x700")

        # --- Style ---
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TButton", padding=6, relief="flat", background="#007bff", foreground="white")
        style.map("TButton", background=[('active', '#0056b3')])
        style.configure("TFrame", background="#f0f0f0")
        style.configure("TLabel", background="#f0f0f0", font=("Helvetica", 10))

        # --- Main Frame ---
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Top Frame for PDF selection ---
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=5)

        self.pdf_label = ttk.Label(top_frame, text="No PDF selected.", font=("Helvetica", 10, "italic"))
        self.pdf_label.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

        self.pdf_button = ttk.Button(top_frame, text="Select PDF", command=self.select_pdf)
        self.pdf_button.pack(side=tk.RIGHT)

        # --- Chat Display Area ---
        self.chat_area = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, state=tk.DISABLED, font=("Helvetica", 11))
        self.chat_area.pack(padx=5, pady=5, expand=True, fill=tk.BOTH)
        self.chat_area.tag_config('user', foreground="#003366")
        self.chat_area.tag_config('bot', foreground="#006400")
        self.chat_area.tag_config('system', foreground="#808080", font=("Helvetica", 9, "italic"))

        # --- Bottom Frame for user input ---
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.X, pady=5)

        self.user_input = ttk.Entry(bottom_frame, font=("Helvetica", 11))
        self.user_input.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        self.user_input.bind("<Return>", self.send_message_event)

        self.send_button = ttk.Button(bottom_frame, text="Send", command=self.send_message)
        self.send_button.pack(side=tk.RIGHT)
        
        # --- Backend Initialization ---
        self.response_queue = queue.Queue()
        self.root.after(100, self.process_queue)
        
        self.pdf_path = None
        self.text_chunks = []
        self.chunk_embeddings = None
        self.chatbot_model = None

        self.add_message("System: Welcome! Please select a PDF to begin.", "system")
        self.init_backend()

    def init_backend(self):
        """
        Initializes the heavy backend models in a separate thread.
        """
        self.add_message("System: Loading AI models... This may take a moment.", "system")
        self.send_button.config(state=tk.DISABLED)
        
        threading.Thread(target=self._load_models_thread, daemon=True).start()

    def _load_models_thread(self):
        """
        Thread target for loading models to avoid freezing the GUI.
        """
        model = rag_core.ChatbotModel()
        self.response_queue.put(("models_loaded", model))

    def select_pdf(self):
        """
        Opens a file dialog to select a PDF and processes it.
        """
        path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if not path:
            return
        
        self.pdf_path = path
        self.pdf_label.config(text=os.path.basename(path))
        self.add_message(f"System: Processing '{os.path.basename(path)}'...", "system")
        self.send_button.config(state=tk.DISABLED)
        
        threading.Thread(target=self._process_pdf_thread, daemon=True).start()

    def _process_pdf_thread(self):
        """
        Thread target for processing the PDF.
        """
        chunks = rag_core.load_and_chunk_pdf(self.pdf_path)
        if self.chatbot_model and self.chatbot_model.retriever_model:
            embeddings = rag_core.create_vector_store(chunks, self.chatbot_model.retriever_model)
            self.response_queue.put(("pdf_processed", (chunks, embeddings)))
        else:
            self.response_queue.put(("error", "Retriever model not loaded."))

    def send_message(self, event=None):
        """
        Handles sending a message from the user input field.
        """
        query = self.user_input.get().strip()
        if not query:
            return

        self.add_message(f"You: {query}", "user")
        self.user_input.delete(0, tk.END)
        self.send_button.config(state=tk.DISABLED)

        # Generate response in a separate thread
        threading.Thread(target=self._generate_response_thread, args=(query,), daemon=True).start()

    def _generate_response_thread(self, query):
        """
        Thread target for generating a bot response.
        """
        if not self.pdf_path:
            self.response_queue.put(("bot_response", "Please select a PDF file first."))
            return
        if self.chatbot_model is None:
            self.response_queue.put(("bot_response", "The chatbot model is not ready yet."))
            return

        context = rag_core.retrieve_context(query, self.chatbot_model.retriever_model, self.text_chunks, self.chunk_embeddings)
        response = self.chatbot_model.generate_response(query, context)
        self.response_queue.put(("bot_response", response))

    def add_message(self, message, tag):
        """
        Adds a message to the chat display area.
        """
        self.chat_area.config(state=tk.NORMAL)
        self.chat_area.insert(tk.END, message + "\n\n", (tag,))
        self.chat_area.config(state=tk.DISABLED)
        self.chat_area.yview(tk.END)

    def process_queue(self):
        """
        Processes messages from the background threads.
        """
        try:
            message_type, data = self.response_queue.get_nowait()
            
            if message_type == "models_loaded":
                self.chatbot_model = data
                self.add_message("System: AI models loaded successfully.", "system")
                if self.pdf_path: # Re-enable button if PDF was selected while loading
                    self.send_button.config(state=tk.NORMAL)
            
            elif message_type == "pdf_processed":
                self.text_chunks, self.chunk_embeddings = data
                self.add_message("System: PDF processed. You can now ask questions.", "system")
                self.send_button.config(state=tk.NORMAL)

            elif message_type == "bot_response":
                self.add_message(f"Bot: {data}", "bot")
                self.send_button.config(state=tk.NORMAL)
            
            elif message_type == "error":
                self.add_message(f"System Error: {data}", "system")
                self.send_button.config(state=tk.NORMAL)

        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.process_queue)

    def send_message_event(self, event):
        self.send_message()

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatbotGUI(root)
    root.mainloop()

