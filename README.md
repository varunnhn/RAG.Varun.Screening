**

### Step-by-Step Guide to Running Your Chatbot

Follow these straightforward instructions to get your Python RAG chatbot with a graphical user interface up and running.

#### Step 1: Prerequisites

Ensure you have Python installed on your computer. You can download it from the official Python website: [python.org](https://www.python.org/downloads/). It's crucial to note that any recent version of Python 3 (e.g., 3.8 or newer) is required for the chatbot to function properly.

#### Step 2: Save the Project Files

It's important to save the three files we've created into the same folder or directory on your computer. This ensures that all the necessary components are easily accessible and organized.

1. rag_core.py: This file contains the backend logic for the chatbot.
    
2. gui_chatbot.py: This file contains the code for the graphical user interface.
    
3. requirements.txt: This file lists all the necessary Python libraries.
    

Your folder should look like this:

/your_project_folder/  
|-- rag_core.py  
|-- gui_chatbot.py  
|-- requirements.txt  
  

#### Step 3: Install the Required Libraries

Next, you need to install the Python libraries listed in requirements.txt.

1. Open your terminal or command prompt.
    

- On Windows, search for cmd or PowerShell.
    
- On macOS, search for Terminal.
    
- On Linux, use your default terminal application.
    

2. Navigate to the folder where you saved your files. You can do this using the cd (change directory) command. For example:  
    cd path/to/your_project_folder  
      
    
3. Once you are in the correct directory, run the following command to install all the dependencies automatically:  
    pip install -r requirements.txt  
      
    This command tells pip (Python's package installer) to read the requirements.txt file and install every library listed inside it.
    

#### Step 4: Run the Application

After the installation is complete, you can run the application.

1. Make sure you are still in the same directory in your terminal.
    
2. Run the gui_chatbot.py script using the following command:  
    python gui_chatbot.py  
      
    

#### Step 5: Interacting with the Chatbot

1. When you run the script, a window titled "RAG Chatbot" will appear.
    
2. You will see a system message indicating that the AI models are loading. This might take a minute, especially on the first run, as the models need to be downloaded.
    
3. Once loaded, click the "Select PDF" button to choose a PDF document from your computer.
    
4. After the PDF is processed, you can type your questions into the input box at the bottom and press Enter or click the "Send" button.
    
5. The chatbot will find relevant information in your PDF and generate an answer.
    

That's it! You now have a fully functional RAG chatbot running on your machine, ready to assist you.

**