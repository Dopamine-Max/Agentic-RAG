# Smart File-Analyzer

Multi-modal AI-powered document analysis system with support for numerous file types.

## Features
- 📄 Document Q&A (if you want images in images in your files to be processed ensure you conver them to .pdf)
- 🖼️ Image description using gemma3
- 🎤 Audio transcription with Whisper
- 🔍 Web search integration
- 🧠 Context-aware conversations

## Installation
1. Install [Python 3.11^](https://www.python.org/downloads/) [Ollama](https://ollama.com/download/windows) and [Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer)
2. Setup Accounts on [Qdrant](https://qdrant.tech), [Groq](https://groq.com), [Ngrok](https://ngrok.com)    
3. Clone repo:
   ```bash
   git clone https://github.com/Dopamine-Max/smart-file-analyzer.git
   cd smart-file-analyzer
   ```
4. Setup the dependencies:
    ```bash
    poetry install
    poetry run pip install git+https://github.com/openai/whisper.git@v20231117
    ```
5. Setup a virtual environment (venv) to run the project
    ```bash
    poetry env use python3
    ```
5. Run the kaggle/colab notebook (.ipynb file) to get ngrok link
6. Run the file api_service.py (current issue requires restart of API after each search)
7. Put all the updated keys and urls in .env file based on .env.example
8. Run config.py file once
9. Through terminal:
    ```bash
    set OLLAMA_HOST=ngrok_url # or http://localhost:11434
    ollama pull nomic-embed-text
    ollama pull gemma3:4b
    ```
10. Run:
    ```bash
    streamlit run interface.py
    ```
