# RAG-LLM-MediBot
## Medical ChatBot - RAG using LLM Model Setup  


Built using LangChain, Hugging Face models, and FAISS vector storage, this chatbot efficiently retrieves relevant information and generates responses using [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3).

---

### Features

- RAG-Based Response Generation: Retrieves relevant context before generating responses.

- Hugging Face LLM Integration: Uses mistralai/Mistral-7B-Instruct-v0.3 for accurate answers.

- FAISS Vector Database: Efficiently stores and retrieves embeddings for improved context.

- Custom Prompt Engineering: Ensures precise and focused responses.

- Streamlit UI: Simple and interactive web-based chatbot interface.

---

### Setup and installation
Make sure you have

- Python3 installed.
- pip installed for package management.
1. Creation of virtualenv
```sh
python3 -m venv env
```
2. activate virtual env
```sh
env/Scripts/activate #Windows
source env/bin/activate #Linux
```
3. install requirements.
```sh
pip install -r requirements.txt
```
4. Add your Knowledge Base

    4.1 Based on your requirements of chatbot add resources (pdf files) in the context folder
  
    4.2 PDFs of books, research papers, articles, documents,
  
    Optional -> in context_maker.py modify the chunk_size and chunk_overlap based on the size of your data to fine tune the embeddings based on your requirements.
  
    4.3 run context_maker.py file
5. Preapre to Load LLM and load the llm with memory of the vector database

    5.1 Make sure to use your huggingface API token after creating an account on Hugging Face and generating an API key with appropiate User permissions [enable repo and inference permissions] and then use that API   key in code
  
    5.2 You can modify the parameters based on your requirements in memory_with_llm.py

6. Run the main.py file using streamlit to serve the RAG
```sh
streamlit run main.py
```
