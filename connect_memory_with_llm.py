import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGING_FACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    """Setup LLM with HuggingFace endpoint."""
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        huggingfacehub_api_token=HF_TOKEN,
        task="text-generation",  # Specify the task
        temperature=0.5,
        top_p=0.95,
        max_new_tokens=512
    )
    return llm

# Create custom prompt template
custom_prompt_template = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Don't provide anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    """Create prompt template for QA."""
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )
    return prompt

def setup_qa_chain():
    """Setup the complete QA chain with FAISS and LLM."""
    try:
        # Load Database
        DB_FAISS_PATH = "vectorstore/db_faiss"
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Load FAISS database
        db = FAISS.load_local(
            DB_FAISS_PATH,
            embedding_model,
            allow_dangerous_deserialization=True
        )
        
        # Create QA Chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=load_llm(HUGGING_FACE_REPO_ID),
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": set_custom_prompt(custom_prompt_template)}
        )
        
        return qa_chain
        
    except Exception as e:
        print(f"Error setting up QA chain: {str(e)}")
        raise

def process_query(qa_chain, query):
    """Process a single query using the QA chain."""
    try:
        # Use invoke instead of direct call
        response = qa_chain.invoke({"query": query})
        return response
    except Exception as e:
        raise RuntimeError(f"Error processing query: {str(e)}")

def main():
    """Main execution function."""
    print("\nInitializing QA System...")
    
    try:
        # Setup QA chain
        qa_chain = setup_qa_chain()
        print("QA System initialized successfully!")
        
        # Get user input and process query
        while True:
            try:
                user_query = input("\nAsk a question (or 'quit' to exit): ").strip()
                
                if user_query.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!")
                    break
                    
                if not user_query:
                    print("Please enter a valid question.")
                    continue
                
                print("\nProcessing your question...")
                response = process_query(qa_chain, user_query)
                
                print("\nRESULT:", response["result"])
                print("\nSOURCE DOCUMENTS:")
                for i, doc in enumerate(response["source_documents"], 1):
                    print(f"\n{i}. {doc.page_content[:200]}...")
                    
            except KeyboardInterrupt:
                print("\nOperation cancelled by user. Goodbye!")
                break
            except Exception as e:
                print(f"Error processing query: {str(e)}")
                print("Please try again with a different question.")

    except Exception as e:
        print(f"Failed to initialize QA System: {str(e)}")
        print("Please check your environment setup and try again.")

if __name__ == "__main__":
    main()