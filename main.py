import sys
sys.path.append(r" C:\Users\NISHANT\AppData\Local\Programs\Python\Python313\Lib\site-packages")
sys.path.append(r"  C:\Users\NISHANT\AppData\Local\Programs\Python\Python313\Lib\site-packages")
import os 
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

DB_FAISS_PATH = "vectorstore/db_faiss"
HF_TOKEN = os.environ.get("HF_TOKEN")
@st.cache_resource

def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization = True)
    return db

def set_custom_prompt(custom_prompt_template):
    """Create prompt template for QA."""
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )
    return prompt

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




def main():
    st.title("Medi-Bot! 🤖🩺")

    if 'message' not in st.session_state:
        st.session_state.message = []

    # Display chat history
    for message in st.session_state.message:
        st.chat_message(message['role']).markdown(message['content'])  # ✅ FIXED

    # Get user input
    prompt = st.chat_input("Pass your prompt here")

    if prompt:
        # Display user message
        st.chat_message('user').markdown(prompt)
        st.session_state.message.append({'role': 'user', 'content': prompt})


        custom_prompt_template = """
            Use the pieces of information provided in the context to answer user's question.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            Don't provide anything out of the given context.

            Context: {context}
            Question: {question}

            Start the answer directly. No small talk please.
            """
        
        HUGGING_FACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.environ.get("HF_TOKEN")
        


        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load vectorstore")
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGING_FACE_REPO_ID),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": set_custom_prompt(custom_prompt_template)}
            )

            response = qa_chain.invoke({'query': prompt})

            if isinstance(response, dict) and 'result' in response:
                result_to_show = response['result']  # ✅ Only show the result
            else:
                result_to_show = str(response)  # ✅ Handle cases where response is a string

            # ✅ Display only the result (without source documents)
            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.message.append({'role': 'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"An error occurred: {e}")



if __name__ == "__main__":
    main()
