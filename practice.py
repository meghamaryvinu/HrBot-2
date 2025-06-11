import streamlit as st
import os
from dotenv import load_dotenv
import time

# Langchain imports
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader


load_dotenv()


openai_api_key = os.getenv("HR_CHATBOT_OPENAI_KEY") 
groq_api_key = os.getenv('HR_CHATBOT_GROQ_KEY') 


if openai_api_key:
    print('Yesss, OPENAI_API_KEY loaded.') 
else:
    st.error("Error: OPENAI_API_KEY environment variable not set. Please set it in your .env file.")
    st.stop() 

if not groq_api_key:
    st.error("Error: HR_CHATBOT_GROQ_KEY environment variable not set. Please set it in your .env file.")
    st.stop()


st.title("HR CHATBOT")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    If the answer is not available in the provided context, politely state that you don't have enough information.

    <context>
    {context}
    </context>

    Question: {input}
    """
)


def vector_embedding():

    if "vectors" not in st.session_state or st.session_state.vectors is None:
        st.info("Initializing AI components and processing HR documents...")
        with st.spinner("Please wait, loading and processing documents... This might take a moment."):
            try:
                st.session_state.embeddings = OpenAIEmbeddings(api_key=os.getenv('HR_CHATBOT_OPENAI_KEY'))
                print('embeddings done')
                hr_docs_path = '/policy_docs'
                print('loading docs')
                if not os.path.isdir(hr_docs_path):
                    st.error(f"Error: Document folder '{hr_docs_path}' not found. Please check the path.")

                st.session_state.loader = PyPDFDirectoryLoader(hr_docs_path)
                st.session_state.docs = st.session_state.loader.load()

                if not st.session_state.docs:
                    st.warning("No PDF documents found in the specified directory. Please check the folder and its contents.")
                    return

                st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

                st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)

                if not st.session_state.final_documents:
                    st.warning("No text chunks were generated from the documents. Documents might be empty or formatting issue.")
                    return

                st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
                st.success("Documents processed and Vector Store DB is ready!")

            except Exception as e:
                st.error(f"An error occurred during vector embedding setup: {e}")
                st.write("Please check your API keys, document path, and network connection. Also ensure `PyPDFDirectoryLoader` can access and read the PDFs.")

                for key in ["embeddings", "loader", "docs", "text_splitter", "final_documents", "vectors"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.session_state.vectors = None 

if 'vectors' not in st.session_state:
    st.session_state.vectors = None 


if st.button("Load and Process HR Documents"):
    vector_embedding()


if st.session_state.vectors:
    st.success("Vector Store DB is ready for queries.")
else:
    st.warning("Vector Store DB is not yet loaded. Click 'Load and Process HR Documents'.")



prompt1 = st.text_input("Enter Your Question about HR Documents:")

if prompt1:
    if st.session_state.vectors: 
        with st.spinner("Generating response..."):
            try:
                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = st.session_state.vectors.as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)

                start_time = time.process_time() # Use start_time for clarity
                response = retrieval_chain.invoke({'input': prompt1})
                end_time = time.process_time()
                
                print(f"Response time: {end_time - start_time:.2f} seconds") # Formatted print
                st.write(response['answer'])

                # With a streamlit expander for context
                with st.expander("Document Context (Click to expand)"):
                    if response.get("context"): # Ensure context exists
                        for i, doc in enumerate(response["context"]):
                            st.write(f"**Chunk {i+1}**")
                            st.write(doc.page_content)
                            st.write("---")
                    else:
                        st.write("No relevant context found for this query.")

            except Exception as e:
                st.error(f"An error occurred while retrieving/generating response: {e}")
                st.write("Please ensure all components are initialized and your query is valid.")
    else:
        st.warning("Please load and process the HR documents first by clicking the button above.")
