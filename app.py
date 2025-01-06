import json
import os
import boto3
import streamlit as st
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


# Sidebar for AWS credentials
AWS_KEY = st.sidebar.text_input(label="AWS Key", placeholder="Enter your access key", type="password")
AWS_SECRET_KEY = st.sidebar.text_input(label="AWS Secret", placeholder="Enter your secret key", type="password")


# Check AWS connection
def check_connection():
    try:
        st.info("Trying to connect...")
        if not AWS_KEY or not AWS_SECRET_KEY:
            st.error("Not connected to AWS. Please provide valid keys.")
        else:
            st.success("Connected to AWS successfully.")
    except Exception as e:
        st.error(f"Error in connecting: {e}")


# Data ingestion function
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    return docs


# Create FAISS vector store
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")


# Define Bedrock LLM models
def get_claude_llm():
    return Bedrock(
        model_id="anthropic.claude-v2:1",
        client=Bedrock_Client,
        model_kwargs={"max_tokens_to_sample": 300}
    )


def mistral():
    return Bedrock(
        model_id="mistral.mistral-small-2402-v1:0",
        client=Bedrock_Client,
        model_kwargs={"max_tokens": 200}
    )


def get_llama3_8_llm():
    return Bedrock(
        model_id="meta.llama3-8b-instruct-v1:0",
        client=Bedrock_Client,
        model_kwargs={"max_gen_len": 512}
    )


# Prompt template
prompt_template = """
Human: Use the following pieces of context to provide a 
concise answer to the question at the end. Summarize with 
more than 250 words and provide detailed explanations. If you don't know the answer, 
just say that you don't know; don't try to make up an answer.

<context>
{context}
</context>

Question: {question}

Assistant:
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])


# Generate response using LLM
def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']


# Main application
def main():
    #st.set_page_config(page_title="Chat with Your Data using AWS Bedrock", layout="wide")

    if st.sidebar.button("Check Connection"):
        with st.spinner("Processing..."):
            check_connection()

    uploaded_file = st.file_uploader(
        label="Upload data (CSV, Excel, or PDF)", type=["csv", "xlsx", "pdf"]
    )

    if uploaded_file:
        file_path = os.path.join("data", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        st.success(f"File saved successfully at: {file_path}")

    st.header("Chat with Your Data using AWS Bedrock 💁")
    user_question = st.text_input("Ask a Question from the Uploaded Files")

    with st.sidebar:
        st.title("Vector Store Management")
        if st.button("Update Vectors"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Vector store updated successfully.")

    if st.button("Claude Output"):
        check_connection()
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm = get_claude_llm()
            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success("Response generated successfully.")

    if st.button("Mistral Output"):
        check_connection()
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm = mistral()
            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success("Response generated successfully.")

    if st.button("Llama Output"):
        check_connection()
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm = get_llama3_8_llm()
            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success("Response generated successfully.")


# Initialize Bedrock client
Bedrock_Client = boto3.client(
    service_name="bedrock-runtime",
    aws_access_key_id=AWS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name="us-east-1"
)

bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=Bedrock_Client)

if __name__ == "__main__":
    main()
