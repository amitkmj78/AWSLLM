import json
import os
import sys
import boto3
import streamlit as st

## We will be suing Titan Embeddings Model To generate Embedding

from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
AWS_KEY = st.sidebar.text_input(label="AWS Key",placeholder="enter your access key", type="password")
AWS_SECRET_KEY = st.sidebar.text_input(label="AWS secret",placeholder="enter your secret key",type="password")
## Data Ingestion

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Vector Embedding And Vector Store

from langchain.vectorstores import FAISS

## LLm Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

## Bedrock Clients
Bedrock_Client=boto3.client(service_name="bedrock-runtime", aws_access_key_id=AWS_KEY, 
aws_secret_access_key=AWS_SECRET_KEY, region_name='us-east-1')
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=Bedrock_Client)


#Data ingestion function
  
def DataIngestion():
    loader= PyPDFDirectoryLoader("data")
    document=loader.load()
    text_splitter= RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs= text_splitter.split_documents(document)
    return docs

#created Vector store
def get_vector_store(docs):
    vectorstore_faiss=FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    vectorstore_faiss.save_local("faiss_index")



#Claude Model
def get_claude_llm():
    ##create the Anthropic Model
       llm = Bedrock(
        model_id="anthropic.claude-v2:1",
        client=Bedrock_Client,
        model_kwargs={"max_tokens_to_sample": 300}
      )
       return llm
    

def mistral():
    ##create the Anthropic Model
    llm=Bedrock(model_id="mistral.mistral-small-2402-v1:0",client=Bedrock_Client,
                model_kwargs={'max_tokens':200})
    
    return llm


def get_llama3_8_llm():
    ##create the Anthropic Model
    llm=Bedrock(model_id="meta.llama3-8b-instruct-v1:0",client=Bedrock_Client,
                model_kwargs={'max_gen_len':512})
    
    return llm


#Prompt template
prompt_template = """

Human: Use the following pieces of context to provide a 
concise answer to the question at the end but usse atleast summarize with 
more than 250 words with detailed explaantions. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


def get_response_llm(llm,vectorstore_faiss,query):
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore_faiss.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
    answer=qa({"query":query})
    return answer['result']



def main():
    #st.set_page_config("Chat PDF")
    uploaded_file=st.file_uploader( label="Please upload data in CSV, Excel and PDF to convert into Context", type=["csv", "xlsx", "PDF"])
    st.write(uploaded_file)

    if uploaded_file is not None:
    # Save the uploaded file in the 'fdata' folder
        file_path = os.path.join("data", uploaded_file.name)
        if file_path is not None:
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())
    
                st.success(f"File saved successfully at: {file_path}")
    
    st.header("Chat with your Data using AWS Bedrock💁")

   
    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Update Or Create Vector Store:")
        
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = DataIngestion()
                get_vector_store(docs)
                st.success("Done")

    if st.button("Claude Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings,allow_dangerous_deserialization=True)
            llm=get_claude_llm()
            
            #faiss_index = get_vector_store(docs)
            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")

    if st.button("Mistral Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings,allow_dangerous_deserialization=True)
            llm=mistral()
            
            #faiss_index = get_vector_store(docs)
            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")


            #get_llama3_2_llm
    if st.button("Lama Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings,allow_dangerous_deserialization=True)
            llm=get_llama3_8_llm()
            
            #faiss_index = get_vector_store(docs)
            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")

if __name__ == "__main__":
    main()














