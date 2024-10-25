import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from typing import List, Optional

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

@st.cache_data
def get_pdf_text(pdf_doc) -> str:
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

@st.cache_data
def get_text_chunks(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    return splitter.split_text(text)

@st.cache_resource
def get_vector_store(chunks: List[str]) -> FAISS:
    embeddings = HuggingFaceEmbeddings()
    return FAISS.from_texts(chunks, embedding=embeddings)

@st.cache_data
def generate_summary(text: str) -> str:
    prompt = f"""
    You are an AI assistant specializing in creating detailed summaries of academic documents for literature reviews. 
    Your task is to summarize the document following these guidelines:

    1. Identify the main theories or concepts discussed.
    2. Summarize the key findings from relevant studies.
    3. Highlight areas of agreement or consensus in the research.
    4. Summarize the methodologies used in the research.
    5. Provide an overview of the potential implications of the research.
    6. Suggest possible directions for future research based on the current literature.
    7. If there is any architecture used then explain the architecture of model stepwise.
    
    Additionally, provide detailed explanations of the mathematical aspects of the research paper:
    
    8. Describe and explain the key mathematical models, theorems, or equations used in the paper.
    
    Document text:
    {text}


    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text.strip()

@st.cache_resource
def get_qa_chain():
    prompt_template = """
    You are an AI research assistant. Use the provided context from research papers to answer the question as accurately as possible. If the answer is not available in the context, respond with, "The information is not available in the provided context."

    Context: {context}
    Question: {question}
    Answer:
    """
 
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)

def process_document(pdf_doc) -> None:
    with st.spinner("Your Research Paper is getting processed..."):
        st.session_state.raw_text = get_pdf_text(pdf_doc)
        text_chunks = get_text_chunks(st.session_state.raw_text)
        st.session_state.vector_store = get_vector_store(text_chunks)
    st.success("Document processed successfully!")

def generate_document_summary() -> None:
    with st.spinner("Your Summary is getting generated..."):
        st.session_state.summary = generate_summary(st.session_state.raw_text)
    st.success("Summary generated!")

def answer_question(question: str) -> Optional[str]:
    if st.session_state.vector_store is None:
        st.error("Please upload a document first.")
        return None
    
    with st.spinner("Thinking..."):
        docs = st.session_state.vector_store.similarity_search(question)
        chain = get_qa_chain()
        response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    return response["output_text"]

def main():
    st.title("📄 Understand Any Research Paper")
    
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'raw_text' not in st.session_state:
        st.session_state.raw_text = ""
    if 'summary' not in st.session_state:
        st.session_state.summary = ""

    with st.sidebar:
        st.header("Document Upload")
        pdf_doc = st.file_uploader("Upload your Research Documentation (PDF)", type="pdf")
        
        if pdf_doc:
            process_document(pdf_doc)
            if st.button("Generate Document Summary"):
                generate_document_summary()

    if st.session_state.summary:
        st.header("Document Summary (Literature Review Style)")
        st.write(st.session_state.summary)
    
    st.header("Ask research-related questions about the document")
    question = st.text_input("Enter your question:")
    
    if question:
        answer = answer_question(question)
        if answer:
            st.subheader("Answer:")
            st.write(answer)

if __name__ == "__main__":
    main()
