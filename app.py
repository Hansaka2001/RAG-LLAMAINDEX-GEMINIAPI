import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables and configure Google Generative AI
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdfs):
    """
    Extract text from uploaded PDF files.
    
    Args:
    pdfs (list): List of uploaded PDF files.
    
    Returns:
    str: Extracted text from all PDFs.
    """
    text = ""
    for pdf in pdfs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """
    Split the extracted text into chunks.
    
    Args:
    text (str): The text to be split.
    
    Returns:
    list: List of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    """
    Create and save a vector store from text chunks.
    
    Args:
    text_chunks (list): List of text chunks.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    """
    Create a conversational chain for question answering.
    
    Returns:
    Chain: A question-answering chain.
    """
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    """
    Process user input and generate a response.
    
    Args:
    user_question (str): The user's question.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Warning: Only set this to True if you trust the source of your vector store
    allow_dangerous = True
    
    if allow_dangerous:
        st.warning("Warning: Allowing dangerous deserialization. Only do this if you trust the source of your vector store.")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    else:
        st.error("Cannot load the vector store due to security restrictions.")
        return
    
    docs = new_db.similarity_search(user_question)
    
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    print(response)
    st.write("Reply: ", response["output_text"])

def main():
    """
    Main function to run the Streamlit app.
    """
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")
    
    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        user_input(user_question)
    
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button",
            accept_multiple_files=True
        )
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()