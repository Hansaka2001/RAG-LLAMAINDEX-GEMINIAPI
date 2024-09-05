import os
import logging
from dotenv import load_dotenv
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import LangChainLLM
from langchain.llms import GooglePalm
from llama_index.node_parser import SimpleNodeParser
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores import SimpleVectorStore
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Set up the Gemini LLM
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
llm = LangChainLLM(llm=GooglePalm(temperature=0.1))

# Create a ServiceContext with the LLM
service_context = ServiceContext.from_defaults(llm=llm)

def create_or_load_index(pdf_directory, index_file="index.pickle"):
    if os.path.exists(index_file):
        logging.info(f"Loading existing index from {index_file}")
        with open(index_file, "rb") as f:
            return pickle.load(f)
    
    logging.info(f"Creating new index from PDF documents in {pdf_directory}")
    try:
        documents = SimpleDirectoryReader(pdf_directory, file_extractor={
            ".pdf": "PyPDFFileReader"
        }).load_data()
        
        parser = SimpleNodeParser.from_defaults(chunk_size=1000, chunk_overlap=200)
        nodes = parser.get_nodes_from_documents(documents)
        
        index = VectorStoreIndex(nodes, service_context=service_context)
        
        with open(index_file, "wb") as f:
            pickle.dump(index, f)
        
        logging.info(f"Index created and saved to {index_file}")
        return index
    except Exception as e:
        logging.error(f"Error creating index: {str(e)}")
        raise

def query_index(index, query):
    try:
        query_engine = index.as_query_engine()
        response = query_engine.query(query)
        return response
    except Exception as e:
        logging.error(f"Error querying index: {str(e)}")
        raise

if __name__ == "__main__":
    pdf_directory = "pdfs"
    index = create_or_load_index(pdf_directory)
    
    query = "What is the main topic of the document?"
    response = query_index(index, query)
    
    print(f"Query: {query}")
    print(f"Response: {response}")