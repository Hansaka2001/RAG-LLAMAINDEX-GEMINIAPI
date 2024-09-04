import os
from dotenv import load_dotenv
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import LangChainLLM
from langchain.llms import GooglePalm
from llama_index.node_parser import SimpleNodeParser

# Load environment variables
load_dotenv()

# Set up the Gemini LLM
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
llm = LangChainLLM(llm=GooglePalm(temperature=0.1))

# Create a ServiceContext with the LLM
service_context = ServiceContext.from_defaults(llm=llm)

def create_index(pdf_directory):
    # Load PDF documents from the specified directory
    documents = SimpleDirectoryReader(pdf_directory, file_extractor={
        ".pdf": "PyPDFFileReader"
    }).load_data()
    
    # Create a parser for splitting the documents into smaller chunks
    parser = SimpleNodeParser.from_defaults(chunk_size=1000, chunk_overlap=200)
    
    # Parse the documents into nodes
    nodes = parser.get_nodes_from_documents(documents)
    
    # Create an index from the nodes
    index = VectorStoreIndex(nodes, service_context=service_context)
    
    return index

def query_index(index, query):
    # Create a query engine from the index
    query_engine = index.as_query_engine()
    
    # Query the index
    response = query_engine.query(query)
    
    return response

if __name__ == "__main__":
    # Specify the directory containing your PDF files
    pdf_directory = "pdfs"
    
    # Create the index
    print("Creating index from PDF documents...")
    index = create_index(pdf_directory)
    print("Index created successfully.")
    
    # Example query
    query = "What is the main topic of the document?"
    response = query_index(index, query)
    
    print(f"Query: {query}")
    print(f"Response: {response}")