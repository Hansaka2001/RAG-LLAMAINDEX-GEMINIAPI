import os
import logging
from dotenv import load_dotenv
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
)
from llama_index.llms import LangChainLLM
from langchain.llms import GooglePalm
from llama_index.node_parser import SimpleNodeParser
from config import PDF_DIRECTORY, LLM_TEMPERATURE, CHUNK_SIZE, CHUNK_OVERLAP

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Set up the Gemini LLM
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
llm = LangChainLLM(llm=GooglePalm(temperature=LLM_TEMPERATURE))

# Create a ServiceContext with the LLM
service_context = ServiceContext.from_defaults(llm=llm)


def create_or_load_index(pdf_directory=PDF_DIRECTORY, persist_dir="./storage"):
    if os.path.exists(persist_dir):
        logging.info(f"Loading existing index from {persist_dir}")
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        return load_index_from_storage(storage_context)

    logging.info(f"Creating new index from PDF documents in {pdf_directory}")
    try:
        documents = SimpleDirectoryReader(pdf_directory, file_extractor={
            ".pdf": "PyPDFFileReader"
        }).load_data()

        parser = SimpleNodeParser.from_defaults(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        nodes = parser.get_nodes_from_documents(documents)

        index = VectorStoreIndex(nodes, service_context=service_context)

        index.storage_context.persist(persist_dir=persist_dir)

        logging.info(f"Index created and saved to {persist_dir}")
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
    index = create_or_load_index()

    query = "What is the main topic of the document?"
    response = query_index(index, query)

    print(f"Query: {query}")
    print(f"Response: {response}")
