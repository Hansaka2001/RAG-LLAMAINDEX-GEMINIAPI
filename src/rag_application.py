from config import PDF_DIRECTORY, LLM_TEMPERATURE, CHUNK_SIZE, CHUNK_OVERLAP
from llama_index.node_parser import SimpleNodeParser
from langchain.llms import GooglePalm
from llama_index.llms import LangChainLLM
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
)
from dotenv import load_dotenv
import logging
import os
from llama_index.embeddings import HuggingFaceEmbedding
import nltk
nltk.download('punkt')


# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Set up the Gemini LLM
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
llm = LangChainLLM(llm=GooglePalm(temperature=LLM_TEMPERATURE))

# Create a ServiceContext with the LLM
llm = GooglePalm(api_key=os.getenv("GOOGLE_API_KEY"))
embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2")
service_context = ServiceContext.from_defaults(
    llm=llm, embed_model=embed_model)


def create_or_load_index(pdf_directory=PDF_DIRECTORY, persist_dir="./storage"):
    if os.path.exists(persist_dir):
        logging.info(f"Loading existing index from {persist_dir}")
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        return load_index_from_storage(storage_context, service_context=service_context)

    logging.info(f"Creating new index from PDF documents in {pdf_directory}")
    try:
        if not os.path.exists(pdf_directory):
            raise FileNotFoundError(f"Directory not found: {pdf_directory}")

        files = os.listdir(pdf_directory)
        if not files:
            raise ValueError(f"No files found in directory: {pdf_directory}")

        logging.info(f"Found {len(files)} files in {pdf_directory}")
        logging.info(f"Files: {files}")

        documents = SimpleDirectoryReader(pdf_directory).load_data()
        logging.info(f"Loaded {len(documents)} documents")

        index = VectorStoreIndex.from_documents(
            documents, service_context=service_context)

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
