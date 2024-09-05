import logging
from src.rag_application import create_or_load_index, query_index

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    print("Welcome to the PDF Question Answering System!")
    
    pdf_directory = "pdfs"
    
    try:
        index = create_or_load_index(pdf_directory)
        print("Knowledge base loaded. You can now ask questions about the PDF content.")

        while True:
            user_query = input("\nEnter your question (or 'quit' to exit): ")
            
            if user_query.lower() == 'quit':
                print("Thank you for using the PDF Question Answering System. Goodbye!")
                break
            
            try:
                response = query_index(index, user_query)
                print(f"\nResponse: {response}")
                
                # Simple feedback mechanism
                feedback = input("Was this response helpful? (y/n): ")
                if feedback.lower() == 'n':
                    logging.info(f"Unhelpful response logged for query: {user_query}")
                    print("Thank you for your feedback. We'll work on improving our responses.")
            except Exception as e:
                print(f"An error occurred while processing your query: {str(e)}")
                logging.error(f"Error processing query '{user_query}': {str(e)}")

    except Exception as e:
        print(f"An error occurred while setting up the system: {str(e)}")
        logging.error(f"Error setting up the system: {str(e)}")

if __name__ == "__main__":
    main()