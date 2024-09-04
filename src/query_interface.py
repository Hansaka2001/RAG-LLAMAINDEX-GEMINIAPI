from src.rag_application import create_index, query_index

def main():
    print("Welcome to the PDF Question Answering System!")
    
    # Specify the directory containing your PDF files
    pdf_directory = "pdfs"
    
    print(f"Loading knowledge base from PDF documents in '{pdf_directory}'...")
    index = create_index(pdf_directory)
    print("Knowledge base loaded. You can now ask questions about the PDF content.")

    while True:
        user_query = input("\nEnter your question (or 'quit' to exit): ")
        
        if user_query.lower() == 'quit':
            print("Thank you for using the PDF Question Answering System. Goodbye!")
            break
        
        response = query_index(index, user_query)
        print(f"\nResponse: {response}")

if __name__ == "__main__":
    main()