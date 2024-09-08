# PDF Chat with Gemini AI

This Streamlit application allows users to chat with their PDF documents using Google's Gemini AI. It extracts text from uploaded PDFs, processes the content, and uses it to answer user queries.

## Features

- PDF text extraction
- Text processing and chunking
- Vector store creation using FAISS
- Question answering using Google's Gemini AI
- User-friendly Streamlit interface

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.7 or higher
- A Google API key for Gemini AI

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/your-username/pdf-chat-gemini.git
   cd pdf-chat-gemini
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root and add your Google API key:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and go to `http://localhost:8501`.

3. Use the sidebar to upload PDF files and process them.

4. Once processed, you can ask questions about the content of the PDFs in the main chat interface.

## How it Works

1. The app extracts text from uploaded PDF files.
2. The extracted text is split into smaller chunks.
3. These chunks are used to create a FAISS vector store.
4. When a user asks a question, the app searches the vector store for relevant information.
5. The relevant information is then passed to the Gemini AI model to generate a response.

## Security Note

This application uses `allow_dangerous_deserialization=True` when loading the FAISS index. This is safe as long as you trust the source of your vector store (i.e., you created it yourself and no one else has modified it). However, be cautious when using vector stores from untrusted sources.

## Contributing

Contributions to this project are welcome. Please follow these steps:

1. Fork the repository.
2. Create a new branch: `git checkout -b <branch_name>`.
3. Make your changes and commit them: `git commit -m '<commit_message>'`
4. Push to the original branch: `git push origin <project_name>/<location>`
5. Create the pull request.

Alternatively, see the GitHub documentation on [creating a pull request](https://help.github.com/articles/creating-a-pull-request/).

## License

This project uses the following license: [MIT License](https://opensource.org/licenses/MIT).

## Contact

If you want to contact me, you can reach me at <your_email@example.com>.

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [LangChain](https://github.com/hwchase17/langchain)
- [Google Generative AI](https://ai.google/discover/generativeai/)
- [FAISS](https://github.com/facebookresearch/faiss)