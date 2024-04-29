## PDF Mastermind: Unleash AI Power for Document Management!📚💡
PDF Mastermind is an innovative tool that harnesses the power of AI to manage, search, summarize, and even engage in conversations about PDF documents.Leveraging cutting-edge AI technologies, PDF Mastermind aims to streamline the process of handling PDF files, making information retrieval and analysis hassle-free. Whether you're a student, researcher, or professional, PDF Mastermind is here to revolutionize how you interact with PDFs.This project utilizes various open-source libraries and AI models, contributing to its functionality and effectiveness.

## Prerequisites
Before you begin, ensure you have met the following requirements:

* Python 3.7+ installed on your system.
* A valid Google API Key. You can obtain one from the [Google AI Studio](https://aistudio.google.com/app/apikey)
* Google Cloud SDK installed. You can download it from [Google Cloud SDK](https://cloud.google.com/sdk?hl=en)
* Access to Vertex AI services for advanced AI capabilities.

## Problem Description
1.	Managing and extracting insights from large volumes of PDF documents is tedious and time-consuming.
2.	Traditional methods lack efficiency and intuitive interaction with PDF content.
3.	Challenges faced by professionals across various domains, including researchers, educators, and knowledge workers.

## Solution Provided
1.	PDF Mastermind automates PDF management tasks.
2.	Users can upload PDF files and extract text effortlessly.
3.	Search indexes are created to facilitate efficient document retrieval.
4.	Document summaries are generated to provide quick insights.
5.	Conversational interactions enable users to obtain relevant information seamlessly.
6.	Harnesses AI and natural language processing to streamline access, analysis, and sharing of knowledge from PDF documents.

## Features
* PDF Text Extraction: The application can extract text from uploaded PDF documents, enabling further analysis and processing.
* Document Summarization: PDF Mastermind provides a summarization feature that condenses lengthy documents into concise summaries, facilitating quick understanding of the document's key points.
* Keyword-Based Search: Users can search through uploaded PDF documents using keywords or phrases. The application employs advanced indexing and search algorithms to retrieve relevant documents efficiently.
* Conversational AI Interaction: PDF Mastermind incorporates a conversational AI component, allowing users to interact with the system in natural language. Users can ask questions or seek information related to the uploaded PDF documents, and the AI assistant provides relevant responses based on the document content.

## Components:
The application consists of several key components:
* Text Extraction and Processing: Utilizes PyPDF2 library for text extraction from PDF documents. The extracted text is then processed and split into manageable chunks for further analysis.
* Document Summarization: Implements the Sumy library for document summarization using Latent Semantic Analysis (LSA) technique. This component generates concise summaries of PDF documents to aid in quick comprehension.
* Keyword-Based Search: Employs the Whoosh library to create a search index for PDF documents. This index enables fast and accurate keyword-based search functionality.
* Conversational AI Integration: Integrates with Google Generative AI for conversational capabilities. The application uses pre-trained models to understand user queries and provide context-aware responses based on the content of uploaded PDF documents.
* User Interface: Built using Streamlit, the application provides an intuitive user interface for uploading PDF documents, interacting with the AI assistant, performing searches, and accessing document summaries.

## Additional Notes:
* The application requires authentication with Google Cloud Platform to access AI services.
* Environment variables such as the Google API key need to be configured for proper functioning.
* Users can clear the chat history using the provided button in the sidebar.

## Getting Started
Follow these steps to set up and run the project on your local machine.

1. Install the required packages:
```
pip install -r requirements.txt
```

2. Create a .env file in the project root directory and add the following line:

   Once you have the API Key you can add it in the ```.env.example``` file and rename it ```.env```.
```
GOOGLE_API_KEY=<your_google_api_key>
```

## How to run the ChatPDF (GUI mode)

Here the instructions to run LLM ChatBOT in GUI mode:

1. Git clone the repository on your local machine:
  ```
  git clone https://github.com/kittu-122/PDF-MasterMind.git
  cd pdfmastermind
  ```

2. Create a Python Virtual environment in your current folder so that you don't corrupt the global python environment creating conflicts with other python applications:
  ```
  python -m venv pdf
  ```

3. Activate the Python virtual environment:
  ```
  pdf/bin/activate
  ```

4. Install the Python libraries in your Python virtual environment:
  ```
  pip install -r requirements.txt
  ```

5. Run the PDF-MasterMind streamlit app:
  ```
  streamlit run pdfmastermind.py
  ```

## Usage:
To use PDF Mastermind, follow these steps:
* Upload PDF Files: Choose the "Upload PDF Files" option and select one or more PDF documents for processing.
* Process PDFs: Click the "Process PDF" button to extract text, create document summaries, and index the documents for search.
* Search PDFs: Use the "Search PDFs" option to enter keywords or phrases and search for relevant documents among the uploaded PDFs.
* View Summaries: Select the "Summary" option to view summaries of uploaded PDF documents, generated by the application.
* Interact with AI Assistant: Engage in conversation with the AI assistant by typing questions or prompts related to the uploaded PDF documents. The assistant will provide informative responses based on the document content.

## Dependencies:
* PyPDF2
* langchain
* Streamlit
* Sumy
* Whoosh
* Google Cloud AI Platform

## Accessing the Website
You can access PDF Mastermind by visiting [PDF-MasterMind](https://pdf-mastermind-wjp8hbpdd3qfygsolmysqe.streamlit.app/)

## Contributing
Contributions to the project are welcome! Feel free to submit pull requests, report issues, or suggest enhancements to improve PDF Mastermind.

**Thank you for choosing this project. Hoping that this project proves useful and delivers a seamless experience for your needs!**
