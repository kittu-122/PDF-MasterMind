import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import streamlit as st
from dotenv import load_dotenv
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID
from whoosh.query import Every
from whoosh.qparser import QueryParser
from google.cloud import aiplatform

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# Authenticate with Vertex AI
aiplatform.init(project="pdf-mastermind", location="us-central1")

# Function to extract text from PDF documents
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            try:
                text += page.extract_text()
            except Exception as e:
                st.warning(f"Error extracting text from PDF: {e}")
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create vector store for text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(api_key=google_api_key, model="models/embedding-001")
    try:
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
    except Exception as e:
        st.warning(f"Error creating FAISS index: {e}")

# Function to load conversational AI chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 0,
        "max_output_tokens": 8192,
    }
    runnable_model = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro-latest",
        generation_config=generation_config,
    )
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm=runnable_model, chain_type="stuff", prompt=prompt)
    return chain

# Function to summarize document
def summarize_document(text, summary_length=3):
    LANGUAGE = "english"
    tokenizer = Tokenizer(LANGUAGE)
    stemmer = Stemmer(LANGUAGE)
    parser = PlaintextParser.from_string(text, tokenizer)
    summarizer = LsaSummarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)
    summary = []
    for sentence in summarizer(parser.document, summary_length):
        summary.append(str(sentence))
    return summary

# Function to create search index for PDF documents
def create_search_index(pdf_docs):
    schema = Schema(title=TEXT(stored=True), content=TEXT)
    if not os.path.exists("index"):
        os.mkdir("index")
    ix = create_in("index", schema)
    writer = ix.writer()
    for uploaded_file in pdf_docs:
        reader = PdfReader(uploaded_file)
        title = uploaded_file.name
        content = ""
        for page in reader.pages:
            content += page.extract_text()
        writer.add_document(title=title, content=content)
    writer.commit()

# Function to search documents based on query
def search_documents(query):
    ix = open_dir("index")
    results = []
    found = False
    with ix.searcher() as searcher:
        query_parser = QueryParser("content", ix.schema)
        parsed_query = query_parser.parse(query)
        hits = searcher.search(parsed_query)
        for hit in hits:
            found = True
            results.append({"title": hit["title"], "score": hit.score})
    if not found:
        google_search_link = f"https://www.google.com/search?q={query.replace(' ', '+')}"
        return None, google_search_link
    return results, None

# Function to display search results
def display_search_results(results):
    if results:
        st.write("Search Results:")
        for result in results:
            st.write(f"- {result['title']} (Score: {result['score']})")
    else:
        st.write("No results found.")

# Function to clear chat history
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Provide PDFs for Unique Rewrite"}]

# Function to display PDF summaries
def display_pdf_summaries(pdf_docs):
    for pdf in pdf_docs:
        with st.expander(f"Summary for {pdf.name}"):
            summary_length = st.slider("Summary Length", key=f"summary_slider_{pdf.name}", min_value=1, max_value=10, value=3)
            summary = summarize_document(get_pdf_text([pdf]), summary_length)
            
            st.markdown("### Summary:")
            for i, sentence in enumerate(summary, start=1):
                st.write(f"- {sentence}")

# Function to handle user input
def user_input(user_question, pdf_docs):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(api_key=google_api_key, model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True) 
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain(
            {"input_documents": docs, "question": user_question}, return_only_outputs=True, )
        return response
    except Exception as e:
        st.error(f"Error processing user input: {e}")
        return None

# Main function
def main():
    # Check if 'pdf_docs' is not already in session_state
    if 'pdf_docs' not in st.session_state:
        st.session_state.pdf_docs = None  # Initialize pdf_docs attribute to None or an empty list
    # Check if 'pdf_docs' is None or not iterable
    if st.session_state.pdf_docs is None or not isinstance(st.session_state.pdf_docs, list):
        st.session_state.pdf_docs = []  # Initialize pdf_docs as an empty list

    st.set_page_config(
        page_title="PDF Mastermind",
        page_icon="🤖"
    )

    st.sidebar.title("PDF Mastermind:")
    option = st.sidebar.radio(
        "Select an option:",
        ("Upload PDF Files", "Search PDFs", "Summary")
    )

    search_section = st.empty()
    summary_section = st.empty()

    with st.sidebar:
        if option == "Upload PDF Files":
            st.write("📁 Upload PDF Files and Unleash the Power of AI")
            pdf_docs = st.file_uploader(
                "", accept_multiple_files=True)

            if st.button("Process PDF"):
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    create_search_index(pdf_docs)
                    st.success("PDFs Processed Successfully")
                    st.session_state.pdf_docs = pdf_docs
                    st.session_state.show_summary = False

    if option == "Upload PDF Files" and "pdf_docs" in st.session_state:
        st.sidebar.write("Uploaded PDFs:")
        for pdf in st.session_state.pdf_docs:
            st.sidebar.write(pdf.name)

    if option == "Search PDFs":
        summary_section.empty()
        st.title("Search PDFs")
        query = st.text_input("Enter your search query:")
        if st.button("Search"):
            if st.session_state.pdf_docs:
                results, google_search_link = search_documents(query)
                if results:
                    display_search_results(results)
                else:
                    st.write("Keyword couldn't be extracted from any of the PDFs.")
                    if google_search_link:
                        st.write(f"You can try searching on Google: [{query}]({google_search_link})")
                search_section.write("")

    elif option == "Summary":
        search_section.empty()
        st.title("PDF Summaries")
        if st.session_state.pdf_docs:
            display_pdf_summaries(st.session_state.pdf_docs)
            summary_section.write("")

    st.title("Unlock PDF Knowledge 💡")
    st.write("Welcome to the future of PDF management!")

    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Provide PDFs for Unique Rewrite"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Extracting..."):
                response = user_input(prompt, st.session_state.pdf_docs)
                placeholder = st.empty()
                full_response = ''
                for item in response['output_text']:
                    full_response += item
                    placeholder.markdown(full_response)

        if response is not None:
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)

if __name__ == "__main__":
    main()
