import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
import os
from dotenv import load_dotenv
from google.api_core.exceptions import NotFound

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="Document Genie", layout="wide")
st.title("Document Genie 🧞")
st.write("Upload your PDF files and ask any question related to them.")

def show_api_key_error():
    st.error("Google API Key not found. Set GOOGLE_API_KEY in your .env file and restart the app.")
    st.info("Visit [Google AI Studio](https://aistudio.google.com/app/apikey) to generate an API key.")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content: text += content
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    if not api_key:
        show_api_key_error()
        return

    try:
        embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001', google_api_key=api_key)
        vector_store = FAISS.from_texts(text_chunks, embeddings)
        vector_store.save_local("faiss_index")
        st.success("Vector store created and saved!")
    except Exception as e:
        st.error(f"Error creating vector store: {e}. Ensure API key is valid and 'embedding-001' model is accessible.")
        st.info("Check Google AI Studio/Cloud Console for API key and model permissions.")

def get_conversational_chain():
    if not api_key:
        show_api_key_error()
        return None

    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not present in the provided context, say: "Answer is not available in the context."
    Do not make up answers.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    try:
        model = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=0.3, google_api_key=api_key)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        return load_qa_chain(model, chain_type="stuff", prompt=prompt)
    except NotFound as e:
        st.error(f"Model Not Found Error: {e}. 'gemini-2.0-flash' might not be available or accessible.")
        st.markdown("""
        **Please verify:**
        1.  **API Key Validity:** `GOOGLE_API_KEY` in `.env` is correct.
        2.  **Gemini API Enabled:** In Google Cloud, "Gemini API" or "Generative Language API" is enabled.
        3.  **Model Availability:** Check [Google AI Studio](https://aistudio.google.com/app/apikey) for models in your region.
        4.  **Permissions:** API key has access to specified models.
        """)
        return None
    except Exception as e:
        st.error(f"Error initializing model: {e}")
        return None

def handle_user_input(user_question):
    index_path = "faiss_index"

    if not os.path.exists(index_path):
        st.error("No vector store found. Please upload and process PDFs first.")
        return
    if not api_key:
        show_api_key_error()
        return

    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        new_db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    except NotFound as e:
        st.error(f"Embedding Model Not Found: {e}. 'embedding-001' might not be available.")
        st.info("Verify your API key and model permissions.")
        return
    except Exception as e:
        st.error(f"Error loading vector store/embeddings: {e}. Try re-processing PDFs.")
        return

    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    if chain:
        with st.spinner("Generating answer..."):
            try:
                response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
                st.write("Answer:", response["output_text"])
            except NotFound as e:
                st.error(f"Model Not Found during QA: {e}. Model might be unavailable. Re-check settings.")
            except Exception as e:
                st.error(f"Error generating answer: {e}")
    else:
        st.error("Could not initialize conversational chain. Check previous error messages.")

def main():
    with st.sidebar:
        st.header("Upload PDFs")
        pdf_docs = st.file_uploader("Upload PDF files", accept_multiple_files=True)
        if st.button("Process PDFs"):
            if not api_key:
                show_api_key_error()
                return
            if not pdf_docs:
                st.error("Please upload at least one PDF file.")
                return
            with st.spinner("Processing PDFs and creating vector store..."):
                raw_text = get_pdf_text(pdf_docs)
                if raw_text:
                    get_vector_store(get_text_chunks(raw_text))
                else:
                    st.warning("No text extracted from PDFs. Ensure they contain selectable text.")
                    st.info("For image-based PDFs, an OCR step might be needed.")

    if os.path.exists("faiss_index"):
        user_question = st.text_input("Ask a question about your uploaded PDFs:")
        if user_question:
            handle_user_input(user_question)
    else:
        st.info("Upload and process your PDF documents in the sidebar to start asking questions.")
        if not api_key:
            st.warning("No API key detected.")
            st.markdown("For more help on setting up your API key, visit: [Google AI Studio](https://aistudio.google.com/app/apikey)")

if __name__ == "__main__":
    main()
